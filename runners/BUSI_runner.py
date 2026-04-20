import os
import cv2
import torch
import numpy as np
from runners.model_BUSI import MultiTaskModel
from runners.config_BUSI import CFG
from pathlib import Path
import os


CLASS_MAP = {
    0: "normal",
    1: "benign",
    2: "malignant"
}

# ===== 全局模型（只加载一次）=====
models = None


def load_models_once():
    global models
    if models is not None:
        return models
    script_dir = Path(__file__).parent.absolute()
    backward_dir = script_dir.parent
    models_base_dir = backward_dir / 'models' / 'BUSI' / 'checkpoints'
    model_paths = [
        str(models_base_dir / f'fold{i}_best.cptk') 
        for i in range(5)
    ]

    models = []
    for path in model_paths:
        model = MultiTaskModel(CFG.num_classes).to(CFG.device)
        model.load_state_dict(torch.load(path, map_location=CFG.device))
        model.eval()
        models.append(model)

    print(f"[BUSI] Loaded {len(models)} models")
    return models


# ===== 对外接口（给FastAPI用）=====
def run_breast(input_path, output_path):
    print("进入 run_breast")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    models = load_models_once()

    img = cv2.imread(input_path, 0)
    orig = img.copy()

    img_resized = cv2.resize(img, (CFG.img_size, CFG.img_size))
    img_resized = img_resized / 255.0

    img_tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float().to(CFG.device)

    cls_preds = []
    seg_preds = []

    with torch.no_grad():
        for model in models:
            cls_pred, seg_pred = model(img_tensor)
            cls_preds.append(torch.softmax(cls_pred, dim=1))
            seg_preds.append(torch.sigmoid(seg_pred))

    cls_pred = torch.mean(torch.stack(cls_preds), dim=0)
    seg_pred = torch.mean(torch.stack(seg_preds), dim=0)

    cls_idx = torch.argmax(cls_pred, dim=1).item()
    cls_name = CLASS_MAP[cls_idx]
    cls_prob = cls_pred[0, cls_idx].item()

    mask = seg_pred[0, 0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    if cls_name != "normal":
        cv2.drawContours(vis, clean_contours, -1, (0, 255, 0), 2)

    text = f"{cls_name} ({cls_prob:.2f})"
    cv2.putText(vis, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    success = cv2.imwrite(output_path, vis)
    print("保存结果:", success)
    if not success:
        raise RuntimeError(f"保存失败: {output_path}")
