import torch

class CFG:
    data_dir = "./data/BUSI/"
    img_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 类别映射
    classes = ["normal", "benign", "malignant"]
    num_classes = 3

    # loss 权重
    lambda_seg = 0.5

    save_dir = "./checkpoints"
    resume = True