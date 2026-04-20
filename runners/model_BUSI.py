import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ---------- Attention ----------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 尺寸对齐（防炸）
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        psi = self.psi(torch.relu(g1 + x1))
        return x * psi


# ---------- Up Block ----------
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.att = AttentionBlock(out_ch, skip_ch, out_ch // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 尺寸对齐（关键）
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=False)

        x2 = self.att(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ---------- 主模型 ----------
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        base = resnet34(weights=ResNet34_Weights.DEFAULT)

        # 修改输入通道（1通道）
        old_weight = base.conv1.weight
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = old_weight.mean(dim=1, keepdim=True)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.layer1 = base.layer1   # 64
        self.layer2 = base.layer2   # 128
        self.layer3 = base.layer3   # 256
        self.layer4 = base.layer4   # 512

        # 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        # Decoder（🔥 正确参数）
        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(128, 64, 64)
        self.up4 = Up(64, 64, 64)

        self.out_seg = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0 = self.layer0(x)   # 64
        x1 = self.layer1(x0)  # 64
        x2 = self.layer2(x1)  # 128
        x3 = self.layer3(x2)  # 256
        x4 = self.layer4(x3)  # 512

        # 分类
        cls = self.pool(x4).view(x.size(0), -1)
        cls = self.fc(cls)

        # 分割
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        seg = self.out_seg(x)

        seg = F.interpolate(seg, size=(256, 256), mode='bilinear', align_corners=False)

        return cls, seg
