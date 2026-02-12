import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisionCNN(nn.Module):
    """Image branch: ResNet18 trunk → projection."""
    def __init__(self, out_dim: int = 128, in_ch: int = 3, train_backbone: bool = False):

        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B,512,1,1]
        if in_ch != 3:
            old = self.backbone[0]
            new_conv = nn.Conv2d(
                in_ch, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None
            )
            # Initialize new conv layer weights
            with torch.no_grad():
                # Copy pretrained weights for first 3 channels
                new_conv.weight[:, :3, :, :] = old.weight
                # Initialize additional channels with zeros or small random values
                if in_ch > 3:
                    new_conv.weight[:, 3:, :, :] = 0.0
                # Copy bias if it exists
                if old.bias is not None:
                    new_conv.bias = old.bias
            self.backbone[0] = new_conv
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)      # [B,512]
        return F.relu(self.proj(h))          # [B,out_dim]

class AudioCNN(nn.Module):
    """Audio branch: small 2D CNN over log-mel spectrograms."""
    def __init__(self, out_dim: int = 128, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)              # [B,M,T] → [B,1,M,T]
        h = self.net(x).flatten(1)           # [B,128]
        return F.relu(self.proj(h))          # [B,out_dim]

class ExplexNet(nn.Module):
    """Explosion-or-Explanation classifier (late fusion)."""
    def __init__(self, vid_dim: int = 128, aud_dim: int = 128, vid_in_ch: int = 9):
        super().__init__()
        self.vision = VisionCNN(out_dim=vid_dim, in_ch=vid_in_ch)
        self.audio  = AudioCNN(out_dim=aud_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vid_dim + aud_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # sigmoid
        )

    def forward(self, img: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        zv = self.vision(img)
        za = self.audio(mel)
        z = torch.cat([zv, za], dim=1)
        return self.classifier(z)  # [B,1] (logits)
