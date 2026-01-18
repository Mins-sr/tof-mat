"""
Two-Stream U-Net for Material Segmentation

Geometry Stream: Depth + Normals
Photometric Stream: IR + Raw (Q1-Q4)
Decoder: Up-sampling with skip connections to output segmentation masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Optional, Tuple, List

from src.models.two_stream_cnn import ResNetEncoder, AttentionGate


class DecoderBlock(nn.Module):
    """
    U-Netのデコーダブロック
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # 偶数解像度でない場合のパディング調整
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TwoStreamUNet(nn.Module):
    """
    Two-Stream U-Net for Pixel-wise Material Segmentation
    """
    def __init__(
        self,
        num_classes: int = 5,
        use_attention: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Geometry Encoder (Depth: 1, Normals: 3 -> 4ch)
        self.geo_enc = ResNetEncoder(in_channels=4)
        
        # Photometric Encoder (IR: 1, Raw: 4 -> 5ch)
        self.photo_enc = ResNetEncoder(in_channels=5)
        
        # Fusion at the bottleneck
        if use_attention:
            self.fusion = AttentionGate(512)
        else:
            self.fusion = lambda g, p: torch.cat([g, p], dim=1)
            # fusion_out_channels would be 1024
            self.bottleneck_conv = nn.Conv2d(1024, 512, kernel_size=1)
        
        self.use_attention = use_attention
        
        # Decoder Layers
        # ResNet-18 feature map sizes (for 640x480):
        # adapter: 320x240, 64ch
        # layer1: 160x120, 64ch
        # layer2: 80x60, 128ch
        # layer3: 40x30, 256ch
        # layer4: 20x15, 512ch (bottleneck)
        
        self.dec4 = DecoderBlock(512, 256, 256) # layer4 -> layer3
        self.dec3 = DecoderBlock(256, 128, 128) # layer3 -> layer2
        self.dec2 = DecoderBlock(128, 64, 64)   # layer2 -> layer1
        self.dec1 = DecoderBlock(64, 64, 64)    # layer1 -> adapter
        
        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(
        self,
        depth: torch.Tensor,
        ir: torch.Tensor,
        raw: torch.Tensor,
        normals: torch.Tensor
    ) -> torch.Tensor:
        # Encoder passes with intermediate features for skip connections
        # Geometry
        g_x = self.geo_enc.adapter(torch.cat([depth, normals], dim=1))
        g_s1 = self.geo_enc.layer1(self.geo_enc.maxpool(g_x))
        g_s2 = self.geo_enc.layer2(g_s1)
        g_s3 = self.geo_enc.layer3(g_s2)
        g_feat = self.geo_enc.layer4(g_s3)
        
        # Photometric
        p_x = self.photo_enc.adapter(torch.cat([ir, raw], dim=1))
        p_s1 = self.photo_enc.layer1(self.photo_enc.maxpool(p_x))
        p_s2 = self.photo_enc.layer2(p_s1)
        p_s3 = self.photo_enc.layer3(p_s2)
        p_feat = self.photo_enc.layer4(p_s3)
        
        # Fusion at bottleneck
        if self.use_attention:
            fused = self.fusion(g_feat, p_feat)
        else:
            fused = self.bottleneck_conv(torch.cat([g_feat, p_feat], dim=1))
            
        # Decoder with skip connections (Fusing features from both streams)
        # For simplicity, we add the features from geo and photo streams for skip connections
        x = self.dec4(fused, g_s3 + p_s3)
        x = self.dec3(x, g_s2 + p_s2)
        x = self.dec2(x, g_s1 + p_s1)
        x = self.dec1(x, g_x + p_x)
        
        x = self.final_upsample(x)
        # Ensure exact match to input size if necessary
        if x.shape[2:] != depth.shape[2:]:
            x = F.interpolate(x, size=depth.shape[2:], mode='bilinear', align_corners=True)
            
        logits = self.final_conv(x)
        return logits


if __name__ == '__main__':
    # Test
    model = TwoStreamUNet(num_classes=5)
    bs, h, w = 2, 480, 640
    d = torch.randn(bs, 1, h, w)
    i = torch.randn(bs, 1, h, w)
    r = torch.randn(bs, 4, h, w)
    n = torch.randn(bs, 3, h, w)
    
    out = model(d, i, r, n)
    print(f"Input: {d.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (bs, 5, h, w)
    print("Segmentation model test PASSED")
