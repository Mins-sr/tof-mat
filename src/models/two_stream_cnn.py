"""
Two-Stream CNN for Material Classification

Geometry Stream: Depth + Normals
Photometric Stream: IR + Raw (Q1-Q4)
Fusion: Feature concatenation with optional attention gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Optional, Tuple


class ChannelAdapter(nn.Module):
    """
    入力チャネル数を調整するアダプタ
    ResNet-18は3チャネル入力を想定しているため、任意チャネル数に対応
    """
    
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResNetEncoder(nn.Module):
    """
    ResNet-18ベースの特徴抽出器
    """
    
    def __init__(self, in_channels: int, pretrained: bool = False):
        super().__init__()
        
        # ResNet-18のバックボーン
        base_model = resnet18(pretrained=pretrained)
        
        # 入力チャネルを調整
        self.adapter = ChannelAdapter(in_channels, 64)
        
        # ResNetの各層（最初のconv層は adapter で置き換え）
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1  # 64ch
        self.layer2 = base_model.layer2  # 128ch
        self.layer3 = base_model.layer3  # 256ch
        self.layer4 = base_model.layer4  # 512ch
        
        self.out_channels = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class AttentionGate(nn.Module):
    """
    2つのストリームの特徴を適応的に重み付けするAttention Gate
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self, 
        feat_geo: torch.Tensor, 
        feat_photo: torch.Tensor
    ) -> torch.Tensor:
        # 両ストリームを結合してAttention weightを計算
        combined = torch.cat([feat_geo, feat_photo], dim=1)
        weights = self.attention(combined)
        
        # 重み付け融合
        w_geo = weights[:, 0:1, :, :]
        w_photo = weights[:, 1:2, :, :]
        
        fused = w_geo * feat_geo + w_photo * feat_photo
        return fused


class TwoStreamMaterialNet(nn.Module):
    """
    Two-Stream CNN for Material Classification
    
    Architecture:
    - Geometry Stream: Depth (1ch) + Normals (3ch) = 4ch → ResNet-18
    - Photometric Stream: IR (1ch) + Raw (4ch) = 5ch → ResNet-18
    - Fusion: Concatenate + Attention Gate → FC Classifier
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        use_attention: bool = True,
        pretrained: bool = False,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Geometry Stream: Depth (1) + Normals (3) = 4 channels
        self.geometry_encoder = ResNetEncoder(in_channels=4, pretrained=pretrained)
        
        # Photometric Stream: IR (1) + Raw (4) = 5 channels
        self.photometric_encoder = ResNetEncoder(in_channels=5, pretrained=pretrained)
        
        # Attention Gate for fusion
        if use_attention:
            self.attention_gate = AttentionGate(512)
            fc_in_features = 512
        else:
            fc_in_features = 512 * 2  # Simple concatenation
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(
        self,
        depth: torch.Tensor,      # (B, 1, H, W)
        ir: torch.Tensor,         # (B, 1, H, W)
        raw: torch.Tensor,        # (B, 4, H, W)
        normals: Optional[torch.Tensor] = None  # (B, 3, H, W)
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            depth: 距離マップ
            ir: IR強度
            raw: Raw相関データ (Q1-Q4)
            normals: 法線マップ（省略時は深度から計算しない、入力に含める必要あり）
        
        Returns:
            logits: クラス予測 (B, num_classes)
        """
        # Geometry Stream入力を構築
        if normals is not None:
            geo_input = torch.cat([depth, normals], dim=1)  # (B, 4, H, W)
        else:
            # 法線がない場合は深度を複製して4chに
            geo_input = depth.repeat(1, 4, 1, 1)
        
        # Photometric Stream入力を構築
        photo_input = torch.cat([ir, raw], dim=1)  # (B, 5, H, W)
        
        # 各ストリームでエンコード
        feat_geo = self.geometry_encoder(geo_input)
        feat_photo = self.photometric_encoder(photo_input)
        
        # 特徴融合
        if self.use_attention:
            fused = self.attention_gate(feat_geo, feat_photo)
        else:
            fused = torch.cat([feat_geo, feat_photo], dim=1)
        
        # Global Average Pooling
        pooled = self.gap(fused)
        pooled = pooled.view(pooled.size(0), -1)
        
        # 分類
        logits = self.classifier(pooled)
        
        return logits
    
    def forward_from_dict(self, batch: dict) -> torch.Tensor:
        """Dictバッチから推論"""
        return self.forward(
            depth=batch['depth'],
            ir=batch['ir'],
            raw=batch['raw'],
            normals=batch.get('normals')
        )


class SimpleMaterialNet(nn.Module):
    """
    シンプルな単一ストリームモデル（ベースライン比較用）
    全チャネルを結合して単一のResNetで処理
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Depth (1) + IR (1) + Raw (4) + Normals (3) = 9 channels
        self.encoder = ResNetEncoder(in_channels=9, pretrained=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(
        self,
        depth: torch.Tensor,
        ir: torch.Tensor,
        raw: torch.Tensor,
        normals: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([depth, ir, raw, normals], dim=1)
        x = self.encoder(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == '__main__':
    # テスト
    model = TwoStreamMaterialNet(num_classes=5)
    
    # ダミー入力
    batch_size = 2
    depth = torch.randn(batch_size, 1, 480, 640)
    ir = torch.randn(batch_size, 1, 480, 640)
    raw = torch.randn(batch_size, 4, 480, 640)
    normals = torch.randn(batch_size, 3, 480, 640)
    
    # 順伝播
    output = model(depth, ir, raw, normals)
    print(f'Output shape: {output.shape}')
    assert output.shape == (batch_size, 5), 'Output shape mismatch'
    print('Model forward pass test PASSED')
