"""
Training script for ToF Material Recognition
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from typing import Tuple, Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ToFMaterialDataset, InMemoryToFDataset
from src.data.synthetic_tof import SyntheticToFGenerator, NUM_CLASSES
from src.models.two_stream_cnn import TwoStreamMaterialNet


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """1エポック分の学習"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        # デバイスに転送
        depth = batch['depth'].to(device)
        ir = batch['ir'].to(device)
        raw = batch['raw'].to(device)
        normals = batch['normals'].to(device)
        labels = batch['label'].to(device)
        
        # 順伝播
        optimizer.zero_grad()
        outputs = model(depth, ir, raw, normals)
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        optimizer.step()
        
        # 統計
        total_loss += loss.item() * depth.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.1f}%'
        })
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """評価"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            depth = batch['depth'].to(device)
            ir = batch['ir'].to(device)
            raw = batch['raw'].to(device)
            normals = batch['normals'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(depth, ir, raw, normals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * depth.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def create_in_memory_dataset(
    num_samples: int,
    seed: int = 42
) -> InMemoryToFDataset:
    """メモリ上でデータセットを生成（ファイルI/O不要）"""
    generator = SyntheticToFGenerator(seed=seed)
    
    depths, irs, raws, normals_list, labels = [], [], [], [], []
    
    samples_per_class = num_samples // NUM_CLASSES
    class_ids = []
    for c in range(NUM_CLASSES):
        class_ids.extend([c] * samples_per_class)
    remainder = num_samples - len(class_ids)
    class_ids.extend(np.random.randint(0, NUM_CLASSES, remainder).tolist())
    np.random.shuffle(class_ids)
    
    for material_class in class_ids:
        depth, ir, raw, normal, label = generator.generate_sample(material_class)
        depths.append(depth)
        irs.append(ir)
        raws.append(raw)
        normals_list.append(normal)
        labels.append(label)
    
    return InMemoryToFDataset(depths, irs, raws, normals_list, labels)


def train(
    data_dir: Optional[str] = None,
    num_samples: int = 500,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = False,
    checkpoint_dir: str = './checkpoints',
    device_str: str = 'auto'
) -> str:
    """
    学習メイン関数
    
    Args:
        data_dir: データディレクトリ（省略時はメモリ上で生成）
        num_samples: メモリ生成時のサンプル数
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        use_focal_loss: Focal Lossを使用するか
        checkpoint_dir: チェックポイント保存先
        device_str: デバイス ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        best_model_path: 最良モデルのパス
    """
    # デバイス設定
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    print(f'Using device: {device}')
    
    # データセット作成
    if data_dir and os.path.exists(data_dir):
        print(f'Loading dataset from {data_dir}')
        full_dataset = ToFMaterialDataset(data_dir)
    else:
        print(f'Generating {num_samples} samples in memory')
        full_dataset = create_in_memory_dataset(num_samples)
    
    # Train/Val分割
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # メモリデータセットでは0推奨
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # モデル作成
    model = TwoStreamMaterialNet(num_classes=NUM_CLASSES, use_attention=True)
    model = model.to(device)
    
    # 損失関数
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # オプティマイザ
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # チェックポイントディレクトリ
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # 学習ループ
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%')
        
        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Saved best model (Val Acc: {val_acc * 100:.2f}%)')
    
    print(f'\nTraining complete. Best Val Acc: {best_val_acc * 100:.2f}%')
    print(f'Best model saved to: {best_model_path}')
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train ToF Material Recognition Model')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (if not specified, generate in memory)')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to generate in memory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use Focal Loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_focal_loss=args.focal_loss,
        checkpoint_dir=args.checkpoint_dir,
        device_str=args.device
    )


if __name__ == '__main__':
    main()
