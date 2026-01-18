"""
Evaluation script for ToF Material Recognition
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ToFMaterialDataset
from src.data.synthetic_tof import MATERIAL_CLASSES, NUM_CLASSES
from src.models.two_stream_cnn import TwoStreamMaterialNet


def load_model(
    model_path: str,
    device: torch.device
) -> TwoStreamMaterialNet:
    """学習済みモデルをロード"""
    model = TwoStreamMaterialNet(num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Loaded model from {model_path}')
    print(f'Checkpoint Val Acc: {checkpoint.get("val_acc", "N/A")}')
    
    return model


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> dict:
    """モデル評価"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            depth = batch['depth'].to(device)
            ir = batch['ir'].to(device)
            raw = batch['raw'].to(device)
            normals = batch['normals'].to(device)
            labels = batch['label']
            
            outputs = model(depth, ir, raw, normals)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # メトリクス計算
    accuracy = (all_predictions == all_labels).mean()
    
    # クラス名取得
    class_names = [MATERIAL_CLASSES[i].name for i in range(NUM_CLASSES)]
    
    # 混同行列
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 分類レポート
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_names': class_names,
        'predictions': all_predictions,
        'labels': all_labels
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = None
) -> None:
    """混同行列を可視化"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # 数値を表示
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Confusion matrix saved to {save_path}')
    else:
        plt.show()
    
    plt.close()


def print_report(results: dict) -> None:
    """評価結果を表示"""
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    
    print(f'\nOverall Accuracy: {results["accuracy"] * 100:.2f}%')
    
    print('\n--- Per-Class Metrics ---')
    report = results['classification_report']
    for class_name in results['class_names']:
        metrics = report[class_name]
        print(f'{class_name:10s}: Precision={metrics["precision"]:.3f}, '
              f'Recall={metrics["recall"]:.3f}, F1={metrics["f1-score"]:.3f}')
    
    print('\n--- Confusion Matrix ---')
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    # ヘッダー
    header = '          ' + ''.join([f'{name[:6]:>8s}' for name in class_names])
    print(header)
    
    # 各行
    for i, name in enumerate(class_names):
        row = f'{name[:8]:8s}: ' + ''.join([f'{cm[i, j]:8d}' for j in range(len(class_names))])
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ToF Material Recognition Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to evaluation data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--save_cm', type=str, default=None,
                        help='Path to save confusion matrix plot')
    
    args = parser.parse_args()
    
    # デバイス
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # データセット
    dataset = ToFMaterialDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # モデル
    model = load_model(args.model_path, device)
    
    # 評価
    results = evaluate_model(model, loader, device)
    
    # 結果表示
    print_report(results)
    
    # 混同行列を保存
    if args.save_cm:
        plot_confusion_matrix(results['confusion_matrix'], results['class_names'], args.save_cm)


if __name__ == '__main__':
    main()
