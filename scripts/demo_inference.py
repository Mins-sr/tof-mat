#!/usr/bin/env python3
"""
Inference Demo Script

学習済みモデルを使用して推論を行い、結果を可視化する
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_tof import SyntheticToFGenerator, MATERIAL_CLASSES, NUM_CLASSES
from src.models.two_stream_cnn import TwoStreamMaterialNet


def visualize_sample(
    depth: np.ndarray,
    ir: np.ndarray,
    raw: np.ndarray,
    true_label: int,
    pred_label: int,
    pred_probs: np.ndarray,
    save_path: str = None
):
    """サンプルの可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    class_names = [MATERIAL_CLASSES[i].name for i in range(NUM_CLASSES)]
    
    # Depth Map
    ax = axes[0, 0]
    im = ax.imshow(depth, cmap='viridis')
    ax.set_title('Depth Map')
    plt.colorbar(im, ax=ax, label='Distance (m)')
    
    # IR Intensity
    ax = axes[0, 1]
    im = ax.imshow(ir, cmap='gray')
    ax.set_title('IR Intensity')
    plt.colorbar(im, ax=ax)
    
    # Raw Q1
    ax = axes[0, 2]
    im = ax.imshow(raw[0], cmap='RdBu')
    ax.set_title('Raw Correlation (Q1)')
    plt.colorbar(im, ax=ax)
    
    # Raw Amplitude (computed from Q1-Q4)
    phase = np.arctan2(raw[3] - raw[1], raw[0] - raw[2])
    amplitude = np.sqrt((raw[0] - raw[2])**2 + (raw[3] - raw[1])**2) / 2
    
    ax = axes[1, 0]
    im = ax.imshow(amplitude, cmap='hot')
    ax.set_title('Raw Amplitude (from Q1-Q4)')
    plt.colorbar(im, ax=ax)
    
    # Raw Phase
    ax = axes[1, 1]
    im = ax.imshow(phase, cmap='hsv')
    ax.set_title('Raw Phase (from Q1-Q4)')
    plt.colorbar(im, ax=ax)
    
    # Prediction probabilities
    ax = axes[1, 2]
    colors = ['green' if i == pred_label else 'steelblue' for i in range(NUM_CLASSES)]
    bars = ax.bar(class_names, pred_probs, color=colors)
    ax.set_ylabel('Probability')
    ax.set_title(f'Prediction: {class_names[pred_label]}\n'
                 f'(True: {class_names[true_label]})')
    ax.set_ylim(0, 1)
    
    # 正解/不正解の表示
    result = "✓ Correct" if true_label == pred_label else "✗ Wrong"
    result_color = "green" if true_label == pred_label else "red"
    fig.suptitle(f'ToF Material Recognition Demo - {result}', 
                 fontsize=14, color=result_color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved visualization to {save_path}')
    else:
        plt.show()
    
    plt.close()


def run_inference(
    model_path: str = None,
    num_samples: int = 5,
    device_str: str = 'auto',
    save_dir: str = None
):
    """推論デモ実行"""
    # デバイス
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
    
    # モデルロード
    model = TwoStreamMaterialNet(num_classes=NUM_CLASSES)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {model_path}')
    else:
        print('No model checkpoint provided. Using random weights for demo.')
    
    model = model.to(device)
    model.eval()
    
    # 合成データ生成
    generator = SyntheticToFGenerator(seed=123)
    
    class_names = [MATERIAL_CLASSES[i].name for i in range(NUM_CLASSES)]
    correct = 0
    
    print(f'\nRunning inference on {num_samples} samples...')
    
    for i in range(num_samples):
        # ランダムな材質クラス
        true_class = np.random.randint(0, NUM_CLASSES)
        depth, ir, raw, normals, label = generator.generate_sample(true_class)
        
        # テンソル化と正規化
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
        ir_t = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).float()
        raw_t = torch.from_numpy(raw).unsqueeze(0).float()
        normals_t = torch.from_numpy(normals).unsqueeze(0).float()
        
        # 正規化
        depth_t = (depth_t - 0.5) / 3.5
        ir_t = ir_t / (ir_t.max() + 1e-6)
        raw_min = raw_t.min()
        raw_max = raw_t.max()
        raw_t = 2 * (raw_t - raw_min) / (raw_max - raw_min + 1e-6) - 1
        
        # 推論
        with torch.no_grad():
            depth_t = depth_t.to(device)
            ir_t = ir_t.to(device)
            raw_t = raw_t.to(device)
            normals_t = normals_t.to(device)
            
            logits = model(depth_t, ir_t, raw_t, normals_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = logits.argmax(dim=1).item()
        
        is_correct = pred_class == true_class
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f'Sample {i+1}: True={class_names[true_class]:10s}, '
              f'Pred={class_names[pred_class]:10s} [{status}] '
              f'(conf={probs[pred_class]:.2f})')
        
        # 可視化
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'sample_{i+1}.png')
        else:
            save_path = None
        
        visualize_sample(depth, ir, raw, true_class, pred_class, probs, save_path)
    
    print(f'\nAccuracy: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)')


def main():
    parser = argparse.ArgumentParser(description='ToF Material Recognition Inference Demo')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        num_samples=args.num_samples,
        device_str=args.device,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
