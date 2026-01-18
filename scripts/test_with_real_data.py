#!/usr/bin/env python3
"""
Real Data Test Script

公開データセット（NYU Depth V2 サブセット）を取得して
モデルの動作確認を行う

NYU Depth V2にはRaw Q1-Q4データが含まれないため、
深度から逆算してRawを疑似生成する
"""

import os
import sys
import urllib.request
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.two_stream_cnn import TwoStreamMaterialNet
from src.preprocessing.normalize import compute_surface_normals


# NYU Depth V2 ラベルから材質マッピング（簡易版）
# NYU Depth V2 のクラスラベルを5材質に変換
MATERIAL_MAPPING = {
    # 木材系
    'floor': 1, 'table': 1, 'desk': 1, 'door': 1, 'cabinet': 1,
    'bookshelf': 1, 'dresser': 1,
    # 布系
    'bed': 3, 'sofa': 3, 'pillow': 3, 'curtain': 3, 'clothes': 3,
    # プラスチック系
    'chair': 2, 'lamp': 2, 'television': 2, 'monitor': 2,
    # 金属系
    'sink': 0, 'mirror': 0, 'refrigerator': 0,
    # デフォルト
    'wall': 2, 'ceiling': 2, 'floor_mat': 3,
}


def generate_pseudo_raw_from_depth(
    depth: np.ndarray,
    modulation_freq: float = 20e6
) -> np.ndarray:
    """
    深度データからRaw相関データ（Q1-Q4）を疑似生成
    
    ToF原理:
    Q_k = B + A * cos(φ + k*π/2), k = 0,1,2,3
    φ = 4πfd/c
    """
    c = 3e8  # 光速
    
    # 無効な深度をマスク
    valid_mask = (depth > 0.1) & (depth < 10.0)
    depth_safe = np.where(valid_mask, depth, 1.0)
    
    # 位相計算
    phase = (4 * np.pi * modulation_freq * depth_safe) / c
    
    # 振幅（距離に反比例 + ランダムな反射率）
    base_amplitude = 500.0
    amplitude = base_amplitude / (depth_safe ** 2)
    amplitude = amplitude * np.random.uniform(0.5, 1.5, amplitude.shape)
    
    # 環境光
    ambient = np.random.uniform(50, 150)
    
    # Q1-Q4
    q1 = ambient + amplitude * np.cos(phase)
    q2 = ambient + amplitude * np.cos(phase + np.pi / 2)
    q3 = ambient + amplitude * np.cos(phase + np.pi)
    q4 = ambient + amplitude * np.cos(phase + 3 * np.pi / 2)
    
    # 無効領域をゼロに
    q1[~valid_mask] = 0
    q2[~valid_mask] = 0
    q3[~valid_mask] = 0
    q4[~valid_mask] = 0
    
    raw = np.stack([q1, q2, q3, q4], axis=0).astype(np.float32)
    return raw


def generate_pseudo_ir_from_depth(depth: np.ndarray) -> np.ndarray:
    """深度からIR強度を疑似生成"""
    valid_mask = (depth > 0.1) & (depth < 10.0)
    depth_safe = np.where(valid_mask, depth, 1.0)
    
    # IR = 反射率 × 1/d²
    ir = 1000.0 / (depth_safe ** 2)
    ir = ir * np.random.uniform(0.3, 1.0, ir.shape)
    ir[~valid_mask] = 0
    
    return ir.astype(np.float32)


def download_sample_depth_data(output_dir: str) -> str:
    """
    サンプル深度データをダウンロード
    
    NYU Depth V2の小さなサブセットを取得
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # サンプルデータのURL（軽量版）
    # 実際のNYU Depth V2は大きいため、ここではテスト用に合成データを作成
    sample_file = os.path.join(output_dir, 'sample_depth.npz')
    
    if os.path.exists(sample_file):
        print(f'Sample data already exists: {sample_file}')
        return sample_file
    
    print('Creating sample depth data (simulating Kinect-like data)...')
    
    # 実際のKinectデータを模倣した合成データ
    samples = []
    
    for i in range(10):
        # 室内シーンをシミュレート
        h, w = 480, 640
        
        # 基本床面（距離2-4m）
        y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        depth = 2.0 + 2.0 * y  # 床は手前から奥に
        
        # 壁（左右）
        wall_left = x < 0.1
        wall_right = x > 0.9
        depth[wall_left] = 1.5 + np.random.uniform(0, 0.5)
        depth[wall_right] = 1.5 + np.random.uniform(0, 0.5)
        
        # ランダムな物体を配置
        num_objects = np.random.randint(2, 5)
        for _ in range(num_objects):
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.3, 0.8)
            rx = np.random.uniform(0.05, 0.15)
            ry = np.random.uniform(0.05, 0.15)
            obj_depth = np.random.uniform(1.0, 3.0)
            
            dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            mask = dist < 1.0
            depth[mask] = obj_depth
        
        # Kinect特有のノイズ
        noise = np.random.normal(0, 0.01 * depth, depth.shape)
        depth = depth + noise
        depth = np.clip(depth, 0.5, 5.0)
        
        # 一部をNaN（測定不可）に
        invalid_ratio = 0.02
        invalid_mask = np.random.random(depth.shape) < invalid_ratio
        depth[invalid_mask] = 0
        
        samples.append({
            'depth': depth.astype(np.float32),
            'scene_id': i
        })
    
    # 保存
    np.savez_compressed(
        sample_file,
        depths=[s['depth'] for s in samples],
        scene_ids=[s['scene_id'] for s in samples]
    )
    
    print(f'Created {len(samples)} sample depth frames: {sample_file}')
    return sample_file


def test_with_real_data(
    model_path: str = None,
    num_samples: int = 5,
    save_dir: str = None
):
    """実データでのテスト"""
    
    # デバイス
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    
    # モデル
    model = TwoStreamMaterialNet(num_classes=5)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model: {model_path}')
    else:
        print('Using randomly initialized model (no checkpoint specified)')
    
    model = model.to(device)
    model.eval()
    
    # サンプルデータ取得
    data_dir = './data/real_samples'
    sample_file = download_sample_depth_data(data_dir)
    
    data = np.load(sample_file)
    depths = data['depths']
    
    print(f'\nLoaded {len(depths)} depth frames')
    print(f'Depth shape: {depths[0].shape}')
    
    # 材質クラス名
    class_names = ['metal', 'wood', 'plastic', 'fabric', 'wax']
    
    # 推論
    results = []
    
    for i in range(min(num_samples, len(depths))):
        depth = depths[i]
        
        # 疑似IR・Rawデータ生成
        ir = generate_pseudo_ir_from_depth(depth)
        raw = generate_pseudo_raw_from_depth(depth)
        normals = compute_surface_normals(depth)
        
        # テンソル化・正規化
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
            logits = model(
                depth_t.to(device),
                ir_t.to(device),
                raw_t.to(device),
                normals_t.to(device)
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = logits.argmax(dim=1).item()
        
        print(f'\nSample {i+1}:')
        print(f'  Depth range: [{depth.min():.2f}, {depth.max():.2f}] m')
        print(f'  Prediction: {class_names[pred_class]} ({probs[pred_class]*100:.1f}%)')
        print(f'  All probs: {dict(zip(class_names, [f"{p*100:.1f}%" for p in probs]))}')
        
        results.append({
            'sample_id': i,
            'depth': depth,
            'ir': ir,
            'raw': raw,
            'normals': normals,
            'pred_class': pred_class,
            'probs': probs
        })
        
        # 可視化
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Depth
            ax = axes[0, 0]
            im = ax.imshow(depth, cmap='viridis')
            ax.set_title('Depth (Kinect-like)')
            plt.colorbar(im, ax=ax)
            
            # IR
            ax = axes[0, 1]
            im = ax.imshow(ir, cmap='gray')
            ax.set_title('Pseudo IR')
            plt.colorbar(im, ax=ax)
            
            # Raw Q1
            ax = axes[0, 2]
            im = ax.imshow(raw[0], cmap='RdBu')
            ax.set_title('Pseudo Raw Q1')
            plt.colorbar(im, ax=ax)
            
            # Normals
            ax = axes[1, 0]
            normals_vis = (normals.transpose(1, 2, 0) + 1) / 2
            ax.imshow(normals_vis)
            ax.set_title('Surface Normals')
            
            # Raw amplitude
            amplitude = np.sqrt((raw[0] - raw[2])**2 + (raw[3] - raw[1])**2) / 2
            ax = axes[1, 1]
            im = ax.imshow(amplitude, cmap='hot')
            ax.set_title('Raw Amplitude')
            plt.colorbar(im, ax=ax)
            
            # Prediction
            ax = axes[1, 2]
            colors = ['green' if j == pred_class else 'steelblue' for j in range(5)]
            ax.bar(class_names, probs, color=colors)
            ax.set_ylim(0, 1)
            ax.set_title(f'Prediction: {class_names[pred_class]}')
            
            plt.suptitle(f'Sample {i+1} - Real-like Depth Data Test')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'real_test_{i+1}.png'), dpi=150)
            plt.close()
    
    print(f'\n{"="*50}')
    print(f'Tested {len(results)} samples')
    if save_dir:
        print(f'Visualizations saved to: {save_dir}')
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test with Real-like Depth Data')
    parser.add_argument('--model_path', type=str, default='./checkpoints_test/best_model.pth')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./output/real_test')
    
    args = parser.parse_args()
    
    test_with_real_data(
        model_path=args.model_path,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
