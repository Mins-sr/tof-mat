#!/usr/bin/env python3
"""
Quick Test Script

全体動作確認（データ生成 → 学習 → 推論）を一括実行
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_quick_test(epochs: int = 3, samples: int = 200, batch_size: int = 8):
    """End-to-End動作確認"""
    
    print('=' * 60)
    print('ToF Material Recognition - Quick Test')
    print('=' * 60)
    
    start_time = time.time()
    
    # ===== Step 1: モデル順伝播テスト =====
    print('\n[1/3] Testing model forward pass...')
    
    import torch
    from src.models.two_stream_cnn import TwoStreamMaterialNet
    
    model = TwoStreamMaterialNet(num_classes=5)
    
    # ダミー入力（フルサイズ 640x480 は重いので、テストでは縮小）
    test_size = (120, 160)  # 1/4サイズ
    depth = torch.randn(2, 1, *test_size)
    ir = torch.randn(2, 1, *test_size)
    raw = torch.randn(2, 4, *test_size)
    normals = torch.randn(2, 3, *test_size)
    
    output = model(depth, ir, raw, normals)
    assert output.shape == (2, 5), f'Expected shape (2, 5), got {output.shape}'
    print(f'  ✓ Model forward pass OK (output shape: {output.shape})')
    
    # ===== Step 2: 合成データ生成テスト =====
    print('\n[2/3] Testing synthetic data generation...')
    
    from src.data.synthetic_tof import SyntheticToFGenerator, NUM_CLASSES, MATERIAL_CLASSES
    
    generator = SyntheticToFGenerator(width=160, height=120, seed=42)  # 縮小サイズ
    
    depth, ir, raw, normal, label = generator.generate_sample(material_class=0)
    
    assert depth.shape == (120, 160), f'Depth shape mismatch: {depth.shape}'
    assert ir.shape == (120, 160), f'IR shape mismatch: {ir.shape}'
    assert raw.shape == (4, 120, 160), f'Raw shape mismatch: {raw.shape}'
    assert normal.shape == (3, 120, 160), f'Normal shape mismatch: {normal.shape}'
    
    print(f'  ✓ Data generation OK')
    print(f'    Depth range: [{depth.min():.2f}, {depth.max():.2f}] m')
    print(f'    IR range: [{ir.min():.1f}, {ir.max():.1f}]')
    print(f'    Materials: {[MATERIAL_CLASSES[i].name for i in range(NUM_CLASSES)]}')
    
    # ===== Step 3: 学習テスト =====
    print(f'\n[3/3] Training test ({epochs} epochs, {samples} samples)...')
    
    from src.train import train
    
    # 縮小サイズのジェネレータ用にモンキーパッチ
    original_init = SyntheticToFGenerator.__init__
    
    def patched_init(self, width=640, height=480, **kwargs):
        original_init(self, width=160, height=120, **kwargs)
    
    SyntheticToFGenerator.__init__ = patched_init
    
    # 縮小モデル用にモンキーパッチ（テスト高速化）
    checkpoint_dir = './checkpoints_test'
    
    try:
        best_model_path = train(
            data_dir=None,  # メモリ生成
            num_samples=samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            checkpoint_dir=checkpoint_dir,
            device_str='cpu'  # テストはCPUで
        )
        
        print(f'  ✓ Training completed')
        print(f'    Best model: {best_model_path}')
        
    except Exception as e:
        print(f'  ✗ Training failed: {e}')
        raise
    finally:
        # パッチを元に戻す
        SyntheticToFGenerator.__init__ = original_init
    
    # ===== 完了 =====
    elapsed = time.time() - start_time
    
    print('\n' + '=' * 60)
    print(f'All tests PASSED! (Elapsed: {elapsed:.1f}s)')
    print('=' * 60)
    
    print('\nNext steps:')
    print('  1. Generate full dataset:')
    print('     python scripts/generate_synthetic_data.py --num_samples 1000')
    print('  2. Train with full resolution:')
    print('     python src/train.py --epochs 20 --batch_size 4')
    print('  3. Run inference demo:')
    print('     python scripts/demo_inference.py --model_path ./checkpoints/best_model.pth')


def main():
    parser = argparse.ArgumentParser(description='Quick Test for ToF Material Recognition')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of samples for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    
    args = parser.parse_args()
    
    run_quick_test(
        epochs=args.epochs,
        samples=args.samples,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
