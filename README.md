# ToF Material Recognition - 最小検証基盤

深度・アクティブIR・Rawデータを統合した材質認識の検証基盤プログラム

## セットアップ

```bash
cd /Users/fushimit/project/tof-mat
pip install -r requirements.txt
```

## クイックスタート（動作確認）

```bash
# 全体テスト（データ生成 → 学習 → 推論）
python scripts/run_quick_test.py --epochs 3 --samples 200
```

## 個別実行

```bash
# 1. 合成データ生成
python scripts/generate_synthetic_data.py --num_samples 1000 --output_dir ./data/train

# 2. 学習
python src/train.py --data_dir ./data/train --epochs 10

# 3. 推論デモ
python scripts/demo_inference.py --model_path ./checkpoints/best_model.pth
```

## 材質セグメンテーション（ピクセル単位）

画像全体ではなく、ピクセル単位で材質を分類するセグメンテーション機能です。

```bash
# 1. セグメンテーション学習（合成データ）
PYTHONPATH=. python src/train_seg.py --samples 500 --epochs 10

# 2. 推論結果の可視化
PYTHONPATH=. python scripts/demo_seg.py
```

## tof実機データとの統合

SDKで録画された `depth`, `ir`, `raw1~4` を読み込むための `SDKDataLoader` を実装しています。

```python
from src.data.segmentation_dataset import SDKDataLoader

loader = SDKDataLoader("path/to/recordings")
frame_data = loader.load_frame("0001") # frame_idを指定
```

### アノテーション
セグメンテーション学習には、ピクセルラベル（0=背景, 1-N=材質ID）が書き込まれた画像ファイルが必要です。

#### 16bit記録データからのアノテーション作成
付属のツール `scripts/maskgenerator.html` を使用して、記録データから直接マスクを作成できます。

1. ブラウザで [maskgenerator.html](scripts/maskgenerator.html) を開きます。
2. `RecInfo.json` と `RecImage_*.raw` を選択して読み込みます。
3. 材質ラベルを選択し、マウスで領域を塗りつぶします。
4. 「マスクを保存」ボタンで `frame_XXXX_mask.png` を出力します。

出力されたマスクファイルは、データディレクトリ内に配置してください。

## プロジェクト構造（拡張後）

```
tof-mat/
├── src/
│   ├── data/           # SDKDataLoader, SegmentationDataset
│   ├── models/         # TwoStreamUNet (Segmentation)
│   └── train_seg.py    # セグメンテーション学習
├── scripts/
│   └── demo_seg.py     # セグメンテーション可視化デモ
```

## 材質クラス

| ID | 材質 | 特性 |
|----|------|------|
| 0 | Metal | 高反射率、表面散乱のみ |
| 1 | Wood | 中程度反射率 |
| 2 | Plastic | 軽度SSS |
| 3 | Fabric | 低反射率、粗い表面 |
| 4 | Wax | 強い表面下散乱 |
