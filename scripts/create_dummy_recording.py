import json
import numpy as np
import os

def create_dummy_recording(output_dir="data/dummy_rec"):
    os.makedirs(output_dir, exist_ok=True)
    
    width, height = 224, 172  # 一般的なToF解像度
    num_frames = 5
    
    # 画像の種類とレイアウト定義
    # IR: 1, Depth: 2, RAW G1-G4: 3-6
    image_kinds = [2, 1, 3, 4, 5, 6] # Depth, IR, G1, G2, G3, G4
    image_names = ["Depth", "IR", "RAW G1", "RAW G2", "RAW G3", "RAW G4"]
    bytes_per_pixel = 2
    
    frame_size = width * height * len(image_kinds) * bytes_per_pixel
    
    info = {
        "ImageWidth": width,
        "ImageHeight": height,
        "FrameSize": frame_size,
        "ImageInfos": []
    }
    
    for i, kind in enumerate(image_kinds):
        info["ImageInfos"].append({
            "ImageKind": kind,
            "BytesPerPixel": bytes_per_pixel,
            "Offset": i * width * height * bytes_per_pixel
        })
        
    # JSON保存
    with open(os.path.join(output_dir, "RecInfo.json"), "w") as f:
        json.dump(info, f, indent=4)
        
    # RAWデータ生成
    raw_data = bytearray()
    for f in range(num_frames):
        for i, name in enumerate(image_names):
            if name == "Depth":
                # 距離画像: 0.5mから2.0mの勾配
                data = np.linspace(500, 2000, width * height).astype(np.uint16)
            elif name == "IR":
                # 赤外線画像: 同心円
                yy, xx = np.mgrid[:height, :width]
                dist = np.sqrt((xx - width/2)**2 + (yy - height/2)**2)
                data = (dist * 100).astype(np.uint16)
            else:
                # RAWデータ: ノイズ
                data = np.random.randint(0, 1000, (height, width), dtype=np.uint16).flatten()
            
            raw_data.extend(data.tobytes())
            
    # RAW保存
    with open(os.path.join(output_dir, "RecImage_00000000.raw"), "wb") as f:
        f.write(raw_data)
        
    print(f"Dummy recording created in {output_dir}")
    print(f"Files: RecInfo.json, RecImage_00000000.raw")

if __name__ == "__main__":
    create_dummy_recording()
