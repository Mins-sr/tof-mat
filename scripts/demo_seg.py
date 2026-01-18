"""
Segmentation Inference Demo

Generate synthetic samples and visualize the segmentation results.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.models.segmentation_net import TwoStreamUNet
from src.data.synthetic_tof import SyntheticToFGenerator, MATERIAL_CLASSES

def visualize_segmentation(model_path, num_samples=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = TwoStreamUNet(num_classes=5).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    model.eval()
    
    generator = SyntheticToFGenerator(width=320, height=240, seed=44)
    
    for i in range(num_samples):
        # Generate sample
        depth, ir, raw, normals, mask_gt = generator.generate_sample()
        
        # Preprocess
        depth_t = torch.from_numpy((depth - 0.5) / 3.5).unsqueeze(0).unsqueeze(0).float().to(device)
        ir_t = torch.from_numpy(ir / (ir.max() + 1e-6)).unsqueeze(0).unsqueeze(0).float().to(device)
        raw_t = torch.from_numpy(2 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-6) - 1).unsqueeze(0).float().to(device)
        normals_t = torch.from_numpy(normals).unsqueeze(0).float().to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(depth_t, ir_t, raw_t, normals_t)
            pred_mask = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            
        # Visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original (IR or Depth)
        axes[0].imshow(ir, cmap='gray')
        axes[0].set_title('Input IR Intensity')
        
        # Ground Truth Mask
        axes[1].imshow(mask_gt, cmap='tab10', vmin=0, vmax=4)
        axes[1].set_title('Ground Truth Mask')
        
        # Predicted Mask
        axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=4)
        axes[2].set_title('Predicted Mask')
        
        # Overlay
        overlay = cv2.addWeighted(ir.astype(np.uint8), 0.7, (pred_mask * 50).astype(np.uint8), 0.3, 0)
        axes[3].imshow(overlay)
        axes[3].set_title('Inference Overlay')
        
        for ax in axes:
            ax.axis('off')
            
        os.makedirs('./output/demo_seg', exist_ok=True)
        plt.savefig(f'./output/demo_seg/res_{i}.png')
        plt.close()
        print(f"Result saved to ./output/demo_seg/res_{i}.png")

if __name__ == '__main__':
    visualize_segmentation('./checkpoints_seg/best_seg_model.pth')
