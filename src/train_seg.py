"""
Training script for ToF Material Segmentation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import argparse

from src.models.segmentation_net import TwoStreamUNet
from src.data.dataset import ToFMaterialDataset, create_in_memory_dataset

def dice_loss(pred, target, num_classes, smooth=1e-6):
    """
    Dice Loss for multi-class segmentation
    """
    pred = torch.softmax(pred, dim=1)
    dice = 0
    for i in range(num_classes):
        p = pred[:, i, ...]
        t = (target == i).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return 1 - (dice / num_classes)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc='  Training')
    for batch in pbar:
        depth = batch['depth'].to(device)
        ir = batch['ir'].to(device)
        raw = batch['raw'].to(device)
        normals = batch['normals'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(depth, ir, raw, normals)
        
        loss_ce = criterion(outputs, mask)
        loss_dice = dice_loss(outputs, mask, model.num_classes)
        loss = loss_ce + loss_dice
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            depth = batch['depth'].to(device)
            ir = batch['ir'].to(device)
            raw = batch['raw'].to(device)
            normals = batch['normals'].to(device)
            mask = batch['mask'].to(device)
            
            outputs = model(depth, ir, raw, normals)
            
            loss_ce = criterion(outputs, mask)
            loss_dice = dice_loss(outputs, mask, model.num_classes)
            loss = loss_ce + loss_dice
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == mask).sum().item()
            total += mask.numel()
            
    return running_loss / len(loader), correct / total

def train_seg(
    num_samples=500,
    epochs=10,
    batch_size=4,
    lr=1e-4,
    checkpoint_dir='./checkpoints_seg'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset
    full_dataset = create_in_memory_dataset(num_samples)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = TwoStreamUNet(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Pixel Acc: {val_acc*100:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_seg_model.pth'))
            print('  Saved best model')
            
    return os.path.join(checkpoint_dir, 'best_seg_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    train_seg(num_samples=args.samples, epochs=args.epochs)
