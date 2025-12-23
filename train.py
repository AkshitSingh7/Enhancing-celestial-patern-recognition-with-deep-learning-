import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.dataset import StarDataset, collate_fn
from src.model import HybridStarDetector

# --- Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        pt = torch.where(targets==1, p, 1-p)
        fl = self.alpha * (1-pt)**self.gamma * bce
        return fl.mean() if self.reduction=="mean" else fl.sum()

# --- Epoch Loop ---
def train_one_epoch(model, dl, opt, device, loss_heat, loss_cls, image_size=512):
    model.train(); total = 0
    for batch in tqdm(dl, desc="Training"):
        imgs = batch['image'].to(device)
        points = batch['points']
        B = imgs.shape[0]
        
        target_heat = torch.zeros((B, 1, image_size, image_size), device=device)
        Pmax = max([p.shape[0] for p in points]) if len(points) > 0 else 0
        cls_target = torch.zeros((B, Pmax), device=device) if Pmax > 0 else None
        
        for i in range(B):
            pts = points[i].to(device)
            for j in range(pts.shape[0]):
                x = int(round(pts[j, 0].item()))
                y = int(round(pts[j, 1].item()))
                if 0 <= x < image_size and 0 <= y < image_size:
                    target_heat[i, 0, y, x] = 1
                    if cls_target is not None: cls_target[i, j] = 1
                    
        opt.zero_grad()
        heat_logits, cls_logits = model(imgs, points)
        l = loss_heat(heat_logits, target_heat)
        if cls_target is not None and cls_logits is not None:
            # Handle potential shape mismatch if batching was weird
            if cls_logits.shape == cls_target.shape:
                l += 0.5 * loss_cls(cls_logits, cls_target)
                
        l.backward()
        opt.step()
        total += l.item() * imgs.size(0)
    return total / len(dl.dataset)

def evaluate(model, dl, device):
    model.eval()
    preds = []; gts = []
    with torch.no_grad():
        for batch in dl:
            imgs = batch['image'].to(device)
            points = batch['points']
            heat_logits, _ = model(imgs, points)
            B = imgs.shape[0]
            for i in range(B):
                gt = torch.zeros((512, 512), device=device)
                for p in points[i]:
                    x = int(round(p[0].item()))
                    y = int(round(p[1].item()))
                    if 0 <= x < 512 and 0 <= y < 512: gt[y, x] = 1
                gts.append(gt.cpu())
                preds.append(torch.sigmoid(heat_logits[i, 0]).cpu())
    
    # Simple F1 Calculation
    thresholds = [0.5]
    best = {"f1": 0}
    for t in thresholds:
        tp = fp = fn = 0
        for p, g in zip(preds, gts):
            pred = (p > t).float()
            tp += ((pred * g) == 1).sum().item()
            fp += ((pred == 1) & (g == 0)).sum().item()
            fn += ((pred == 0) & (g == 1)).sum().item()
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        if f1 > best["f1"]: best = {"precision": prec, "recall": rec, "f1": f1}
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4) # Small batch for CPU/testing
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_ds = StarDataset(os.path.join(args.data_dir, "train/images"), os.path.join(args.data_dir, "train/labels"))
    val_ds = StarDataset(os.path.join(args.data_dir, "val/images"), os.path.join(args.data_dir, "val/labels"))
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = HybridStarDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_heat = FocalLoss()
    loss_cls = nn.BCEWithLogitsLoss()

    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, device, loss_heat, loss_cls)
        metrics = evaluate(model, val_dl, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val F1: {metrics['f1']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Saved Best Model!")

if __name__ == "__main__":
    main()
