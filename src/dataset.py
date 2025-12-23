import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

def read_yolo_label(path):
    boxes = []
    if not os.path.exists(path): return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            c, x, y, w, h = map(float, parts)
            boxes.append((c, x, y, w, h))
    return boxes

class StarDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_size=512, max_points=1024):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.max_points = max_points

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        # Load .npy file
        img = np.load(self.images[idx]).astype(np.float32)
        
        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = np.clip(img, -6, 6)
        img_t = torch.from_numpy(img).unsqueeze(0)
        
        # Load Labels
        base = os.path.splitext(os.path.basename(self.images[idx]))[0]
        boxes = read_yolo_label(os.path.join(self.labels_dir, f"{base}.txt"))
        
        pts = []
        for c, x_c, y_c, w_rel, h_rel in boxes:
            x_pix = x_c * self.image_size
            y_pix = y_c * self.image_size
            # Format: x, y, 0, 0, class
            pts.append([x_pix, y_pix, 0, 0, c])
            
        if len(pts) > self.max_points: 
            pts = random.sample(pts, self.max_points)
            
        pts = np.array(pts, dtype=np.float32) if len(pts) > 0 else np.zeros((0, 5), dtype=np.float32)
        
        return {"image": img_t, "points": pts, "name": base}

def collate_fn(samples):
    images = torch.stack([s["image"] for s in samples], 0)
    points = [torch.from_numpy(s["points"]).float() for s in samples]
    names = [s["name"] for s in samples]
    return {"image": images, "points": points, "names": names}
