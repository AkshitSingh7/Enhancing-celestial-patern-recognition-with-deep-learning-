import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- Helper Functions ---
def sample_image_features(feat_map, points_pix, image_size):
    B, C, Hf, Wf = feat_map.shape
    device = feat_map.device
    Pmax = max([p.shape[0] for p in points_pix]) if len(points_pix) > 0 else 0
    sampled = torch.zeros((B, Pmax, C), device=device)
    for i in range(B):
        pts = points_pix[i]
        if pts.shape[0] == 0: continue
        scale_x = Wf / float(image_size)
        scale_y = Hf / float(image_size)
        x_f = pts[:, 0] * scale_x
        y_f = pts[:, 1] * scale_y
        x_norm = (x_f / (Wf - 1)) * 2 - 1
        y_norm = (y_f / (Hf - 1)) * 2 - 1
        grid = torch.stack([x_norm, y_norm], 1).to(device).view(1, -1, 1, 2)
        feat_i = F.grid_sample(feat_map[i:i+1], grid, align_corners=True)
        feat_i = feat_i.view(C, -1).transpose(0, 1)
        sampled[i, :feat_i.shape[0], :] = feat_i
    return sampled

def splat_points_to_heatmap(points_pix, scores, image_size, sigma=1.5, kernel_size=9):
    B, P, _ = points_pix.shape
    point_map = torch.zeros((B, 1, image_size, image_size), device=points_pix.device)
    coords = torch.arange(kernel_size, device=points_pix.device) - kernel_size//2
    xg, yg = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(xg**2 + yg**2) / (2*sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    for b in range(B):
        pts = points_pix[b]
        scr = scores[b]
        valid_mask = ~torch.isnan(pts[:, 0]) & ~torch.isnan(pts[:, 1])
        pts = pts[valid_mask]
        scr = scr[valid_mask]
        for i in range(len(pts)):
            x = int(round(pts[i, 0].item()))
            y = int(round(pts[i, 1].item()))
            if 0 <= x < image_size and 0 <= y < image_size:
                point_map[b, 0, y, x] += scr[i]

    blurred_heatmaps = F.conv2d(point_map, kernel, padding=kernel_size//2)
    return blurred_heatmaps

# --- Network Modules ---
class ImageEncoder(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        res = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.conv1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1, self.layer2, self.layer3 = res.layer1, res.layer2, res.layer3
        self.proj = nn.Conv2d(256, out_channels, 1)
    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        x = self.conv1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.proj(x)

class PointTransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mlp2 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm1(x + self.mlp1(x))
        x_t = x.transpose(0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x = self.norm2(x + attn_out.transpose(0, 1))
        return x + self.mlp2(x)

class PointBranch(nn.Module):
    def __init__(self, in_feat=3, d_model=128, num_blocks=3):
        super().__init__()
        self.input_mlp = nn.Sequential(nn.Linear(in_feat, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([PointTransformerBlock(d_model) for _ in range(num_blocks)])
    def forward(self, pts):
        x = self.input_mlp(pts)
        for b in self.blocks: x = b(x)
        return x

class FusionHead(nn.Module):
    def __init__(self, point_dim=128, img_feat_dim=256, hidden=256):
        super().__init__()
        self.cross_mlp = nn.Sequential(nn.Linear(point_dim+img_feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, point_dim))
        self.point_cls = nn.Sequential(nn.Linear(point_dim, point_dim//2), nn.ReLU(), nn.Linear(point_dim//2, 1))
        self.point_offset = nn.Sequential(nn.Linear(point_dim, point_dim//2), nn.ReLU(), nn.Linear(point_dim//2, 2))
        self.heat_conv = nn.Sequential(nn.Conv2d(img_feat_dim, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
    def forward(self, image_feat, point_feats, sampled_img_feats):
        fused = torch.cat([point_feats, sampled_img_feats], -1)
        fused = self.cross_mlp(fused)
        point_out = point_feats + fused
        cls_logits = self.point_cls(point_out).squeeze(-1)
        offsets = self.point_offset(point_out)
        heat = self.heat_conv(image_feat)
        return cls_logits, offsets, heat

class HybridStarDetector(nn.Module):
    def __init__(self, image_size=512, img_feat_dim=256, point_feat_dim=128):
        super().__init__()
        self.image_size = image_size
        self.encoder = ImageEncoder(out_channels=img_feat_dim)
        self.point_branch = PointBranch(in_feat=3, d_model=point_feat_dim)
        self.fusion = FusionHead(point_dim=point_feat_dim, img_feat_dim=img_feat_dim)
        
    def forward(self, image, points_list):
        feat = self.encoder(image)
        B, C, Hf, Wf = feat.shape
        Pmax = max([p.shape[0] for p in points_list]) if len(points_list) > 0 else 0
        pts_feats_padded = torch.zeros((B, Pmax, 3), device=image.device)
        pts_pix_t = []
        
        for i in range(B):
            pts = points_list[i].to(image.device)
            if pts.shape[0] == 0:
                pts_pix_t.append(torch.zeros((0, 2), device=image.device))
                continue
            x = pts[:, 0]; y = pts[:, 1]
            mag = pts[:, 4] if pts.shape[1] > 4 else torch.zeros_like(x)
            x_norm = x / self.image_size
            y_norm = y / self.image_size
            mag_norm = torch.clamp((mag - 5.0) / 15.0, 0, 1)
            feats = torch.stack([x_norm, y_norm, mag_norm], 1)
            pts_feats_padded[i, :feats.shape[0], :] = feats
            pts_pix_t.append(torch.stack([x, y], 1))
            
        point_embs = self.point_branch(pts_feats_padded)
        sampled_img_feats = sample_image_features(feat, pts_pix_t, self.image_size)
        cls_logits, offsets, heat_feat = self.fusion(feat, point_embs, sampled_img_feats)
        
        heatmap_full = F.interpolate(heat_feat, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        
        pts_pad = torch.full((B, Pmax, 2), float('nan'), device=image.device)
        for i in range(B):
            if pts_pix_t[i].shape[0] > 0:
                pts_pad[i, :pts_pix_t[i].shape[0], :] = pts_pix_t[i]
                
        probs = torch.sigmoid(cls_logits)
        splat = splat_points_to_heatmap(pts_pad, probs, self.image_size)
        heatmap_logits = heatmap_full + splat
        return heatmap_logits, cls_logits
