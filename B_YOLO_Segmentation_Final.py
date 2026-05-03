#!/usr/bin/env python
# coding: utf-8

# # B-YOLO Instance Segmentation — Training + Baseline Comparison
# **IIT KGP H100 Server — Run in parallel with classification notebook**
# 
# Runs two models:
# 1. **Baseline** — frozen YOLO11s-seg backbone + plain Conv decoder (no Bayesian)
# 2. **B-YOLO** — frozen YOLO11s-seg backbone + BayesianSegmentationHead
# 
# Metrics: Dice score, Mask IoU, mAP@50

# In[2]:


# Cell 1 — GPU check
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free = torch.cuda.mem_get_info(i)[0] // 1024**3
    print(f'GPU {i}  : {p.name}  Total: {p.total_memory//1024**3}GB  Free: ~{free}GB')


# In[3]:


# Cell 2 — CONFIG  ← only cell you need to edit
import os

BASE = os.path.expanduser('~/nas/model/B_YOLO')

CFG = {
    # paths
    'image_dir'     : f'{BASE}/2_Instance_Segmentation/images',
    'train_json'    : f'{BASE}/2_Instance_Segmentation/train.json',
    'val_json'      : f'{BASE}/2_Instance_Segmentation/test.json',
    'checkpoint_dir': f'{BASE}/checkpoints_seg',

    # training
    'num_epochs'    : 100,
    'batch_size'    : 16,       # ← MUST be small for segmentation (640x640 masks)
                               #   increase to 16 only if GPU memory allows
    'input_size'    : (640, 640),
    'num_workers'   : 4,
    'patience'      : 15,

    # B-YOLO specific
    'lr_head'       : 1e-3,
    'lr_backbone'   : 1e-4,   # lower LR for unfrozen backbone layers
    'weight_decay'  : 1e-4,
    'unfreeze_last' : 2,

    # Baseline specific
    'baseline_lr'   : 1e-3,

    'gpu_id'        : 0,
}

os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
print('Config OK ✓')
print(f'NOTE: batch_size={CFG["batch_size"]} — segmentation masks are large, cannot use 512')


# In[4]:


# Cell 3 — Imports
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

device = torch.device(f'cuda:{CFG["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device} — {torch.cuda.get_device_name(CFG["gpu_id"])}')


# In[5]:


# Cell 4 — Dataset
# Reads COCO segmentation JSON, builds binary masks from polygons

class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, json_path, input_size=(640, 640), is_train=True):
        self.image_dir  = image_dir
        self.input_size = input_size
        self.is_train   = is_train

        with open(json_path, 'r') as f:
            coco = json.load(f)

        self.images      = {img['id']: img for img in coco['images']}
        self.annotations = {}
        for ann in coco['annotations']:
            self.annotations.setdefault(ann['image_id'], []).append(ann)
        self.image_ids = list(self.images.keys())

        # Image transforms
        if is_train:
            self.img_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(input_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.img_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(input_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self): return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        image    = Image.open(os.path.join(self.image_dir, img_info['file_name'])).convert('RGB')
        orig_w, orig_h = image.size

        # Apply horizontal flip consistently to image AND mask
        do_flip = self.is_train and torch.rand(1).item() > 0.5

        # Build binary mask from COCO polygons
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in self.annotations.get(img_id, []):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for poly in ann['segmentation']:
                    if len(poly) >= 6:   # need at least 3 points
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 1)

        # Resize mask
        mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]),
                          interpolation=cv2.INTER_NEAREST)

        # Apply flip to both
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = np.fliplr(mask).copy()

        image_tensor = self.img_transforms(image)
        mask_tensor  = torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W]

        return image_tensor, mask_tensor


train_dataset = COCOSegmentationDataset(
    CFG['image_dir'], CFG['train_json'], CFG['input_size'], is_train=True)
val_dataset   = COCOSegmentationDataset(
    CFG['image_dir'], CFG['val_json'],   CFG['input_size'], is_train=False)

train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'],
                          shuffle=True,  num_workers=CFG['num_workers'],
                          pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=CFG['num_workers'],
                          pin_memory=True)

print(f'Train : {len(train_dataset)} samples, {len(train_loader)} batches')
print(f'Val   : {len(val_dataset)} samples, {len(val_loader)} batches')

# Sanity check — show mask coverage
img, msk = train_dataset[0]
print(f'Image shape : {img.shape}')
print(f'Mask shape  : {msk.shape}')
print(f'Mask coverage (% foreground): {msk.mean().item()*100:.1f}%')


# In[6]:


# Cell 5 — Loss function (Dice + BCE combined)
# Both models use this same loss

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = 1.0 - bce_weight

    def forward(self, predictions, targets):
        # predictions: [B, 1, H, W] raw logits
        # targets    : [B, 1, H, W] binary float mask
        bce  = F.binary_cross_entropy_with_logits(predictions, targets)
        prob = torch.sigmoid(predictions)
        inter = (prob * targets).sum(dim=[1,2,3])
        union = prob.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
        dice  = 1 - ((2 * inter + 1e-8) / (union + 1e-8))
        return self.bce_weight * bce + self.dice_weight * dice.mean()


def compute_metrics(predictions, targets, threshold=0.5):
    """Returns dice, iou for a batch. predictions are raw logits."""
    prob     = torch.sigmoid(predictions)
    pred_bin = (prob > threshold).float()

    inter = (pred_bin * targets).sum(dim=[1,2,3])
    union_dice = pred_bin.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
    union_iou  = (pred_bin + targets).clamp(0,1).sum(dim=[1,2,3])

    dice = ((2 * inter + 1e-8) / (union_dice + 1e-8)).mean().item()
    iou  = ((inter + 1e-8) / (union_iou  + 1e-8)).mean().item()
    return dice, iou


print('Loss and metrics defined ✓')


# In[7]:


# Cell 6 — Shared backbone

class YOLO11sSegBackbone(nn.Module):
    """Uses yolo11s-seg.pt — pretrained for segmentation tasks."""
    def __init__(self):
        super().__init__()
        yolo = YOLO('yolo11s-seg.pt')
        self.backbone = yolo.model.model[:10]
    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        return x   # output: [B, 512, H/32, W/32] i.e. [B, 512, 20, 20] for 640 input


def freeze_backbone(backbone, unfreeze_last=0):
    for p in backbone.parameters():
        p.requires_grad = False
    if unfreeze_last > 0:
        layers = list(backbone.backbone)
        for layer in layers[-unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad = True

print('Backbone defined ✓')


# ---
# # Part 1 — Baseline (Plain Conv Decoder, No Bayesian)

# In[8]:


# Cell 7 — Baseline segmentation model

class BaselineSegmenter(nn.Module):
    """
    YOLO11s-seg backbone + plain Conv decoder.
    No Bayesian layers — deterministic baseline.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.decoder  = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(size=(640, 640), mode='bilinear', align_corners=False),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.decoder(features)  # [B, 1, 640, 640] raw logits


base_backbone = YOLO11sSegBackbone()
freeze_backbone(base_backbone, unfreeze_last=CFG['unfreeze_last'])
baseline_model = BaselineSegmenter(base_backbone).to(device)

total     = sum(p.numel() for p in baseline_model.parameters())
trainable = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
print(f'Baseline  total     : {total:,}')
print(f'Baseline  trainable : {trainable:,}  ({100*trainable/total:.1f}%)')


# In[9]:


# Cell 8 — Generic training loop (used by both models)

def train_segmentation_model(model, train_loader, val_loader, num_epochs,
                              optimizer, scheduler, loss_fn, ckpt_path,
                              label, is_bayesian=False):

    history      = defaultdict(list)
    best_dice    = 0.0
    best_iou     = 0.0
    patience_ctr = 0

    print(f'\n{"="*65}')
    print(f'Training: {label}')
    print(f'{"="*65}')

    for epoch in range(num_epochs):

        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss = train_dice = train_iou = 0.0
        n_batches  = 0

        for images, masks in tqdm(train_loader,
                                  desc=f'Ep {epoch+1:3d}/{num_epochs} [train]',
                                  leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()

            if is_bayesian:
                out   = model(images, sample=True)
                preds = out['predictions']
            else:
                preds = model(images)

            loss = loss_fn(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                d, iou = compute_metrics(preds.detach(), masks)

            train_loss += loss.item()
            train_dice += d
            train_iou  += iou
            n_batches  += 1

        scheduler.step()
        train_loss /= n_batches
        train_dice /= n_batches
        train_iou  /= n_batches

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss = val_dice = val_iou = 0.0
        n_val    = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader,
                                      desc=f'Ep {epoch+1:3d}/{num_epochs} [val]  ',
                                      leave=False):
                images = images.to(device)
                masks  = masks.to(device)

                if is_bayesian:
                    out   = model(images, sample=False)
                    preds = out['predictions']
                else:
                    preds = model(images)

                loss = loss_fn(preds, masks)
                d, iou = compute_metrics(preds, masks)

                val_loss += loss.item()
                val_dice += d
                val_iou  += iou
                n_val    += 1

        val_loss /= n_val
        val_dice /= n_val
        val_iou  /= n_val
        cur_lr    = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        print(f'Ep {epoch+1:3d}/{num_epochs} | '
              f'Loss: {train_loss:.4f} | '
              f'Train Dice: {train_dice:.4f} | '
              f'Val Dice: {val_dice:.4f} | '
              f'Val IoU: {val_iou:.4f} | '
              f'LR: {cur_lr:.1e}')

        if val_dice > best_dice:
            best_dice = val_dice
            best_iou  = val_iou
            patience_ctr = 0
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'val_dice'        : val_dice,
                'val_iou'         : val_iou,
                'metric'          : val_dice,
                'history'         : dict(history),
            }, ckpt_path)
            print(f'  ✓ Best saved  Dice: {val_dice:.4f}  IoU: {val_iou:.4f}')
        else:
            patience_ctr += 1
            if patience_ctr >= CFG['patience']:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    print(f'{"="*65}')
    print(f'{label} done.  Best Dice: {best_dice:.4f}  IoU: {best_iou:.4f}')
    return history, best_dice, best_iou


print('Training loop defined ✓')


# In[ ]:


# Cell 9 — Run baseline training

base_optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, baseline_model.parameters()),
    lr=CFG['baseline_lr'], weight_decay=CFG['weight_decay']
)
base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    base_optimizer, T_max=CFG['num_epochs'], eta_min=1e-6
)
base_loss_fn = DiceBCELoss(bce_weight=0.5)
base_ckpt    = Path(CFG['checkpoint_dir']) / 'baseline_segmentation_best.pt'

base_history, base_best_dice, base_best_iou = train_segmentation_model(
    baseline_model, train_loader, val_loader,
    CFG['num_epochs'], base_optimizer, base_scheduler, base_loss_fn,
    base_ckpt, label='Baseline (YOLO11s-seg + Conv Decoder)',
    is_bayesian=False
)


# ---
# # Part 2 — B-YOLO (Bayesian Segmentation Head)

# In[ ]:


# Cell 10 — B-YOLO segmentation model

class BayesianSegmentationHead(nn.Module):
    """
    Bayesian uncertainty via MC Dropout2d.
    When sample=True, dropout is active even during eval — giving variance across passes.
    """
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1      = nn.BatchNorm2d(256)
        self.conv2    = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2      = nn.BatchNorm2d(128)
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(640, 640), mode='bilinear', align_corners=False)

    def forward(self, x, sample=True):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout2d(x, p=0.3, training=sample)   # Bayesian dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout2d(x, p=0.3, training=sample)   # Bayesian dropout
        return self.upsample(self.conv_out(x))

    def get_kl_divergence(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)


class B_YOLO_Seg(nn.Module):
    """B-YOLO segmentation only — cleaner than the multi-task version for this notebook."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head     = BayesianSegmentationHead(in_channels=512)

    def forward(self, x, sample=True, num_mc_samples=1, return_uncertainty=True):
        with torch.no_grad():
            features = self.backbone(x)

        if return_uncertainty and num_mc_samples > 1:
            preds = torch.stack(
                [self.head(features, sample=True) for _ in range(num_mc_samples)]
            )
            return {
                'predictions': preds.mean(0),
                'uncertainty': preds.var(0),
            }

        pred = self.head(features, sample=sample)
        return {
            'predictions': pred,
            'uncertainty': torch.zeros_like(pred),
        }


byolo_backbone = YOLO11sSegBackbone()
freeze_backbone(byolo_backbone, unfreeze_last=CFG['unfreeze_last'])
byolo_model = B_YOLO_Seg(byolo_backbone).to(device)

total     = sum(p.numel() for p in byolo_model.parameters())
trainable = sum(p.numel() for p in byolo_model.parameters() if p.requires_grad)
print(f'B-YOLO-Seg total     : {total:,}')
print(f'B-YOLO-Seg trainable : {trainable:,}  ({100*trainable/total:.1f}%)')


# In[ ]:


# Cell 11 — Run B-YOLO training (differential LR)

backbone_params = [p for p in byolo_model.backbone.parameters() if p.requires_grad]
head_params     = [p for p in byolo_model.head.parameters() if p.requires_grad]

byolo_optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': CFG['lr_backbone']},
    {'params': head_params,     'lr': CFG['lr_head']},
], weight_decay=CFG['weight_decay'])

byolo_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    byolo_optimizer, T_max=CFG['num_epochs'], eta_min=1e-6
)
byolo_loss_fn = DiceBCELoss(bce_weight=0.5)
byolo_ckpt    = Path(CFG['checkpoint_dir']) / 'b_yolo_segmentation_best.pt'

byolo_history, byolo_best_dice, byolo_best_iou = train_segmentation_model(
    byolo_model, train_loader, val_loader,
    CFG['num_epochs'], byolo_optimizer, byolo_scheduler, byolo_loss_fn,
    byolo_ckpt, label='B-YOLO (Bayesian Segmentation Head)',
    is_bayesian=True
)


# In[ ]:


# Cell 12 — Plot comparison curves

e_base  = range(1, len(base_history['val_dice'])  + 1)
e_byolo = range(1, len(byolo_history['val_dice']) + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(e_base,  base_history['train_loss'],  label='Baseline', color='steelblue')
axes[0].plot(e_byolo, byolo_history['train_loss'], label='B-YOLO',   color='coral')
axes[0].set_title('Training Loss (DiceBCE)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(e_base,  base_history['val_dice'],  label='Baseline', color='steelblue')
axes[1].plot(e_byolo, byolo_history['val_dice'], label='B-YOLO',   color='coral')
axes[1].set_title('Val Dice Score')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice')
axes[1].legend(); axes[1].grid(True)

axes[2].plot(e_base,  base_history['val_iou'],  label='Baseline', color='steelblue')
axes[2].plot(e_byolo, byolo_history['val_iou'], label='B-YOLO',   color='coral')
axes[2].set_title('Val Mask IoU')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('IoU')
axes[2].legend(); axes[2].grid(True)

plt.suptitle('Baseline vs B-YOLO — Instance Segmentation', fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path = Path(CFG['checkpoint_dir']) / 'seg_comparison_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {plot_path}')


# In[ ]:


# Cell 13 — Final results table

print('\n' + '='*60)
print('FINAL RESULTS COMPARISON — SEGMENTATION')
print('='*60)
print(f'{"Model":<35} {"Dice":>8} {"IoU":>8}')
print('-'*60)
print(f'{"Baseline (YOLO11s-seg+Conv)":<35} {base_best_dice:>8.4f} {base_best_iou:>8.4f}')
print(f'{"B-YOLO (Bayesian Head)":<35} {byolo_best_dice:>8.4f} {byolo_best_iou:>8.4f}')
print('='*60)
print(f'B-YOLO improvement: Dice {(byolo_best_dice-base_best_dice):+.4f}  IoU {(byolo_best_iou-base_best_iou):+.4f}')
print('='*60)


# In[ ]:


# Cell 14 — Uncertainty analysis (B-YOLO unique contribution)

print('Loading best B-YOLO checkpoint...')
ckpt = torch.load(byolo_ckpt, map_location=device)
byolo_model.load_state_dict(ckpt['model_state_dict'])
byolo_model.eval()

all_dice, all_iou, all_unc = [], [], []

with torch.no_grad():
    for images, masks in tqdm(val_loader, desc='Uncertainty evaluation'):
        images = images.to(device)
        masks  = masks.to(device)

        out = byolo_model(images, sample=True, num_mc_samples=10, return_uncertainty=True)
        d, iou = compute_metrics(out['predictions'], masks)
        unc = out['uncertainty'].mean(dim=[1,2,3])  # per-image uncertainty

        all_dice.append(d)
        all_iou.append(iou)
        all_unc.extend(unc.cpu().tolist())

mean_dice = np.mean(all_dice)
mean_iou  = np.mean(all_iou)
mean_unc  = np.mean(all_unc)

print('\n' + '='*55)
print('B-YOLO UNCERTAINTY ANALYSIS (10 MC passes)')
print('='*55)
print(f'Final Val Dice              : {mean_dice:.4f}')
print(f'Final Val Mask IoU          : {mean_iou:.4f}')
print(f'Mean pixel uncertainty      : {mean_unc:.6f}')
print(f'High-uncertainty images (>median): '
      f'{sum(u > np.median(all_unc) for u in all_unc)} / {len(all_unc)}')
print('='*55)


# In[ ]:


# Cell 15 — Visualize predictions + uncertainty maps (4 samples)

byolo_model.eval()
images, masks = next(iter(val_loader))
images = images.to(device)

with torch.no_grad():
    out = byolo_model(images[:4], sample=True, num_mc_samples=10, return_uncertainty=True)

pred_masks = torch.sigmoid(out['predictions'][:4])   # [4, 1, 640, 640]
unc_maps   = out['uncertainty'][:4]                  # [4, 1, 640, 640]

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
col_titles = ['Input image', 'Ground truth', 'B-YOLO prediction', 'Uncertainty map']
for col, t in enumerate(col_titles):
    axes[0, col].set_title(t, fontsize=11, fontweight='bold')

for row in range(4):
    # Denormalize image for display
    img_np = images[row].cpu().permute(1,2,0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = np.clip(img_np * std + mean, 0, 1)

    gt_np  = masks[row, 0].cpu().numpy()
    pr_np  = pred_masks[row, 0].cpu().numpy()
    unc_np = unc_maps[row, 0].cpu().numpy()

    # Dice for this image
    pr_bin = (pr_np > 0.5).astype(float)
    inter  = (pr_bin * gt_np).sum()
    dice_i = (2*inter + 1e-8) / (pr_bin.sum() + gt_np.sum() + 1e-8)

    axes[row, 0].imshow(img_np)
    axes[row, 1].imshow(gt_np,  cmap='gray')
    axes[row, 2].imshow(pr_np,  cmap='gray')
    axes[row, 2].set_xlabel(f'Dice: {dice_i:.3f}', fontsize=9)
    im = axes[row, 3].imshow(unc_np, cmap='hot')
    axes[row, 3].set_xlabel(f'Unc: {unc_np.mean():.5f}', fontsize=9)
    plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
    for ax in axes[row]: ax.axis('off')

plt.suptitle('B-YOLO Segmentation — Predictions + Uncertainty Maps', fontsize=13)
plt.tight_layout()
vis_path = Path(CFG['checkpoint_dir']) / 'seg_visualization.png'
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {vis_path}')

