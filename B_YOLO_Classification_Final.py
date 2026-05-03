#!/usr/bin/env python
# coding: utf-8

# # B-YOLO Classification — Retraining + Baseline Comparison
# **IIT KGP H100 Server**
# 
# Runs two models back to back:
# 1. **Baseline** — frozen YOLO11s backbone + plain Linear head (no Bayesian layers)
# 2. **B-YOLO** — frozen YOLO11s backbone + Bayesian head + classification head
# 
# Both trained for 100 epochs. Results printed and plotted together.

# In[1]:


# Cell 1 — GPU check
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}  : {p.name}  {p.total_memory//1024**3} GB')


# In[2]:


# Cell 2 — CONFIG  ← only cell you need to edit
import os

BASE = os.path.expanduser('~/nas/model/B_YOLO')

CFG = {
    # paths
    'image_dir'     : f'{BASE}/5_Classification/images',
    'train_json'    : f'{BASE}/5_Classification/train.json',
    'val_json'      : f'{BASE}/5_Classification/test.json',
    'checkpoint_dir': f'{BASE}/checkpoints',

    # training
    'num_epochs'    : 100,
    'batch_size'    : 512,
    'num_workers'   : 8,
    'patience'      : 15,

    # B-YOLO specific
    'lr_head'       : 1e-3,   # LR for Bayesian head + classification head
    'lr_backbone'   : 1e-4,   # Lower LR for unfrozen backbone layers
    'weight_decay'  : 1e-4,
    'kl_weight'     : 1e-5,   # was 0.01
    'mc_samples'    : 5,
    'unfreeze_last' : 2,      # 2 is better than 4 based on results

    # Baseline specific
    'baseline_lr'   : 1e-3,

    'num_classes'   : None,   # auto-detected
    'gpu_id'        : 0,
}

os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
print('Config OK ✓')


# In[3]:


# Cell 3 — Imports
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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


# In[4]:


# Cell 4 — Auto-detect num_classes
with open(CFG['train_json'], 'r') as f:
    data = json.load(f)

if isinstance(data, dict) and 'annotations' in data:
    all_ids = [ann['category_id'] for ann in data['annotations']]
else:
    all_ids = [s.get('label', s.get('category_id')) for s in data]

CFG['num_classes'] = max(all_ids) + 1
print(f'Category IDs : {min(all_ids)} → {max(all_ids)}')
print(f'num_classes  : {CFG["num_classes"]}')
print(f'Annotations  : {len(all_ids)}')


# In[5]:


# Cell 5 — Dataset

class JSONClassificationDataset(Dataset):
    def __init__(self, image_dir, json_path, transforms=None):
        self.image_dir  = image_dir
        self.transforms = transforms
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = []
        if isinstance(data, dict) and 'annotations' in data:
            img_dict = {img['id']: img['file_name'] for img in data['images']}
            for ann in data['annotations']:
                self.samples.append({
                    'file_name': img_dict[ann['image_id']],
                    'label'    : ann['category_id']
                })
        elif isinstance(data, list):
            self.samples = data

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        fname = s.get('file_name', s.get('image', s.get('filename')))
        label = s.get('label', s.get('category_id', s.get('class_id')))
        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(label, dtype=torch.long)


train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256)),
    v2.RandomCrop((224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = JSONClassificationDataset(CFG['image_dir'], CFG['train_json'], train_transforms)
val_dataset   = JSONClassificationDataset(CFG['image_dir'], CFG['val_json'],   val_transforms)

train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'],
                          shuffle=True,  num_workers=CFG['num_workers'], pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)

print(f'Train : {len(train_dataset)} samples, {len(train_loader)} batches')
print(f'Val   : {len(val_dataset)} samples, {len(val_loader)} batches')


# In[6]:


# Cell 6 — Shared backbone builder (used by both models)

class YOLO11sBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        yolo = YOLO('yolo11s.pt')
        self.backbone = yolo.model.model[:10]
    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        return x

def freeze_backbone(backbone, unfreeze_last=0):
    for p in backbone.parameters():
        p.requires_grad = False
    if unfreeze_last > 0:
        layers = list(backbone.backbone)
        for layer in layers[-unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad = True

print('Backbone class defined ✓')


# ---
# # Part 1 — Baseline Model (Plain YOLO11s + Linear Head)

# In[7]:


# Cell 7 — Baseline model definition

class BaselineClassifier(nn.Module):
    """Plain YOLO11s backbone + single Linear head. No Bayesian layers."""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # Slightly deeper head to match parameter count more fairly
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        return self.head(features)


# Build baseline
base_backbone = YOLO11sBackbone()
freeze_backbone(base_backbone, unfreeze_last=2)
baseline_model = BaselineClassifier(base_backbone, CFG['num_classes']).to(device)

total     = sum(p.numel() for p in baseline_model.parameters())
trainable = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
print(f'Baseline  total params    : {total:,}')
print(f'Baseline  trainable params: {trainable:,}  ({100*trainable/total:.1f}%)')


# In[ ]:


# # Cell 8 — Train baseline

# def train_model(model, train_loader, val_loader, num_epochs, optimizer,
#                 scheduler, loss_fn, ckpt_path, label, is_bayesian=False, kl_weight=0.0):
#     """Generic training loop — works for both baseline and B-YOLO."""
#     history   = defaultdict(list)
#     best_top1 = 0.0
#     best_top5 = 0.0
#     patience_ctr = 0

#     print(f'\n{"="*65}')
#     print(f'Training: {label}')
#     print(f'{"="*65}')

#     for epoch in range(num_epochs):
#         # ── Train ──────────────────────────────────────────────────
#         model.train()
#         train_loss = train_correct = train_total = 0

#         for images, targets in tqdm(train_loader,
#                                     desc=f'Ep {epoch+1:3d}/{num_epochs} [train]',
#                                     leave=False):
#             images, targets = images.to(device), targets.to(device)
#             optimizer.zero_grad()

#             if is_bayesian:
#                 out   = model(images, sample=True, num_mc_samples=1, return_uncertainty=False)
#                 preds = out['predictions']
#                 kl    = model.get_kl_divergence() / len(train_loader)
#                 loss  = loss_fn(preds, targets) + kl_weight * kl
#             else:
#                 preds = model(images)
#                 loss  = loss_fn(preds, targets)

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#             train_loss    += loss.item()
#             train_correct += (preds.argmax(dim=1) == targets).sum().item()
#             train_total   += targets.size(0)

#         scheduler.step()
#         train_acc  = train_correct / train_total
#         train_loss = train_loss / len(train_loader)

#         # ── Validate ───────────────────────────────────────────────
#         model.eval()
#         v_top1 = v_top5 = v_total = 0

#         with torch.no_grad():
#             for images, targets in tqdm(val_loader,
#                                         desc=f'Ep {epoch+1:3d}/{num_epochs} [val]  ',
#                                         leave=False):
#                 images, targets = images.to(device), targets.to(device)

#                 if is_bayesian:
#                     out   = model(images, sample=False, num_mc_samples=1, return_uncertainty=False)
#                     preds = out['predictions']
#                 else:
#                     preds = model(images)

#                 v_top1  += (preds.argmax(dim=1) == targets).sum().item()
#                 top5     = preds.topk(min(5, CFG['num_classes']), dim=1)[1]
#                 v_top5  += sum(targets[i] in top5[i] for i in range(targets.size(0)))
#                 v_total += targets.size(0)

#         val_top1 = v_top1 / v_total
#         val_top5 = v_top5 / v_total
#         cur_lr   = optimizer.param_groups[0]['lr']

#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_top1'].append(val_top1)
#         history['val_top5'].append(val_top5)

#         print(f'Ep {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.3f} | '
#               f'Train: {train_acc*100:.1f}% | '
#               f'Val Top-1: {val_top1*100:.2f}% | '
#               f'Val Top-5: {val_top5*100:.2f}% | LR: {cur_lr:.1e}')

#         if val_top1 > best_top1:
#             best_top1 = val_top1
#             best_top5 = val_top5
#             patience_ctr = 0
#             torch.save({
#                 'epoch'           : epoch,
#                 'model_state_dict': model.state_dict(),
#                 'val_top1'        : val_top1,
#                 'val_top5'        : val_top5,
#                 'metric'          : val_top1,
#                 'num_classes'     : CFG['num_classes'],
#                 'history'         : dict(history),
#             }, ckpt_path)
#             print(f'  ✓ Best saved  Top-1: {val_top1*100:.2f}%  Top-5: {val_top5*100:.2f}%')
#         else:
#             patience_ctr += 1
#             if patience_ctr >= CFG['patience']:
#                 print(f'  Early stopping at epoch {epoch+1}')
#                 break

#     print(f'{"="*65}')
#     print(f'{label} done.  Best Top-1: {best_top1*100:.2f}%  Top-5: {best_top5*100:.2f}%')
#     return history, best_top1, best_top5


# # Optimizer — single LR for baseline (no backbone fine-tuning trick needed)
# base_optimizer = optim.AdamW(
#     filter(lambda p: p.requires_grad, baseline_model.parameters()),
#     lr=CFG['baseline_lr'], weight_decay=CFG['weight_decay']
# )
# base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     base_optimizer, T_max=CFG['num_epochs'], eta_min=1e-6
# )
# base_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
# base_ckpt    = Path(CFG['checkpoint_dir']) / 'baseline_classification_best.pt'

# base_history, base_best_top1, base_best_top5 = train_model(
#     baseline_model, train_loader, val_loader,
#     CFG['num_epochs'], base_optimizer, base_scheduler, base_loss_fn,
#     base_ckpt, label='Baseline (YOLO11s + Linear Head)',
#     is_bayesian=False
# )


# In[12]:


# ══════════════════════════════════════════════════════════════════
# STOP BASELINE NOW — best already saved at epoch 71 (32.70% Top-1)
# Run THIS cell to start B-YOLO immediately
# ══════════════════════════════════════════════════════════════════

# # Record baseline result (already saved in checkpoint)
# base_best_top1 = 0.3270
# base_best_top5 = 0.6160
# print(f'Baseline result (from epoch 71): Top-1 {base_best_top1*100:.2f}%  Top-5 {base_best_top5*100:.2f}%')
# print('Starting B-YOLO training...\n')



# ---
# # Part 2 — B-YOLO (Bayesian Head)

# In[14]:


# Cell 9 — B-YOLO model definition

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.1):
        super().__init__()
        self.prior_std     = prior_std
        self.weight_mean   = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mean     = nn.Parameter(torch.zeros(out_features))           # randn → zeros
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -3.0))  # was randn*0.1-5
        self.bias_logvar   = nn.Parameter(torch.full((out_features,), -3.0))              # was randn*0.1-5
        
    def forward(self, x, sample=True):
        if sample:
            w = self.weight_mean + torch.exp(0.5 * self.weight_logvar) * torch.randn_like(self.weight_mean)
            b = self.bias_mean   + torch.exp(0.5 * self.bias_logvar)   * torch.randn_like(self.bias_mean)
        else:
            w, b = self.weight_mean, self.bias_mean
        return F.linear(x, w, b)

    def get_kl_divergence(self):
        wv  = torch.exp(self.weight_logvar)
        bv  = torch.exp(self.bias_logvar)
        pv  = torch.tensor(self.prior_std**2, device=wv.device)
        kl  = lambda v, m, lv: 0.5 * torch.sum((v + m**2)/pv - 1 + torch.log(pv) - lv)
        return kl(wv, self.weight_mean, self.weight_logvar) + kl(bv, self.bias_mean, self.bias_logvar)


class BayesianHead(nn.Module):
    def __init__(self, input_channels=512, hidden1=512, hidden2=256):
        super().__init__()
        self.fc1     = BayesianLinear(input_channels, hidden1)
        self.fc2     = BayesianLinear(hidden1, hidden2)
        self.bn1     = nn.BatchNorm1d(hidden1)
        self.bn2     = nn.BatchNorm1d(hidden2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, sample=True):
        x = self.dropout(self.relu(self.bn1(self.fc1(x, sample))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x, sample))))
        return x

    def get_kl_divergence(self):
        return self.fc1.get_kl_divergence() + self.fc2.get_kl_divergence()


class B_YOLO(nn.Module):
    """
    Cleaner single-task B-YOLO for classification only.
    No unused task heads wasting memory.
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head     = BayesianHead(512, 512, 256)
        self.cls_head = nn.Linear(256, num_classes)
 
    def forward(self, x, sample=True, num_mc_samples=1, return_uncertainty=True):
        with torch.no_grad():
            features = self.backbone(x)
        features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
 
        if return_uncertainty and num_mc_samples > 1:
            preds = torch.stack(
                [self.cls_head(self.head(features, sample=True)) for _ in range(num_mc_samples)]
            )
            return {'predictions': preds.mean(0), 'uncertainty': preds.var(0)}
 
        pred = self.cls_head(self.head(features, sample=sample))
        return {'predictions': pred, 'uncertainty': torch.zeros_like(pred)}
 
    def get_kl_divergence(self):
        return self.head.get_kl_divergence()


# Build B-YOLO
byolo_backbone = YOLO11sBackbone()
freeze_backbone(byolo_backbone, unfreeze_last=CFG['unfreeze_last'])
byolo_model = B_YOLO(byolo_backbone, num_classes=CFG['num_classes'])
byolo_model.to(device)

total     = sum(p.numel() for p in byolo_model.parameters())
trainable = sum(p.numel() for p in byolo_model.parameters() if p.requires_grad)
print(f'B-YOLO total params    : {total:,}')
print(f'B-YOLO trainable params: {trainable:,}  ({100*trainable/total:.1f}%)')


# In[ ]:


# Cell 10 — Train B-YOLO
# Key fix: separate LR for backbone layers vs head layers

# Cell 10 — Train B-YOLO (fixed: KL annealing + corrected optimizer)

backbone_params = [p for p in byolo_model.backbone.parameters() if p.requires_grad]
head_params     = [p for p in byolo_model.parameters()
                   if p.requires_grad and not any(p is bp for bp in backbone_params)]

byolo_optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': CFG['lr_backbone']},
    {'params': head_params,     'lr': CFG['lr_head']},
], weight_decay=CFG['weight_decay'])

byolo_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    byolo_optimizer, T_max=CFG['num_epochs'], eta_min=1e-6
)
byolo_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
byolo_ckpt    = Path(CFG['checkpoint_dir']) / 'b_yolo_classification_best.pt'

# ── Custom training loop with KL annealing ──────────────────────
from collections import defaultdict

history   = defaultdict(list)
best_top1 = 0.0
best_top5 = 0.0
patience_ctr = 0
KL_WARMUP_EPOCHS = 20   # ramp KL from 0 → kl_weight over 20 epochs

print(f'\n{"="*65}')
print(f'Training: B-YOLO (Bayesian Head) — with KL annealing')
print(f'{"="*65}')

for epoch in range(CFG['num_epochs']):

    # ── KL weight for this epoch ──────────────────────────────────
    kl_scale      = min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)
    effective_kl  = CFG['kl_weight'] * kl_scale

    # ── Train ─────────────────────────────────────────────────────
    byolo_model.train()
    train_loss = train_correct = train_total = 0

    for images, targets in tqdm(train_loader,
                                desc=f'Ep {epoch+1:3d}/{CFG["num_epochs"]} [train]',
                                leave=False):
        images, targets = images.to(device), targets.to(device)
        byolo_optimizer.zero_grad()

        # MC samples during training for better gradient estimates
        preds_list = [byolo_model(images, sample=True, num_mc_samples=1,
                                   return_uncertainty=False)['predictions']
                      for _ in range(CFG['mc_samples'])]
        preds = torch.stack(preds_list).mean(0)

        kl   = byolo_model.get_kl_divergence() / len(train_loader)
        loss = byolo_loss_fn(preds, targets) + effective_kl * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(byolo_model.parameters(), 1.0)
        byolo_optimizer.step()

        train_loss    += loss.item()
        train_correct += (preds.argmax(dim=1) == targets).sum().item()
        train_total   += targets.size(0)

    byolo_scheduler.step()
    train_acc  = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # ── Validate ──────────────────────────────────────────────────
    byolo_model.eval()
    v_top1 = v_top5 = v_total = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader,
                                    desc=f'Ep {epoch+1:3d}/{CFG["num_epochs"]} [val]  ',
                                    leave=False):
            images, targets = images.to(device), targets.to(device)
            out   = byolo_model(images, sample=False, num_mc_samples=1,
                                return_uncertainty=False)
            preds = out['predictions']

            v_top1 += (preds.argmax(dim=1) == targets).sum().item()
            top5    = preds.topk(min(5, CFG['num_classes']), dim=1)[1]
            v_top5 += sum(targets[i] in top5[i] for i in range(targets.size(0)))
            v_total += targets.size(0)

    val_top1 = v_top1 / v_total
    val_top5 = v_top5 / v_total
    cur_lr   = byolo_optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_top1'].append(val_top1)
    history['val_top5'].append(val_top5)

    print(f'Ep {epoch+1:3d}/{CFG["num_epochs"]} | Loss: {train_loss:.3f} | '
          f'Train: {train_acc*100:.1f}% | '
          f'Val Top-1: {val_top1*100:.2f}% | Val Top-5: {val_top5*100:.2f}% | '
          f'LR: {cur_lr:.1e} | KL×: {kl_scale:.2f}')

    if val_top1 > best_top1:
        best_top1    = val_top1
        best_top5    = val_top5
        patience_ctr = 0
        torch.save({
            'epoch': epoch, 'model_state_dict': byolo_model.state_dict(),
            'val_top1': val_top1, 'val_top5': val_top5,
            'num_classes': CFG['num_classes'], 'history': dict(history),
        }, byolo_ckpt)
        print(f'  ✓ Best saved  Top-1: {val_top1*100:.2f}%  Top-5: {val_top5*100:.2f}%')
    else:
        patience_ctr += 1
        if patience_ctr >= CFG['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

print(f'{"="*65}')
print(f'B-YOLO done.  Best Top-1: {best_top1*100:.2f}%  Top-5: {best_top5*100:.2f}%')
byolo_best_top1 = best_top1
byolo_best_top5 = best_top5
byolo_history   = history


# In[ ]:
# Restore baseline results from saved checkpoint
base_best_top1 = 0.3283   # from epoch 71
base_best_top5 = 0.6120

# Cell 11 — Plot comparison curves
# Cell 11 — Plot comparison curves

# Load baseline history from checkpoint (Cell 8 was skipped)
base_ckpt_path = Path(CFG['checkpoint_dir']) / 'baseline_classification_best.pt'
base_ckpt_data = torch.load(base_ckpt_path, map_location='cpu')
base_history   = base_ckpt_data['history']

e_base  = range(1, len(base_history['val_top1'])  + 1)
e_byolo = range(1, len(byolo_history['val_top1']) + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(e_base,  base_history['train_loss'],  label='Baseline', color='steelblue')
axes[0].plot(e_byolo, byolo_history['train_loss'],  label='B-YOLO',   color='coral')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True)

# Top-1
axes[1].plot(e_base,  [x*100 for x in base_history['val_top1']],  label='Baseline Top-1', color='steelblue')
axes[1].plot(e_byolo, [x*100 for x in byolo_history['val_top1']], label='B-YOLO Top-1',   color='coral')
axes[1].set_title('Val Top-1 Accuracy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(True)

# Top-5
axes[2].plot(e_base,  [x*100 for x in base_history['val_top5']],  label='Baseline Top-5', color='steelblue')
axes[2].plot(e_byolo, [x*100 for x in byolo_history['val_top5']], label='B-YOLO Top-5',   color='coral')
axes[2].set_title('Val Top-5 Accuracy')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy (%)')
axes[2].legend(); axes[2].grid(True)

plt.suptitle('Baseline vs B-YOLO — Classification on COCO', fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path = Path(CFG['checkpoint_dir']) / 'comparison_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {plot_path}')

# Cell 12 — Final side-by-side results table
# Cell 12 — Final side-by-side results table
# DELETE these two lines that are currently at the top of this cell:
# base_best_top1 = 0.3283
# base_best_top5 = 0.6120
# Those are already set in the cell above Cell 9 (the hardcoded restore cell)

print('\n' + '='*60)
print('FINAL RESULTS COMPARISON')
print('='*60)
print(f'{"Model":<30} {"Top-1":>8} {"Top-5":>8}')
print('-'*60)
print(f'{"Baseline (YOLO11s+Linear)":<30} {base_best_top1*100:>7.2f}% {base_best_top5*100:>7.2f}%')
print(f'{"B-YOLO (Bayesian Head)":<30} {byolo_best_top1*100:>7.2f}% {byolo_best_top5*100:>7.2f}%')
print('='*60)
delta1 = (byolo_best_top1 - base_best_top1) * 100
delta5 = (byolo_best_top5 - base_best_top5) * 100
print(f'B-YOLO improvement: Top-1 {delta1:+.2f}%  Top-5 {delta5:+.2f}%')
print('='*60)

# In[ ]:


# Cell 13 — B-YOLO uncertainty evaluation (the unique contribution of Bayesian approach)

print('Loading best B-YOLO checkpoint...')
ckpt = torch.load(byolo_ckpt, map_location=device)
byolo_model.load_state_dict(ckpt['model_state_dict'], strict=False)
byolo_model.eval()

all_preds, all_targets, all_confs, all_unc, all_correct = [], [], [], [], []

with torch.no_grad():
    for images, targets in tqdm(val_loader, desc='Uncertainty evaluation'):
        images = images.to(device)

        out  = byolo_model(images, sample=True, num_mc_samples=30, return_uncertainty=True)
        prob = torch.softmax(out['predictions'], dim=1)
        pred = prob.argmax(dim=1)
        conf = prob.max(dim=1).values
        unc  = out['uncertainty'].mean(dim=1)

        all_preds.append(pred.cpu())
        all_targets.append(targets)
        all_confs.append(conf.cpu())
        all_unc.append(unc.cpu())
        all_correct.append((pred.cpu() == targets).float())

all_preds   = torch.cat(all_preds)
all_targets = torch.cat(all_targets)
all_confs   = torch.cat(all_confs)
all_unc     = torch.cat(all_unc)
all_correct = torch.cat(all_correct)

# Key insight: correct predictions should have lower uncertainty than wrong ones
unc_correct = all_unc[all_correct == 1].mean().item()
unc_wrong   = all_unc[all_correct == 0].mean().item()

print('\n' + '='*55)
print('B-YOLO UNCERTAINTY ANALYSIS')
print('='*55)
print(f'Final Top-1 Accuracy     : {all_correct.mean().item()*100:.2f}%')
print(f'Mean confidence          : {all_confs.mean().item():.4f}')
print(f'Mean uncertainty (all)   : {all_unc.mean().item():.6f}')
print(f'Mean uncertainty (correct predictions) : {unc_correct:.6f}')
print(f'Mean uncertainty (wrong  predictions)  : {unc_wrong:.6f}')
print(f'Uncertainty ratio wrong/correct        : {unc_wrong/unc_correct:.2f}x')
print('  → Higher ratio means Bayesian uncertainty is meaningful')
print('='*55)

