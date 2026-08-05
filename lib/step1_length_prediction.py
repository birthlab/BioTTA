"""Step 1 network architecture and preprocessing utilities."""

import os
import pandas as pd
import numpy as np
import nibabel as nb
from natsort import natsorted
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from typing import Sequence


def extract_brain(data, inds, sz_brain):
    if isinstance(sz_brain, int):
        sz_brain = [sz_brain, sz_brain, sz_brain]
    xsz_brain = inds[1] - inds[0] + 1
    ysz_brain = inds[3] - inds[2] + 1
    zsz_brain = inds[5] - inds[4] + 1
    brain = np.zeros((sz_brain[0], sz_brain[1], sz_brain[2]))
    x_start = int((sz_brain[0] - xsz_brain) / 2)
    y_start = int((sz_brain[1] - ysz_brain) / 2)
    z_start = int((sz_brain[2] - zsz_brain) / 2)
    brain[x_start:x_start+xsz_brain, y_start:y_start+ysz_brain,
          z_start:z_start+zsz_brain] = data[inds[0]:inds[1]+1, inds[2]:inds[3]+1, inds[4]:inds[5]+1]
    return brain

def block_ind(mask):
    tmp = np.nonzero(mask)
    xmin, xmax = np.min(tmp[0]), np.max(tmp[0])
    ymin, ymax = np.min(tmp[1]), np.max(tmp[1])
    zmin, zmax = np.min(tmp[2]), np.max(tmp[2])
    return [xmin, xmax, ymin, ymax, zmin, zmax]


class LocalAppearance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.,
        mode: str = 'add',
    ):
        super().__init__()
        self.mode = mode
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.in_conv = self.Block(in_channels, filters)
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False)
        self.enc1 = self.Block(filters, filters, dropout)
        self.enc2 = self.Block(filters, filters, dropout)
        self.enc3 = self.Block(filters, filters, dropout)
        self.enc4 = self.Block(filters, filters, dropout)
        if mode == 'add':
            self.dec3 = self.Block(filters, filters, dropout)
            self.dec2 = self.Block(filters, filters, dropout)
            self.dec1 = self.Block(filters, filters, dropout)
        else:
            self.dec3 = self.Block(2*filters, filters, dropout)
            self.dec2 = self.Block(2*filters, filters, dropout)
            self.dec1 = self.Block(2*filters, filters, dropout)
        nn.init.trunc_normal_(self.out_conv.weight, 0, 1e-4)

    def Block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        if self.mode == 'add':
            d3 = self.dec3(self.up(e4)+e3)
            d2 = self.dec2(self.up(d3)+e2)
            d1 = self.dec1(self.up(d2)+e1)
        else:
            d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        out = self.out_conv(d1)
        return d1, out

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        return x * channel_att * spatial_att


class FPNFusion(nn.Module):
    """Fuse multi-scale features with a feature pyramid network."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i, in_channels in enumerate(in_channels_list):
            self.lateral_convs.append(nn.Conv3d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv3d(out_channels, out_channels, 3, padding=1))
        
    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i], scale_factor=2, mode='trilinear', align_corners=True
            )
        
        return [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]


class EnhancedLengthPredictor(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.4
    ):
        super().__init__()
        self.pool = pretrained_model.pool
        self.in_conv = pretrained_model.in_conv
        self.enc1 = pretrained_model.enc1
        self.enc2 = pretrained_model.enc2
        self.enc3 = pretrained_model.enc3
        self.enc4 = pretrained_model.enc4

        self.fpn = FPNFusion(
            in_channels_list=[filters, filters, filters, filters],
            out_channels=filters // 2
        )
        
        self.ventricle_attention = CBAM(filters * 2)
        
        self.ventricle_feature_extractor = nn.Sequential(
            nn.Conv3d(filters * 2, filters, 3, padding=1),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(filters, filters, 3, padding=1),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.spp = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        self.spp_proj = nn.Linear(filters * 2 * (1 + 8 + 64), filters * 2)
        
        self.main_fc = nn.Sequential(
            nn.Linear(filters * 2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.ventricle_fc = nn.Sequential(
            nn.Linear(filters, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 4)
        )
        
        self.fusion_weights = nn.Parameter(torch.ones(2, 4))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        fpn_features = self.fpn([e1, e2, e3, e4])
        
        fpn_features_upsampled = []
        for i, feat in enumerate(fpn_features):
            scale_factor = 2 ** i
            if scale_factor > 1:
                feat = F.interpolate(feat, scale_factor=scale_factor, mode='trilinear', align_corners=True)
            fpn_features_upsampled.append(feat)
        
        fused_features = torch.cat(fpn_features_upsampled, dim=1)
        
        ventricle_attn_features = self.ventricle_attention(fused_features)
        
        ventricle_features = self.ventricle_feature_extractor(ventricle_attn_features)
        
        global_features = []
        for pool in self.spp:
            pooled = pool(fused_features)
            global_features.append(pooled.view(pooled.size(0), -1))
        
        global_features_flat = torch.cat(global_features, dim=1)
        global_features_flat = self.spp_proj(global_features_flat)
        
        ventricle_pooled = F.adaptive_avg_pool3d(ventricle_features, 1)
        ventricle_features_flat = ventricle_pooled.view(ventricle_pooled.size(0), -1)
        
        main_lengths = self.main_fc(global_features_flat)
        ventricle_lengths = self.ventricle_fc(ventricle_features_flat)
        
        ventricle_weights = torch.sigmoid(self.fusion_weights[0])
        main_weights = torch.sigmoid(self.fusion_weights[1])
        
        total_weights = ventricle_weights + main_weights
        ventricle_weights = ventricle_weights / total_weights
        main_weights = main_weights / total_weights
        
        fused_ventricle_lengths = (
            ventricle_weights * ventricle_lengths + 
            main_weights * main_lengths[:, :4]
        )
        
        combined_lengths = torch.cat([fused_ventricle_lengths, main_lengths[:, 4:]], dim=1)
        
        return combined_lengths, ventricle_attn_features
    

class TestImageDataset(Dataset):
    def __init__(self, image_dir, label_csv_path):
        self.image_paths = []
        self.image_names = []
        self.label_mapping = {}
        
        img_files = natsorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        print(f"Found {len(img_files)} images in test folder")
        
        for img_file in img_files:
            img_name = img_file[:-7] if img_file.endswith('.nii.gz') else img_file[:-4]
            self.image_paths.append(os.path.join(image_dir, img_file))
            self.image_names.append(img_name)
        
        if label_csv_path and os.path.exists(label_csv_path):
            df = pd.read_csv(label_csv_path, dtype={'label_name':str})
            for _, row in df.iterrows():
                img_name = row['label_name']
                lengths = row[biometry_list].values.astype(float).tolist()
                self.label_mapping[img_name] = lengths
        else:
            print("Warning: Label CSV not provided or not found. Loss will not be computed.")
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        
        X_T1 = nb.load(img_path).get_fdata()
        mask = X_T1 > 0
        
        X_T1_brain = X_T1[mask]
        mean = np.mean(X_T1_brain) if len(X_T1_brain) > 0 else 0
        std = np.std(X_T1_brain) if len(X_T1_brain) > 0 else 1
        normalized_brain = (X_T1_brain - mean) / std if len(X_T1_brain) > 0 else X_T1_brain
        X_T1[mask] = normalized_brain
        X_T1[~mask] = 0
        
        ind_brain = block_ind(mask)
        X_T1 = extract_brain(X_T1, ind_brain, [128,160,128])
        data = X_T1.reshape((1,)+X_T1.shape)
        data = torch.tensor(data, dtype=torch.float32)

        label = self.label_mapping.get(img_name, None)
        if label is not None:
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.full((len(biometry_list),), float('nan'), dtype=torch.float32)
            
        return data, label, img_name


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None, ventricle_indices=None):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else torch.ones(11)
        self.ventricle_indices = ventricle_indices if ventricle_indices is not None else []
    
    def forward(self, pred, target):
        per_feature_loss = (pred - target)**2
        
        weights_tensor = self.weights.to(pred.device)
        
        weighted_loss = per_feature_loss * weights_tensor
        
        if self.ventricle_indices:
            ventricle_loss = per_feature_loss[self.ventricle_indices] * 5.0
            weighted_loss[self.ventricle_indices] = ventricle_loss
        
        return weighted_loss.mean()
