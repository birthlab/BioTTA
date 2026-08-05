"""Data, loss, registration, and visualization helpers for BioTTA Step 2."""

import os
from pathlib import Path
import ants
import numpy as np
import nibabel as nb
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TENTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self._freeze_non_bn_params()
        
        self._configure_bn_layers()
        
        self._set_special_layers_eval()

    def _freeze_non_bn_params(self):
        """Freeze every parameter except batch-normalization affine terms."""
        for param in self.model.parameters():
            param.requires_grad = False
            
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)

    def _configure_bn_layers(self):
        """Use current-batch statistics without accumulating running values."""
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = False
                m.reset_running_stats()
                m.momentum = 0.0

    def _set_special_layers_eval(self):
        """Keep stochastic layers such as dropout in evaluation mode."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()

    def forward(self, x):
        self.model.train()  
        return self.model(x)


def extract_brain(data, inds, sz_brain):
    if isinstance(sz_brain, int):
        sz_brain = [sz_brain, sz_brain, sz_brain]
    xsz_brain = inds[1] - inds[0] + 1
    ysz_brain = inds[3] - inds[2] + 1
    zsz_brain = inds[5] - inds[4] + 1
    # print(data.shape)
    # print(inds)
    brain = np.zeros((sz_brain[0], sz_brain[1], sz_brain[2]))
    x_start = int((sz_brain[0] - xsz_brain) / 2)
    y_start = int((sz_brain[1] - ysz_brain) / 2)
    z_start = int((sz_brain[2] - zsz_brain) / 2)
    brain[x_start:x_start+xsz_brain,y_start:y_start+ysz_brain,
          z_start:z_start+zsz_brain] = data[inds[0]:inds[1]+1,inds[2]:inds[3]+1,inds[4]:inds[5]+1]
    return brain

def convert_coords_to_original(coords, inds, sz_brain):
    """Map coordinates from the centered crop back to the original volume."""
    if isinstance(sz_brain, int):
        sz_brain = [sz_brain, sz_brain, sz_brain]
    
    coords = np.asarray(coords)
    was_1d = coords.ndim == 1
    if was_1d:
        coords = coords.reshape(1, -1)
    
    xsz_brain = inds[1] - inds[0] + 1
    ysz_brain = inds[3] - inds[2] + 1
    zsz_brain = inds[5] - inds[4] + 1
    
    x_start = int((sz_brain[0] - xsz_brain) / 2)
    y_start = int((sz_brain[1] - ysz_brain) / 2)
    z_start = int((sz_brain[2] - zsz_brain) / 2)
    
    original_coords = coords.copy()
    original_coords[:, 0] = coords[:, 0] - x_start + inds[0]
    original_coords[:, 1] = coords[:, 1] - y_start + inds[2]
    original_coords[:, 2] = coords[:, 2] - z_start + inds[4]
    
    if was_1d:
        return original_coords[0]
    
    return original_coords

def block_ind(mask):
    tmp = np.nonzero(mask);
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]
    return ind_brain

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, target_image_path, pred_length_df):
        self.target_image_path = target_image_path
        self.target_image = [nb.load(self.target_image_path).get_fdata()]
        self.affine = [nb.load(self.target_image_path).affine]
        target_name = Path(target_image_path).name
        self.target_image_name = target_name[:-7] if target_name.lower().endswith('.nii.gz') else Path(target_name).stem
        self.pred_length_df = pred_length_df
        
    def __len__(self):
        return len(self.target_image)

    def detect_orientation(self, affine_matrix):
        """Classify an affine as neurological or radiological orientation."""
        orientation_vec = affine_matrix[:3, :3]
        
        if orientation_vec[0, 0] > 0:
            # print('RAS')
            return 'radiological'
        else:
            # print('LPS')
            return 'neurological'
    
    def __getitem__(self, idx):
        image_data = self.target_image[idx]
        X_T1  = image_data.copy()
        mask = X_T1 > 0
        X_T1_brain = X_T1[mask]
        mean = np.mean(X_T1_brain)
        std = np.std(X_T1_brain)
        normalized_brain = (X_T1_brain - mean) / std
        X_T1[mask] = normalized_brain
        X_T1[~mask] = 0
        ind_brain = block_ind(mask)
        X_T1 = extract_brain(X_T1, ind_brain, [128,160,128])
        
        # orientation = self.detect_orientation(img_affine)
        # if orientation == 'neurological': 
        #     swap_matrix = np.eye(4)
        #     img_affine = swap_matrix

        data = X_T1.reshape((1,)+X_T1.shape)
        data = torch.tensor(data, dtype=torch.float32) # torch.Size([1, 128, 160, 128])
        row = self.pred_length_df.loc[
            self.pred_length_df['label_name'] == self.target_image_name, 
            self.pred_length_df.columns != 'label_name'
        ].values.astype(np.float32)
        pred_length = torch.tensor(row.squeeze(), dtype=torch.float32) # torch.Size([11])
        data_pair = {'image': data, 'length': pred_length}
        return data_pair


class EntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        x_flat = x.view(B * C, -1)
        max_vals = x_flat.max(dim=1, keepdim=True).values
        data_exp = torch.exp(x_flat - max_vals)
        sum_exp = data_exp.sum(dim=1, keepdim=True)
        data_softmax = data_exp / sum_exp
        data_clipped = torch.clamp(data_softmax, min=self.epsilon)
        p_logp = - data_clipped * torch.log(data_clipped)  # [B*C, N]
        channel_entropies = p_logp.sum(dim=1)
        return channel_entropies.view(B, C).mean(dim=[0,1])
    

class DifferentiableLandmarkDetector(nn.Module):
    def __init__(self, init_temp, topk):
        """Create a sparse, differentiable landmark detector."""
        super().__init__()
        # self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.temperature = init_temp
        self.topk = topk
        
    def forward(self, heatmap):
        """Convert heatmaps shaped [B, C, D, H, W] into [B, C, 3] coordinates."""
        B, C, D, H, W = heatmap.shape
        # print(torch.max(heatmap),torch.min(heatmap))
        device = heatmap.device
        x, y, z = torch.meshgrid(
            torch.arange(D, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        coords = torch.stack((x, y, z), dim=-1).float()  # [D, H, W, 3]
        coords = coords.permute(3, 0, 1, 2)
        
        coords_flat = coords.reshape(3, -1).permute(1, 0)  # [D*H*W, 3]
        heat_flat = heatmap.view(B, C, -1)  # [B, C, D*H*W]
        actual_topk = min(self.topk, heat_flat.size(-1))
        values, indices = torch.topk(heat_flat, actual_topk, dim=-1)
        
        weights = values / self.temperature 
        weights = weights - weights.max(dim=-1, keepdim=True).values
        exp_weights = torch.exp(weights)
        probs = exp_weights / (exp_weights.sum(dim=-1, keepdim=True) + 1e-20)

        selected_coords = coords_flat[indices.view(-1)].view(B, C, actual_topk, 3)
        weighted_coords = torch.sum(probs.unsqueeze(-1) * selected_coords, dim=2)

        return weighted_coords


class LengthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, keypoints, length_pseudo_label):
        """Match lengths from 22 keypoints to 11 pseudo-label measurements."""
        C = keypoints.shape[1]
        pred_lengths = []
        for i in range(0, C, 2):
            dist = torch.norm(keypoints[:, i, :] - keypoints[:, i+1, :], dim=-1)
            pred_lengths.append(dist)
        pred_lengths = torch.stack(pred_lengths, dim=1)
        # print(pred_lengths)
        # print(length_pseudo_label)
        
        loss = F.mse_loss(pred_lengths, length_pseudo_label)
        return loss

class BoundaryGradientLoss(nn.Module):
    """Encourage selected landmark pairs to lie on strong image gradients."""
    
    def __init__(self, 
                 point_pairs: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 kernel_type: str = 'scharr',
                 smooth_sigma: float = 1.0,
                 grad_sigma: float = 1.0,
                 adaptive_threshold: bool = True,
                 min_grad: float = 0.3):
        super().__init__()
        
        if kernel_type not in ['sobel', 'scharr']:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        self.point_indices = []
        for pair_idx in point_pairs:
            if pair_idx < 1:
                raise ValueError("Point pair indices start from 1")
            start_idx = (pair_idx - 1) * 2
            self.point_indices.extend([start_idx, start_idx + 1])
        
        self.kernel_type = kernel_type
        self.smooth_sigma = smooth_sigma
        self.grad_sigma = grad_sigma
        self.adaptive_threshold = adaptive_threshold
        self.min_grad = min_grad
        
        self.smooth_kernel = self._create_smooth_kernel()

        self.gradient_kernel = self._create_gradient_kernel()
    
    def _create_smooth_kernel(self) -> torch.Tensor:
        """Create a normalized one-dimensional Gaussian kernel."""
        kernel_size = max(3, int(2 * 2 * self.smooth_sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        k = torch.arange(kernel_size) - kernel_size//2
        k = torch.exp(-k**2 / (2 * self.smooth_sigma**2))
        return k / k.sum()
    
    def _gaussian_smooth(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply separable Gaussian smoothing to a 3D volume."""
        smoothed = volume.clone()
        kernel = self.smooth_kernel.to(volume.device)
        
        for dim, padding_size in zip([2, 3, 4], 
                                [kernel.size(0)//2, kernel.size(0)//2, kernel.size(0)//2]):
            kernel_shape = [1, 1, 1, 1, 1]
            kernel_shape[dim] = -1
            conv_kernel = kernel.view(*kernel_shape)
            
            pad_size = kernel.size(0) - 1
            pad_before = pad_size // 2
            pad_after = pad_size - pad_before
            
            if dim == 2:
                smoothed = F.pad(smoothed, (0, 0, 0, 0, pad_before, pad_after), mode='replicate')
            elif dim == 3:
                smoothed = F.pad(smoothed, (0, 0, pad_before, pad_after, 0, 0), mode='replicate')
            else:
                smoothed = F.pad(smoothed, (pad_before, pad_after, 0, 0, 0, 0), mode='replicate')
            
            smoothed = F.conv3d(
                smoothed, 
                conv_kernel,
                padding=0
            )
        
        return smoothed
    
    def _create_gradient_kernel(self) -> torch.Tensor:
        """Create 3D gradient kernels for the Z, Y, and X axes."""
        if self.kernel_type == 'sobel':
            kx = torch.tensor([
                [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]],
                
                [[2, 0, -2],
                [4, 0, -4],
                [2, 0, -2]],
                
                [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]]
            ], dtype=torch.float32)
            
            ky = torch.tensor([
                [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]],
                
                [[2, 4, 2],
                [0, 0, 0],
                [-2, -4, -2]],
                
                [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]
            ], dtype=torch.float32)
            
            kz = torch.tensor([
                [[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]],
                
                [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
                
                [[-1, -2, -1],
                [-2, -4, -2],
                [-1, -2, -1]]
            ], dtype=torch.float32)
        else:  # scharr
            kx = torch.tensor([
                [[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]],
                
                [[10, 0, -10],
                [30, 0, -30],
                [10, 0, -10]],
                
                [[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]]
            ], dtype=torch.float32)
            
            ky = torch.tensor([
                [[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]],
                
                [[10, 30, 10],
                [0, 0, 0],
                [-10, -30, -10]],
                
                [[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]]
            ], dtype=torch.float32)
            
            kz = torch.tensor([
                [[3, 10, 3],
                [10, 30, 10],
                [3, 10, 3]],
                
                [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
                
                [[-3, -10, -3],
                [-10, -30, -10],
                [-3, -10, -3]]
            ], dtype=torch.float32)
        
        kx = kx.reshape(3, 3, 3)
        ky = ky.reshape(3, 3, 3)
        kz = kz.reshape(3, 3, 3)
        
        kernel = torch.stack([kz, ky, kx])[:, None, ...]
        
        if self.grad_sigma > 0:
            z, y, x = torch.meshgrid(
                torch.arange(3), 
                torch.arange(3), 
                torch.arange(3), 
                indexing='ij'
            )
            center = torch.tensor([1.0, 1.0, 1.0])
            dist = torch.sqrt((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2)
            gauss = torch.exp(-dist**2 / (2 * self.grad_sigma**2))
            kernel = kernel * gauss[None, None, ...]
        
        return kernel / kernel.abs().sum(dim=(2, 3, 4), keepdim=True)
    
    def _compute_gradient_magnitude(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute a smoothed 3D gradient-magnitude volume."""
        smoothed = self._gaussian_smooth(volume)
        
        gradients = F.conv3d(
            smoothed, 
            self.gradient_kernel.to(volume.device), 
            padding=1
        )
        
        grad_magnitude = torch.sqrt(torch.sum(gradients**2, dim=1, keepdim=True) + 1e-8)
        
        return grad_magnitude
    
    def _sample_points(self, 
                      volume: torch.Tensor, 
                      points: torch.Tensor) -> torch.Tensor:
        """Sample volume values at points with trilinear interpolation."""
        if volume.dim() == 4:
            volume = volume.unsqueeze(1)
        D, H, W = volume.shape[2:]
        grid = points.clone()
        grid[..., 0] = 2.0 * points[..., 0] / (D - 1) - 1.0  # Z
        grid[..., 1] = 2.0 * points[..., 1] / (H - 1) - 1.0  # Y
        grid[..., 2] = 2.0 * points[..., 2] / (W - 1) - 1.0  # X
        
        grid = grid.unsqueeze(2).unsqueeze(2)
        
        return F.grid_sample(volume, grid, 
                            mode='bilinear', 
                            align_corners=True).squeeze(-1).squeeze(-1)
    
    def forward(self,
               volume: torch.Tensor, 
               pred_points: torch.Tensor) -> torch.Tensor:
        """Compute boundary-gradient loss for predicted 3D points."""
        B, C, D, H, W = volume.shape
        
        selected_points = pred_points[:, self.point_indices]  # [B, M, 3]
        
        grad_map = self._compute_gradient_magnitude(volume)  # [B, 1, D, H, W]
        
        sampled_grads = self._sample_points(grad_map, selected_points)  # [B, M]
        # print(sampled_grads)
        
        if self.adaptive_threshold:
            with torch.no_grad():
                threshold = torch.median(grad_map.view(B, -1), dim=1)[0]
                threshold = threshold.view(B, 1)
        else:
            threshold = self.min_grad
            
        loss_per_point = F.relu(threshold - sampled_grads)
        
        return loss_per_point.mean()
    

def register_template_label(template_image_path, template_label_path, target_image_path, output_path):
    template_data = ants.image_read(template_image_path)    
    image_data = ants.image_read(target_image_path)
    reg_result = ants.registration(
        fixed = template_data,
        moving = image_data,
        type_of_transform = 'SyN', 
        reg_iterations = (150, 100, 70),
        syn_sampling = 16,
        verbose=False
    )
    template_label = ants.image_read(template_label_path)
    registered_label = ants.apply_transforms( # apply_transforms
        fixed  = image_data,
        moving = template_label,  
        transformlist = reg_result['invtransforms'],
        interpolator = 'nearestNeighbor'
    )
    ants.image_write(registered_label, output_path)
    if 'fwdtransforms' in reg_result:
        for file in reg_result['fwdtransforms']:
            if os.path.exists(file):
                os.unlink(file)
    if 'invtransforms' in reg_result:
        for file in reg_result['invtransforms']:
            if os.path.exists(file):
                os.unlink(file)
    
def crop_template_label(registered_label, image_data):
    mask = image_data > 0
    ind_brain = block_ind(mask)
        # crop
    cropped_template_label = extract_brain(registered_label,ind_brain,[128,160,128]) 
    return cropped_template_label

def constrain_heatmap(cropped_template_label, data, label_value, radius, template_label_scale, template_label_pan):
    heatmap_mask = np.zeros_like(cropped_template_label, dtype=np.int32)
    voxels = np.argwhere(cropped_template_label == label_value)
    if len(voxels) > 0:
        geometric_center = np.mean(voxels, axis=0)
        z, y, x = np.indices(cropped_template_label.shape)
        distances = np.sqrt(
            (z - geometric_center[0])**2 +
            (y - geometric_center[1])**2 +
            (x - geometric_center[2])**2
        )
        heatmap_mask[distances < radius] = 1
    original_data = data.copy()
    data[heatmap_mask == 1] = template_label_scale * original_data[heatmap_mask == 1] + template_label_pan
    data[heatmap_mask == 0] = -1000
    return data



def get_landmark_from_heatmap(path_or_data, slice_index, detector=None, init_temp=0.05, topk=50):
    if detector is None:
        detector = DifferentiableLandmarkDetector(init_temp=init_temp, topk=topk)
        detector = detector.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if isinstance(path_or_data, str):
        path = path_or_data
        img = nb.load(path)
        heatmap_data = img.get_fdata()  # [D, H, W]
    elif isinstance(path_or_data, np.ndarray):
        heatmap_data = path_or_data
    
    heatmap = torch.tensor(heatmap_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [B=1, C=1, D, H, W]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    heatmap = heatmap.to(device)
    detector = detector.to(device)
    
    with torch.no_grad():
        peak_coord = detector(heatmap)  # [B=1, C=1, 3]
    coordinate = np.round(peak_coord[0, 0].cpu().numpy())
    slice_sum_max_index = coordinate[slice_index].astype(int) 
    
    return coordinate, slice_sum_max_index


def plot_slice(mask, img, slice_index, orientation, up_bound, low_bound, coordinate):
    if orientation == 2:
        mask_slc = mask[:, :, slice_index]
        img_slc = img[:, :, slice_index]
    elif orientation == 0:
        mask_slc = mask[slice_index, :, :]
        img_slc = img[slice_index, :, :]
    elif orientation == 1:
        mask_slc = mask[:, slice_index, :]
        img_slc = img[:, slice_index, :]
    else:
        raise ValueError("Invalid orientation value. Must be 0, 1, or 2.")

    img_ups = ((img_slc - low_bound) / (up_bound - low_bound) * 255).astype(np.uint8)
    mask_ups = mask_slc

    image_rgb = np.zeros((*img_ups.shape, 3), dtype=np.uint8)
    image_rgb[..., 0] = img_ups  # R
    image_rgb[..., 1] = img_ups  # G
    image_rgb[..., 2] = img_ups  # B
    image_rgb[~mask_ups] = [0, 0, 0]

    cross_positions = []
    for ii in range(coordinate.shape[0]):
        if orientation == 2:
            x = coordinate[ii, 0]
            y = coordinate[ii, 1]
        elif orientation == 0:
            x = coordinate[ii, 1]
            y = coordinate[ii, 2]
        elif orientation == 1:
            x = coordinate[ii, 0]
            y = coordinate[ii, 2]
        else:
            continue

        if 0 <= x < image_rgb.shape[0] and 0 <= y < image_rgb.shape[1]:
            cross_positions.append((x, y))

    return image_rgb, cross_positions

def rotate_coordinate(x, y, k, img_width, img_height):
    """Rotate a two-dimensional coordinate by k quarter turns."""
    if k == 1:
        x_new = y
        y_new = img_width - 1 - x
    elif k == 2:
        x_new = img_height - 1 - x
        y_new = img_width - 1 - y
    elif k == 3:
        x_new = img_height - 1 - y
        y_new = x
    else:
        raise ValueError("Invalid rotation parameter k. Must be 1, 2, or 3.")
    return x_new, y_new
