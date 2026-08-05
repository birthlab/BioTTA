"""Step 2: localize 22 landmarks with test-time adaptation."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nb
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import scipy.stats as stats

from lib.step2_source_network import LocalAppearance
from lib.step2_supp import (
    TENTWrapper, TestDataset, DifferentiableLandmarkDetector,
    EntropyLoss, LengthLoss, BoundaryGradientLoss,
    register_template_label, constrain_heatmap, get_landmark_from_heatmap,
    extract_brain, block_ind, crop_template_label,
    plot_slice, rotate_coordinate, convert_coords_to_original
)


def run_tta_and_predict(
    image_path: str,
    pred_lengths: pd.Series,
    model_path: str,
    device: torch.device,
    config: Dict,
    template_paths: Optional[Dict] = None,
    age: Optional[float] = None,
    skip_tta: bool = False,
    tta_model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    curve_dir: Optional[str] = None,
    save_intermediate: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """Run TTA for one image and return 22 landmark coordinates plus metadata."""
    batch_size = config.get('batch_size', 1)
    if batch_size != 1:
        raise ValueError("Step 2 processes one image at a time and requires batch_size=1.")
    channel = config.get('channel', 22)
    init_temp = config.get('init_temp', 0.5)
    topk = config.get('topk', 50)
    learning_rate = config.get('learning_rate', 0.0005)
    num_epoch = config.get('num_epoch', 5)
    lambda_entropy_loss = config.get('lambda_entropy_loss', 0.5)
    lambda_length_loss = config.get('lambda_length_loss', 1.5)
    lambda_boundarygrad_loss = config.get('lambda_boundarygrad_loss', 300)
    point_pairs = config.get('point_pairs', list(range(1, 12)))
    radius = config.get('radius', [20, 20, 20, 20, 20, 20, 5, 5, 5, 5, 5])
    template_label_scale = config.get('template_label_scale', 1)
    template_label_pan = config.get('template_label_pan', 0.25)
    orientations = config.get('orientations', [2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    template_heatmap_number = config.get('template_heatmap_number', 
        [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 13, 14, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30])
    
    image_name = Path(image_path).name
    if image_name.lower().endswith('.nii.gz'):
        image_name = image_name[:-7]
    elif image_name.lower().endswith('.nii'):
        image_name = image_name[:-4]
    
    biometry_list = ['RLV_A', 'LLV_A', 'RLV_C', 'LLV_C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']
    pred_length_dict = {'label_name': image_name}
    for i, biometry in enumerate(biometry_list):
        pred_length_dict[biometry] = pred_lengths.iloc[i] if isinstance(pred_lengths, pd.Series) else pred_lengths[i]
    pred_length_df = pd.DataFrame([pred_length_dict])
    
    dataset_target = TestDataset(image_path, pred_length_df)
    target_loader = DataLoader(
        dataset_target, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == 'cuda'
    )
    
    net = LocalAppearance(1, channel).to(device)
    
    if not skip_tta and tta_model_path is None:
        if Path(model_path).is_file():
            pretrained_model = torch.load(model_path, map_location=device, weights_only=True)
            net.load_state_dict(pretrained_model.get('net', pretrained_model), strict=False)
            print(f"[OK] Step 2 model loaded: {model_path}")
        else:
            raise FileNotFoundError(f"Step 2 model was not found: {model_path}")
        
        tent_net = TENTWrapper(net)
        
        detector = DifferentiableLandmarkDetector(init_temp=init_temp, topk=topk)
        entropy_loss_f = EntropyLoss()
        length_loss_f = LengthLoss()
        boundarygrad_loss_f = BoundaryGradientLoss(
            point_pairs=point_pairs,
            kernel_type='scharr',
            smooth_sigma=1.0,
            grad_sigma=0.9,
            adaptive_threshold=False,
            min_grad=0.4
        )
        
        optim = torch.optim.Adam(
            list(tent_net.parameters()) + list(length_loss_f.parameters()),
            lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=num_epoch)
        
        tent_net.train()
        min_loss = float('inf')
        best_model_state = None
        
        print(f"Starting test-time adaptation ({num_epoch} epochs)...")
        for epoch in range(num_epoch):
            epoch_losses = []
            for i, batch_data in enumerate(target_loader):
                image = batch_data['image'].to(device)
                length_pseudo_label = batch_data['length'].to(device)
                
                _, HLA = tent_net(image)
                
                all_keypoints = []
                for c in range(channel):
                    channel_heatmap = HLA[:, c:c+1, :, :, :]
                    keypoints = detector(channel_heatmap)
                    all_keypoints.append(keypoints.squeeze(1))
                keypoints = torch.stack(all_keypoints, dim=1)
                
                entropy_loss = entropy_loss_f(HLA)
                length_loss = length_loss_f(keypoints, length_pseudo_label)
                boundarygrad_loss = boundarygrad_loss_f(image, keypoints)
                loss = (lambda_entropy_loss * entropy_loss + 
                       lambda_length_loss * length_loss + 
                       lambda_boundarygrad_loss * boundarygrad_loss)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_losses.append(loss.item())
                if i == 0:
                    print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
            
            if not epoch_losses:
                raise RuntimeError("Step 2 produced no training batches.")
            avg_loss = float(np.mean(epoch_losses))
            if avg_loss < min_loss:
                min_loss = avg_loss
                best_model_state = {
                    key: value.detach().clone() for key, value in tent_net.state_dict().items()
                }
            scheduler.step()
        
        if best_model_state is not None:
            tent_net.load_state_dict(best_model_state)
        print("[OK] Test-time adaptation completed")
        
        if save_intermediate is not None and save_intermediate.get('step2_tta_models', False):
            if output_dir is not None:
                tta_model_path = os.path.join(output_dir, 'tent_TTA.pth')
                torch.save({
                    'net': tent_net.state_dict(),
                    'optim': optim.state_dict()
                }, tta_model_path)
                print(f"[OK] Adapted model saved to: {tta_model_path}")
        
    elif tta_model_path is not None and Path(tta_model_path).is_file():
        TTA_model = torch.load(tta_model_path, map_location=device, weights_only=True)
        tent_net = TENTWrapper(net)
        tent_net.load_state_dict(TTA_model.get('net', TTA_model), strict=False)
        print(f"[OK] Existing adapted model loaded: {tta_model_path}")
    else:
        pretrained_model = torch.load(model_path, map_location=device, weights_only=True)
        net.load_state_dict(pretrained_model.get('net', pretrained_model), strict=False)
        tent_net = TENTWrapper(net)
        print("[WARNING] TTA skipped; using the pretrained model")
    
    heatmap = None
    for batch_data in target_loader:
        image = batch_data['image'].to(device)
        _, HLA = tent_net(image)
        heatmap = HLA.detach().cpu().numpy().astype(np.float32).squeeze()
        break
    if heatmap is None:
        raise RuntimeError("Step 2 produced no inference batches.")
    
    data = nb.load(image_path).get_fdata()
    mask = data > 0
    if not np.any(mask):
        raise ValueError(f"The input image has no positive-valued foreground: {image_path}")
    ind_brain = block_ind(mask)
    sized_data = extract_brain(data, ind_brain, [128, 160, 128])
    sized_mask = sized_data > 0
    max_value = np.max(sized_data)
    min_value = np.min(sized_data)
    
    cropped_template_heatmap_data = None
    if template_paths and age is not None:
        atlas_age = int(round(age))
        template_image_path = Path(template_paths['image_folder']) / f'STA{atlas_age}.nii.gz'
        template_label_path = Path(template_paths['label_folder']) / f'{atlas_age}_m.nii.gz'
        if template_image_path.is_file() and template_label_path.is_file():
            fallback_folder = Path(output_dir or '.') / 'registered_templates'
            registered_folder = Path(template_paths.get('registered_folder') or fallback_folder)
            registered_folder.mkdir(parents=True, exist_ok=True)
            registered_template_path = (
                registered_folder / f'{image_name}_registered_{atlas_age}_m.nii.gz'
            )
            if not registered_template_path.is_file():
                register_template_label(
                    str(template_image_path),
                    str(template_label_path),
                    image_path,
                    str(registered_template_path),
                )
                print(f"[OK] Atlas registration completed for week {atlas_age}")
            else:
                print(f"[OK] Reusing registered atlas label: {registered_template_path}")

            registered_template_label = nb.load(registered_template_path).get_fdata()
            cropped_template_heatmap_data = crop_template_label(registered_template_label, data)
        else:
            print(
                f"[WARNING] Atlas templates are unavailable for week {atlas_age}; "
                "continuing without the atlas constraint."
            )
    
    coordinates = []
    if cropped_template_heatmap_data is not None:
        for i in range(11):
            heatmap_data_1 = constrain_heatmap(
                cropped_template_heatmap_data, heatmap[2*i], 
                template_heatmap_number[2*i], radius[i], 
                template_label_scale, template_label_pan
            )
            heatmap_data_2 = constrain_heatmap(
                cropped_template_heatmap_data, heatmap[2*i+1], 
                template_heatmap_number[2*i+1], radius[i], 
                template_label_scale, template_label_pan
            )
            
            coordinate_1, _ = get_landmark_from_heatmap(heatmap_data_1, orientations[i], init_temp=init_temp, topk=topk)
            coordinate_2, _ = get_landmark_from_heatmap(heatmap_data_2, orientations[i], init_temp=init_temp, topk=topk)
            coordinate = np.array([coordinate_1, coordinate_2])
            coordinates.append(coordinate)
    else:
        for i in range(11):
            coordinate_1, _ = get_landmark_from_heatmap(heatmap[2*i], orientations[i], init_temp=init_temp, topk=topk)
            coordinate_2, _ = get_landmark_from_heatmap(heatmap[2*i+1], orientations[i], init_temp=init_temp, topk=topk)
            coordinate = np.array([coordinate_1, coordinate_2])
            coordinates.append(coordinate)
    
    landmarks = np.vstack(coordinates)
    
    landmarks_original = convert_coords_to_original(landmarks, ind_brain, [128, 160, 128])
    
    if save_intermediate is not None and save_intermediate.get('step2_landmarks_txt', False):
        if output_dir is not None:
            landmarks_txt_path = os.path.join(output_dir, 'landmarks.txt')
            with open(landmarks_txt_path, 'w', encoding='utf-8') as f:
                for i in range(11):
                    coordinate = np.array([landmarks[2*i], landmarks[2*i+1]])
                    merged_coordinates = coordinate.flatten()
                    coordinates_str = ' '.join(map(str, merged_coordinates))
                    f.write(f"{coordinates_str}\n")
            print(f"[OK] Landmark coordinates saved to: {landmarks_txt_path}")
    
    if save_intermediate is not None and save_intermediate.get('step2_landmarks_txt_original', False):
        if output_dir is not None:
            landmarks_original_txt_path = os.path.join(output_dir, 'landmarks_original.txt')
            with open(landmarks_original_txt_path, 'w', encoding='utf-8') as f:
                for i in range(11):
                    coordinate = np.array([landmarks_original[2*i], landmarks_original[2*i+1]])
                    merged_coordinates = coordinate.flatten()
                    coordinates_str = ' '.join(map(str, merged_coordinates))
                    f.write(f"{coordinates_str}\n")
            print(f"[OK] Original-space landmarks saved to: {landmarks_original_txt_path}")
    
    if save_intermediate is not None and save_intermediate.get('step2_landmarks_image', False):
        if output_dir is not None:
            save_landmarks_visualization(
                image_path, landmarks, sized_mask, sized_data, 
                max_value, min_value, config, output_dir
            )
    
    if age is not None and curve_dir is not None and output_dir is not None:
        calculated_lengths = calculate_lengths_from_landmarks(landmarks)
        
        save_csv = save_intermediate is not None and save_intermediate.get('step2_length_centile_csv', False)
        centile_df = calculate_centiles(calculated_lengths, age, curve_dir, output_dir, save_csv=save_csv)
        
        if save_intermediate is not None and save_intermediate.get('step2_length_centile_plot', False):
            plot_centile_curves(calculated_lengths, age, curve_dir, output_dir, save_intermediate)
    
    metadata = {
        'image_name': image_name,
        'age': age,
        'heatmap_shape': heatmap.shape,
        'landmarks_shape': landmarks.shape
    }
    
    return landmarks, metadata


def calculate_lengths_from_landmarks(landmarks: np.ndarray, resolution: float = 0.8) -> np.ndarray:
    """Calculate 11 physical lengths from 22 paired landmark coordinates."""
    lengths = []
    for i in range(11):
        point1 = landmarks[2*i]
        point2 = landmarks[2*i+1]
        
        distance = np.sqrt(np.sum((point1 - point2)**2))
        
        actual_length = distance * resolution
        lengths.append(actual_length)
    
    return np.array(lengths)


def calculate_centiles(lengths: np.ndarray, age: float, curve_dir: str, output_dir: str, save_csv: bool = True) -> pd.DataFrame:
    """Map 11 measurements to gestational-age-specific centiles."""
    biometry_list = ['RLV-A', 'LLV-A', 'RLV-C', 'LLV-C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']
    centile_data = []
    
    for i, (length_name, length_value) in enumerate(zip(biometry_list, lengths)):
        csv_path = os.path.join(curve_dir, f"{length_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"[WARNING] Development curve was not found: {csv_path}")
            centile_value = np.nan
        else:
            curve_df = pd.read_csv(csv_path)
            
            closest_idx = (curve_df['GA'] - age).abs().idxmin()
            closest_row = curve_df.iloc[closest_idx]
            
            p50_value = closest_row['50th']
            
            p5_value = closest_row['5th']
            p95_value = closest_row['95th']
            
            estimated_std = (p95_value - p5_value) / 3.29
            
            if estimated_std > 0:
                z_score = (length_value - p50_value) / estimated_std
                centile_value = stats.norm.cdf(z_score) * 100
            else:
                centile_value = 50.0
        
        centile_data.append({
            'Biometry': length_name,
            'Length_mm': length_value,
            'Centile': centile_value
        })
    
    centile_df = pd.DataFrame(centile_data)
    
    if save_csv:
        output_path = os.path.join(output_dir, 'length_centile.csv')
        centile_df.to_csv(output_path, index=False)
        print(f"[OK] Measurement centiles saved to: {output_path}")
    
    return centile_df


def plot_centile_curves(lengths: np.ndarray, age: float, curve_dir: str, output_dir: str, save_intermediate: Optional[Dict] = None):
    """Plot the 11 measurements against their development curves."""
    del save_intermediate
    biometry_list = ['RLV-A', 'LLV-A', 'RLV-C', 'LLV-C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']
    
    y_axis_settings = {
        "RLV-A": (2, 18, 4),
        "LLV-A": (2, 18, 4),
        "RLV-C": (2, 18, 4),
        "LLV-C": (2, 18, 4),
        "TCD": (18, 58, 10),
        "FOD": (40, 120, 20),
        "BBD": (40, 100, 15),
        "CBD": (35, 95, 15),
        "AVD": (4, 20, 4),
        "VH": (8, 24, 4),
        "CCL": (18, 50, 8),
    }
    
    def draw_plot(is_dark_mode=False):
        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        axes_flat = axes.flatten()
        
        if is_dark_mode:
            fig.patch.set_facecolor('#1a1a1a')
            bg_color = '#1a1a1a'
            text_color = 'white'
            grid_color = 'gray'
            spine_color = 'white'
        else:
            fig.patch.set_facecolor('white')
            bg_color = 'white'
            text_color = 'black'
            grid_color = 'gray'
            spine_color = 'black'
        
        roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi']
        
        for i, length_name in enumerate(biometry_list):
            ax = axes_flat[i]
            
            ax.set_facecolor(bg_color)
            
            y_pos = -0.1 if i < 6 else -0.18
            ax.text(0.5, y_pos, f'({roman_numerals[i]})', transform=ax.transAxes, 
                    fontsize=20, fontweight='bold', ha='center', va='top', color=text_color)
            
            ax.grid(True, linestyle="-", linewidth=1.5, color=grid_color, alpha=0.5)
            ax.tick_params(axis="both", direction="in", width=1.5, length=6, color=text_color, labelcolor=text_color)
            ax.tick_params(axis="x", labelsize=18, pad=10)
            ax.tick_params(axis="y", labelsize=18)
            
            csv_path = os.path.join(curve_dir, f"{length_name}.csv")
            if os.path.exists(csv_path):
                curve_df = pd.read_csv(csv_path)
                
                ax.plot(
                    curve_df["GA"],
                    curve_df["50th"],
                    color="red",
                    linewidth=4.5,
                    alpha=1.0,
                    zorder=5,
                )
                ax.plot(
                    curve_df["GA"],
                    curve_df["5th"],
                    color="darkblue" if not is_dark_mode else "lightblue",
                    linewidth=4.5,
                    linestyle="--",
                    alpha=1.0,
                    zorder=5,
                )
                ax.plot(
                    curve_df["GA"],
                    curve_df["95th"],
                    color="darkblue" if not is_dark_mode else "lightblue",
                    linewidth=4.5,
                    linestyle="--",
                    alpha=1.0,
                    zorder=5,
                )
                
                x_min = np.nanmin(curve_df["GA"].to_numpy())
                x_max = np.nanmax(curve_df["GA"].to_numpy())
            else:
                x_min, x_max = 20.0, 40.0
            
            ax.scatter(
                [age],
                [lengths[i]],
                color="black" if not is_dark_mode else "white",
                alpha=1,
                s=150,
                zorder=10,
                linewidth=2
            )
            
            y_start, y_end, y_gap = y_axis_settings.get(length_name, (0, 50, 10))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_start, y_end)
            ax.set_xticks(np.arange(np.floor(x_min / 5) * 5, np.ceil(x_max / 5) * 5 + 0.1, 5))
            ax.set_yticks(np.arange(y_start, y_end + 0.1, y_gap))
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_color(spine_color)
            ax.spines["bottom"].set_color(spine_color)
            
            ax.text(
                0.02,
                0.95,
                length_name,
                transform=ax.transAxes,
                fontsize=22,
                fontweight="bold",
                va="top",
                color=text_color
            )
            
            if length_name in ["RLV-A", "TCD"]:
                ax.set_ylabel("Length (mm)", fontsize=22, color=text_color)
            if length_name in ["TCD", "FOD", "AVD", "VH", "CCL"]:
                ax.set_xlabel("Gestational Age (weeks)", fontsize=22, color=text_color)
        
        if len(axes_flat) > len(biometry_list):
            axes_flat[len(biometry_list)].set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    fig = draw_plot(is_dark_mode=False)
    output_path_pdf = os.path.join(output_dir, 'length_centile.png')
    fig.savefig(output_path_pdf, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Print centile plot saved to: {output_path_pdf}")
    
    fig = draw_plot(is_dark_mode=True)
    output_path_web = os.path.join(output_dir, 'length_centile_web.png')
    fig.savefig(output_path_web, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Web centile plot saved to: {output_path_web}")


def save_landmarks_visualization(
    image_path: str,
    landmarks: np.ndarray,
    sized_mask: np.ndarray,
    sized_data: np.ndarray,
    max_value: float,
    min_value: float,
    config: Dict,
    output_dir: str
):
    """Save a compact 11-panel landmark visualization."""
    del image_path
    orientations = config.get('orientations', [2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    location_image_names = ['1+2', '3+4', '5+6', '7+8', '9+10', '11+12', '13+14', '15+16', '17+18', '19+20', '21+22']
    
    os.makedirs(output_dir, exist_ok=True)
    
    coordinates = []
    images_ups = []
    aspect_ratios = []
    cross_positions_list = []
    
    for i in range(11):
        coordinate_1 = landmarks[2*i]
        coordinate_2 = landmarks[2*i+1]
        coordinate = np.array([coordinate_1, coordinate_2])
        coordinates.append(coordinate)
        
        orientation = orientations[i]
        if orientation == 0:
            slice_sum_max_index_1 = int(round(coordinate_1[0]))
            slice_sum_max_index_2 = int(round(coordinate_2[0]))
        elif orientation == 1:
            slice_sum_max_index_1 = int(round(coordinate_1[1]))
            slice_sum_max_index_2 = int(round(coordinate_2[1]))
        else:  # orientation == 2
            slice_sum_max_index_1 = int(round(coordinate_1[2]))
            slice_sum_max_index_2 = int(round(coordinate_2[2]))
        
        slice_sum_max_index = round((slice_sum_max_index_1 + slice_sum_max_index_2) / 2)
        
        img_ups, cross_positions = plot_slice(
            sized_mask, sized_data, slice_sum_max_index, 
            orientations[i], max_value, min_value, coordinate
        )
        
        img_ups = np.rot90(img_ups, k=1)
        rotated_cross_positions = []
        for (x, y) in cross_positions:
            x_new, y_new = rotate_coordinate(
                x, y, k=3, 
                img_width=img_ups.shape[1], 
                img_height=img_ups.shape[0]
            )
            rotated_cross_positions.append((x_new, y_new))
        cross_positions = rotated_cross_positions
        
        non_zero_indices = np.nonzero(img_ups)
        if len(non_zero_indices[0]) > 0 and len(non_zero_indices[1]) > 0:
            leftmost = np.min(non_zero_indices[1])
            rightmost = np.max(non_zero_indices[1])
            cropped_width = rightmost - leftmost + 1
            new_width = cropped_width + 20
            h, w = img_ups.shape[:2]
            new_img = np.zeros((h, new_width, 3), dtype=img_ups.dtype)
            paste_left = 10
            new_img[:, paste_left:paste_left + cropped_width] = img_ups[:, leftmost:rightmost + 1]
            img_ups = new_img
            cross_positions = [(x, y-leftmost+paste_left) for (x, y) in cross_positions]
        
        images_ups.append(img_ups)
        cross_positions_list.append(cross_positions)
        h, w = img_ups.shape[:2]
        aspect_ratios.append(w / h)
    
    total_width_ratio = sum(aspect_ratios)
    fig_height = 3
    wspace = -0.05
    fig = plt.figure(figsize=(fig_height * (total_width_ratio + wspace * 10), fig_height))
    gs = gridspec.GridSpec(1, 11, width_ratios=aspect_ratios, wspace=wspace)
    
    for i in range(11):
        img_ups = images_ups[i]
        cross_positions = cross_positions_list[i]
        if np.max(img_ups) > 1 or np.min(img_ups) < 0:
            value_range = np.max(img_ups) - np.min(img_ups)
            if value_range > 0:
                img_ups = (img_ups - np.min(img_ups)) / value_range
        img_ups = img_ups.astype(np.float32)
        ax = fig.add_subplot(gs[i])
        ax.imshow(img_ups, aspect='auto')
        for (x, y) in cross_positions:
            ax.plot(y, x, 'kx', markersize=7, markeredgewidth=4)
        for (x, y) in cross_positions:
            ax.plot(y, x, 'rx', markersize=6, markeredgewidth=2)
        ax.axis('off')
        ax.set_adjustable('datalim')
    
    landmarks_image_path = os.path.join(output_dir, 'landmarks.png')
    plt.savefig(landmarks_image_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()
    print(f"[OK] Landmark visualization saved to: {landmarks_image_path}")
