import numpy as np
import os
import glob
import joblib
import torch
import h5py
import json
import argparse
from sklearn.neighbors import KDTree
from scipy.stats import mode
from scipy.ndimage import distance_transform_edt, sobel, uniform_filter

DEFAULT_TILE_SIZE = 256
SPATIAL_NEIGHBORS = 8

def collect_slide_files(root_dir):
    slide_files = {}
    duplicate_slides = []
    all_pt_files = glob.glob(os.path.join(root_dir, "**", "pt_files", "*.pt"), recursive=True)

    for pt_path in sorted(all_pt_files):
        slide_id = os.path.splitext(os.path.basename(pt_path))[0]
        base_dir = os.path.dirname(os.path.dirname(pt_path))
        h5_path = os.path.join(base_dir, "h5_files", f"{slide_id}.h5")

        if slide_id in slide_files:
            duplicate_slides.append(slide_id)
            continue

        slide_files[slide_id] = (pt_path, h5_path)

    return slide_files, duplicate_slides

def ordered_fold_slide_ids(splits):
    ordered_ids = []
    seen = set()
    preferred_keys = ["train", "val", "test"]
    split_keys = preferred_keys + [key for key in splits.keys() if key not in preferred_keys]

    for split_key in split_keys:
        for slide_id in splits.get(split_key, []):
            if slide_id in seen:
                continue
            ordered_ids.append(slide_id)
            seen.add(slide_id)

    return ordered_ids

def warn_if_model_train_count_mismatch(kmeans, fold_name, splits):
    train_count = len(splits.get("train", []))
    model_steps = getattr(kmeans, "n_steps_", None)

    if model_steps is None:
        print("Warning: KMeans model does not expose n_steps_; cannot compare against train split count.")
        return

    if model_steps != train_count:
        print(
            f"Warning: {fold_name} KMeans n_steps_={model_steps}, "
            f"but CV train split has {train_count} slides."
        )

def load_slide_data(pt_path, h5_path):
    try:
        features = torch.load(pt_path, map_location='cpu')
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        
        if not os.path.exists(h5_path):
            return None, None, None

        with h5py.File(h5_path, 'r') as f:
            if 'coords' not in f:
                return None, None, None
            
            coords = f['coords'][:]
            
            if 'patch_size' in f.attrs:
                patch_size = int(f.attrs['patch_size'])
            elif 'tile_size' in f.attrs:
                patch_size = int(f.attrs['tile_size'])
            else:
                patch_size = DEFAULT_TILE_SIZE

        if len(features) != len(coords):
            if features.shape[0] != coords.shape[0]:
                return None, None, None
            
        return features, coords, patch_size

    except Exception as e:
        print(f"  [Error] {os.path.basename(pt_path)}: {e}")
        return None, None, None

def clean_spatial_noise_robust(coords, labels, k=8):
    if len(coords) < k + 1:
        return labels 
        
    tree = KDTree(coords)
    dists, inds = tree.query(coords, k=k+1)
    
    grid_step = np.median(dists[:, 1]) 
    max_dist = grid_step * 1.5
    smoothed_labels = labels.copy()
    
    for i in range(len(labels)):
        valid_mask = dists[i] <= max_dist
        valid_indices = inds[i][valid_mask]
        valid_labels = labels[valid_indices]
        
        if len(valid_labels) > 0:
            vote_result = mode(valid_labels, keepdims=False)
            vote = vote_result.mode if np.ndim(vote_result.mode) == 0 else vote_result.mode[0]
            smoothed_labels[i] = vote
            
    return smoothed_labels

def create_tissue_grid(coords, labels, tile_size):
    if len(coords) == 0: return None, None
    
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    cols = int(np.round((x_max - x_min) / tile_size)) + 10
    rows = int(np.round((y_max - y_min) / tile_size)) + 10
    
    grid = np.full((rows, cols), -1, dtype=int)
    
    c_idx = np.round((coords[:, 0] - x_min) / tile_size).astype(int) + 5
    r_idx = np.round((coords[:, 1] - y_min) / tile_size).astype(int) + 5
    
    c_idx = np.clip(c_idx, 0, cols - 1)
    r_idx = np.clip(r_idx, 0, rows - 1)
    
    grid[r_idx, c_idx] = labels
    return grid, {'r': r_idx, 'c': c_idx, 'step': tile_size}

def calculate_enhanced_spatial_features(grid, transform, n_clusters):
    if grid is None: return None
        
    pixel_scale = transform['step']
    r_idx, c_idx = transform['r'], transform['c']
    gammas = np.logspace(-7, -2, 10)
    all_features = []

    for cid in range(n_clusters):
        mask = (grid == cid)
        
        if not np.any(mask):
            all_features.append(np.zeros((len(r_idx), 14), dtype=np.float32)) 
            continue
            
        dist_func = distance_transform_edt(~mask) - distance_transform_edt(mask)
        sdf = dist_func * pixel_scale
        
        rbfs = [np.exp(-g * (sdf**2)) for g in gammas]
        gx = sobel(sdf, axis=1)
        gy = sobel(sdf, axis=0)
        
        local_mu = uniform_filter(sdf, size=3)
        local_sq_mu = uniform_filter(sdf**2, size=3)
        local_std = np.sqrt(np.maximum(local_sq_mu - local_mu**2, 0))
        
        cluster_stack = rbfs + [gx, gy, local_mu, local_std]
        sampled = np.stack([feat[r_idx, c_idx] for feat in cluster_stack], axis=1)
        all_features.append(sampled)

    return np.concatenate(all_features, axis=1)


def main(args):
    with open(args.cv_json, 'r') as f:
        cv_splits = json.load(f)

    slide_files, duplicate_slides = collect_slide_files(args.root_dir)

    print(f"Total unique .pt slides found in workspace: {len(slide_files)}")
    if duplicate_slides:
        print(f"Warning: found {len(duplicate_slides)} duplicate slide IDs; using first path for each.")

    for fold_name, splits in cv_splits.items():
        print(f"\n==================================================")
        print(f"Generating SDF Features for {fold_name} | {args.model_name}")
        print(f"==================================================")

        # 1. Load Fold-Specific KMeans Model
        model_path = os.path.join(args.model_dir, f"{args.model_name}_kmeans_model_{fold_name}.pkl")
        if not os.path.exists(model_path):
            print(f"KMeans model missing for fold: {model_path} -> Skipping.")
            continue

        print(f"Loading Model: {model_path}")
        kmeans = joblib.load(model_path)
        warn_if_model_train_count_mismatch(kmeans, fold_name, splits)
        n_clusters = kmeans.n_clusters

        # 2. Setup Fold-Specific Output Directory
        fold_output_dir = os.path.join(args.root_dir, f"sdf_70d_features_{fold_name}")
        os.makedirs(fold_output_dir, exist_ok=True)

        fold_slide_ids = ordered_fold_slide_ids(splits)
        missing_slide_ids = [slide_id for slide_id in fold_slide_ids if slide_id not in slide_files]
        process_slide_ids = [slide_id for slide_id in fold_slide_ids if slide_id in slide_files]

        if missing_slide_ids:
            print(f"Warning: {len(missing_slide_ids)} CV slides are missing .pt files for {fold_name}.")
            print(f"First missing slide: {missing_slide_ids[0]}")

        print(f"\n--- Processing Fold {fold_name} ({len(process_slide_ids)} slides from CV JSON) ---")

        for i, slide_id in enumerate(process_slide_ids):
            pt_path, h5_path = slide_files[slide_id]
            output_path = os.path.join(fold_output_dir, f"{slide_id}{args.sdf_suffix}.npz")

            if os.path.exists(output_path):
                continue

            print(f"[{i+1}/{len(process_slide_ids)}] {slide_id}", end=" ", flush=True)

            # A. Load
            features, coords, patch_size = load_slide_data(pt_path, h5_path)
            if features is None:
                print("-> Load Error")
                continue

            # B. Predict using Fold-Specific KMeans
            try:
                if features.ndim == 1: features = features.reshape(1, -1)
                labels = kmeans.predict(features)
            except Exception as e:
                print(f"-> Predict Error: {e}")
                continue

            # C. Compute SDF
            labels = clean_spatial_noise_robust(coords, labels, k=SPATIAL_NEIGHBORS)
            grid, transform = create_tissue_grid(coords, labels, patch_size)
            
            if grid is not None:
                spatial_70d = calculate_enhanced_spatial_features(grid, transform, n_clusters)
                
                np.savez_compressed(
                    output_path, 
                    spatial_features=spatial_70d.astype(np.float32),
                    coords=coords,
                    labels=labels
                )
                print("-> Done")
            else:
                print("-> Grid Error")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help="Base directory of features (e.g. /scratch/.../VA_Cross_Validation/uni)")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the KMeans .pkl models")
    parser.add_argument('--cv_json', type=str, required=True, help="Path to CV split JSON")
    parser.add_argument('--model_name', type=str, required=True, help="E.g., uni or resnet50_layer3_norm")
    parser.add_argument('--sdf_suffix', type=str, default='_sdf70', help="Suffix for SDF feature files (default: _sdf70)")
    
    args = parser.parse_args()
    main(args)
