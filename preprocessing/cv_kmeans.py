import os
import glob
import torch
import joblib
import numpy as np
import json
import argparse
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def find_pt_file(slide_id, root_dir):
    """
    Finds the .pt file for a given slide_id by searching the root directory.
    """
    search_pattern = os.path.join(root_dir, '**', 'pt_files', f'{glob.escape(slide_id)}.pt')
    files = glob.glob(search_pattern, recursive=True)
    if files:
        return files[0]
    return None

def load_feature_tensor(path):
    try:
        tensor = torch.load(path, map_location='cpu')
        if isinstance(tensor, torch.Tensor):
            return tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main(args):
    OPTIMAL_K = args.clusters
    BATCH_SIZE = 1024
    RANDOM_STATE = 42
    
    with open(args.cv_json, 'r') as f:
        cv_splits = json.load(f)
        
    for fold_name, splits in cv_splits.items():
        print(f"\n==================================================")
        print(f"Training KMeans for {fold_name} | {args.model_name}")
        print(f"==================================================")
        
        train_slides = splits['train']
        print(f"Number of training slides for {fold_name}: {len(train_slides)}")
        
        output_model_path = os.path.join(args.out_dir, f"{args.model_name}_kmeans_model_{fold_name}.pkl")
        
        # Skip if a valid, fitted model already exists
        if os.path.exists(output_model_path):
            try:
                existing_model = joblib.load(output_model_path)
                if hasattr(existing_model, 'cluster_centers_'):
                    print(f"Valid model already exists at {output_model_path}. Skipping KMeans fitting.")
                    continue
            except Exception as e:
                print(f"Existing model corrupted or invalid: {e}. Re-fitting.")
        
        final_model = MiniBatchKMeans(n_clusters=OPTIMAL_K, 
                                      batch_size=BATCH_SIZE, 
                                      random_state=RANDOM_STATE, 
                                      n_init=10)

        valid_slides_count = 0
        loop = tqdm(train_slides, desc=f"Fitting {fold_name}")
        for slide_id in loop:
            fpath = find_pt_file(slide_id, args.root_dir)
            
            if fpath is None:
                continue
                
            features = load_feature_tensor(fpath)
            if features is not None and len(features) > 0:
                final_model.partial_fit(features)
                valid_slides_count += 1
                
        output_model_path = os.path.join(args.out_dir, f"{args.model_name}_kmeans_model_{fold_name}.pkl")
        joblib.dump(final_model, output_model_path)
        print(f"\nDone training {fold_name} on {valid_slides_count} slides.")
        print(f"Model saved to: {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help="Base directory of features")
    parser.add_argument('--out_dir', type=str, required=True, help="Where to save the .pkl models")
    parser.add_argument('--cv_json', type=str, required=True, help="Path to CV split JSON")
    parser.add_argument('--model_name', type=str, required=True, help="E.g., uni or resnet50_layer3_norm")
    parser.add_argument('--clusters', type=int, default=5, help="Number of KMeans clusters")
    
    args = parser.parse_args()
    main(args)
