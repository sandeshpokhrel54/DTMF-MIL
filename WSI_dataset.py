import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import json
import glob
from pathlib import Path

class SpatialMILDataset(Dataset):
    def __init__(self, 
                 base_root: str, 
                 model_name: str,
                 mode: str = 'train', 
                 label_json: str = None, 
                 split_json: str = None,
                 sdf_suffix: str = '_sdf70',
                 tile_size: int = 256,
                 class_map: dict = {'luad': 0, 'lusc': 1},
                 sdf_model_name: str = None):
        """
        Args:
            base_root: The root directory containing the model folder (e.g., .../TCGA_NSCLC_Features)
            model_name: The specific visual feature folder name (e.g., 'resnet50_layer3_norm')
            sdf_model_name: Optional model folder to source SDF70 features from. Defaults to model_name.
            mode: 'train', 'val', or 'test'
            label_json: Path to .json file for test labels (required for test mode)
            split_json: Path to .json file containing train/val split slide IDs
            sdf_suffix: Suffix used in SDF filenames (e.g., '_sdf70')
        """
        self.base_root = Path(base_root)
        self.model_name = model_name
        self.sdf_model_name = sdf_model_name or model_name
        self.model_root = self.base_root / model_name
        self.sdf_model_root = self.base_root / self.sdf_model_name
        self.mode = mode
        self.sdf_suffix = sdf_suffix
        self.tile_size = tile_size
        self.class_map = {k.lower(): v for k, v in class_map.items()} 
        self.data = []
        self.cv_fold = None

        # Define specific paths based on structure
        if self.mode in ['train', 'val']:
            self.feat_dir = self.model_root / "training"
            self.sdf_dir = self.sdf_model_root / "sdf_70d_features"
            
            # Load split lists if provided
            self.split_list = None
            if split_json:
                with open(split_json, 'r') as f:
                    splits = json.load(f)
                    self.split_list = set(splits.get(self.mode, []))
                    
            self._build_train_list()
        elif self.mode == 'test':
            self.feat_dir = self.model_root / "testing"
            self.sdf_dir = self.sdf_model_root / "sdf_70d_testing"
            
            # Handle Test Labels
            if label_json is None:
                print("Warning: label_json not provided for test mode. Labels will be -1.")
                self.test_labels = {}
            else:
                with open(label_json, 'r') as f:
                    self.test_labels = json.load(f)
            self._build_test_list()
        elif self.mode == 'all':
            self.split_list = None
            
            # Load Training Data
            self.feat_dir = self.model_root / "training"
            self.sdf_dir = self.sdf_model_root / "sdf_70d_features"
            if self.feat_dir.exists():
                self._build_train_list()
                
            # Load Testing Data
            self.feat_dir = self.model_root / "testing"
            self.sdf_dir = self.sdf_model_root / "sdf_70d_testing"
            if self.feat_dir.exists():
                if label_json is None:
                    print("Warning: label_json not provided for test mode in 'all'. Labels will be -1.")
                    self.test_labels = {}
                else:
                    with open(label_json, 'r') as f:
                        self.test_labels = json.load(f)
                self._build_test_list()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _build_train_list(self):
        # Recursively find all .pt files in training directory (inside LUAD/LUSC folders)
        # Structure: training/LUAD/pt_files/*.pt
        all_pt_files = sorted(list(self.feat_dir.rglob("*.pt")))
        
        if not all_pt_files:
            raise FileNotFoundError(f"No .pt files found in {self.feat_dir}")

        for pt_path in all_pt_files:
            slide_id = pt_path.stem
            
            # Filter by split list if applicable
            if self.split_list is not None and slide_id not in self.split_list:
                continue

            # Infer label from parent directory name (e.g., LUAD)
            # Parent of pt_files is the class folder
            class_name = pt_path.parent.parent.name.lower()
            
            if class_name in self.class_map:
                label = self.class_map[class_name]
            else:
                continue # Skip if folder name doesn't match known classes

            self.data.append({
                'slide_id': slide_id,
                'pt_path': pt_path,
                'label': label,
                'class_name': class_name,
                'sdf_dir': self.sdf_dir
            })

    def _build_test_list(self):
        # Testing folder is usually flat: testing/pt_files/*.pt
        all_pt_files = sorted(list(self.feat_dir.rglob("*.pt")))
        
        if not all_pt_files:
             raise FileNotFoundError(f"No .pt files found in {self.feat_dir}")

        for pt_path in all_pt_files:
            slide_id = pt_path.stem
            
            # Look up label in JSON
            # JSON format: { "slide_id.svs": "LUAD" } or { "slide_id": "LUAD" }
            label = -1
            found_key = None
            
            # Try to match slide_id with or without extensions in the JSON keys
            for key in self.test_labels:
                if slide_id in key: # naive match
                    val = self.test_labels[key].lower()
                    if val in self.class_map:
                        label = self.class_map[val]
                        found_key = val
                        break
            
            self.data.append({
                'slide_id': slide_id,
                'pt_path': pt_path,
                'label': label,
                'class_name': found_key if found_key else 'unknown',
                'sdf_dir': self.sdf_dir
            })

    def _load_h5_coords(self, slide_id, pt_path):
        """Finds the corresponding .h5 file for coordinates."""
        # Logic: 
        # 1. Check ../h5_files/relative_to_pt
        # 2. Check standard training structure (sibling folder)
        
        parent_dir = pt_path.parent.parent # e.g., .../training/LUAD
        
        # Try 'h5_files'
        h5_path = parent_dir / "h5_files" / f"{slide_id}.h5"
        if h5_path.exists(): return h5_path
        
        # Try 'h5pyfiles' (legacy)
        h5_path = parent_dir / "h5pyfiles" / f"{slide_id}.h5"
        if h5_path.exists(): return h5_path
        
        # Try recursive search if standard fails
        found = list(parent_dir.rglob(f"{slide_id}.h5"))
        if found: return found[0]
        
        return None

    def _load_sdf(self, slide_id, class_name, sdf_dir=None):
        """Finds the corresponding .npz SDF file."""
        import glob
        filename = f"{slide_id}{self.sdf_suffix}.npz"
        
        sdf_base = sdf_dir if sdf_dir is not None else self.sdf_dir
        # print(f"DEBUG _load_sdf: slide_id={slide_id}, class_name={class_name}, sdf_base={sdf_base}")
        
        # 1. Try flat directory structure first (since directories were recently flattened)
        flat_path = sdf_base / filename
        if flat_path.exists(): 
            # print(f"DEBUG _load_sdf: Found at {flat_path}")
            return flat_path
        
        # 2. Try nested directory structure (original format)
        nested_path = sdf_base / class_name / filename
        if nested_path.exists(): 
            # print(f"DEBUG _load_sdf: Found at {nested_path}")
            return nested_path
            
        # 3. Fallback to recursive search
        found = list(sdf_base.rglob(filename))
        if found: 
            # print(f"DEBUG _load_sdf: Found via rglob at {found[0]}")
            return found[0]
            
        # print(f"DEBUG _load_sdf: NOT FOUND!")
        return None

    def set_cv_fold(self, fold_name):
        """Sets the active cross-validation fold to route to the correct fold-specific SDF directory."""
        self.cv_fold = fold_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pt_path = item['pt_path']
        slide_id = item['slide_id']
        label = item['label']
        class_name = item['class_name']
        
        if self.cv_fold is not None:
            active_sdf_dir = self.sdf_model_root / f"sdf_70d_features_{self.cv_fold}"
            # Backward compatibility: fallback to generic folder if fold-specific one doesn't exist
            if not active_sdf_dir.exists():
                active_sdf_dir = item.get('sdf_dir', self.sdf_dir)
        else:
            active_sdf_dir = item.get('sdf_dir', self.sdf_dir)
        
        # 1. Load Visual Features (.pt)
        try:
            features = torch.load(pt_path, map_location='cpu')
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)
        except Exception as e:
            # Return dummy/zeros or raise error depending on preference
            print(f"Error loading {pt_path}: {e}")
            features = torch.zeros(1, 1024)

        # 2. Load Coordinates (.h5)
        h5_path = self._load_h5_coords(slide_id, pt_path)
        if h5_path and h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                coords_raw = f['coords'][:]
        else:
            # Fallback if H5 missing (rare): create dummy coords or fail
            coords_raw = np.zeros((features.shape[0], 2))

        # 3. Load Spatial SDF Features (.npz)
        sdf_path = self._load_sdf(slide_id, item['class_name'], sdf_dir=active_sdf_dir)
        
        cluster_labels = None

        if sdf_path and sdf_path.exists():
            try:
                with np.load(sdf_path) as data:
                    if 'spatial_features' in data:
                        sdf_features = data['spatial_features']
                    else:
                        # Fallback if key is different (e.g. arr_0)
                        key = list(data.keys())[0]
                        sdf_features = data[key]
                    if 'labels' in data:
                        cluster_labels = data['labels']
            except:
                 sdf_features = np.zeros((features.shape[0], 70))
        else:
            # If SDF missing, return zeros (allows training to proceed)
            sdf_features = np.zeros((features.shape[0], 70))

        if cluster_labels is None:
            if sdf_features.shape[-1] >= 70:
                # Existing SDF layout: 5 clusters x [10 RBFs, gx, gy, local_mu, local_std].
                local_mu = sdf_features[:, :70].reshape(-1, 5, 14)[..., 12]
                cluster_labels = np.argmin(local_mu, axis=1)
            else:
                cluster_labels = np.zeros((features.shape[0],), dtype=np.int64)

        # 4. Alignment & Normalization
        # Ensure all lengths match the minimum length (data integrity)
        min_len = min(features.shape[0], coords_raw.shape[0], sdf_features.shape[0], cluster_labels.shape[0])
        
        features = features[:min_len]
        coords_raw = coords_raw[:min_len]
        sdf_features = sdf_features[:min_len]
        cluster_labels = cluster_labels[:min_len]
        
        # Grid Coordinates for Transformers
        coords_grid = np.round((coords_raw - coords_raw.min(axis=0)) / self.tile_size).astype(int)

        return {
            'slide_id': slide_id,
            'features': features.float(),                    # [N, 1024]
            'distances': torch.from_numpy(sdf_features).float(), # [N, 70]
            'coords': torch.from_numpy(coords_grid).long(),      # [N, 2]
            'cluster_labels': torch.from_numpy(cluster_labels).long(), # [N]
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

# --- Testing the Class ---
if __name__ == "__main__":
    
    script_dir = Path(__file__).resolve().parent
    from torch.utils.data import Subset

    BASE = "BASE_DIR"
    MODEL = "MODEL_NAME"
    CV_SPLIT_JSON = script_dir / "splits" / "cv_splits.json"
    LABEL_JSON = script_dir.parent / "test_label_va.json"

    all_ds = SpatialMILDataset(
        base_root=BASE,
        model_name=MODEL,
        mode='all',
        label_json=str(LABEL_JSON),
        class_map={'non_metastatic': 0, 'metastatic': 1}
    )

    all_ds.set_cv_fold(FOLD_NAME)

    with open(CV_SPLIT_JSON, 'r') as f:
        cv_splits = json.load(f)

    available_ids = {item['slide_id'] for item in all_ds.data}
    target_slide_id = None
    for split_name in ["train", "val", "test"]:
        for slide_id in cv_splits[FOLD_NAME][split_name]:
            if slide_id in available_ids:
                target_slide_id = slide_id
                break
        if target_slide_id is not None:
            break

    print("\n=============================================")
    print(f"--- Current CV Logic (mode='all', {FOLD_NAME}) ---")
    print(f"Model root:   {all_ds.model_root}")
    print(f"Loaded slides:{len(all_ds)}")
    print(f"Active SDF:   {all_ds.model_root / f'sdf_70d_features_{FOLD_NAME}'}")

    if target_slide_id is not None:
        target_idx = next(
            i for i, item in enumerate(all_ds.data)
            if item['slide_id'] == target_slide_id
        )

        target_subset = Subset(all_ds, [target_idx])
        cv_loader = DataLoader(target_subset, batch_size=1, shuffle=False)
        cv_batch = next(iter(cv_loader))

        expected_sdf = all_ds.model_root / f"sdf_70d_features_{FOLD_NAME}" / f"{target_slide_id}_sdf70.npz"

        print(f"Slide ID:     {target_slide_id}")
        print(f"SDF Path:     {expected_sdf}")
        print(f"SDF Exists:   {expected_sdf.exists()}")
        print(f"Visual Shape: {cv_batch['features'].shape}")
        print(f"SDF Shape:    {cv_batch['distances'].shape}")
        print(f"SDF values:   {cv_batch['distances'][0][0]}")
        print(f"Coords Shape: {cv_batch['coords'].shape}")
        print(f"Cluster Lbl Shape: {cv_batch['cluster_labels'].shape}")
        print(f"Cluster Lbl values:{cv_batch['cluster_labels'][0][:10]}")
        print(f"Label:        {cv_batch['label'].item()}")
    else:
        print(f"Could not find any {FOLD_NAME} split slide in all_ds!")
