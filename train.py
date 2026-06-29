import argparse
import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.builder import create_model
from WSI_dataset import SpatialMILDataset

SUPPORTED_MODEL_FAMILIES = [
    "abmil",
    "clam",
    "dsmil",
    "dftd",
    "ilra",
    "rrt",
    "transmil",
    "transformer",
    "wikg",
    "spatial_abmil_orig",
]

BACKBONE_ENCODER_DIM = {
    "conch": ("conch", 512),
    "resnet50": ("resnet50", 1024),
    "resnet50_layer3_norm": ("resnet50", 1024),
    "uni": ("uni", 1024),
    "lego_uni": ("uni", 1024),
    "uni2": ("uni_v2", 1536),
    "uni2h": ("uni_v2", 1536),
    "uni_v2": ("uni_v2", 1536),
    "prov-gigapath": ("gigapath", 1536),
    "gigapath": ("gigapath", 1536),
    "phikon": ("phikon", 768),
    "phikon-v2": ("phikon2", 1024),
    "virchow-1280": ("virchow_1280", 1280),
    "virchow-2560": ("virchow", 2560),
    "virchow2-1280": ("virchow2_1280", 1280),
    "virchow2-2560": ("virchow2", 2560),
}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_class_map(value):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"class map must be JSON, got: {value}") from exc
    return {str(k): int(v) for k, v in parsed.items()}


def get_loggers(save_dir):
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    main_logger = logging.getLogger("dtmf_main_logger")
    main_logger.setLevel(logging.INFO)
    if main_logger.hasHandlers():
        main_logger.handlers.clear()

    main_log_file = os.path.join(save_dir, f"train_all_multirun_summary_{timestamp}.log")
    fh1 = logging.FileHandler(main_log_file, mode="w")
    fh1.setFormatter(formatter)
    main_logger.addHandler(fh1)

    sh1 = logging.StreamHandler(sys.stdout)
    sh1.setFormatter(formatter)
    main_logger.addHandler(sh1)

    epoch_logger = logging.getLogger("dtmf_epoch_logger")
    epoch_logger.setLevel(logging.INFO)
    if epoch_logger.hasHandlers():
        epoch_logger.handlers.clear()

    epoch_log_file = os.path.join(save_dir, f"train_all_multirun_epochs_{timestamp}.log")
    fh2 = logging.FileHandler(epoch_log_file, mode="w")
    fh2.setFormatter(formatter)
    epoch_logger.addHandler(fh2)

    main_logger.info(f"Summary logging to: {main_log_file}")
    main_logger.info(f"Epoch logging to: {epoch_log_file}")
    return main_logger, epoch_logger


def forward_step(model, batch, criterion, device, model_name):
    features = batch["features"].to(device)
    label = batch["label"].to(device)

    if "spatial" in model_name:
        model_kwargs = {
            "precomputed_distances": batch["distances"].to(device),
            "coords": batch["coords"].to(device),
            "loss_fn": criterion,
            "label": label,
        }
        if "cluster_labels" in batch:
            model_kwargs["cluster_labels"] = batch["cluster_labels"].to(device)
        results, _ = model(features, **model_kwargs)
    else:
        results, _ = model(features, loss_fn=criterion, label=label)

    return results, label


def train_one_epoch(model, loader, optimizer, criterion, device, model_name, accum_iter):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        results, label = forward_step(model, batch, criterion, device, model_name)
        loss = results["loss"] / accum_iter
        loss.backward()

        if (i + 1) % accum_iter == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        current_loss = loss.item() * accum_iter
        total_loss += current_loss
        logits = results["logits"].detach().cpu()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    epoch_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, epoch_acc


def evaluate(model, loader, criterion, device, model_name):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            results, label = forward_step(model, batch, criterion, device, model_name)
            total_loss += results["loss"].item()
            logits = results["logits"]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    avg_loss = total_loss / max(len(loader), 1)

    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    if all_probs.shape[1] == 2:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auc = 0.0
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    else:
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
        except ValueError:
            auc = 0.0
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, bacc, auc, conf_mat, f1, qwk


def path_has_cv_data(model_root):
    model_root = Path(model_root)
    has_features = (model_root / "training").exists() and (model_root / "testing").exists()
    has_fold_sdf = any(
        child.is_dir() and child.name.startswith("sdf_70d_features_")
        for child in model_root.iterdir()
    ) if model_root.exists() else False
    return has_features and has_fold_sdf


def discover_backbones(base_root, requested):
    base_root = Path(base_root)
    if requested != ["auto"]:
        return [b for b in requested if path_has_cv_data(base_root / b)]

    if not base_root.exists():
        return []
    return [child.name for child in sorted(base_root.iterdir()) if child.is_dir() and path_has_cv_data(child)]


def encoder_dim_for_backbone(backbone):
    if backbone in BACKBONE_ENCODER_DIM:
        return BACKBONE_ENCODER_DIM[backbone]
    if "resnet50" in backbone:
        return "resnet50", 1024
    if "conch" in backbone:
        return "conch", 512
    if "gigapath" in backbone:
        return "gigapath", 1536
    if "phikon-v2" in backbone:
        return "phikon2", 1024
    if "phikon" in backbone:
        return "phikon", 768
    if "uni2" in backbone or "uni_v2" in backbone:
        return "uni_v2", 1536
    if "uni" in backbone:
        return "uni", 1024
    raise ValueError(f"Could not infer encoder/in_dim for backbone folder '{backbone}'.")


def class_dist_string(labels, class_map):
    names_by_label = {v: k for k, v in class_map.items()}
    parts = []
    for label_id in sorted(set(class_map.values())):
        parts.append(f"{names_by_label.get(label_id, label_id)}: {labels.count(label_id)}")
    unknown_count = labels.count(-1)
    if unknown_count:
        parts.append(f"unknown: {unknown_count}")
    return ", ".join(parts)


def model_names_for_encoder(encoder, families):
    unsupported = sorted(set(families) - set(SUPPORTED_MODEL_FAMILIES))
    if unsupported:
        raise ValueError(f"Unsupported model families: {unsupported}")
    return [f"{family}.base.{encoder}.none" for family in families]


def run_one_backbone(args, backbone, main_save_dir):
    encoder, in_dim = encoder_dim_for_backbone(backbone)
    run_name = f"{args.dataset_name}_{backbone}_all_cv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_save_dir = main_save_dir / run_name
    run_save_dir.mkdir(parents=True, exist_ok=True)

    main_logger, epoch_logger = get_loggers(str(run_save_dir))
    main_logger.info(f"Dataset: {args.dataset_name}")
    main_logger.info(f"Backbone folder: {backbone}")
    main_logger.info(f"Encoder: {encoder} | in_dim: {in_dim}")
    main_logger.info(f"Base root: {args.base_root}")
    main_logger.info(f"CV splits: {args.cv_split_json}")
    main_logger.info(f"Labels: {args.label_json}")

    all_ds = SpatialMILDataset(
        base_root=args.base_root,
        model_name=backbone,
        mode="all",
        label_json=args.label_json,
        class_map=args.class_map,
        sdf_suffix=args.sdf_suffix,
    )
    with open(args.cv_split_json, "r", encoding="utf-8") as f:
        cv_splits = json.load(f)

    model_names = model_names_for_encoder(encoder, args.model_families)
    main_logger.info(f"Total cohort size loaded: {len(all_ds)}")
    main_logger.info(f"Model names: {', '.join(model_names)}")

    overall_results = {}
    device = torch.device(args.device)

    for model_name in model_names:
        main_logger.info("\n=========================================================")
        main_logger.info(f"Model: {model_name} | Running {len(cv_splits)}-fold CV")
        main_logger.info("=========================================================")
        overall_results[model_name] = {
            "f1": [], "auc": [], "acc": [], "bacc": [], "qwk": [],
            "best_val_auc": [], "best_val_bacc": [], "best_val_loss": [],
            "conf_mat": [], "best_val_conf_mat": [],
        }

        for fold_name in cv_splits.keys():
            if hasattr(all_ds, "set_cv_fold"):
                all_ds.set_cv_fold(fold_name)
            seed_everything(args.seed)

            train_ids = set(cv_splits[fold_name]["train"])
            val_ids = set(cv_splits[fold_name]["val"])
            test_ids = set(cv_splits[fold_name]["test"])

            train_idx = [i for i, item in enumerate(all_ds.data) if item["slide_id"] in train_ids]
            val_idx = [i for i, item in enumerate(all_ds.data) if item["slide_id"] in val_ids]
            test_idx = [i for i, item in enumerate(all_ds.data) if item["slide_id"] in test_ids]

            train_labels = [all_ds.data[i]["label"] for i in train_idx]
            val_labels = [all_ds.data[i]["label"] for i in val_idx]
            test_labels = [all_ds.data[i]["label"] for i in test_idx]

            main_logger.info(f"\n--- Fold {fold_name} for {model_name} ---")
            main_logger.info(f"  Train Size: {len(train_idx)} | {class_dist_string(train_labels, args.class_map)}")
            main_logger.info(f"  Val Size:   {len(val_idx)} | {class_dist_string(val_labels, args.class_map)}")
            main_logger.info(f"  Test Size:  {len(test_idx)} | {class_dist_string(test_labels, args.class_map)}")

            train_loader = DataLoader(Subset(all_ds, train_idx), batch_size=1, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(Subset(all_ds, val_idx), batch_size=1, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(Subset(all_ds, test_idx), batch_size=1, shuffle=False, num_workers=args.num_workers)

            model_kwargs = {"num_classes": args.num_classes, "in_dim": in_dim}
            if "spatial" in model_name:
                model_kwargs["spatial_input_dim"] = args.spatial_input_dim
            model = create_model(model_name, **model_kwargs).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()

            best_val_auc = float("-inf")
            best_val_bacc = 0.0
            best_val_loss = float("inf")
            best_val_qwk = float("-inf")
            best_val_conf_mat = None
            best_val_f1 = 0.0
            best_val_acc = 0.0

            model_save_dir = run_save_dir / model_name
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = model_save_dir / f"best_val_model_{model_name}_{fold_name}.pth"

            for epoch in range(args.epochs):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, model_name, args.accum_iter
                )
                val_loss, val_acc, val_bacc, val_auc, val_conf_mat, val_f1, val_qwk = evaluate(
                    model, val_loader, criterion, device, model_name
                )
                epoch_logger.info(
                    f"Epoch {epoch + 1:02d} | Trn_Loss: {train_loss:.4f} Trn_Acc: {train_acc:.4f} | "
                    f"Val_Loss: {val_loss:.4f} Val_Acc: {val_acc:.4f} Val_BACC: {val_bacc:.4f} "
                    f"Val_AUC: {val_auc:.4f} F1_score:{val_f1:.4f} QWK:{val_qwk:.4f}"
                )

                if args.num_classes > 2:
                    is_best = val_qwk > best_val_qwk or (np.isclose(val_qwk, best_val_qwk) and val_loss < best_val_loss)
                else:
                    is_best = val_auc > best_val_auc or (np.isclose(val_auc, best_val_auc) and val_loss < best_val_loss)

                if is_best:
                    best_val_auc = val_auc
                    best_val_bacc = val_bacc
                    best_val_loss = val_loss
                    best_val_qwk = val_qwk
                    best_val_conf_mat = val_conf_mat
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_save_path)

            model.load_state_dict(torch.load(model_save_path, map_location=device))
            test_loss, test_acc, test_bacc, test_auc, test_conf_mat, test_f1, test_qwk = evaluate(
                model, test_loader, criterion, device, model_name
            )

            main_logger.info(f"BEST VAL Result for {model_name} ({fold_name}):")
            main_logger.info(f"  F1:  {best_val_f1:.4f}")
            main_logger.info(f"  AUC: {best_val_auc:.4f}")
            main_logger.info(f"  ACC: {best_val_acc:.4f}")
            main_logger.info(f"  BACC:{best_val_bacc:.4f}")
            main_logger.info(f"  QWK: {best_val_qwk:.4f}")
            main_logger.info(f"  Confusion Matrix:\n{best_val_conf_mat}")

            main_logger.info(f"TEST Results for {model_name} ({fold_name}):")
            main_logger.info(f"  F1:  {test_f1:.4f}")
            main_logger.info(f"  AUC: {test_auc:.4f}")
            main_logger.info(f"  ACC: {test_acc:.4f}")
            main_logger.info(f"  BACC:{test_bacc:.4f}")
            main_logger.info(f"  QWK: {test_qwk:.4f}")
            main_logger.info(f"  Confusion Matrix:\n{test_conf_mat}")

            overall_results[model_name]["best_val_auc"].append(best_val_auc)
            overall_results[model_name]["best_val_bacc"].append(best_val_bacc)
            overall_results[model_name]["best_val_loss"].append(best_val_loss)
            overall_results[model_name]["f1"].append(test_f1)
            overall_results[model_name]["auc"].append(test_auc)
            overall_results[model_name]["acc"].append(test_acc)
            overall_results[model_name]["bacc"].append(test_bacc)
            overall_results[model_name]["qwk"].append(test_qwk)
            overall_results[model_name]["conf_mat"].append(test_conf_mat)
            overall_results[model_name]["best_val_conf_mat"].append(best_val_conf_mat)

            with open(model_save_dir / f"results_{model_name}_{fold_name}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "best_val_f1": best_val_f1,
                    "best_val_auc": best_val_auc,
                    "best_val_acc": best_val_acc,
                    "best_val_bacc": best_val_bacc,
                    "best_val_qwk": best_val_qwk,
                    "best_val_loss": best_val_loss,
                    "best_val_conf_mat": best_val_conf_mat.tolist() if isinstance(best_val_conf_mat, np.ndarray) else best_val_conf_mat,
                    "test_f1": test_f1,
                    "test_auc": test_auc,
                    "test_acc": test_acc,
                    "test_bacc": test_bacc,
                    "test_qwk": test_qwk,
                    "test_loss": test_loss,
                    "test_conf_mat": test_conf_mat.tolist() if isinstance(test_conf_mat, np.ndarray) else test_conf_mat,
                }, f, indent=2)

        main_logger.info(f"\n--- Aggregated Results for {model_name} ---")
        main_logger.info(f"  Mean TEST F1:   {np.mean(overall_results[model_name]['f1']):.4f} +/- {np.std(overall_results[model_name]['f1']):.4f}")
        main_logger.info(f"  Mean TEST AUC:  {np.mean(overall_results[model_name]['auc']):.4f} +/- {np.std(overall_results[model_name]['auc']):.4f}")
        main_logger.info(f"  Mean TEST ACC:  {np.mean(overall_results[model_name]['acc']):.4f} +/- {np.std(overall_results[model_name]['acc']):.4f}")
        main_logger.info(f"  Mean TEST BACC: {np.mean(overall_results[model_name]['bacc']):.4f} +/- {np.std(overall_results[model_name]['bacc']):.4f}")
        main_logger.info(f"  Mean TEST QWK:  {np.mean(overall_results[model_name]['qwk']):.4f} +/- {np.std(overall_results[model_name]['qwk']):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DTMf-MIL/spatial_abmil_orig and standard MIL baselines with CV.")
    parser.add_argument("--base-root", required=True, help="Root containing one folder per feature backbone.")
    parser.add_argument("--dataset-name", default="dataset", help="Name used in output run folders.")
    parser.add_argument("--backbones", nargs="+", default=["auto"], help="Backbone folders to run, or auto.")
    parser.add_argument("--cv-split-json", required=True, help="CV split JSON with fold keys (e.g. fold_1, fold_2, ...).")
    parser.add_argument("--label-json", required=True, help="JSON labels for held-out/test slides.")
    parser.add_argument("--class-map", type=parse_class_map, default={"non_metastatic": 0, "metastatic": 1})
    parser.add_argument("--save-dir", default="runs", help="Output directory for logs, checkpoints, and JSON results.")
    parser.add_argument("--model-families", nargs="+", default=SUPPORTED_MODEL_FAMILIES)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--spatial-input-dim", type=int, default=70)
    parser.add_argument("--sdf-suffix", default="_sdf70", help="Suffix for SDF feature file names.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accum-iter", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    backbones = discover_backbones(args.base_root, args.backbones)
    if not backbones:
        raise SystemExit("No usable backbones found. Check --base-root, --backbones, and fold-specific SDF folders.")

    print("Planned CV runs:")
    for backbone in backbones:
        encoder, in_dim = encoder_dim_for_backbone(backbone)
        print(f"  - {args.dataset_name} | {backbone} | encoder={encoder} | in_dim={in_dim}")

    for backbone in backbones:
        run_one_backbone(args, backbone, save_dir)


if __name__ == "__main__":
    main()
