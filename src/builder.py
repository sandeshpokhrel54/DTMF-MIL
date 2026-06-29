import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn
import yaml
from loguru import logger

from src.builders.ModelDict import ModelDict
from src.builder_utils import _cfg, build_model_with_cfg
from src._global_mappings import (
    REPO_PATH,
    CONFIG_PATH,
    ENCODER_DIM_MAPPING,
    MODEL_ENTRYPOINTS,
    MODEL_SAVE_PATH,
)


def save_model(model: nn.Module, model_name: str, save_folder: str = MODEL_SAVE_PATH, save_pretrained: bool = False):
    from transformers import PreTrainedModel
    import torch

    model_path = os.path.join(save_folder, model_name)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    if isinstance(model, PreTrainedModel) and save_pretrained:
        model.save_pretrained(model_path)
        if hasattr(model, "config"):
            model.config.save_pretrained(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
    print(f"Model saved to {model_path}")


def create_model(
    model_name: str,
    num_classes: int = 2,
    checkpoint_path: str = "",
    hf_base_repo: str = "mahmoodlab/",
    from_pretrained: bool = False,
    pretrained_strict: bool = False,
    keep_classifier: bool = False,
    **kwargs,
):
    model_dict = ModelDict.from_string(model_name)
    pretrained_cfg = _create_pretrained_config(
        model_dict,
        hf_source=hf_base_repo,
        local_source=MODEL_SAVE_PATH,
    )
    pretrained_cfg = _update_checkpoint_path(checkpoint_path, pretrained_cfg)
    from_pretrained = from_pretrained and model_dict.is_pretrained()

    return build_model(
        model_name=model_dict.model_name,
        model_config=model_dict.model_config,
        pretrained=model_dict.is_pretrained(),
        encoder=model_dict.encoder,
        num_classes=num_classes,
        pretrained_cfg=pretrained_cfg,
        from_pretrained=from_pretrained,
        pretrained_strict=pretrained_strict,
        keep_classifier=keep_classifier,
        **kwargs,
    )


def _update_checkpoint_path(checkpoint_path: str, pretrained_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.warning(
            f"Checkpoint path manually provided ({checkpoint_path}). "
            f"Overwriting previous local filepath {pretrained_cfg.get('local_path', '')}."
        )
        pretrained_cfg["file"] = checkpoint_path
    return pretrained_cfg


def build_model(
    model_name: str,
    model_config: str,
    pretrained: bool,
    encoder: str,
    num_classes: int,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    from_pretrained: bool = False,
    pretrained_strict: bool = False,
    keep_classifier: bool = False,
    **kwargs,
) -> nn.Module:
    if model_name not in MODEL_ENTRYPOINTS:
        supported = ", ".join(sorted(MODEL_ENTRYPOINTS))
        raise KeyError(f"Unsupported model '{model_name}'. Supported models: {supported}")
    if encoder not in ENCODER_DIM_MAPPING:
        supported = ", ".join(sorted(ENCODER_DIM_MAPPING))
        raise KeyError(f"Unsupported encoder '{encoder}'. Supported encoders: {supported}")

    config = _load_model_config(model_name, model_config)
    config["in_dim"] = ENCODER_DIM_MAPPING[encoder]
    config["num_classes"] = num_classes
    config.update(kwargs)

    model_cls, config_cls = MODEL_ENTRYPOINTS[model_name]
    return build_model_with_cfg(
        model_cls,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        model_cfg=config_cls(**config),
        from_pretrained=from_pretrained,
        pretrained_strict=pretrained_strict,
        keep_classifier=keep_classifier,
        **kwargs,
    )


def _load_model_config(model_name: str, model_config: str) -> Dict[str, Any]:
    config_path = os.path.join(CONFIG_PATH, model_name, f"{model_config}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _create_pretrained_config(
    name_dict: ModelDict,
    hf_source: str = "MahmoodLab/",
    local_source: str = "model_weights",
) -> Dict[str, Any]:
    default_cfg = _cfg()
    model_path = name_dict.to_string()
    default_cfg["hf_hub_id"] = os.path.join(hf_source, model_path).replace("\\", "/")
    default_cfg["local_path_parent"] = os.path.join(REPO_PATH, local_source, model_path)
    default_cfg["local_path"] = os.path.join(REPO_PATH, local_source, model_path)
    return default_cfg
