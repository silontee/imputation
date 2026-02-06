"""TOTEM VQVAE model loader with singleton caching."""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any
import importlib

import torch

logger = logging.getLogger(__name__)


@dataclass
class VQVAEConfig:
    """Configuration for TOTEM VQVAE model."""
    compression_factor: int = 4
    code_dim: int = 64
    num_codes: int = 512
    model_path: str = ""
    device: str = "cpu"


class TOTEMModelLoader:
    """Singleton model loader for TOTEM VQVAE.

    Caches loaded models to avoid repeated loading overhead.
    """

    _instance: Optional["TOTEMModelLoader"] = None
    _model: Optional[Any] = None
    _config: Optional[VQVAEConfig] = None
    _current_path: Optional[str] = None

    def __new__(cls) -> "TOTEMModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_device(cls) -> str:
        """Auto-detect best available device: CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def load(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        code_path: Optional[str] = None,
    ) -> Tuple[Any, VQVAEConfig]:
        """Load TOTEM VQVAE model and config.

        Args:
            model_path: Path to the directory containing model files or the model .pt file directly.
            config_path: Optional path to config JSON. If not provided, auto-discovers from model_path.
            device: Target device. If None, auto-detects.
            code_path: Path to TOTEM code root (must contain `lib/models`).

        Returns:
            Tuple of (model, config).
        """
        loader = cls()

        if device is None:
            device = cls.get_device()

        # Resolve paths
        model_path = Path(model_path)

        # Ensure pickled model classes can be imported (torch.save(model) format).
        cls._ensure_totem_imports(code_path)

        # Find the actual model file
        if model_path.is_file() and model_path.suffix in {".pt", ".pth"}:
            model_file = model_path
            model_dir = model_path.parent
        elif model_path.is_dir():
            # Look for .pt/.pth files in the directory
            model_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth"))
            if not model_files:
                # Check subdirectories
                model_files = list(model_path.glob("**/*.pt")) + list(model_path.glob("**/*.pth"))
            if not model_files:
                raise FileNotFoundError(f"No .pt/.pth model files found in {model_path}")

            # Prefer final_model first if present.
            model_files = sorted(model_files, key=lambda p: (0 if p.name == "final_model.pth" else 1, str(p)))
            model_file = model_files[0]
            model_dir = model_path
            logger.info(f"Found model file: {model_file}")
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

        model_file_str = str(model_file)

        # Check cache
        if loader._model is not None and loader._current_path == model_file_str:
            # Move to requested device if different
            if loader._config and loader._config.device != device:
                loader._model.to(device)
                loader._config.device = device
            return loader._model, loader._config

        # Load config
        config = VQVAEConfig(device=device, model_path=model_file_str)

        if config_path:
            config_file = Path(config_path)
        else:
            # Auto-discover config file
            config_candidates = [
                model_dir / "config.json",
                model_dir / "config_file.json",
                model_dir / "configs" / "config_file.json",
            ]
            config_file = None
            for candidate in config_candidates:
                if candidate.exists():
                    config_file = candidate
                    break

        if config_file and config_file.exists():
            with open(config_file, "r") as f:
                cfg_data = json.load(f)
            vq_cfg = cfg_data.get("vqvae_config", cfg_data)
            config.compression_factor = int(vq_cfg.get("compression_factor", 4))
            config.code_dim = int(vq_cfg.get("embedding_dim", vq_cfg.get("code_dim", 64)))
            config.num_codes = int(vq_cfg.get("num_embeddings", vq_cfg.get("num_codes", 512)))
            logger.info(f"Loaded config from {config_file}")
        else:
            logger.warning(f"Config file not found, using defaults")

        # Load model
        logger.info(f"Loading TOTEM model from {model_file} to {device}")
        model = torch.load(model_file, map_location=device, weights_only=False)
        model.to(device)
        model.eval()

        # Cache
        loader._model = model
        loader._config = config
        loader._current_path = model_file_str

        logger.info(f"TOTEM model loaded successfully (compression_factor={config.compression_factor})")

        return model, config

    @classmethod
    def _ensure_totem_imports(cls, code_path: Optional[str]) -> None:
        """Ensure `lib.models.*` can be imported for pickled TOTEM models."""
        if code_path:
            code_dir = Path(code_path)
            if code_dir.exists():
                code_dir_str = str(code_dir)
                if code_dir_str not in sys.path:
                    sys.path.insert(0, code_dir_str)
            else:
                logger.warning(f"TOTEM code path does not exist: {code_path}")

        try:
            importlib.import_module("lib.models.vqvae")
        except Exception as e:
            raise ImportError(
                "Failed to import TOTEM model modules (`lib.models.vqvae`). "
                "Set TOTEM_CODE_PATH to the folder containing `lib/` (e.g., /app/TOTEM/imputation)."
            ) from e

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached model."""
        loader = cls()
        loader._model = None
        loader._config = None
        loader._current_path = None
