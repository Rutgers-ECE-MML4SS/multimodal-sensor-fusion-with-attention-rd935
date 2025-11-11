"""
Training Script for Multimodal Sensor Fusion

Uses PyTorch Lightning for training with Hydra configuration.
Most infrastructure is provided - students need to integrate their fusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import pathlib
import typing
import torch.serialization as serialization
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Dict  # <<< added so we can type-hint _fuse

serialization.add_safe_globals([
    pathlib.WindowsPath,
    DictConfig,
    ListConfig,
    ContainerMetadata,
    typing.Any,
])

from data import create_dataloaders
from fusion import build_fusion_model
from encoders import build_encoder


# -------------------------------------------------------------------------
# make torch.load safe for Hydra objects (WindowsPath, DictConfig, etc.)
# -------------------------------------------------------------------------
_orig_torch_load = torch.load


def _safe_load(*args, **kwargs):
    # if not specified, force old behavior
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _safe_load


class MultimodalFusionModule(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal fusion training.

    Handles training loop, validation, and logging.
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra configuration object
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # ---------------------------------------------------------
        # Build encoders for each modality
        # ---------------------------------------------------------
        self.encoders = nn.ModuleDict()
        modality_output_dims = {}

        for modality in config.dataset.modalities:
            # each modality can have its own encoder config
            encoder_config = dict(config.model.encoders.get(modality, {}))
            input_dim = encoder_config.pop("input_dim", 64)
            output_dim = config.model.output_dim
            # remove "type" if present in YAML
            encoder_config.pop("type", None)

            # map dataset-specific names → canonical names used in encoders.py
            canonical_modality = modality
            if modality.startswith("imu_"):
                # imu_hand, imu_chest, imu_ankle → 'imu'
                canonical_modality = "imu"
            elif modality in ["heart_rate", "hr"]:
                # HR is still time-series → treat as imu-like
                canonical_modality = "imu"

            self.encoders[modality] = build_encoder(
                modality=canonical_modality,
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_config=encoder_config,
            )
            modality_output_dims[modality] = output_dim

        print(">>> Using num_classes =", config.dataset.get("num_classes", "NOT FOUND"))

        # ---------------------------------------------------------
        # Build fusion model
        # Students need to ensure their fusion implementation works here
        # ---------------------------------------------------------
        fusion_kwargs = dict(
            fusion_type=config.model.fusion_type,
            modality_dims=modality_output_dims,
            num_classes=config.dataset.get("num_classes", 11),
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )

        # only attention-based fusion needs num_heads
        if config.model.fusion_type in ("hybrid", "uncertainty"):
            fusion_kwargs["num_heads"] = config.model.get("num_heads", 2)

        self.fusion_model = build_fusion_model(**fusion_kwargs)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # (you had these but weren’t using them – leaving for compatibility)
        self.train_metrics = []
        self.val_metrics = []

    # ------------------------------------------------------------------
    # NEW: small helper to normalize fusion outputs
    # some fusions return just logits
    # hybrid returns (logits, extras)
    # ------------------------------------------------------------------
    def _fuse(
        self, encoded_features: Dict[str, torch.Tensor], mask: torch.Tensor = None
    ):
        """
        Run the fusion model and always return (logits, extras_or_None).
        This keeps train/val/test code simple.
        """
        # only ask for attention if we know model is hybrid
        wants_attention = (
            hasattr(self.config.model, "fusion_type")
            and self.config.model.fusion_type == "hybrid"
        )

        if wants_attention:
            out = self.fusion_model(encoded_features, mask, return_attention=True)
        else:
            out = self.fusion_model(encoded_features, mask)

        # normalize output
        if isinstance(out, tuple):
            logits, extras = out
        else:
            logits, extras = out, None
        return logits, extras

    def forward(self, features, mask=None):
        """
        Forward pass through encoders and fusion model.

        Args:
            features: Dict of {modality_name: features}
            mask: Optional modality availability mask

        Returns:
            logits: Class predictions
        """
        # Encode each modality
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded_features[modality] = encoder(features[modality])

        # Fusion (we ignore extras at top-level forward)
        logits, _ = self._fuse(encoded_features, mask)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        features, labels, mask = batch

        # encode each modality first
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded_features[modality] = encoder(features[modality])

        # Fusion (hybrid may return (logits, extras))
        logits, _ = self._fuse(encoded_features, mask)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        features, labels, mask = batch

        # encode each modality
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded_features[modality] = encoder(features[modality])

        # fusion
        logits, extras = self._fuse(encoded_features, mask)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Get confidence for calibration
        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        # Log metrics
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

        # we can still return a dict; lightning 2 will just collect it
        out = {
            "val_loss": loss,
            "val_acc": acc,
            "preds": preds,
            "labels": labels,
            "confidences": confidences,
        }

        # if you want to look at extras later, you can stash them here
        # but we do NOT use validation_epoch_end anymore (removed in PL 2.x)
        if extras is not None and "attention_matrix" in extras:
            out["attention_matrix"] = extras["attention_matrix"].detach().cpu()
        if extras is not None and "fusion_weights" in extras:
            out["fusion_weights"] = extras["fusion_weights"].detach().cpu()

        return out

    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        features, labels, mask = batch

        # encode
        encoded_features = {}
        for modality, encoder in self.encoders.items():
            if modality in features:
                encoded_features[modality] = encoder(features[modality])

        # fusion
        logits, _ = self._fuse(encoded_features, mask)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        probs = F.softmax(logits, dim=1)
        confidences, _ = torch.max(probs, dim=1)

        self.log("test/acc", acc, on_epoch=True)

        return {
            "preds": preds,
            "labels": labels,
            "confidences": confidences,
        }

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        # Learning rate scheduler
        if self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate / 100,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif self.config.training.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """
    Main training function.

    Args:
        config: Hydra configuration
    """
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    # Create output directories
    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        modality_dropout=config.training.augmentation.modality_dropout,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = MultimodalFusionModule(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{epoch}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=config.experiment.save_top_k,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name="logs",
    )

    # mixed precision toggle
    precision_mode = "16-mixed" if getattr(config, "mixed_precision", False) else "32-true"

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices=1,
        precision=precision_mode,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config.experiment.log_every_n_steps,
        gradient_clip_val=config.training.gradient_clip_norm,
        deterministic=True,
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test on best model
    print("\nTesting best model...")
    best_model_path = str(checkpoint_callback.best_model_path)
    print(f"Loading best model from: {best_model_path}")

    trainer.test(model, test_loader, ckpt_path=best_model_path)

    # Save final results
    results = {
        "best_model_path": str(best_model_path),
        "best_val_loss": float(checkpoint_callback.best_model_score),
        "config": OmegaConf.to_container(config, resolve=True),
    }

    results_file = save_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete! Results saved to: {results_file}")
    print(f"Best model: {best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
