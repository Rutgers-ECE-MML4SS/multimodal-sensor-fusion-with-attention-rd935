import os
import sys
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from train import MultimodalFusionModule
from data import create_dataloaders
from analysis import plot_calibration_diagram


def parse_args():
    p = argparse.ArgumentParser(description="Make calibration / reliability diagram")
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "base.yaml"),
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "runs" / "a2_hybrid_pamap2" / "checkpoints" / "last.ckpt"),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "analysis" / "calibration.png"),
    )
    return p.parse_args()


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


def main():
    args = parse_args()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f">>> loading model from {args.checkpoint}")
    model = MultimodalFusionModule.load_from_checkpoint(args.checkpoint, config=cfg)
    model.to(device)
    model.eval()

    ds = cfg.dataset
    train_loader, val_loader, test_loader = create_dataloaders(
        ds.name,
        ds.data_dir,
        ds.modalities,
        batch_size=ds.batch_size,
        num_workers=ds.num_workers,
        modality_dropout=cfg.training.augmentation.modality_dropout,
    )

    loader = test_loader or val_loader

    all_conf, all_pred, all_lab = [], [], []

    print(">>> collecting predictions for calibration...")
    for batch in loader:
        # collate: (modalities_dict, labels, modality_mask)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            modalities_dict, labels, modality_mask = batch
            modalities_dict = move_to_device(modalities_dict, device)
            labels = move_to_device(labels, device)
            modality_mask = move_to_device(modality_mask, device)

            batch_for_model = {
                **modalities_dict,         
                "labels": labels,
                "modality_mask": modality_mask,
            }
        else:
            # fallback: maybe it's already a dict
            batch_for_model = move_to_device(batch, device)
            if "labels" in batch_for_model:
                labels = batch_for_model["labels"]
            elif "label" in batch_for_model:
                labels = batch_for_model["label"]
            else:
                raise KeyError("Could not find labels in batch")

        with torch.no_grad():
            out = model(batch_for_model)

        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        all_conf.append(conf.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_lab.append(labels.cpu().numpy())

    confidences = np.concatenate(all_conf, axis=0)
    predictions = np.concatenate(all_pred, axis=0)
    labels = np.concatenate(all_lab, axis=0)

    print(f">>> got {len(confidences)} examples")

    plot_calibration_diagram(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        num_bins=15,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
