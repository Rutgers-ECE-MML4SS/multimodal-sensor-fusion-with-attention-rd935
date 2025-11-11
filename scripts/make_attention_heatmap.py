import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
import numpy as np

# make src importable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from train import MultimodalFusionModule
from data import create_dataloaders
from analysis import plot_attention_weights  


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cfg_path = ROOT / "config" / "base.yaml"
    ckpt_path = ROOT / "runs" / "a2_hybrid_pamap2" / "checkpoints" / "last.ckpt"
    out_path = ROOT / "analysis" / "attention_viz.png"

    cfg = OmegaConf.load(str(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f">>> loading model from {ckpt_path}")
    model = MultimodalFusionModule.load_from_checkpoint(str(ckpt_path), config=cfg)
    model.to(device)
    model.eval()

    # build dataloaders same as train.py
    ds = cfg.dataset
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=ds.name,
        data_dir=ds.data_dir,
        modalities=ds.modalities,
        batch_size=ds.batch_size,
        num_workers=ds.num_workers,
        modality_dropout=cfg.training.augmentation.modality_dropout,
    )

    # get one batch
    loader = test_loader or val_loader
    features, labels, mask = next(iter(loader))

    # to device
    features = move_to_device(features, device)
    mask = move_to_device(mask, device)

    # 1) encode each modality exactly like your Lightning module does
    encoded_features = {}
    for modality, encoder in model.encoders.items():
        if modality in features:
            encoded_features[modality] = encoder(features[modality])

    # 2) project to common dim using the fusion model's projection layers
    fusion = model.fusion_model  
    modality_names = fusion.modality_names
    proj_feats = {
        m: fusion.proj[m](encoded_features[m]) for m in modality_names
    }  

    # 3) pick the first example in batch and build a cosine-sim matrix
    M = len(modality_names)
    sim_matrix = np.zeros((M, M), dtype=np.float32)

    # get cpu vectors for the first item
    proj_feats_1 = {m: proj_feats[m][0].detach().cpu() for m in modality_names}

    for i, mi in enumerate(modality_names):
        vi = proj_feats_1[mi]
        for j, mj in enumerate(modality_names):
            vj = proj_feats_1[mj]
            num = float((vi * vj).sum())
            den = float(vi.norm() * vj.norm()) + 1e-8
            sim = num / den
            sim_matrix[i, j] = sim

    # 4) normalize to 0..1 for pretty plotting
    sim_min, sim_max = sim_matrix.min(), sim_matrix.max()
    if sim_max > sim_min:
        sim_matrix = (sim_matrix - sim_min) / (sim_max - sim_min)

    # 5) plot with your helper
    plot_attention_weights(
        attention_weights=sim_matrix,
        modality_names=modality_names,
        save_path=str(out_path),
    )


if __name__ == "__main__":
    main()
