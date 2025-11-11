# A2: Multimodal Sensor Fusion with Attention  
**ECE 532 – Fall 2025**  
**Author:** Ritwika Das (rd935)  
**Dataset:** PAMAP2 Physical Activity Monitoring  

This repository implements and evaluates three fusion strategies — **Early**, **Late**, and **Hybrid (Attention-Based)** — for multimodal human activity recognition using the PAMAP2 dataset. It includes training, evaluation, uncertainty quantification, and analysis scripts to reproduce all results and plots shown in the final report.

---

## Setup Instructions

Clone the repository and create the Conda environment:
```bash
git clone <repo-url> repo-name
cd repo-name
conda env create -f environment.yml -n a2
conda activate a2
```

Download the **PAMAP2 Physical Activity Monitoring** dataset from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring) and update the path in `config/base.yaml`:
```yaml
dataset:
  name: pamap2
  data_dir: "C:/Users/<username>/pamap2_data"
```

Verify installation:
```bash
pytest -q
```
**Expected output:** `Setup complete, environment activated`

---

## Training

All three fusion strategies (early, late, hybrid) can be trained using a single command loop:
```bash
for fusion in early late hybrid; do
    python src/train.py model.fusion_type=$fusion
done
```

There is folder called script which includes a lot of the files which I used for training/creating the plots. There
is a file called run_all.ps1 (made for powershell) that includes the lines to train all 3 fusion, in addition other versions
of the fusions which we needed for the assignment (single modality, hybrid - num_heads = 1, 4, 8)

Each run uses `config/base.yaml` and saves checkpoints under:
```
runs/a2_<fusion>_pamap2/checkpoints/last.ckpt
```

**Expected runtime:** approximately 30–40 minutes on GPU (longer on CPU).  
**Expected output:** validation loss and accuracy printed to the console, and a saved model checkpoint.

To retrain a specific model (e.g., Hybrid):
```bash
python src/train.py model.fusion_type=hybrid
```

---

## Evaluation and Analysis

After training, run evaluation to generate results:
```bash
python src/eval.py --checkpoint runs/a2_early_pamap2/checkpoints/last.ckpt --output_dir experiments/early --device cuda
python src/eval.py --checkpoint runs/a2_late_pamap2/checkpoints/last.ckpt --output_dir experiments/late --device cuda
python src/eval.py --checkpoint runs/a2_hybrid_pamap2/checkpoints/last.ckpt --output_dir experiments/hybrid --device cuda
```

To create fusion_comparison.json
```bash
python scripts/make_fusion_comparison.py
```

To create the baselines_comparison.json
```bash
python scripts/run_baselines.py
```

Then generate analysis figures:

For fusion_comparison.png and missing_modality.png
```bash
python src/analysis.py --experiment_dir experiments/ --output_dir analysis/
```

For attention_viz.png
```bash
python scripts/make_attention_heatmap.py 
```

For calibration.png
```bash
python scripts/make_callibration.py
```

For attention_heads_acc.png
```bash
python scripts/run_attention_ablation.py
```

**Expected outputs:**
- JSONs in `experiments/`:  
  `evaluation_results.json` for early, late, and hybrid models  
- Plots in `analysis/`:  
  `fusion_comparison.png`, `missing_modality.png`, `attention_viz.png`, `calibration.png`

---

## Uncertainty Analysis

Calibration and uncertainty metrics (ECE, NLL, reliability diagram) can be computed via:
```bash
python src/uncertainty.py --preds experiments/hybrid/evaluation_results.json --output experiments/uncertainty.json
```

If uncertainty is integrated into evaluation:
```bash
python src/eval.py --checkpoint runs/a2_hybrid_pamap2/checkpoints/last.ckpt --output_dir experiments/hybrid --compute_uncertainty
```

**Expected output:** `experiments/uncertainty.json` with fields:
```json
{
  "dataset": "pamap2",
  "ece": 0.05,
  "nll": 0.31,
  "accuracy_per_bin": [...],
  "bins": [...]
}
```

---

## Results Summary

| Fusion Strategy | Accuracy | F1 (macro) | ECE  |
|-----------------|-----------|------------|------|
| Early Fusion    | 0.931     | 0.849      | 0.043 |
| Late Fusion     | **0.932** | **0.850**  | **0.043** |
| Hybrid Fusion   | 0.926     | 0.845      | 0.051 |

**Dataset used:** PAMAP2 (3 IMUs + heart rate).  
**Best model:** Late Fusion — achieved the highest accuracy and best calibration.  
**Training time:** ~30–40 minutes on local hardware.  
**Key finding:** All three fusion methods performed similarly (within 1%), with Late Fusion slightly best. Hybrid Fusion learned meaningful modality dependencies but showed limited accuracy gains due to preprocessing that made sensor features highly correlated. Late Fusion offers the best trade-off between simplicity, robustness, and calibration.

---

## File Manifest

**Code**
```
src/fusion.py          # Early, Late, Hybrid fusion modules
src/attention.py       # Cross-modal attention mechanism
src/encoders.py        # Modality-specific encoders
src/uncertainty.py     # Calibration and ECE analysis
src/train.py           # Training pipeline
src/eval.py            # Evaluation and JSON generation
src/analysis.py        # Plot generation
```

**Configs**
```
config/base.yaml
config/fusion_strategies.yaml
config/datasets.yaml
```

**Outputs**
```
experiments/early/evaluation_results.json
experiments/late/evaluation_results.json
experiments/hybrid/evaluation_results.json
experiments/uncertainty.json
analysis/fusion_comparison.png
analysis/missing_modality.png
analysis/attention_viz.png
analysis/calibration.png
report.pdf
```

**Checkpoints**
```
runs/a2_early_pamap2/checkpoints/last.ckpt
runs/a2_late_pamap2/checkpoints/last.ckpt
runs/a2_hybrid_pamap2/checkpoints/last.ckpt
```

---

## Summary

- **Dataset:** PAMAP2 Physical Activity Monitoring  
- **Best model:** Late Fusion (Acc = 0.932, F1 = 0.85, ECE = 0.043)  
- **Graceful degradation:** Confirmed under missing-sensor tests  
- **Calibration:** ECE ≈ 0.05 for all fusion types  
- **Insight:** Preprocessing homogenized sensor features, reducing the Hybrid model’s advantage  
- **Recommended deployment model:** Late Fusion — simplest, most robust, and well-calibrated  

---

✅ **Expected final terminal output**
```
Training complete. Checkpoints saved in runs/
Evaluation complete. Results in experiments/
Plots saved in analysis/
Setup complete, environment activated.
```

---

**End of README**  
*(ECE 532 – Multimodal Sensor Fusion with Attention, Fall 2025)*
