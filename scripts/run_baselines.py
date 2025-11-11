import json
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# (config_name_for_training, run_folder_name, label_in_final_json, eval_config_path)
RUNS = [
    ("imu_hand", "a2_single_modality_pamap2", "single_modality_imu_hand", CONFIG_DIR / "single_modality.yaml"),
    ("early_all", "a2_early_pamap2", "naive_concat",                CONFIG_DIR / "base.yaml"),
    ("hybrid_all", "a2_hybrid_pamap2", "hybrid_attention",          CONFIG_DIR / "base.yaml"),
]


def run_cmd(cmd_list):
    """Run a command and print helpful info."""
    print(f"\n>>> Running: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"[!] Command failed: {' '.join(cmd_list)}")
    return result.returncode


def get_latest_ckpt(run_folder: Path):
    ckpt_dir = run_folder / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime, reverse=True)
    return ckpts[0] if ckpts else None


def eval_output_path(label: str) -> Path:
    return EXPERIMENTS_DIR / f"{label}.json"


def main():
    aggregated = {"results": {}}

    for train_cfg, run_folder_name, label, eval_cfg in RUNS:
        run_path = RUNS_DIR / run_folder_name

        # Train if needed
        if not run_path.exists():
            ret = run_cmd([
                "python", "src/train.py",
                "--config-name", train_cfg
            ])
            if ret != 0:
                print(f"[!] Training failed for {train_cfg}, skipping.")
                continue
        else:
            print(f"[✓] Skipping training for {train_cfg} (runs/{run_folder_name} exists).")

        # Get checkpoint
        ckpt = get_latest_ckpt(run_path)
        if ckpt is None:
            print(f"[!] No checkpoint found in {run_path}/checkpoints")
            continue
        print(f"[✓] Using checkpoint: {ckpt}")

        # Evaluate using eval.py’s CLI
        ret = run_cmd([
            "python", "src/eval.py",
            "--checkpoint", ckpt.as_posix(),
            "--config", eval_cfg.as_posix(),
            "--output_dir", EXPERIMENTS_DIR.as_posix()
        ])
        if ret != 0:
            print(f"[!] Eval failed for {train_cfg}, skipping aggregation.")
            continue

        # Read from evaluation_results.json
        result_file = EXPERIMENTS_DIR / "evaluation_results.json"
        if not result_file.exists():
            print(f"[!] Expected eval output not found: {result_file}")
            continue

        with result_file.open("r") as f:
            data = json.load(f)

        aggregated["results"][label] = {
            "dataset": data.get("dataset"),
            "fusion_type": data.get("fusion_type"),
            "accuracy": data.get("test_accuracy"),
            "f1_macro": data.get("test_f1_macro"),
            "loss": data.get("test_loss"),
            "ece": data.get("ece"),
        }

        per_run_file = eval_output_path(label)
        with per_run_file.open("w") as f:
            json.dump(data, f, indent=2)

    # Write final combined JSON
    combined_path = EXPERIMENTS_DIR / "baseline_comparison.json"
    with combined_path.open("w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n✅ Wrote combined baseline results → {combined_path.resolve()}")


if __name__ == "__main__":
    main()
