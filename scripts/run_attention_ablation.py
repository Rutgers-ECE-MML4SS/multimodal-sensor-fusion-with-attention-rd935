import json
import subprocess
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt

# let Windows chill about OpenMP (optional)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

HEADS = [1, 4, 8]

ROOT = Path(__file__).parent.resolve()
RUNS_DIR = ROOT / "runs"
CONFIG_PATH = ROOT / "config" / "base.yaml"   
EXPERIMENTS_DIR = ROOT / "experiments"
ANALYSIS_DIR = ROOT / "analysis"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

COMBINED_JSON = EXPERIMENTS_DIR / "attn_heads_eval.json"
PLOT_PATH = ANALYSIS_DIR / "attn_heads_accuracy.png"


def find_latest_ckpt(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def main():
    combined = {}

    acc_x = []
    acc_y = []

    for h in HEADS:
        run_name = f"a2_hybrid_heads_{h}"
        run_dir = RUNS_DIR / run_name
        print(f"\n=== evaluating {run_name} ===")

        ckpt = find_latest_ckpt(run_dir)
        if ckpt is None:
            print(f"[!] No checkpoint found in {run_dir}/checkpoints, skipping.")
            combined[str(h)] = {"error": "no checkpoint found"}
            continue

        print(f"→ using checkpoint: {ckpt}")

        # call your existing eval.py
        cmd = [
            sys.executable,
            "src/eval.py",
            "--checkpoint", ckpt.as_posix(),
            "--config", CONFIG_PATH.as_posix(),
            "--output_dir", EXPERIMENTS_DIR.as_posix(),
        ]
        print(">>>", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[!] eval failed for heads={h}")
            combined[str(h)] = {"error": "eval failed"}
            continue

        # eval.py should have written experiments/evaluation_results.json
        result_file = EXPERIMENTS_DIR / "evaluation_results.json"
        if not result_file.exists():
            print(f"[!] expected {result_file} not found after eval for heads={h}")
            combined[str(h)] = {"error": "no evaluation_results.json"}
            continue

        with result_file.open("r") as f:
            data = json.load(f)

        # store under this head
        combined[str(h)] = data

        # collect for plotting
        acc = data.get("test_accuracy", None)
        if acc is not None:
            acc_x.append(h)
            acc_y.append(acc)

        # also save a per-head copy so they don't overwrite each other
        per_head_file = EXPERIMENTS_DIR / f"evaluation_results_heads_{h}.json"
        with per_head_file.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ saved per-head eval → {per_head_file}")

    # write combined json
    with COMBINED_JSON.open("w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n✅ wrote combined eval → {COMBINED_JSON}")

    # make accuracy plot if we have data
    if acc_x:
        # sort by heads so the line is nice
        acc_x, acc_y = zip(*sorted(zip(acc_x, acc_y), key=lambda x: x[0]))
        plt.figure(figsize=(5, 4))
        plt.plot(acc_x, acc_y, marker="o")
        plt.xticks(acc_x)
        plt.xlabel("Number of attention heads")
        plt.ylabel("Test accuracy")
        plt.title("Hybrid attention heads vs test accuracy")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=200)
        print(f"✅ saved plot → {PLOT_PATH}")
    else:
        print("[!] no accuracies found, skipping plot")


if __name__ == "__main__":
    main()
