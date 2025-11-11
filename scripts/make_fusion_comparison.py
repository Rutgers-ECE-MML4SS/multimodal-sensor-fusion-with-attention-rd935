import json
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early", required=True, help="path to early evaluation_results.json")
    parser.add_argument("--late", required=True, help="path to late evaluation_results.json")
    parser.add_argument("--hybrid", required=True, help="path to hybrid evaluation_results.json")
    parser.add_argument("--out", default="fusion_comparison.json", help="output json path")
    # if you want to override modalities from CLI:
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["imu_hand", "imu_chest", "imu_ankle", "heart_rate"],
        help="list of modalities used in the runs",
    )
    args = parser.parse_args()

    def load(path):
        with open(path, "r") as f:
            return json.load(f)

    early = load(args.early)
    late = load(args.late)
    hybrid = load(args.hybrid)

    # take dataset name from any one of them
    dataset = early.get("dataset", "unknown")

    out = {
        "dataset": dataset,
        "modalities": args.modalities,
        "results": {
            "early_fusion": {
                "accuracy": early["test_accuracy"],
                "f1_macro": early["test_f1_macro"],
                "ece": early["ece"],
            },
            "late_fusion": {
                "accuracy": late["test_accuracy"],
                "f1_macro": late["test_f1_macro"],
                "ece": late["ece"],
            },
            "hybrid_fusion": {
                "accuracy": hybrid["test_accuracy"],
                "f1_macro": hybrid["test_f1_macro"],
                "ece": hybrid["ece"],
            },
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
