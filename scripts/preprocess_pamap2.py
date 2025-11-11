from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
RAW_ROOT = Path(r"C:\Users\ritwi\PAMAP2_Dataset")
PROTOCOL_DIR = RAW_ROOT / "Protocol"
OPTIONAL_DIR = RAW_ROOT / "Optional"

OUT_ROOT = Path(r"C:\Users\ritwi\pamap2_data")   # output folder

SEQ_LEN = 128
STRIDE = 64

# window-level split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  


def load_subject_file(path: Path) -> np.ndarray:
    # PAMAP2 is space-separated, missing values often -1
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return df.to_numpy()


def load_all_subjects():
    """Load EVERY subject found in Protocol/Optional, stack, and sort by time."""
    all_rows = []
    for folder in (PROTOCOL_DIR, OPTIONAL_DIR):
        for f in folder.glob("subject*.dat"):
            print(f"✓ loading {f}")
            all_rows.append(load_subject_file(f))

    if not all_rows:
        raise FileNotFoundError("No PAMAP2 .dat files found in Protocol/Optional")

    data = np.vstack(all_rows)
    # sort by timestamp so we don't make windows across jumps
    data = data[data[:, 0].argsort()]
    return data


def window_data(arr, window_size=128, stride=64):
    T, F = arr.shape
    windows = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(arr[start:end])
    if not windows:
        return np.empty((0, window_size, F), dtype=arr.dtype)
    return np.stack(windows)


def majority_label_per_window(label_windows: np.ndarray) -> np.ndarray:
    labels = []
    for w in label_windows:
        vals, counts = np.unique(w, return_counts=True)
        labels.append(vals[counts.argmax()])
    return np.array(labels, dtype=np.int64)


def normalize_train_and_apply(train_arrs, val_arrs, test_arrs):
    """
    train_arrs: dict(name -> np.array (N, T, F))
    returns the same dicts but normalized per feature using train stats
    """
    stats = {}
    for name, arr in train_arrs.items():
        flat = arr.reshape(-1, arr.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-8
        stats[name] = (mean, std)

    def apply(d, stats):
        out = {}
        for name, arr in d.items():
            mean, std = stats[name]
            out[name] = (arr - mean) / std
        return out

    return apply(train_arrs, stats), apply(val_arrs, stats), apply(test_arrs, stats)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    data = load_all_subjects()
    print(f"Total raw rows: {data.shape}")

    # --------------------------------------------------------
    # column splits (matching your original script)
    # 0: time
    # 1: activity
    # 2: heart rate
    # 3:20  -> hand (17)
    # 20:37 -> chest (17)
    # 37:54 -> ankle (17)
    # --------------------------------------------------------
    activity_raw = data[:, 1].astype("int64")  # <-- keep as-is, including 0

    heart_rate = np.nan_to_num(data[:, 2].astype("float32"), nan=0.0).reshape(-1, 1)
    imu_hand = np.nan_to_num(data[:, 3:20].astype("float32"), nan=0.0)
    imu_chest = np.nan_to_num(data[:, 20:37].astype("float32"), nan=0.0)
    imu_ankle = np.nan_to_num(data[:, 37:54].astype("float32"), nan=0.0)

    # window everything
    imu_hand_w = window_data(imu_hand, SEQ_LEN, STRIDE)
    imu_chest_w = window_data(imu_chest, SEQ_LEN, STRIDE)
    imu_ankle_w = window_data(imu_ankle, SEQ_LEN, STRIDE)
    heart_rate_w = window_data(heart_rate, SEQ_LEN, STRIDE)

    label_windows = window_data(activity_raw.reshape(-1, 1), SEQ_LEN, STRIDE).squeeze(-1)
    labels_w = majority_label_per_window(label_windows)

    # don't drop label 0; instead, remap all labels that exist to 0..(K-1)
    unique_labels = np.unique(labels_w)
    print("Raw labels present:", unique_labels)

    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels_w = np.array([label_map[l] for l in labels_w], dtype=np.int64)

    print(f"Total windows (no dropping): {labels_w.shape[0]}")
    print(f"Label distribution after remap: {np.bincount(labels_w)}")
    num_classes = len(unique_labels)
    print("num_classes =", num_classes)

    # shuffle
    idx = np.arange(labels_w.shape[0])
    np.random.shuffle(idx)

    imu_hand_w = imu_hand_w[idx]
    imu_chest_w = imu_chest_w[idx]
    imu_ankle_w = imu_ankle_w[idx]
    heart_rate_w = heart_rate_w[idx]
    labels_w = labels_w[idx]

    # split by ratio
    N = labels_w.shape[0]
    n_train = int(N * TRAIN_RATIO)
    n_val = int(N * VAL_RATIO)
    n_test = N - n_train - n_val

    train = {
        "imu_hand": imu_hand_w[:n_train],
        "imu_chest": imu_chest_w[:n_train],
        "imu_ankle": imu_ankle_w[:n_train],
        "heart_rate": heart_rate_w[:n_train],
        "labels": labels_w[:n_train],
    }
    val = {
        "imu_hand": imu_hand_w[n_train:n_train + n_val],
        "imu_chest": imu_chest_w[n_train:n_train + n_val],
        "imu_ankle": imu_ankle_w[n_train:n_train + n_val],
        "heart_rate": heart_rate_w[n_train:n_train + n_val],
        "labels": labels_w[n_train:n_train + n_val],
    }
    test = {
        "imu_hand": imu_hand_w[n_train + n_val:],
        "imu_chest": imu_chest_w[n_train + n_val:],
        "imu_ankle": imu_ankle_w[n_train + n_val:],
        "heart_rate": heart_rate_w[n_train + n_val:],
        "labels": labels_w[n_train + n_val:],
    }

    # normalize using train stats (per modality)
    train_norm, val_norm, test_norm = normalize_train_and_apply(
        {k: v for k, v in train.items() if k != "labels"},
        {k: v for k, v in val.items() if k != "labels"},
        {k: v for k, v in test.items() if k != "labels"},
    )

    # put labels back
    train_norm["labels"] = train["labels"]
    val_norm["labels"] = val["labels"]
    test_norm["labels"] = test["labels"]

    # save
    for split_name, split_data in [("train", train_norm), ("val", val_norm), ("test", test_norm)]:
        split_dir = OUT_ROOT / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "imu_hand.npy", split_data["imu_hand"])
        np.save(split_dir / "imu_chest.npy", split_data["imu_chest"])
        np.save(split_dir / "imu_ankle.npy", split_data["imu_ankle"])
        np.save(split_dir / "heart_rate.npy", split_data["heart_rate"])
        np.save(split_dir / "labels.npy", split_data["labels"])
        print(split_name, {k: v.shape for k, v in split_data.items()})

    # also save the label map so the model knows num_classes
    np.save(OUT_ROOT / "label_map.npy", np.array([[k, v] for k, v in label_map.items()], dtype=object))
    print("Saved label map at", OUT_ROOT / "label_map.npy")


if __name__ == "__main__":
    main()
