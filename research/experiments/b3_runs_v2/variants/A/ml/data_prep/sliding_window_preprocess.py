import numpy as np
import pandas as pd

WINDOW_SIZE = 100
STRIDE = 25

DEFAULT_INPUT_FILE = "../../../datasets/Monday-WorkingHours.pcap_ISCX.csv"
DEFAULT_OUTPUT_FEATURES = "../artifacts/window_features.npy"
DEFAULT_OUTPUT_METADATA = "../artifacts/window_metadata.npy"

DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"]


def build_windows(input_file: str = DEFAULT_INPUT_FILE) -> tuple[np.ndarray, list[dict]]:
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    metadata_df = df[["Source IP", "Destination IP", "Timestamp"]].copy()

    existing_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    metadata_df = metadata_df.loc[df.index]
    print("Clean dataset size:", len(df))

    data = df.values
    windows = []
    window_metadata = []

    print("Generating sliding windows...")
    for start in range(0, len(data) - WINDOW_SIZE, STRIDE):
        window = data[start:start + WINDOW_SIZE]
        feature_vector = np.concatenate([
            np.mean(window, axis=0), np.std(window, axis=0),
            np.max(window, axis=0), np.min(window, axis=0),
        ])
        windows.append(feature_vector)

        meta_slice = metadata_df.iloc[start:start + WINDOW_SIZE]
        window_metadata.append({
            "start_idx": int(start),
            "end_idx": int(start + WINDOW_SIZE),
            "src_ip": str(meta_slice["Source IP"].iloc[0]),
            "dst_ip": str(meta_slice["Destination IP"].iloc[0]),
            "timestamp": str(meta_slice["Timestamp"].iloc[0]),
        })

    return np.array(windows), window_metadata


if __name__ == "__main__":
    windows, metadata = build_windows()
    print("Windows created:", windows.shape)

    np.save(DEFAULT_OUTPUT_FEATURES, windows)
    np.save(DEFAULT_OUTPUT_METADATA, metadata)
    print(f"Saved {DEFAULT_OUTPUT_FEATURES}")
    print(f"Saved {DEFAULT_OUTPUT_METADATA}")
