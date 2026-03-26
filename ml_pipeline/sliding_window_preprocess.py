import pandas as pd
import numpy as np

WINDOW_SIZE = 100
STRIDE = 25

print("Loading dataset...")

df = pd.read_csv("../datasets/Monday-WorkingHours.pcap_ISCX.csv")
df.columns = df.columns.str.strip()

print("Columns detected:")
print(df.columns)
metadata_df = df[[
    "Source IP",
    "Destination IP",
    "Timestamp"
]].copy()

drop_cols = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

existing_cols = [c for c in drop_cols if c in df.columns]

df = df.drop(columns=existing_cols)

df = df.apply(pd.to_numeric, errors='coerce')
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

    mean_vals = np.mean(window, axis=0)
    std_vals = np.std(window, axis=0)
    max_vals = np.max(window, axis=0)
    min_vals = np.min(window, axis=0)

    feature_vector = np.concatenate([mean_vals, std_vals, max_vals, min_vals])

    windows.append(feature_vector)

    meta_slice = metadata_df.iloc[start:start + WINDOW_SIZE]

    metadata = {
        "start_idx": int(start),
        "end_idx": int(start + WINDOW_SIZE),
        "src_ip": str(meta_slice["Source IP"].iloc[0]),
        "dst_ip": str(meta_slice["Destination IP"].iloc[0]),
        "timestamp": str(meta_slice["Timestamp"].iloc[0])
    }

    window_metadata.append(metadata)

windows = np.array(windows)

print("Windows created:", windows.shape)

np.save("window_features.npy", windows)
np.save("window_metadata.npy", window_metadata)  

print("Saved window_features.npy")
print("Saved window_metadata.npy")