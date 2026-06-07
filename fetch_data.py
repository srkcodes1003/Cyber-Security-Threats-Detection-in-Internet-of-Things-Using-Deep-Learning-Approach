import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
import os

def fetch_and_save_data():
    csv_filename = "kddcup99_sample.csv"
    if os.path.exists(csv_filename):
        print(f"[FETCH DATA] Dataset already exists at {csv_filename}. Skipping download.")
        return

    print("[FETCH DATA] Fetching KDDCup99 dataset using scikit-learn...")
    # Fetch the 10% subset of KDDCup99 dataset
    data = fetch_kddcup99(percent10=True, as_frame=False)
    
    # Feature names for KDDCup99
    feature_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    
    X = data.data
    y = data.target
    
    print(f"[FETCH DATA] Loaded {X.shape[0]} records. Creating DataFrame...")
    
    # Construct DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    
    # Clean up byte objects to regular strings
    print("[FETCH DATA] Decoding byte objects to strings...")
    for col in df.columns:
        # Check if the column contains byte strings and decode them
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda val: val.decode("utf-8") if isinstance(val, bytes) else str(val))
            
    # KDDCup99 has some columns that are objects but should be numeric (like duration, src_bytes, dst_bytes, etc.)
    # We will identify the numeric column types and convert them
    numeric_cols = [
        "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    print("[FETCH DATA] Sampling subset for fast end-to-end local execution...")
    # To prevent slow training, we take a balanced sample of 20,000 rows
    # We group by label and sample up to 1000 per class to keep it representative, then pad with normal and neptune
    sampled_dfs = []
    for label, group in df.groupby("label"):
        sample_size = min(len(group), 1500)
        sampled_dfs.append(group.sample(n=sample_size, random_state=42))
        
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the final sample
    sampled_df = sampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"[FETCH DATA] Saving sample of {sampled_df.shape[0]} rows to {csv_filename}...")
    sampled_df.to_csv(csv_filename, index=False)
    print("[FETCH DATA] Download and saving complete!")

if __name__ == "__main__":
    fetch_and_save_data()
