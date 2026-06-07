import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

# Define target category mappings
ATTACK_MAP = {
    'normal': 'normal',
    
    # DDoS (high volume distributed flood attacks)
    'neptune': 'ddos',
    'smurf': 'ddos',
    'apache2': 'ddos',
    'udpstorm': 'ddos',
    
    # DoS (denial of service, single-source or low-rate crashers)
    'back': 'dos',
    'land': 'dos',
    'pod': 'dos',
    'teardrop': 'dos',
    'mailbomb': 'dos',
    'processtable': 'dos',
    'worm': 'dos',
    
    # Probe (port scans, network mapping)
    'ipsweep': 'probe',
    'nmap': 'probe',
    'portsweep': 'probe',
    'satan': 'probe',
    'mscan': 'probe',
    'saint': 'probe',
    
    # R2L (remote to local access attempts)
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'multihop': 'r2l',
    'phf': 'r2l',
    'spy': 'r2l',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
    'sendmail': 'r2l',
    'named': 'r2l',
    'xlock': 'r2l',
    'xsnoop': 'r2l',
    'snmpgetattack': 'r2l',
    'snmpguess': 'r2l',
    'httptunnel': 'r2l',
    
    # U2R (user to root privilege escalation)
    'buffer_overflow': 'u2r',
    'loadmodule': 'u2r',
    'perl': 'u2r',
    'rootkit': 'u2r',
    'sqlattack': 'u2r',
    'xterm': 'u2r',
    'ps': 'u2r'
}

MULTICLASS_LABELS = ['normal', 'dos', 'ddos', 'probe', 'r2l', 'u2r']

def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV dataset using Pandas."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    print(f"[PREPROCESS] Loading dataset from {filepath}...")
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by dropping NaNs, infs, and duplicate records."""
    before_rows = df.shape[0]
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop NaNs
    df = df.dropna()
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    after_rows = df.shape[0]
    print(f"[PREPROCESS] Cleaned data: rows reduced from {before_rows} to {after_rows} (dropped {before_rows - after_rows} duplicates/NaNs/infs).")
    return df

def preprocess_pipeline(filepath: str, mode: str = "binary", test_size: float = 0.2, random_state: int = 42):
    """
    Ingests and preprocesses the CSV data:
    1. Loads dataset.
    2. Cleans NaNs/infs/duplicates.
    3. Maps label column to binary (0/1) or multi-class (0-5) categories.
    4. Label encodes categorical columns.
    5. Normalizes numerical features with MinMaxScaler.
    6. Performs stratified train/test split.
    """
    assert mode in ["binary", "multiclass"], "Mode must be 'binary' or 'multiclass'"
    
    df = load_data(filepath)
    df = clean_data(df)
    
    # Separate features and target
    if "label" not in df.columns:
        # Fallback if label is named target
        label_col = [c for c in df.columns if c.lower() in ["label", "target", "class"]][0]
    else:
        label_col = "label"
        
    X = df.drop(columns=[label_col])
    y_raw = df[label_col].astype(str).str.strip().str.lower()
    
    # Apply label mappings
    if mode == "binary":
        # Binary: normal is 0, any attack is 1
        y = y_raw.apply(lambda label: 0 if "normal" in label else 1).values
        num_classes = 2
    else:
        # Multiclass: normal, dos, ddos, probe, r2l, u2r mapped to 0-5
        def map_multiclass(label):
            # Clean label ending dots
            label_clean = label.rstrip('.')
            mapped = ATTACK_MAP.get(label_clean, None)
            if mapped is None:
                # heuristic fallback
                if "normal" in label_clean:
                    return "normal"
                elif "ddos" in label_clean:
                    return "ddos"
                elif "dos" in label_clean or "mailbomb" in label_clean or "processtable" in label_clean:
                    return "dos"
                elif "satan" in label_clean or "nmap" in label_clean or "scan" in label_clean or "reconnaissance" in label_clean or "portscan" in label_clean:
                    return "probe"
                elif "theft" in label_clean or "bot" in label_clean or "r2l" in label_clean:
                    return "r2l"
                elif "u2r" in label_clean or "buffer_overflow" in label_clean or "rootkit" in label_clean:
                    return "u2r"
                else:
                    # Default to 'dos' for unknown attack types
                    return "dos"
            return mapped

        y_mapped = y_raw.apply(map_multiclass)
        # Convert mapped labels to indices
        label_to_index = {name: idx for idx, name in enumerate(MULTICLASS_LABELS)}
        y = y_mapped.map(label_to_index).values
        num_classes = len(MULTICLASS_LABELS)

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[PREPROCESS] Categorical features to encode: {categorical_cols}")
    
    # Save encoders to load during production test pipelines
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Scale features using MinMaxScaler
    print("[PREPROCESS] Normalizing features using Min-Max Scaling...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, "scaler.joblib")
    
    # Stratified Train/Test split
    print(f"[PREPROCESS] Partitioning data (test_size={test_size}, stratify=True)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[PREPROCESS] Data processed successfully.")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Target classes: {num_classes} ({mode} mode)")
    
    return X_train, X_test, y_train, y_test, num_classes, X.columns.tolist()

if __name__ == "__main__":
    # Test script run
    try:
        X_train, X_test, y_train, y_test, num_classes, features = preprocess_pipeline(
            "kddcup99_sample.csv", mode="binary"
        )
        print("Success! Preprocessing script works correctly.")
    except Exception as e:
        print("Preprocessing test failed (this is expected if 'kddcup99_sample.csv' has not been created yet):", e)
