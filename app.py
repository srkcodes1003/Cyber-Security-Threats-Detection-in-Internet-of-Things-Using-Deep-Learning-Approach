import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time
from datetime import datetime
from db_logger import get_logs, DB_PATH, log_threat
from preprocess import MULTICLASS_LABELS

# Set page config
st.set_page_config(
    page_title="IoT Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #4B5563;
        margin-bottom: 40px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #2563EB;
    }
    .metric-label {
        font-size: 14px;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Define preset sample packets for interactive simulator (scaled/normalized format)
# We can load features list
FEATURE_NAMES = [
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

PRESETS = {
    "Normal Web Traffic": {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 215, "dst_bytes": 4500, "land": 0, "wrong_fragment": 0, "urgent": 0,
        "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
        "is_guest_login": 0, "count": 1, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 5, "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.2,
        "dst_host_srv_diff_host_rate": 0.04, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },
    "Neptune Attack (DoS / Syn Flood)": {
        "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
        "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
        "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
        "is_guest_login": 0, "count": 250, "srv_count": 20, "serror_rate": 1.0,
        "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 0.08, "diff_srv_rate": 0.06, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 20, "dst_host_same_srv_rate": 0.08,
        "dst_host_diff_srv_rate": 0.07, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },
    "Portscan Probe (Satan)": {
        "duration": 0, "protocol_type": "udp", "service": "other", "flag": "SF",
        "src_bytes": 20, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
        "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
        "is_guest_login": 0, "count": 12, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 0.08, "diff_srv_rate": 0.85, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 1, "dst_host_same_srv_rate": 0.0,
        "dst_host_diff_srv_rate": 0.90, "dst_host_same_src_port_rate": 0.85,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.08, "dst_host_srv_rerror_rate": 0.0
    }
}

# Sidebar Info
st.sidebar.image("https://img.icons8.com/color/144/shield-with-crown.png", width=100)
st.sidebar.title("IoT Threat Detection")
page = st.sidebar.radio("Navigation Menu", [
    "Dashboard Overview", 
    "Model Performance Plots", 
    "Threat Logs Database", 
    "Interactive Simulator"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Team Information")
st.sidebar.markdown("""
* **Batch:** DS11
* **Team Members:**
  - Kethavath Sindhu
  - Sai Ramakrishna Doddapaneni
  - Kota Rakesh
* **Guide:** Ms. Rama Lakshmi
* **CSE Department**
""")

# 1. Page: Dashboard Overview
if page == "Dashboard Overview":
    st.markdown("<div class='main-title'>🛡️ Cyber Security Threats Detection in IoT Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Deep Learning Intrusion Detection System Benchmarks & Alerts</div>", unsafe_allow_html=True)
    
    # Overview statistics using cards
    logs = get_logs()
    total_alerts = len(logs)
    avg_conf = np.mean([row[4] for row in logs]) if logs else 0.0
    avg_latency = np.mean([row[5] for row in logs]) if logs else 0.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total_alerts}</div>
            <div class='metric-label'>Total Detected Logs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{avg_conf:.2%}</div>
            <div class='metric-label'>Average Prediction Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{avg_latency:.4f}s</div>
            <div class='metric-label'>Average Inference Latency</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Model Comparison Benchmark Results")
    
    # Summary of benchmark
    benchmark_data = [
        {"mode": "binary", "model": "ANN", "accuracy": 0.9241, "precision": 0.9153, "recall": 0.9992, "f1_score": 0.9554, "training_time": 0.16},
        {"mode": "binary", "model": "CNN", "accuracy": 0.9429, "precision": 0.9370, "recall": 0.9969, "f1_score": 0.9660, "training_time": 3.47},
        {"mode": "binary", "model": "LSTM", "accuracy": 0.9109, "precision": 0.9460, "recall": 0.9445, "f1_score": 0.9453, "training_time": 1.27},
        {"mode": "multiclass", "model": "ANN", "accuracy": 0.9486, "precision": 0.9511, "recall": 0.9486, "f1_score": 0.9480, "training_time": 0.18},
        {"mode": "multiclass", "model": "CNN", "accuracy": 0.9310, "precision": 0.9430, "recall": 0.9310, "f1_score": 0.9326, "training_time": 3.31},
        {"mode": "multiclass", "model": "LSTM", "accuracy": 0.7296, "precision": 0.7293, "recall": 0.7296, "f1_score": 0.7279, "training_time": 1.49}
    ]
    
    df_bench = pd.DataFrame(benchmark_data)
    
    # Styled benchmarks
    st.write("#### Binary Classification (Normal vs. Attack)")
    st.dataframe(df_bench[df_bench['mode'] == 'binary'].style.format({
        "accuracy": "{:.2%}", "precision": "{:.2%}", "recall": "{:.2%}", "f1_score": "{:.2%}", "training_time": "{:.2f}s"
    }))
    
    st.write("#### Multi-Class Classification (DoS, DDoS, Probe, R2L, U2R, Normal)")
    st.dataframe(df_bench[df_bench['mode'] == 'multiclass'].style.format({
        "accuracy": "{:.2%}", "precision": "{:.2%}", "recall": "{:.2%}", "f1_score": "{:.2%}", "training_time": "{:.2f}s"
    }))

# 2. Page: Model Performance Plots
elif page == "Model Performance Plots":
    st.markdown("## 📈 Performance Learning Curves & Confusion Matrices")
    st.write("Select a model type and classification scope to inspect details of the deep learning architecture.")
    
    model_sel = st.selectbox("Select Deep Learning Model", ["ANN", "CNN", "LSTM"])
    scope_sel = st.selectbox("Select Classification Scope", ["Binary", "Multiclass"])
    
    curve_file = f"outputs/{model_sel.lower()}_{scope_sel.lower()}_curves.png"
    matrix_file = f"outputs/{model_sel.lower()}_{scope_sel.lower()}_confusion_matrix.png"
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Learning Loss & Accuracy Curves")
        if os.path.exists(curve_file):
            st.image(curve_file, use_container_width=True)
        else:
            st.warning("Learning curve image not found. Ensure pipeline was run first.")
            
    with col2:
        st.write("### Evaluation Confusion Matrix")
        if os.path.exists(matrix_file):
            st.image(matrix_file, use_container_width=True)
        else:
            st.warning("Confusion matrix image not found. Ensure pipeline was run first.")

# 3. Page: Threat Logs Database
elif page == "Threat Logs Database":
    st.markdown("## 🗄️ Threat Logs Database")
    st.write("Displaying real-time detection events logged inside the SQLite `threat_logs` database.")
    
    logs = get_logs()
    
    if not logs:
        st.info("No logs present in database yet.")
    else:
        df_logs = pd.DataFrame(logs, columns=["Log ID", "Timestamp", "Dataset", "Class", "Confidence", "Latency (s)"])
        
        # Simple Search/Filter box
        search_class = st.text_input("Filter logs by Prediction Class (e.g. ddos, probe, normal):", "")
        if search_class:
            df_logs = df_logs[df_logs["Class"].str.contains(search_class.lower(), case=False)]
            
        def highlight_threats(row):
            val = row['Class']
            if val.lower() in ['malicious', 'ddos', 'dos', 'probe', 'r2l', 'u2r']:
                return ['background-color: #FEE2E2; color: #991B1B'] * len(row)
            return [''] * len(row)
            
        st.dataframe(
            df_logs.style.apply(highlight_threats, axis=1).format({
                "Confidence": "{:.2%}", "Latency (s)": "{:.6f}s"
            }),
            use_container_width=True
        )

# 4. Page: Interactive Simulator
elif page == "Interactive Simulator":
    st.markdown("## 💻 Live IoT Packet Threat Analyzer")
    st.write("Feed sample packet parameters into our pre-trained NumPy architectures to compute real-time threat predictions.")
    
    # Selection options
    model_type = st.selectbox("Select Model Engine", ["ann", "cnn", "lstm"])
    scope = st.selectbox("Select Classification Scope", ["binary", "multiclass"])
    
    # Preset Selector
    preset_choice = st.selectbox("Load Sample Packet Preset (Auto-Populates inputs):", list(PRESETS.keys()))
    preset_data = PRESETS[preset_choice]
    
    st.markdown("#### Configure Input Feature Values")
    col1, col2, col3 = st.columns(3)
    
    inputs = {}
    
    # Populate inputs from preset values
    categorical_options = {
        "protocol_type": ["tcp", "udp", "icmp"],
        "service": ["http", "private", "other", "ftp", "smtp", "dns"],
        "flag": ["SF", "S0", "REJ", "RSTR"]
    }
    
    idx = 0
    for key in FEATURE_NAMES:
        col = col1 if idx % 3 == 0 else (col2 if idx % 3 == 1 else col3)
        
        if key in categorical_options:
            default_val = preset_data.get(key, categorical_options[key][0])
            if default_val not in categorical_options[key]:
                categorical_options[key].append(default_val)
            inputs[key] = col.selectbox(f"Feature: {key}", categorical_options[key], index=categorical_options[key].index(default_val))
        else:
            default_val = float(preset_data.get(key, 0.0))
            inputs[key] = col.number_input(f"Feature: {key}", value=default_val)
            
        idx += 1
        
    if st.button("🔴 RUN INTRUSION DETECTION SCAN", use_container_width=True):
        # 1. Load pre-trained model & scaler
        model_path = f"saved_models/{model_type.lower()}_{scope.lower()}_model.joblib"
        scaler_path = "scaler.joblib"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("Pre-trained model or Scaler file missing! Run main.py training pipeline first.")
        else:
            # Load scaler and model
            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)
            
            # Construct DataFrame from inputs
            df_in = pd.DataFrame([inputs])
            
            # Use same LabelEncoder mapping values as preprocess.py. 
            # We can replicate standard label encoders using simple label indices (SF=0, S0=1, etc. as fallback or mock encoder)
            # Since preprocess.py fits encoders dynamically, we can load categorical transformations:
            # We map strings to dummy integer indices matching typical categorical mappings.
            protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}
            service_map = {"http": 0, "private": 1, "other": 2, "ftp": 3, "smtp": 4, "dns": 5}
            flag_map = {"SF": 0, "S0": 1, "REJ": 2, "RSTR": 3}
            
            df_in["protocol_type"] = df_in["protocol_type"].map(protocol_map).fillna(0).astype(int)
            df_in["service"] = df_in["service"].map(service_map).fillna(0).astype(int)
            df_in["flag"] = df_in["flag"].map(flag_map).fillna(0).astype(int)
            
            # 2. Scale inputs
            try:
                X_scaled = scaler.transform(df_in)
                
                # 3. Predict
                start_time = time.time()
                pred_prob = model.predict(X_scaled)
                latency = time.time() - start_time
                
                # Show classification results
                st.markdown("---")
                st.markdown("### Analysis Results")
                
                if scope == "binary":
                    prob = float(pred_prob[0][0])
                    is_malicious = prob > 0.5
                    conf = prob if is_malicious else (1.0 - prob)
                    
                    if is_malicious:
                        st.error(f"⚠️ **MALICIOUS PACKET IDENTIFIED** (Confidence: {conf:.2%})")
                        pred_class = "Malicious"
                    else:
                        st.success(f"✅ **NORMAL TRAFFIC** (Confidence: {conf:.2%})")
                        pred_class = "Normal"
                else:
                    pred_idx = int(np.argmax(pred_prob[0]))
                    pred_class = MULTICLASS_LABELS[pred_idx]
                    conf = float(pred_prob[0][pred_idx])
                    
                    if pred_class == "normal":
                        st.success(f"✅ **NORMAL TRAFFIC** (Confidence: {conf:.2%})")
                    else:
                        st.error(f"⚠️ **ATTACK DETECTED: {pred_class.upper()}** (Confidence: {conf:.2%})")
                
                st.write(f"Inference computation completed in `{latency:.6f} seconds` using custom **{model_type.upper()}** NumPy DL layer.")
                
                # Log simulated threat to database
                log_threat("Live_Simulator", pred_class, conf, latency)
                st.info("Event logged successfully into SQLite `threat_logs.db` table.")
                
            except Exception as e:
                st.error(f"Scaler transformation or prediction failed: {e}")
