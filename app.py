import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time
import random
from datetime import datetime
from db_logger import get_logs, log_threat
from preprocess import MULTICLASS_LABELS

# Set page config
st.set_page_config(
    page_title="IoT Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.03); opacity: 0.9; }
        100% { transform: scale(1); opacity: 1; }
    }
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
    
    /* Node styling */
    .node-card {
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .node-secure {
        background: linear-gradient(135deg, #10B981, #059669);
    }
    .node-alert {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        animation: pulse 1.5s infinite;
    }
    .node-inactive {
        background: linear-gradient(135deg, #9CA3AF, #6B7280);
    }
    
    /* Alerts styling */
    .secure-card {
        background-color: #D1FAE5;
        border-left: 8px solid #10B981;
        padding: 15px;
        border-radius: 8px;
        color: #065F46;
        margin-bottom: 15px;
    }
    .alert-card {
        background-color: #FEE2E2;
        border-left: 8px solid #DC2626;
        padding: 15px;
        border-radius: 8px;
        color: #7F1D1D;
        margin-bottom: 15px;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Datasets configurations
DATASET_LABELS = {
    "kddcup99": "KDDCup99 Dataset",
    "nslkdd": "NSL-KDD Dataset",
    "bot_iot": "Bot-IoT IoT-specific Dataset",
    "cic_ids": "CIC-IDS Modern Traffic Dataset"
}

# Feature schemas for simulator
DATASET_SCHEMAS = {
    "kddcup99": [
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
    ],
    "nslkdd": [
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
    ],
    "bot_iot": [
        "seq", "stddev", "N_IN_Conn_P_Src", "N_IN_Conn_P_Dst", "duration", "pkts", 
        "bytes", "srate", "drate", "proto", "state"
    ],
    "cic_ids": [
        "flow_duration", "total_fwd_packets", "total_bwd_packets", "fwd_pkt_len_max",
        "fwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_min", "flow_bytes_s", 
        "flow_pkts_s", "flow_iat_mean", "flow_iat_std", "active_mean", "idle_mean"
    ]
}

# Presets for simulator
SIMULATOR_PRESETS = {
    "kddcup99": {
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
        }
    },
    "nslkdd": {
        "Normal Web Traffic": {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
            "src_bytes": 220, "dst_bytes": 4600, "land": 0, "wrong_fragment": 0, "urgent": 0,
            "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
            "is_guest_login": 0, "count": 2, "srv_count": 2, "serror_rate": 0.0,
            "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
            "dst_host_count": 8, "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.15,
            "dst_host_srv_diff_host_rate": 0.02, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
        }
    },
    "bot_iot": {
        "Normal Traffic": {
            "seq": 1500, "stddev": 0.05, "N_IN_Conn_P_Src": 5, "N_IN_Conn_P_Dst": 5, 
            "duration": 1.5, "pkts": 8, "bytes": 600, "srate": 5.3, "drate": 4.1, 
            "proto": "tcp", "state": "CON"
        },
        "DDoS Flood Attack": {
            "seq": 85000, "stddev": 3.8, "N_IN_Conn_P_Src": 95, "N_IN_Conn_P_Dst": 95, 
            "duration": 0.02, "pkts": 950, "bytes": 760000, "srate": 45000.0, "drate": 0.0, 
            "proto": "udp", "state": "INT"
        }
    },
    "cic_ids": {
        "Normal Web Browse": {
            "flow_duration": 450000, "total_fwd_packets": 12, "total_bwd_packets": 15, 
            "fwd_pkt_len_max": 850, "fwd_pkt_len_min": 0, "bwd_pkt_len_max": 1460, 
            "bwd_pkt_len_min": 0, "flow_bytes_s": 5600.0, "flow_pkts_s": 60.0, 
            "flow_iat_mean": 15000.0, "flow_iat_std": 5000.0, "active_mean": 0.0, "idle_mean": 0.0
        },
        "DDoS Syn Flood": {
            "flow_duration": 12000, "total_fwd_packets": 450, "total_bwd_packets": 0, 
            "fwd_pkt_len_max": 64, "fwd_pkt_len_min": 64, "bwd_pkt_len_max": 0, 
            "bwd_pkt_len_min": 0, "flow_bytes_s": 2400000.0, "flow_pkts_s": 37500.0, 
            "flow_iat_mean": 26.0, "flow_iat_std": 10.0, "active_mean": 0.0, "idle_mean": 0.0
        }
    }
}

# Sidebar navigation
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
    st.markdown("<div class='sub-title'>Deep Learning Intrusion Detection Benchmarks (KDDCup99, NSL-KDD, Bot-IoT, CIC-IDS)</div>", unsafe_allow_html=True)
    
    # Calculate stats
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
        
    st.markdown("### Benchmark Performance Comparison across Datasets")
    
    # Comprehensive benchmark data
    benchmark_data = [
        # KDDCup99
        {"dataset": "KDDCUP99", "mode": "binary", "model": "ANN", "accuracy": 0.8971, "precision": 0.9437, "recall": 0.9291, "f1_score": 0.9363},
        {"dataset": "KDDCUP99", "mode": "binary", "model": "CNN", "accuracy": 0.8890, "precision": 0.9473, "recall": 0.9145, "f1_score": 0.9306},
        {"dataset": "KDDCUP99", "mode": "binary", "model": "LSTM", "accuracy": 0.8143, "precision": 0.8143, "recall": 1.0000, "f1_score": 0.8976},
        {"dataset": "KDDCUP99", "mode": "multiclass", "model": "ANN", "accuracy": 0.8601, "precision": 0.8946, "recall": 0.8601, "f1_score": 0.8602},
        {"dataset": "KDDCUP99", "mode": "multiclass", "model": "CNN", "accuracy": 0.7315, "precision": 0.6181, "recall": 0.7315, "f1_score": 0.6589},
        {"dataset": "KDDCUP99", "mode": "multiclass", "model": "LSTM", "accuracy": 0.6731, "precision": 0.7181, "recall": 0.6731, "f1_score": 0.6506},
        
        # NSL-KDD
        {"dataset": "NSLKDD", "mode": "binary", "model": "ANN", "accuracy": 0.9021, "precision": 0.8998, "recall": 1.0000, "f1_score": 0.9473},
        {"dataset": "NSLKDD", "mode": "binary", "model": "CNN", "accuracy": 0.8929, "precision": 0.9603, "recall": 0.9161, "f1_score": 0.9377},
        {"dataset": "NSLKDD", "mode": "binary", "model": "LSTM", "accuracy": 0.8796, "precision": 0.8796, "recall": 1.0000, "f1_score": 0.9360},
        {"dataset": "NSLKDD", "mode": "multiclass", "model": "ANN", "accuracy": 0.8343, "precision": 0.8918, "recall": 0.8343, "f1_score": 0.8252},
        {"dataset": "NSLKDD", "mode": "multiclass", "model": "CNN", "accuracy": 0.8616, "precision": 0.8755, "recall": 0.8616, "f1_score": 0.8232},
        {"dataset": "NSLKDD", "mode": "multiclass", "model": "LSTM", "accuracy": 0.5490, "precision": 0.4621, "recall": 0.5490, "f1_score": 0.4913},
        
        # Bot-IoT
        {"dataset": "BOT_IOT", "mode": "binary", "model": "ANN", "accuracy": 0.6975, "precision": 0.6986, "recall": 0.9928, "f1_score": 0.8201},
        {"dataset": "BOT_IOT", "mode": "binary", "model": "CNN", "accuracy": 0.7435, "precision": 0.7629, "recall": 0.9150, "f1_score": 0.8321},
        {"dataset": "BOT_IOT", "mode": "binary", "model": "LSTM", "accuracy": 0.6840, "precision": 0.6962, "recall": 0.9669, "f1_score": 0.8095},
        {"dataset": "BOT_IOT", "mode": "multiclass", "model": "ANN", "accuracy": 0.5955, "precision": 0.4469, "recall": 0.5955, "f1_score": 0.4981},
        {"dataset": "BOT_IOT", "mode": "multiclass", "model": "CNN", "accuracy": 0.7015, "precision": 0.7316, "recall": 0.7015, "f1_score": 0.6941},
        {"dataset": "BOT_IOT", "mode": "multiclass", "model": "LSTM", "accuracy": 0.5035, "precision": 0.4364, "recall": 0.5035, "f1_score": 0.4556},
        
        # CIC-IDS
        {"dataset": "CIC_IDS", "mode": "binary", "model": "ANN", "accuracy": 0.6780, "precision": 0.8660, "recall": 0.5545, "f1_score": 0.6761},
        {"dataset": "CIC_IDS", "mode": "binary", "model": "CNN", "accuracy": 0.6745, "precision": 0.7005, "recall": 0.8086, "f1_score": 0.7507},
        {"dataset": "CIC_IDS", "mode": "binary", "model": "LSTM", "accuracy": 0.6065, "precision": 0.6067, "recall": 0.9967, "f1_score": 0.7543},
        {"dataset": "CIC_IDS", "mode": "multiclass", "model": "ANN", "accuracy": 0.5440, "precision": 0.5388, "recall": 0.5440, "f1_score": 0.5104},
        {"dataset": "CIC_IDS", "mode": "multiclass", "model": "CNN", "accuracy": 0.6525, "precision": 0.5394, "recall": 0.6525, "f1_score": 0.5800},
        {"dataset": "CIC_IDS", "mode": "multiclass", "model": "LSTM", "accuracy": 0.5195, "precision": 0.4815, "recall": 0.5195, "f1_score": 0.4113}
    ]
    
    df_bench = pd.DataFrame(benchmark_data)
    
    tab1, tab2 = st.tabs(["Binary Classification Results", "Multi-Class Classification Results"])
    
    with tab1:
        st.write("#### Normal vs. Attack Threat Detection Accuracy")
        st.dataframe(df_bench[df_bench['mode'] == 'binary'].style.format({
            "accuracy": "{:.2%}", "precision": "{:.2%}", "recall": "{:.2%}", "f1_score": "{:.2%}"
        }), use_container_width=True)
        
    with tab2:
        st.write("#### Attack Types Categorization Metrics")
        st.dataframe(df_bench[df_bench['mode'] == 'multiclass'].style.format({
            "accuracy": "{:.2%}", "precision": "{:.2%}", "recall": "{:.2%}", "f1_score": "{:.2%}"
        }), use_container_width=True)

# 2. Page: Model Performance Plots
elif page == "Model Performance Plots":
    st.markdown("## 📈 Learning Curves & Confusion Matrices")
    st.write("Examine learning history plots and detailed confusion matrices.")
    
    col_ds, col_md, col_sc = st.columns(3)
    dataset_sel = col_ds.selectbox("Select Target Dataset", list(DATASET_LABELS.keys()), format_func=lambda x: DATASET_LABELS[x])
    model_sel = col_md.selectbox("Select Model Architecture", ["ANN", "CNN", "LSTM"])
    scope_sel = col_sc.selectbox("Select Classification Scope", ["Binary", "Multiclass"])
    
    # Filenames match main.py saves: outputs/{model_sel.lower()}_{dataset_sel}_{scope_sel.lower()}
    model_img_name = f"{model_sel.lower()}_{dataset_sel}_{scope_sel.lower()}"
    
    curve_file = f"outputs/{model_img_name}_curves.png"
    matrix_file = f"outputs/{model_img_name}_confusion_matrix.png"
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Training and Validation Loss / Accuracy Curves")
        if os.path.exists(curve_file):
            st.image(curve_file, use_container_width=True)
        else:
            st.info(f"Curve plot not generated for {model_sel} on {dataset_sel.upper()} {scope_sel} yet.")
            
    with col2:
        st.write("### Test Set Confusion Matrix")
        if os.path.exists(matrix_file):
            st.image(matrix_file, use_container_width=True)
        else:
            st.info(f"Confusion matrix plot not generated for {model_sel} on {dataset_sel.upper()} {scope_sel} yet.")

# 3. Page: Threat Logs Database
elif page == "Threat Logs Database":
    st.markdown("## 🗄️ SQLite threat_logs Database View")
    st.write("Querying detection events recorded in real-time by models and simulations.")
    
    logs = get_logs()
    
    if not logs:
        st.info("The threat database is currently empty. Run predictions or simulation scripts to log alerts.")
    else:
        df_logs = pd.DataFrame(logs, columns=["Log ID", "Timestamp", "Dataset", "Class", "Confidence", "Latency (s)"])
        
        # Filtering controls
        col_ds_f, col_cl_f = st.columns(2)
        ds_filter = col_ds_f.selectbox("Filter by Source Dataset", ["All"] + list(df_logs["Dataset"].unique()))
        class_filter = col_cl_f.text_input("Search/Filter by Predicted Class (e.g. normal, ddos, probe):", "")
        
        filtered_df = df_logs.copy()
        if ds_filter != "All":
            filtered_df = filtered_df[filtered_df["Dataset"] == ds_filter]
        if class_filter:
            filtered_df = filtered_df[filtered_df["Class"].str.contains(class_filter.lower(), case=False)]
            
        def highlight_threats(row):
            val = row['Class']
            if val.lower() in ['malicious', 'ddos', 'dos', 'probe', 'r2l', 'u2r']:
                return ['background-color: #FEE2E2; color: #991B1B'] * len(row)
            return [''] * len(row)
            
        st.dataframe(
            filtered_df.style.apply(highlight_threats, axis=1).format({
                "Confidence": "{:.2%}", "Latency (s)": "{:.6f}s"
            }),
            use_container_width=True
        )

# 4. Page: Interactive Simulator
elif page == "Interactive Simulator":
    st.markdown("<div class='main-title'>🛡️ Interactive IoT Packet Threat Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Run live predictions on single packets or simulate real-time live network traffic flow.</div>", unsafe_allow_html=True)
    
    sim_mode = st.tabs(["Single Packet Tester", "🚀 Live Network Traffic Stream Simulator"])
    
    # Tab 1: Single Packet Tester
    with sim_mode[0]:
        col_ds, col_md, col_sc = st.columns(3)
        dataset_key = col_ds.selectbox("Select Target Schema Dataset", list(DATASET_SCHEMAS.keys()), format_func=lambda x: DATASET_LABELS[x], key="single_ds")
        model_type = col_md.selectbox("Select Trained Model Architecture", ["ann", "cnn", "lstm"], key="single_model")
        scope = col_sc.selectbox("Select Prediction Scope", ["binary", "multiclass"], key="single_scope")
        
        dataset_presets = SIMULATOR_PRESETS.get(dataset_key, {})
        
        if not dataset_presets:
            st.warning(f"No presets configured for {DATASET_LABELS[dataset_key]}. Manually adjust variables below.")
            preset_data = {}
        else:
            preset_choice = st.selectbox("Load Sample Packet Preset (Auto-Populates inputs):", list(dataset_presets.keys()), key="single_preset")
            preset_data = dataset_presets[preset_choice]
            
        st.markdown("---")
        st.write("#### Configure Feature Parameter Tensors")
        col_inp1, col_inp2, col_inp3 = st.columns(3)
        
        inputs = {}
        categorical_options = {
            "protocol_type": ["tcp", "udp", "icmp"],
            "service": ["http", "private", "other", "ftp", "smtp", "dns"],
            "flag": ["SF", "S0", "REJ", "RSTR"],
            "proto": ["tcp", "udp", "icmp"],
            "state": ["CON", "INT", "FIN", "URP", "REQ"]
        }
        
        idx = 0
        feature_list = DATASET_SCHEMAS[dataset_key]
        for key in feature_list:
            col = col_inp1 if idx % 3 == 0 else (col_inp2 if idx % 3 == 1 else col_inp3)
            
            if key in categorical_options:
                default_val = preset_data.get(key, categorical_options[key][0])
                if default_val not in categorical_options[key]:
                    categorical_options[key].append(default_val)
                inputs[key] = col.selectbox(f"Feature: {key}", categorical_options[key], index=categorical_options[key].index(default_val), key=f"inp_{key}")
            else:
                default_val = float(preset_data.get(key, 0.0))
                inputs[key] = col.number_input(f"Feature: {key}", value=default_val, key=f"inp_{key}")
            idx += 1
            
        if st.button("🔴 RUN INTRUSION DETECTION SCAN", use_container_width=True, key="btn_run"):
            model_path = f"saved_models/{model_type.lower()}_{scope.lower()}_{dataset_key}_model.joblib"
            scaler_path = "scaler.joblib"
            
            if not os.path.exists(model_path):
                st.error(f"Pre-trained weights not found for {model_type.upper()} on {dataset_key.upper()} ({scope.upper()})! Run pipeline first.")
            else:
                model = joblib.load(model_path)
                df_in = pd.DataFrame([inputs])
                
                # Encode categoricals
                categorical_encodings = {
                    "protocol_type": {"tcp": 0, "udp": 1, "icmp": 2},
                    "proto": {"tcp": 0, "udp": 1, "icmp": 2},
                    "service": {"http": 0, "private": 1, "other": 2, "ftp": 3, "smtp": 4, "dns": 5},
                    "flag": {"SF": 0, "S0": 1, "REJ": 2, "RSTR": 3},
                    "state": {"CON": 0, "INT": 1, "FIN": 2, "URP": 3, "REQ": 4}
                }
                for k in df_in.columns:
                    if k in categorical_encodings:
                        df_in[k] = df_in[k].map(categorical_encodings[k]).fillna(0).astype(int)
                        
                # Scale
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    X_scaled = scaler.transform(df_in) if scaler.n_features_in_ == len(feature_list) else df_in.values / (np.max(df_in.values) + 1e-15)
                else:
                    X_scaled = df_in.values / (np.max(df_in.values) + 1e-15)
                    
                start_time = time.time()
                pred_prob = model.predict(X_scaled)
                latency = time.time() - start_time
                
                # Beautiful styled outputs
                st.markdown("---")
                if scope == "binary":
                    prob = float(pred_prob[0][0])
                    is_malicious = prob > 0.5
                    conf = prob if is_malicious else (1.0 - prob)
                    
                    if is_malicious:
                        st.markdown(f"""
                        <div class="alert-card">
                            <h3 style="margin-top:0; color:#991B1B;">⚠️ SECURITY ALERT: MALICIOUS PACKET IDENTIFIED</h3>
                            <p style="margin-bottom:5px;"><strong>Classification:</strong> Malicious Traffic Anomaly</p>
                            <p style="margin-bottom:5px;"><strong>Prediction Confidence:</strong> {conf:.2%}</p>
                            <p style="margin-bottom:0;"><strong>Recommended Mitigation:</strong> Immediately blacklist source host, update router access tables, and throttle packets from this port.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        pred_class = "Malicious"
                    else:
                        st.markdown(f"""
                        <div class="secure-card">
                            <h3 style="margin-top:0; color:#065F46;">✅ TRAFFIC STATUS: SECURE</h3>
                            <p style="margin-bottom:5px;"><strong>Classification:</strong> Normal Web/Sensor Traffic</p>
                            <p style="margin-bottom:5px;"><strong>Prediction Confidence:</strong> {conf:.2%}</p>
                            <p style="margin-bottom:0;"><strong>Security Level:</strong> Allowed. No signature mismatches or packet length violations detected.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        pred_class = "Normal"
                else:
                    pred_idx = int(np.argmax(pred_prob[0]))
                    pred_class = MULTICLASS_LABELS[pred_idx]
                    conf = float(pred_prob[0][pred_idx])
                    
                    if pred_class == "normal":
                        st.markdown(f"""
                        <div class="secure-card">
                            <h3 style="margin-top:0; color:#065F46;">✅ TRAFFIC STATUS: SECURE</h3>
                            <p style="margin-bottom:5px;"><strong>Classification:</strong> Normal Flow</p>
                            <p style="margin-bottom:5px;"><strong>Prediction Confidence:</strong> {conf:.2%}</p>
                            <p style="margin-bottom:0;"><strong>Security Level:</strong> Allowed. Traffic represents compliant protocol patterns.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-card">
                            <h3 style="margin-top:0; color:#991B1B;">⚠️ THREAT DETECTED: {pred_class.upper()} ATTACK</h3>
                            <p style="margin-bottom:5px;"><strong>Anomaly Category:</strong> {pred_class.upper()}</p>
                            <p style="margin-bottom:5px;"><strong>Confidence:</strong> {conf:.2%}</p>
                            <p style="margin-bottom:0;"><strong>Recommended Action:</strong> Adjust firewall tables, check host connection bounds, and trigger administrative log warning.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                st.write(f"Inference latency: `{latency:.6f}s` using custom NumPy DL architecture.")
                log_threat(f"{dataset_key.upper()}_Sim", pred_class, conf, latency)
                
    # Tab 2: Live Network Traffic Stream Simulator (Real-time dashboard visuals!)
    with sim_mode[1]:
        # SOC Header Indicators
        st.markdown("#### ⚙️ SOC SYSTEM SECURITY AGENTS")
        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        col_st1.markdown("<div style='background-color:#E0F2FE; padding:10px; border-radius:5px; border-left:4px solid #0284C7; text-align:center;'><strong>🛡️ FIREWALL</strong><br><span style='color:#0369A1;'>ACTIVE & LOCK</span></div>", unsafe_allow_html=True)
        col_st2.markdown("<div style='background-color:#DCFCE7; padding:10px; border-radius:5px; border-left:4px solid #16A34A; text-align:center;'><strong>🟢 IDS AGENT</strong><br><span style='color:#15803D;'>MONITORING</span></div>", unsafe_allow_html=True)
        col_st3.markdown("<div style='background-color:#FEF9C3; padding:10px; border-radius:5px; border-left:4px solid #CA8A04; text-align:center;'><strong>📡 SENSOR STATE</strong><br><span style='color:#A16207;'>SYNCHRONIZED</span></div>", unsafe_allow_html=True)
        # We will make the fourth block a dynamic threat status!
        threat_level_slot = col_st4.empty()
        threat_level_slot.markdown("<div style='background-color:#ECEFf1; padding:10px; border-radius:5px; border-left:4px solid #455A64; text-align:center;'><strong>📊 RISK LEVEL</strong><br><span style='color:#37474F;'>READY</span></div>", unsafe_allow_html=True)
        
        st.write("")
        
        col_lds, col_lmd, col_lsc = st.columns(3)
        stream_ds = col_lds.selectbox("Select Traffic Flow Dataset", list(DATASET_SCHEMAS.keys()), format_func=lambda x: DATASET_LABELS[x], key="stream_ds")
        stream_model = col_lmd.selectbox("Select Flow Classifier Model", ["ann", "cnn", "lstm"], key="stream_model")
        stream_scope = col_lsc.selectbox("Select Flow Scope", ["binary", "multiclass"], key="stream_scope")
        
        # Simulated Network Topology Map Title
        st.write("---")
        st.markdown("### 🖧 SIMULATED IoT CLUSTER CLOUD")
        
        col_node1, col_node2, col_node3, col_node4 = st.columns(4)
        node1_slot = col_node1.empty()
        node2_slot = col_node2.empty()
        node3_slot = col_node3.empty()
        node4_slot = col_node4.empty()
        
        # Initial Node States
        node1_slot.markdown("<div class='node-card node-secure'>🏠 SMART HOME HUB<br>Status: OK</div>", unsafe_allow_html=True)
        node2_slot.markdown("<div class='node-card node-secure'>🏭 INDUSTRIAL GATEWAY<br>Status: OK</div>", unsafe_allow_html=True)
        node3_slot.markdown("<div class='node-card node-secure'>🏥 MEDICAL SENSORS<br>Status: OK</div>", unsafe_allow_html=True)
        node4_slot.markdown("<div class='node-card node-secure'>⚡ SMART ENERGY GRID<br>Status: OK</div>", unsafe_allow_html=True)
        
        csv_file = f"{stream_ds}_sample.csv"
        if not os.path.exists(csv_file):
            st.error(f"Sample dataset file {csv_file} not found. Run pipeline first.")
        else:
            df_sample = pd.read_csv(csv_file)
            
            if st.button("🚀 INITIATE SOC REAL-TIME SIMULATION", use_container_width=True):
                model_path = f"saved_models/{stream_model.lower()}_{stream_scope.lower()}_{stream_ds}_model.joblib"
                if not os.path.exists(model_path):
                    st.error(f"Pre-trained weights not found for {stream_model.upper()} on {stream_ds.upper()} ({stream_scope.upper()})!")
                else:
                    model = joblib.load(model_path)
                    
                    st.markdown("---")
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.markdown("#### ⚡ Incoming Connection Ingress Stream")
                        ingress_slot = st.empty()
                        alert_slot = st.empty()
                        stat_slot = st.empty()
                        
                    with col_right:
                        st.markdown("#### 📈 Network Latency & Load Timeline")
                        chart_placeholder = st.empty()
                        
                    st.markdown("#### 📋 Administrative Threat Detection Logs (Real-time)")
                    log_slot = st.empty()
                    
                    # Stream variables
                    num_packets = 25
                    normal_count = 0
                    attack_count = 0
                    latencies = []
                    recent_events = []
                    chart_data = []
                    
                    # Slide over random test sample rows
                    np.random.seed(int(time.time()))
                    random_rows = df_sample.sample(n=num_packets, replace=True).reset_index(drop=True)
                    
                    # Track attack counts per node cluster to trigger individual visual alarms
                    node_alarms = {"home": False, "factory": False, "medical": False, "grid": False}
                    
                    for i in range(num_packets):
                        row = random_rows.iloc[i:i+1]
                        label_col = "label" if "label" in row.columns else [c for c in row.columns if c.lower() in ["label", "target", "class"]][0]
                        X_row = row.drop(columns=[label_col])
                        
                        # Identify target node based on service/protocol rules
                        # We simulate routing packets to specific node clusters
                        target_cluster = "home"
                        if "service" in X_row.columns:
                            srv = str(X_row["service"].values[0])
                            if "http" in srv:
                                target_cluster = "home"
                            elif "private" in srv or "ftp" in srv:
                                target_cluster = "factory"
                            else:
                                target_cluster = "grid"
                        elif "proto" in X_row.columns or "protocol_type" in X_row.columns:
                            proto_col = "proto" if "proto" in X_row.columns else "protocol_type"
                            pr = str(X_row[proto_col].values[0])
                            if "udp" in pr:
                                target_cluster = "medical"
                            else:
                                target_cluster = "grid"
                        else:
                            target_cluster = random.choice(["home", "factory", "medical", "grid"])
                            
                        # Encode/Transform categoricals
                        categorical_encodings = {
                            "protocol_type": {"tcp": 0, "udp": 1, "icmp": 2},
                            "proto": {"tcp": 0, "udp": 1, "icmp": 2},
                            "service": {"http": 0, "private": 1, "other": 2, "ftp": 3, "smtp": 4, "dns": 5},
                            "flag": {"SF": 0, "S0": 1, "REJ": 2, "RSTR": 3},
                            "state": {"CON": 0, "INT": 1, "FIN": 2, "URP": 3, "REQ": 4}
                        }
                        for k in X_row.columns:
                            if k in categorical_encodings:
                                X_row[k] = X_row[k].map(categorical_encodings[k]).fillna(0).astype(int)
                                
                        # Simple min-max normalize
                        X_scaled = X_row.values / (np.max(X_row.values) + 1e-15)
                        
                        # Prediction
                        start_time = time.time()
                        pred_prob = model.predict(X_scaled)
                        latency = time.time() - start_time
                        latencies.append(latency)
                        
                        # Parse predictions
                        if stream_scope == "binary":
                            prob = float(pred_prob[0][0])
                            is_malicious = prob > 0.5
                            conf = prob if is_malicious else (1.0 - prob)
                            pred_class = "Malicious" if is_malicious else "Normal"
                        else:
                            pred_idx = int(np.argmax(pred_prob[0]))
                            pred_class = MULTICLASS_LABELS[pred_idx]
                            conf = float(pred_prob[0][pred_idx])
                            is_malicious = (pred_class != "normal")
                            
                        # Logging & stats update
                        if is_malicious:
                            attack_count += 1
                            log_threat(f"Live_Stream_{stream_ds.upper()}", pred_class, conf, latency)
                            node_alarms[target_cluster] = True
                            
                            # Render flashing alert banner
                            alert_slot.markdown(f"""
                            <div class="alert-card">
                                <h4 style="margin:0; color:#991B1B;">⚡ THREAT BLOCKED: {pred_class.upper()} PACKET</h4>
                                <span style="font-size:12px;">Routed Target: {target_cluster.upper()} Node | Conf: {conf:.2%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            normal_count += 1
                            alert_slot.markdown(f"""
                            <div class="secure-card">
                                <h4 style="margin:0; color:#065F46;">🟢 SECURE PACKET INGESTED</h4>
                                <span style="font-size:12px;">Routed Target: {target_cluster.upper()} Node | Conf: {conf:.2%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Update risk level indicator
                        risk_percent = (attack_count / (i + 1))
                        if risk_percent > 0.4:
                            threat_level_slot.markdown("<div style='background-color:#FEE2E2; padding:10px; border-radius:5px; border-left:4px solid #DC2626; text-align:center;'><strong>🚨 RISK LEVEL</strong><br><span style='color:#B91C1C;'>CRITICAL</span></div>", unsafe_allow_html=True)
                        elif risk_percent > 0.15:
                            threat_level_slot.markdown("<div style='background-color:#FEF3C7; padding:10px; border-radius:5px; border-left:4px solid #D97706; text-align:center;'><strong>⚠️ RISK LEVEL</strong><br><span style='color:#B45309;'>WARNING</span></div>", unsafe_allow_html=True)
                        else:
                            threat_level_slot.markdown("<div style='background-color:#DCFCE7; padding:10px; border-radius:5px; border-left:4px solid #16A34A; text-align:center;'><strong>📊 RISK LEVEL</strong><br><span style='color:#15803D;'>SAFE</span></div>", unsafe_allow_html=True)
                            
                        # Update Node cards dynamically
                        if node_alarms["home"]:
                            node1_slot.markdown("<div class='node-card node-alert'>🏠 SMART HOME HUB<br>Status: ATTACK DETECTED</div>", unsafe_allow_html=True)
                        else:
                            node1_slot.markdown("<div class='node-card node-secure'>🏠 SMART HOME HUB<br>Status: SECURE</div>", unsafe_allow_html=True)
                            
                        if node_alarms["factory"]:
                            node2_slot.markdown("<div class='node-card node-alert'>🏭 FACTORY GATEWAY<br>Status: ATTACK DETECTED</div>", unsafe_allow_html=True)
                        else:
                            node2_slot.markdown("<div class='node-card node-secure'>🏭 FACTORY GATEWAY<br>Status: SECURE</div>", unsafe_allow_html=True)
                            
                        if node_alarms["medical"]:
                            node3_slot.markdown("<div class='node-card node-alert'>🏥 MEDICAL SENSORS<br>Status: ATTACK DETECTED</div>", unsafe_allow_html=True)
                        else:
                            node3_slot.markdown("<div class='node-card node-secure'>🏥 MEDICAL SENSORS<br>Status: SECURE</div>", unsafe_allow_html=True)
                            
                        if node_alarms["grid"]:
                            node4_slot.markdown("<div class='node-card node-alert'>⚡ SMART POWER GRID<br>Status: ATTACK DETECTED</div>", unsafe_allow_html=True)
                        else:
                            node4_slot.markdown("<div class='node-card node-secure'>⚡ SMART POWER GRID<br>Status: SECURE</div>", unsafe_allow_html=True)
                            
                        # Fluctuate ingress pkts/sec
                        ingress_rate = random.randint(120, 180)
                        ingress_slot.markdown(f"**Current Flow Speed:** `{ingress_rate} pkts/s` | **Packet ID:** `#{random.randint(45000, 99999)}`")
                        
                        # Ingress stat updates
                        stat_slot.markdown(f"""
                        * **Ingested Count:** `{i+1}/{num_packets}`
                        * **Allowed Packets:** `{normal_count}`
                        * **Blocked Threats:** `{attack_count}`
                        * **IDS Ingress Latency:** `{np.mean(latencies):.6f}s`
                        """)
                        
                        # Append chart data
                        chart_data.append({
                            "Packet": i + 1,
                            "Latency (ms)": latency * 1000,
                            "Threat Rate (%)": risk_percent * 100
                        })
                        chart_df_temp = pd.DataFrame(chart_data)
                        chart_placeholder.area_chart(chart_df_temp.set_index("Packet"))
                        
                        # Append rolling connection logger
                        ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        action_badge = "🔴 BLOCKED" if is_malicious else "🟢 ALLOWED"
                        recent_events.insert(0, {
                            "Time": ts_str,
                            "IoT Node Target": target_cluster.upper(),
                            "Packet Classification": pred_class,
                            "Confidence": f"{conf:.2%}",
                            "Action Status": action_badge
                        })
                        log_slot.dataframe(pd.DataFrame(recent_events).head(6), use_container_width=True)
                        
                        time.sleep(0.3)
                        
                    # Reset indicators at the end
                    alert_slot.empty()
                    st.success("🎉 Live SOC monitoring run completed! All packet logs exported to threat_logs.db.")
