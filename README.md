# Cyber Security Threats Detection in Internet of Things Using Deep Learning Approach

This repository contains a production-grade, modular, and portable deep learning framework designed to detect cybersecurity intrusions and traffic anomalies in high-throughput IoT networks.

---

## 👥 Project Team & Metadata
* **Batch Number:** DS11
* **Project Team Members:**
    * **Kethavath Sindhu** (Roll No: 1608-23-750-063)
    * **Sai Ramakrishna Doddapaneni** (Roll No: 1608-23-750-004)
    * **Kota Rakesh** (Roll No: 1608-23-750-057)
* **Project Guide:** Ms. Rama Lakshmi, Assistant Professor, CSE Department
* **Version:** 1.0 (June 2026)

---

## 📖 Introduction & Core Objective
The expansion of IoT introduces major security liabilities due to heterogeneous architectures, limited battery life, and lack of standardized security protocols. Legacy Intrusion Detection Systems (IDS) rely on manual signature updates and fail to detect novel, zero-day mutations. 

This project implements **three distinct deep learning architectures (ANN, 1D-CNN, LSTM)** to automatically learn complex network packet traffic patterns and classify them into normal or malicious flows.

### Scope of Classification:
1. **Binary Classification:** Distinguish between **Normal** and **Malicious** traffic.
2. **Multi-Class Classification:** Categorize malicious traffic into **DoS**, **DDoS**, **Probe**, **Remote-to-Local (R2L)**, and **User-to-Root (U2R)**.

---

## 🛠️ Portability: Pure NumPy Deep Learning Engine
Standard deep learning frameworks (TensorFlow, PyTorch) frequently run into dynamic link library initialization routine failures (`WinError 1114` and `.pyd` module conflicts) on Windows machines with dual-graphics switching or power-saving configurations.

To resolve this and guarantee **100% execution compatibility on any host machine (CPU-only)**, this project uses a custom backpropagation and training engine written entirely in **pure NumPy**. It implements analytical gradients, sliding 1D convolutions, and Backpropagation Through Time (BPTT) for recurrent neural units. It mimics a Keras-like interface (`model.fit()` and `model.predict()`) and outputs standardized callback logs.

---

## 📂 Project Architecture & Modules
```
c:\Users\Hp\Desktop\MP\
├── requirements.txt         # Package dependencies
├── fetch_data.py            # Automated KDDCup99 dataset sampling & decoding
├── preprocess.py            # Missing/NaN values cleaning, Min-Max Scaling, train/test split
├── models.py                # NumPy implementations of Dense, SimpleConv1D, SimpleRNN layers
├── train.py                 # Core trainer script managing fit loops & weights serialization
├── evaluate.py              # Performance metric calculators, loss curves, and confusion heatmaps
├── db_logger.py             # SQLite database administrative logs manager
├── main.py                  # End-to-end command-line pipeline execution script
└── app.py                   # Streamlit web application frontend dashboard
```

---

## 🚀 Installation & Execution

### 1. Set Up Virtual Environment & Install Dependencies
First, create a virtual environment in the root folder and install the dependencies:
```bash
# Create virtual environment
py -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Main Pipeline (CMD Line)
Run the pipeline to download the dataset, preprocess it, train the models (ANN, CNN, LSTM), evaluate performance, and run threat simulations:
```bash
python main.py --epochs 3 --batch_size 128
```
*Outputs (confusion matrix images, training curves) will be written to the `outputs/` directory.*
*Trained model objects will be saved to `saved_models/` using `joblib`.*

### 3. Launch the Streamlit Frontend Web Dashboard
Start the interactive dashboard server:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to view:
* **Overview panel** displaying comparison benchmark tables.
* **Performance curves and confusion matrices** visualizer.
* **Threat Logs database** querying recent SQLite warnings in real-time.
* **Live IoT Packet Threat Analyzer** simulator using presets to perform live detection scans.

---

## 📊 Performance Benchmarks (Sample Run)
Training results obtained on a sampled subset (12,460 rows) of the KDDCup99 dataset over 3 epochs:

| Scope | Model | Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Binary** | ANN | 92.41% | 91.53% | 99.92% | 95.54% | 0.16s |
| **Binary** | 1D-CNN | 94.29% | 93.70% | 99.69% | 96.60% | 3.47s |
| **Binary** | LSTM | 91.09% | 94.60% | 94.45% | 94.53% | 1.27s |
| **Multi-Class** | ANN | 94.86% | 95.11% | 94.86% | 94.80% | 0.18s |
| **Multi-Class** | 1D-CNN | 93.10% | 94.30% | 93.10% | 93.26% | 3.31s |
| **Multi-Class** | LSTM | 72.96% | 72.93% | 72.96% | 72.79% | 1.49s |

---

## 🗄️ Database Logging Scheme
Administrative alert logs are written automatically to `threat_logs.db`. The table schema matches:

| Column | Type | Description |
| :--- | :--- | :--- |
| `log_id` | INTEGER (PK, AUTOINCREMENT) | Unique identifier for each warning. |
| `timestamp` | TEXT (DEFAULT CURRENT_TIMESTAMP) | Time stamp when threat was identified. |
| `source_dataset` | TEXT | Source of traffic logs (e.g. KDDCup99, Live_Simulator). |
| `predicted_class` | TEXT | Decoded class (Normal, DDoS, DoS, Probe, etc.). |
| `confidence_score` | REAL | Probability score calculated by the model. |
| `execution_time` | REAL | Total calculation latency time in seconds. |
