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

### Supported Datasets:
* **KDDCup99** (Standard benchmark)
* **NSL-KDD** (Cleaned benchmark)
* **Bot-IoT** (IoT-specific sensor/actuator network dataset)
* **CIC-IDS** (Modern, diverse packet traffic dataset)

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
├── fetch_data.py            # Automated dataset sampler & generators for all 4 datasets
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
Run the pipeline to download/generate all datasets, train the models (ANN, CNN, LSTM) on all of them, evaluate performance, and run threat simulations:
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
* **Overview panel** displaying comparison benchmark tables across all datasets.
* **Performance curves and confusion matrices** visualizer.
* **Threat Logs database** querying recent SQLite warnings in real-time.
* **Live IoT Packet Threat Analyzer** simulator using presets to perform live detection scans.

---

## 📊 Multi-Dataset Benchmark Results
Training accuracy results obtained across all four standard benchmark datasets (1 epoch run):

| Dataset | Scope | Model | Accuracy | Precision | Recall | F1-Score | Training Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KDDCup99** | Binary | ANN | 89.71% | 94.37% | 92.91% | 93.63% | 0.08s |
| **KDDCup99** | Binary | CNN | 88.90% | 94.73% | 91.45% | 93.06% | 2.16s |
| **KDDCup99** | Binary | LSTM | 81.43% | 81.43% | 100.00% | 89.76% | 0.75s |
| **KDDCup99** | Multiclass | ANN | 86.01% | 89.46% | 86.01% | 86.02% | 0.07s |
| **KDDCup99** | Multiclass | CNN | 73.15% | 61.81% | 73.15% | 65.89% | 2.24s |
| **KDDCup99** | Multiclass | LSTM | 67.31% | 71.81% | 67.31% | 65.06% | 1.12s |
| **NSL-KDD** | Binary | ANN | 90.21% | 89.98% | 100.00% | 94.73% | 0.12s |
| **NSL-KDD** | Binary | CNN | 89.29% | 96.03% | 91.61% | 93.77% | 1.64s |
| **NSL-KDD** | Binary | LSTM | 87.96% | 87.96% | 100.00% | 93.60% | 0.54s |
| **NSL-KDD** | Multiclass | ANN | 83.43% | 89.18% | 83.43% | 82.52% | 0.08s |
| **NSL-KDD** | Multiclass | CNN | 86.16% | 87.55% | 86.16% | 82.32% | 0.88s |
| **NSL-KDD** | Multiclass | LSTM | 54.90% | 46.21% | 54.90% | 49.13% | 0.50s |
| **Bot-IoT** | Binary | ANN | 69.75% | 69.86% | 99.28% | 82.01% | 0.04s |
| **Bot-IoT** | Binary | CNN | 74.35% | 76.29% | 91.50% | 83.21% | 0.16s |
| **Bot-IoT** | Binary | LSTM | 68.40% | 69.62% | 96.69% | 80.95% | 0.09s |
| **Bot-IoT** | Multiclass | ANN | 59.55% | 44.69% | 59.55% | 49.81% | 0.05s |
| **Bot-IoT** | Multiclass | CNN | 70.15% | 73.16% | 70.15% | 69.41% | 0.16s |
| **Bot-IoT** | Multiclass | LSTM | 50.35% | 43.64% | 50.35% | 45.56% | 0.10s |
| **CIC-IDS** | Binary | ANN | 67.80% | 86.60% | 55.45% | 67.61% | 0.04s |
| **CIC-IDS** | Binary | CNN | 67.45% | 70.05% | 80.86% | 75.07% | 0.23s |
| **CIC-IDS** | Binary | LSTM | 60.65% | 60.67% | 99.67% | 75.43% | 0.11s |
| **CIC-IDS** | Multiclass | ANN | 54.40% | 53.88% | 54.40% | 51.04% | 0.03s |
| **CIC-IDS** | Multiclass | CNN | 65.25% | 53.94% | 65.25% | 58.00% | 0.23s |
| **CIC-IDS** | Multiclass | LSTM | 51.95% | 48.15% | 51.95% | 41.13% | 0.11s |

---

## 🗄️ Database Logging Scheme
Administrative alert logs are written automatically to `threat_logs.db`. The table schema matches:

| Column | Type | Description |
| :--- | :--- | :--- |
| `log_id` | INTEGER (PK, AUTOINCREMENT) | Unique identifier for each warning. |
| `timestamp` | TEXT (DEFAULT CURRENT_TIMESTAMP) | Time stamp when threat was identified. |
| `source_dataset` | TEXT | Source of traffic logs (e.g. KDDCUP99_Sim, BOT_IOT_Sim). |
| `predicted_class` | TEXT | Decoded class (Normal, DDoS, DoS, Probe, etc.). |
| `confidence_score` | REAL | Probability score calculated by the model. |
| `execution_time` | REAL | Total calculation latency time in seconds. |
