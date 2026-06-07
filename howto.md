# How-To Run Guide: IoT Threat Detection SOC Framework

This document outlines the step-by-step commands to install dependencies, train the models, run the dashboards, and expose the server publicly using a Cloudflare tunnel.

---
```bash
git clone https://github.com/srkcodes1003/Cyber-Security-Threats-Detection-in-Internet-of-Things-Using-Deep-Learning-Approach.git 
```

## 1. Environment Initialization & Setup

Before running the scripts, set up your Python environment and install the required modules.

### Create Python Virtual Environment (Windows PowerShell / CMD)
```bash
python -m venv venv
```

### Activate the Virtual Environment
* **On Windows CMD:**
  ```cmd
  venv\Scripts\activate
  ```
* **On Windows PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **On Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### Install Dependencies
Ensure the packages in `requirements.txt` are fully installed:
```bash
pip install -r requirements.txt
```

---

## 2. Ingest Data & Train Deep Learning Models

To build and train the deep learning models (Dense ANN, 1D-CNN, LSTM) for KDDCup99, NSL-KDD, Bot-IoT, and CIC-IDS datasets, run the core pipeline:

### Fetch Datasets and Train All Models (24 models total, default)
This command will fetch the raw data, generate high-fidelity samples, pre-process them, train all models for 3 epochs, compile metrics, and run a simulation query.
```bash
python main.py
```

### Custom Pipeline Sweep Configs
You can customize the run using command-line arguments:
* **Train a single dataset (e.g., Bot-IoT) in multiclass mode:**
  ```bash
  python main.py --dataset bot_iot --mode multiclass --epochs 5 --batch_size 64
  ```
* **Train specific models (e.g., only ANN and CNN) in binary mode:**
  ```bash
  python main.py --dataset all --mode binary --models "ann,cnn" --epochs 3
  ```

---

## 3. Run Streamlit Analytics Dashboard

To view diagnostic curves, metrics heatmaps, and simulated log events in the standard Streamlit interface, run:
```bash
streamlit run app.py
```
* **Access URL:** `http://localhost:8501`

---

## 4. Run Flask API & SOC Web Dashboard (Premium UI)

To launch the Security Operations Center (SOC) dashboard featuring a glassmorphism dark theme, live network topology pulse status flags, and rolling Chart.js timeline waveforms:

### Launch the Flask Server
```bash
python server.py
```
* **Access URL:** `http://localhost:5000`

---

## 5. Expose the Server Publicly (Cloudflare Tunnel)

To share the running Flask SOC dashboard over the public internet or view it on other devices (bypassing local firewalls and NATs), run a secure Cloudflare tunnel.

### Command-line Tunnel (Quick-start)
If you have `cloudflared` installed on your machine, run:
```bash
cloudflared tunnel --url http://localhost:5000
```
This will generate a random, secure public URL (e.g., `https://random-words.trycloudflare.com`) that points directly to your locally running dashboard.

---
