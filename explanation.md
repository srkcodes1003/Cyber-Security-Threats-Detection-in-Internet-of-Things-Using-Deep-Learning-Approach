# Detailed Project Technical Explanation: IoT Threat Detection System

This document provides a comprehensive technical overview of the implementation details, mathematical concepts, and system architecture for the **Cyber Security Threat Detection Framework in IoT Networks using custom Deep Learning architectures**.

---

## 1. Modular Architecture Overview

The system is designed with a decoupled modular architecture to separate data retrieval, preprocessing, machine learning execution, log persistence, and visual analytics:

```
c:\Users\Hp\Desktop\MP\
├── requirements.txt         # Package dependencies (Min-Max versions)
├── db_logger.py             # Thread-safe SQLite logger for intrusion events
├── fetch_data.py            # Dataset sample generation (KDDCup99, Bot-IoT, etc.)
├── preprocess.py            # Normalization, categorical encoding, class mapping
├── models.py                # Custom DL layer classes in pure NumPy
├── train.py                 # Fitting models and joblib serialization
├── evaluate.py              # Exporting learning curves and confusion matrices
├── main.py                  # Integration controller (24 sweeps runner)
├── app.py                   # Streamlit analytics frontend dashboard
├── server.py                # Flask API server serving predictions and stream
├── templates/
│   └── index.html           # SOC web HTML5 structure
└── static/
    ├── style.css            # Glassmorphism dark SOC dashboard CSS
    └── script.js            # Chart.js integration, AJAX controllers
```

---

## 2. Lightweight Pure NumPy Deep Learning Engine

Due to host DLL resource loading conflicts and instruction set limits (AVX) that block compiled packages like TensorFlow/PyTorch, the deep learning logic is written in **pure NumPy** (`models.py`). This engine replicates standard feedforward and backpropagation calculus.

### Layer Classes
1. **`Layer` (Base Interface)**: Defines abstract `forward(input)` and `backward(output_gradient, learning_rate)` methods.
2. **`Dense` (Fully Connected)**:
   * **Initialization**: Uses **Glorot/Xavier Initialization** to scale weights:
     $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{input\_size} + \text{output\_size}}}\right)$$
     Biases are initialized to $0$.
   * **Forward Pass**: Evaluates the affine transformation:
     $$Z = X \cdot W + b$$
   * **Backpropagation**: Calculates parameter gradients and inputs derivative:
     $$\frac{\partial L}{\partial W} = X^T \cdot \delta_{out}, \quad \frac{\partial L}{\partial b} = \sum \delta_{out}, \quad \delta_{in} = \delta_{out} \cdot W^T$$
     *Includes gradient clipping between $[-1.0, 1.0]$ to prevent gradient explosion.*
3. **`SimpleConv1D` (1D Convolution)**:
   * Replicates feature extraction over grid vectors:
     $$\text{out}[f, t] = \sum_{k=0}^{K-1} X[t + k] \cdot W[f, k] + b[f]$$
   * Evaluates convolution backward gradients by sliding filter derivatives.
4. **`SimpleRNN` (Recurrent Layer)**:
   * Simulates seq-behavior using a hidden state $h_t$:
     $$h_t = \tanh(x_t \cdot W_{xh} + h_{t-1} \cdot W_{hh} + b_h)$$
   * Computes **Backpropagation Through Time (BPTT)** by traversing sequential cells in reverse index.

### Activation Layers
* **`ReLU`**: Returns $\max(0, X)$. Backward gradient is $1$ for $x > 0$ and $0$ otherwise.
* **`Sigmoid`**: Used for binary classification:
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
   *Safely clips outputs to $[-15, 15]$ to prevent underflow/overflow.*
* **`Softmax`**: Used for multiclass classification:
   $$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$$
   *Uses shifted exponential calculation to guarantee numerical stability.*

### Training Optimization (`NumPyModel.fit`)
* **Loss Functions**: Implements **Binary Cross-Entropy** (BCE) for binary classification and **Categorical Cross-Entropy** (CCE) for multiclass classification:
  $$\text{BCE} = -\frac{1}{N}\sum \left(y\log(p) + (1-y)\log(1-p)\right)$$
  $$\text{CCE} = -\frac{1}{N}\sum \sum y_{ij}\log(p_{ij})$$
* **Optimizer**: Shuffled mini-batch **Stochastic Gradient Descent (SGD)** with adaptive learning rates ($0.01$ for binary, $0.02$ for multiclass).
* **Metrics Track**: Fits callbacks returning histories (`loss`, `accuracy`, `val_loss`, `val_accuracy`) for Keras charting compatibility.

---

## 3. Data Processing Pipeline (`preprocess.py`)

A raw telemetry dataset requires extensive cleaning before models can evaluate feature dimensions:

```
[Raw CSV Packet Input] 
       │
       ▼
[Data Cleaning] ──► Removes duplicate rows, NaNs, and infinite records.
       │
       ▼
[Category Mapping] ──► Binary: normal (0), attack (1)
       │            ──► Multiclass: maps raw threat label to 1 of 6 unified categories:
       │                ['normal', 'dos', 'ddos', 'probe', 'r2l', 'u2r']
       ▼
[Categorical Encode] ──► Label encodes objects (e.g., tcp=0, udp=1, icmp=2)
       │
       ▼
[Normalization] ──► MinMax Scaler fits features to [0, 1] bounds:
       │            X_scaled = (X - X_min) / (X_max - X_min)
       ▼
[Split Data] ──► Stratified Train/Test partitioning (80/20 split)
```

---

## 4. SQLite Database Logger (`db_logger.py`)

For persistent record audits, the framework includes an asynchronous-friendly, thread-safe SQLite connection:
* **Initialization**: Auto-creates a schema table `threat_logs` inside `threat_logs.db`.
* **Attributes Saved**: `log_id` (INTEGER AUTOINCREMENT), `timestamp` (DATETIME DEFAULT CURRENT_TIMESTAMP), `source_dataset` (TEXT), `predicted_class` (TEXT), `confidence_score` (REAL), and `execution_time` (REAL).
* **Write Mechanism**: Commits logs dynamically on prediction events.

---

## 5. Exposing the Dashboards

The system supports two complementary frontend dashboards:

### Streamlit Dashboard (`app.py`)
Exposes detailed evaluation charts, accuracy tables, model weights comparisons, loss curve plots, and custom single packet diagnostics selectors. Intended for deep offline analysis.

### Flask SOC Web Console (`server.py`)
Served over port `5000`, this represents a premium live dashboard designed for network operators:
* **Network Node Flash**: Four network clusters (Smart Home, Factory, Medical, Smart Grid) flash red or orange in real-time as attacks are detected on their respective sub-ports.
* **Rolling Waveform**: Chart.js scrolling plot shows inspection confidence and processing speed.
* **AJAX Streaming**: A JavaScript fetch loop queries `/api/stream` every 500ms. The endpoint picks a random line from the selected CSV sample, feeds it to the active model, logs it to SQLite, and streams the results back to the dashboard table.
* **Diagnostics Simulator**: The dynamic form renders inputs matching the active dataset. Selecting a preset (e.g., "UDP Flood DDoS" or "Normal HTTP Web") populates the fields instantly. Clicking "Inspect" runs predictions using the NumPy engine and triggers gauge visualizations.

---
