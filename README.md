# Cyber Security Threats Detection in Internet of Things Using Deep Learning Approach

This repository houses a production-grade, modular, and portable deep learning framework designed to detect cybersecurity intrusions and traffic anomalies in high-throughput IoT networks.

---

## 👥 Project Team & Metadata
* **Batch Number:** DS11
* **Project Team Members:**
    * **Kethavath Sindhu** (Roll No: 1608-23-750-063)
    * **Sai Ramakrishna Doddapaneni** (Roll No: 1608-23-750-004)
    * **Kota Rakesh** (Roll No: 1608-23-750-057)
* **Project Guide:** Ms. Rama Lakshmi, Assistant Professor, CSE Department
* **Department:** Computer Science & Engineering (CSE)
* **Version:** 1.0 (June 2026)

---

## 📖 Introduction & Core Objective
The expansion of IoT introduces major security liabilities due to heterogeneous architectures, resource-constrained environments, and a lack of standardized security protocols. Legacy Intrusion Detection Systems (IDS) rely on manual signature matching, failing to detect zero-day exploits.

This project implements **three distinct deep learning architectures (Dense ANN, 1D-CNN, LSTM)** to automatically learn network telemetry patterns and classify them into normal or malicious traffic.

### Supported Datasets
* **KDDCup99:** Standard network threat benchmark dataset.
* **NSL-KDD:** Refined/cleaned benchmark addressing statistical issues in KDDCup99.
* **Bot-IoT:** Realistic IoT-specific sensor/actuator network threat dataset.
* **CIC-IDS:** Modern, diverse telemetry dataset modeling complex attack structures.

### Scope of Classification
1. **Binary Classification:** Distinguish between **Normal** (secure) and **Malicious** (attack) traffic.
2. **Multi-Class Classification:** Classify malicious traffic into 5 attack types: **DoS**, **DDoS**, **Probe**, **Remote-to-Local (R2L)**, and **User-to-Root (U2R)**.

---

## 🛠️ Portability: Pure NumPy Deep Learning Engine
Standard frameworks (TensorFlow, PyTorch) can trigger DLL loading failures (`WinError 1114`) on GPU-powered Windows environments due to driver routing conflicts.

To solve this, this framework uses a lightweight deep learning engine built **entirely from scratch in pure NumPy** (`models.py`). It implements Xavier initialization, gradient clipping, custom 1D convolutions, and Backpropagation Through Time (BPTT) with zero framework overhead.

---

## 🧮 Mathematical Foundations (Key Formulas)

### 1. Xavier/Glorot Weight Initialization
To prevent vanishing or exploding gradients during training, weights are initialized from a normal distribution with variance proportional to input/output dimensions:
$$W \sim \mathcal{N}\left(0, \sigma^2\right) \quad \text{where} \quad \sigma = \sqrt{\frac{2}{\text{input\_size} + \text{output\_size}}}$$

### 2. Dense Layer Forward & Backward Propagation
* **Forward Pass (Affine Matrix Multiply):**
  $$Z = X \cdot W + b$$
* **Backward Pass (Gradient Calculations):**
  $$\frac{\partial L}{\partial W} = X^T \cdot \delta_{\text{out}}$$
  $$\frac{\partial L}{\partial b} = \sum_{\text{batch}} \delta_{\text{out}}$$
  $$\delta_{\text{in}} = \delta_{\text{out}} \cdot W^T$$
  *Where $\delta_{\text{out}}$ is the incoming gradient, and weights/biases are clipped within $[-1.0, 1.0]$ before optimization.*

### 3. Activation Functions & Derivatives
* **ReLU (Rectified Linear Unit):**
  $$f(z) = \max(0, z) \quad \implies \quad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
* **Sigmoid (Binary Classifier Output with Numerical Clip):**
  $$\sigma(z) = \frac{1}{1 + e^{-\text{clip}(z, -15, 15)}} \quad \implies \quad \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
* **Softmax (Multiclass Classifier Output with Numerical Shift):**
  $$\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j} e^{z_j - \max(z)}}$$

### 4. SimpleConv1D Layer (1D Convolution)
* **Forward Pass:**
  $$\text{out}[f, t] = \sum_{k=0}^{K-1} X[t + k] \cdot W[f, k] + b[f]$$
  *Where $K$ is the kernel size, $f$ is the filter index, and $t$ is the spatial output step.*

### 5. SimpleRNN Layer & Backpropagation Through Time (BPTT)
* **Forward Sequential Recurrence:**
  $$h_{t+1} = \tanh\left(x_t \cdot W_{xh} + h_t \cdot W_{hh} + b_h\right)$$
* **BPTT Gradient Recurrence:**
  $$\delta_t = dh_{t+1} \cdot \left(1 - h_{t+1}^2\right)$$
  $$\frac{\partial L}{\partial W_{xh}} = \sum_{t} x_t^T \cdot \delta_t, \quad \frac{\partial L}{\partial W_{hh}} = \sum_{t} h_t^T \cdot \delta_t, \quad \frac{\partial L}{\partial b_h} = \sum_{t} \delta_t, \quad dh_t = \delta_t \cdot W_{hh}^T$$

### 6. Numerical Optimization (Cross-Entropy & SGD Updates)
* **Binary Cross-Entropy Loss (BCE):**
  $$\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i + \epsilon) + (1 - y_i) \log(1 - p_i + \epsilon) \right] \quad (\epsilon = 10^{-15})$$
* **Categorical Cross-Entropy Loss (CCE):**
  $$\text{CCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij} + \epsilon) \quad (\epsilon = 10^{-15})$$
* **SGD Parameter Updates (with clipped updates):**
  $$W \leftarrow W - \alpha \cdot \text{clip}(\nabla_W L, -1.0, 1.0)$$
  $$b \leftarrow b - \alpha \cdot \text{clip}(\nabla_b L, -1.0, 1.0)$$

---

## 💻 Key Code Syntaxes

### 1. Custom Dense Layer Implementation (`models.py`)
```python
class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        # Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        
    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias
        
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Gradient clip for stability
        np.clip(weights_gradient, -1.0, 1.0, out=weights_gradient)
        np.clip(bias_gradient, -1.0, 1.0, out=bias_gradient)
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient
```

### 2. Preprocessing Scale & Normalization Pipeline (`preprocess.py`)
```python
# Scale features using MinMaxScaler
print("[PREPROCESS] Normalizing features using Min-Max Scaling...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for static inference pipelines
joblib.dump(scaler, "scaler.joblib")

# Stratified Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
)
```

### 3. Server-Side Live Stream API Handler (`server.py`)
```python
@app.route("/api/stream")
def get_stream_packet():
    # Randomly select a row from the source CSV file
    df_sample = pd.read_csv(csv_file)
    row = df_sample.sample(n=1).reset_index(drop=True)
    X_row = row.drop(columns=[label_col])
    
    # Preprocess categorical inputs and scale
    df_in = X_row.copy()
    for key in df_in.columns:
        if key in CATEGORICAL_MAPPINGS:
            df_in[key] = df_in[key].map(CATEGORICAL_MAPPINGS[key]).fillna(0).astype(int)
            
    X_scaled = df_in.values / (np.max(df_in.values) + 1e-15)
    pred_prob = model.predict(X_scaled)
    # Stream features, true label, prediction, and target routing cluster
```

---

## 📂 Project Architecture & Modules
```
c:\Users\Hp\Desktop\MP\
├── requirements.txt         # Package dependencies (Flask, Streamlit, Pandas, Scikit-Learn)
├── db_logger.py             # Thread-safe SQLite logger for intrusion events
├── fetch_data.py            # Automated dataset sampler and generators
├── preprocess.py            # Normalization, categorical encoding, class mapping
├── models.py                # Pure NumPy Deep Learning architectures (ANN, CNN, RNN)
├── train.py                 # Fitting models and joblib weights serialization
├── evaluate.py              # Exporting learning curves and confusion matrices
├── main.py                  # Integration controller (Runs sweeps for 24 configurations)
├── app.py                   # Streamlit analytics frontend dashboard
├── server.py                # Flask API server serving predictions and streaming events
├── templates/
│   └── index.html           # SOC web HTML5 structure
└── static/
    ├── style.css            # Dark SOC glassmorphism styles
    └── script.js            # Chart.js connection logic and API polling controllers
```

---

## 🚀 Installation & Execution

### 1. Initialize Virtual Environment & Install Dependencies
```bash
# Create Python Virtual Environment
python -m venv venv

# Activate Virtual Environment (Windows CMD)
venv\Scripts\activate

# Activate Virtual Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
Run the main controller script to download sample datasets, fit all models, and write performance graphs to the workspace:
```bash
python main.py --epochs 3 --batch_size 128
```

### 3. Launch Streamlit Analytics Interface
```bash
streamlit run app.py
```
*Accessible on [http://localhost:8501](http://localhost:8501)*

### 4. Launch Flask SOC Web App Console
```bash
python server.py
```
*Accessible on [http://localhost:5000](http://localhost:5000)*

### 5. Expose Server Publicly (Cloudflare Quick Tunnel)
```bash
cloudflared tunnel --url http://localhost:5000
```

---

## 📊 Multi-Dataset Benchmark Results
Training scores achieved across datasets (1 epoch baseline run):

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
Threat events logged during single tests or streaming simulations are stored in `threat_logs.db` under the `threat_logs` table:

| Column | Type | Description |
| :--- | :--- | :--- |
| `log_id` | INTEGER (PK, AUTOINCREMENT) | Unique identifier for each alert. |
| `timestamp` | TEXT (DEFAULT CURRENT_TIMESTAMP) | Log timestamp of detection. |
| `source_dataset` | TEXT | Running module or simulator data scheme name. |
| `predicted_class` | TEXT | Class label verdict (Normal, DDoS, DoS, Probe, R2L, U2R). |
| `confidence_score` | REAL | Numerical prediction confidence of the neural layer. |
| `execution_time` | REAL | Core network execution latency (in seconds). |
