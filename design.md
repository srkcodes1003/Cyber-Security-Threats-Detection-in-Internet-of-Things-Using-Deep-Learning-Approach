# System Design Document (SDD)

## 1. System Architecture Overview
The deep learning-based IoT Intrusion Detection System (IDS) uses a modular processing pipeline. Raw high-dimensional network logs enter the pipeline, go through mathematical transformations, pass through advanced neural feature-extractors, and produce deterministic threat vectors and real-time alerts.

```
       +---------------------------------------------+
       |   IoT Network Traffic Data Source (CSV)     |
       |  (KDDCup99, NSL-KDD, Bot-IoT, CIC-IDS)      |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |       Data Preprocessing Module            |
       |  - Cleaning & Outlier Removal               |
       |  - Label Encoding (Categorical Data)        |
       |  - Min-Max Feature Normalization            |
       |  - Stratified Train/Test Partitioning       |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |        Deep Learning Model Core             |
       |   +-----------+ +-----------+ +---------+   |
       |   | ANN Engine| | CNN Engine| |  LSTM   |   |
       |   +-----------+ +-----------+ +---------+   |
       |       (TensorFlow / Keras / PyTorch)        |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |       Classification & Inference Layer      |
       |  - Multi-class / Binary Decoders            |
       |  - Normal vs Malicious (DDoS, DoS, etc)     |
       +----------------------+----------------------+
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
+-----------+-----------+           +-----------+-----------+
|    Alert Generation   |           | Performance Analysis  |
|  - Real-time Console  |           |  - Confusion Matrix   |
|    Notification       |           |  - Accuracy/Loss Plots|
|  - Log Storage (DB)   |           |  - Precision / Recall |
+-----------------------+           +-----------------------+
```

---

## 2. Component Breakdown

### 2.1 Data Ingestion & Preprocessing Subsystem
This module converts raw network metrics into normalized tensors suitable for deep neural pathways.
* **Parser Component:** Uses `pandas` to read network flows, mapping column strings to defined matrices.
* **Encoder Component:** Scans structural categorical values (e.g., `protocol_type`, `service`, `flag`) and translates them into uniform numerical indices via Scikit-learn encoding mechanisms.
* **Scaler Component:** Applies feature transformations to prevent high-magnitude metrics (e.g., `num_root`, `duration`) from skewing gradient calculations.
    $$\mathbf{X}_{norm} = \frac{\mathbf{X} - \mathbf{X}_{min}}{\mathbf{X}_{max} - \mathbf{X}_{min}}$$
* **Splitter Component:** Divides datasets into stratified configurations to preserve minority attack frequencies across training ($80\%$) and validation ($20\%$) sets.

### 2.2 Model Training Architecture Subsystem
The engine provides three parallel deep configurations to capture varied data relationships:
1.  **Artificial Neural Network (ANN):**
    * Composed of sequential Fully Connected (Dense) Layers.
    * Uses Rectified Linear Unit (ReLU) activations internally, paired with Dropout regularization layers to mitigate overfitting.
    * Targets global non-linear feature interactions.
2.  **1D Convolutional Neural Network (1D-CNN):**
    * Applies localized 1D kernel convolution windows over the multi-feature rows.
    * Captures structural relationships between adjacent network attributes.
    * Followed by MaxPooling layers to isolate high-activation features.
3.  **Long Short-Term Memory (LSTM):**
    * Processes features as sequenced steps or sequential blocks to handle temporal network flows.
    * Utilizes internal gate states (forget, input, output gates) to discover sustained multi-packet correlations over time.

### 2.3 Classification, Decision & Evaluation Subsystem
* **Output Output Configuration:** Employs a `Softmax` layer for multi-class identification or a `Sigmoid` layer for binary tracking.
* **Loss Optimization Function:** Employs Categorical Cross-Entropy for multi-class target labels:
    $$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$
* **Evaluation Engine:** Computes confusion matrices and updates performance analytics including True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).

### 2.4 Alerting & Persistence Layer (Optional)
* **Console System:** Issues automated warning triggers to administrators when prediction probabilities pass specified malicious thresholds.
* **Data Archiving (Optional):** Stores event logs containing Timestamp, Predicted Class, Confidence Rate, and Target IP information inside an underlying database instance (MySQL or MongoDB).

---

## 3. Data Flow Diagram (DFD) Flow Sequence
1.  **Ingestion:** The system reads background training or test packets from file formats.
2.  **Preprocessing Transformation:** The raw matrix runs through row dropping, text alignment encoding, and Min-Max value normalizations.
3.  **Instantiation:** The system loads selected configuration models (ANN/CNN/LSTM) with compiled parameters.
4.  **Forward/Backward Training Iteration:** The network processes features, evaluates loss variations, and runs optimization sweeps (e.g., Adam Optimizer) to adjust internal layer weights.
5.  **Weight Saving:** The system exports static weight variables to memory disks.
6.  **Inference Testing:** Unseen evaluation lines pass through saved weights to calculate threat output scores and generate operational charts.

---

## 4. Database Schema Structure (Optional Logging Layer)
For implementations that include log archiving, the database tables follow these definitions:

### Table: `threat_logs` (Relational SQL Schema example)
| Attribute Field | Variable Format | Rules & Constraints | Description |
| :--- | :--- | :--- | :--- |
| `log_id` | INT | PRIMARY KEY, AUTO_INCREMENT | Unique identifier for the logging row. |
| `timestamp` | DATETIME | DEFAULT CURRENT_TIMESTAMP | Date and time the threat packet was identified. |
| `source_dataset`| VARCHAR(50) | NOT NULL | Name of the referenced testing dataset (e.g., Bot-IoT). |
| `predicted_class`| VARCHAR(30) | NOT NULL | Categorized result (Normal, DDoS, DoS, Probe, etc.). |
| `confidence_score`| FLOAT | NOT NULL | Probability score generated by the model activation layer. |
| `execution_time`| FLOAT | NOT NULL | Total calculation latency time in seconds. |
