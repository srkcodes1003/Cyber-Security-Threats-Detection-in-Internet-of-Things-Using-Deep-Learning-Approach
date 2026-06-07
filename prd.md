# Product Requirements Document (PRD)

## 1. Document Overview
* **Project Title:** Cyber Security Threats Detection in Internet of Things Using Deep Learning Approach
* **Batch Number:** DS11
* **Project Team Members:**
    * Kethavath Sindhu (Roll No: 1608-23-750-063)
    * Sai Ramakrishna Doddapaneni (Roll No: 1608-23-750-004)
    * Kota Rakesh (Roll No: 1608-23-750-057)
* **Project Guide:** Ms. Rama Lakshmi, Assistant Professor, CSE Department
* **Version:** 1.0
* **Date:** June 2026

---

## 2. Introduction & Background
The Internet of Things (IoT) has rapidly integrated into modern infrastructure, spanning healthcare, smart homes, industrial automation, and transport networks. Despite its benefits, this ubiquitous connectivity exposes ecosystems to critical cyber vulnerabilities. IoT nodes typically feature heterogeneous architectures, limited battery and processing capabilities, and lack standardized built-in security protocols. Consequently, they serve as prime targets for sophisticated cyber-attacks. Traditional security tools are unequipped to manage this scale, requiring automated, intelligent, and real-time security intrusion detection.

---

## 3. Problem Statement
The expansion of IoT environments introduces major security liabilities that legacy Intrusion Detection Systems (IDS) fail to mitigate effectively:
1.  **Inadequacy of Traditional Systems:** Conventional systems rely on signature-based or rule-based mechanisms, depending on pre-configured attack signatures and manual feature engineering. This restricts their capacity to identify novel, zero-day, or rapidly mutating attack variants.
2.  **High False Alarm Rates:** Dynamic IoT network traffic produces high false positive and false negative rates in traditional models, leading to operational fatigue.
3.  **High-Dimensional Data Handling:** IoT networks produce large volumes of multi-structured, high-dimensional traffic data that traditional machine learning algorithms struggle to process efficiently at scale.
4.  **Resource Constraints:** Limited computing and memory assets on edge IoT devices hinder the direct execution of complex local protection systems, mandating a scalable, high-throughput backend architecture.

---

## 4. Project Objectives & Scope
The main objective of this project is to develop and validate a robust deep learning-based cybersecurity threat detection framework tailored for high-throughput IoT networks.

### Specific Goals:
* **Automated Feature Extraction:** Eliminate manual feature engineering by utilizing deep architectures that learn representations directly from raw preprocessed network traffic data.
* **Multi-Model Analysis:** Design, implement, and benchmark three distinct deep learning architectures: Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), and Long Short-Term Memory (LSTM) networks.
* **Traffic Classification:** Preprocess and categorize incoming network traffic into "Normal" or "Malicious", with granular sub-classification of specific attack types (DDoS, DoS, Probe, R2L, U2R).
* **Rigorous Metric Evaluation:** Authenticate system performance using multi-dimensional metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
* **Benchmark Validation:** Train and evaluate the models using industry-standard large-scale benchmark datasets (such as KDDCup99, NSL-KDD, Bot-IoT, and CIC-IDS).

---

## 5. Functional Requirements
### 5.1 Data Ingestion & Preprocessing Module
* **FR-1.1:** The system must accept industry benchmark IoT network traffic datasets in CSV format.
* **FR-1.2:** The system must execute automated data cleaning, handling missing fields, inf values, and duplicate records.
* **FR-1.3:** The system must perform categorical feature transformations using Label Encoding or One-Hot Encoding for non-numeric fields.
* **FR-1.4:** The system must implement feature scaling via Min-Max Normalization or Standard Scaling to stabilize deep learning gradient descents.
* **FR-1.5:** The system must partition data into stratified Train/Test splits to guarantee un-biased model testing.

### 5.2 Deep Learning Model Engine
* **FR-2.1 (ANN):** Implement a Multi-Layer Perceptron (MLP/ANN) to learn dense non-linear parameter interactions.
* **FR-2.2 (CNN):** Implement a 1D Convolutional Neural Network (1D-CNN) to capture spatial hierarchies and localized correlations from sequence features.
* **FR-2.3 (LSTM):** Implement a Long Short-Term Memory network to evaluate temporal patterns and long-term sequential dependencies within network packets.
* **FR-2.4:** The framework must provide a configuration interface for tuning model hyper-parameters (e.g., learning rates, batch size, epochs, dropout rates).

### 5.3 Inference & Threat Classification
* **FR-3.1:** The system must classify binary traffic status (Normal vs. Malicious).
* **FR-3.2:** For malicious evaluations, the framework must identify the attack category (e.g., DoS, DDoS, Probe, Remote-to-Local [R2L], User-to-Root [U2R]).
* **FR-3.3:** The core engine must execute real-time simulation runs by feeding batch lines of unseen test data into the saved model weights.

### 5.4 Reporting & Alert Generation
* **FR-4.1:** The system must output clear textual or graphical alert flags immediately upon detecting malicious packets.
* **FR-4.2:** The system must generate a visual performance report including training logs, loss curves, accuracy plots, and confusion matrices.
* **FR-4.3 (Optional):** Maintain an administrative logger system to write threat event logs into an underlying database (MySQL/MongoDB).

---

## 6. Non-Functional Requirements
### 6.1 Performance & Latency
* **NFR-1.1:** Inference time for classified validation batches must be minimized to support near-real-time alert responses.
* **NFR-1.2:** Data pipeline preparation must handle files containing over 100,000 traffic records without memory exhaustion.

### 6.2 Accuracy & Reliability
* **NFR-2.1:** The deep learning architectures aim to minimize false alarms (False Positives), achieving precision benchmarks comparable with contemporary literature review baselines (~95%+ accuracy targets depending on the specific dataset).
* **NFR-2.2:** The model evaluation must be resilient against data imbalance, measured via F1-Score and True Negative Rates.

### 6.3 Scalability & Maintainability
* **NFR-3.1:** The framework must use a modular code design to support adding new dataset configurations or alternative neural structures without refactoring the pipeline.
* **NFR-3.2:** The model storage layer must save trained model weight signatures (`.h5` or `.keras` or `.pt`) for independent offline deployment.

### 6.4 Usability
* **NFR-4.1:** The model training and verification sequences must be executable through clear interactive scripts or structured Jupyter Notebooks with inline documentation.
