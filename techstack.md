# Technology Stack Document (TSD)

This document details the software libraries, framework runtimes, hardware configurations, and execution tools specified for the IoT deep learning intrusion detection framework.

## 1. Core Language & Runtimes
* **Programming Language:** Python 3.8 or above
    * *Justification:* Python provides a mature ecosystem for data science and deep learning, with native support for tensor computations and data preprocessing pipelines.
* **Operating Systems:** Windows 10/11, Ubuntu 20.04 or later, or macOS
    * *Justification:* Universal support ensures cross-platform compatibility for both development and testing across diverse host workstations.

---

## 2. Frameworks & Deep Learning Acceleration
* **TensorFlow 2.x / Keras:**
    * *Role:* Primary runtime engine for creating and managing deep neural layers.
    * *Justification:* Keras provides a clean API for compiling sequential layers (ANN, 1D-CNN) and handling complex recurrent configurations (LSTM), while ensuring reproducible research designs.
* **PyTorch (Alternative Option):**
    * *Role:* Dynamic graph execution environment for building advanced tensor calculations.
* **CUDA Toolkit & cuDNN (Optional but Recommended):**
    * *Role:* Provides parallel computing acceleration on NVIDIA GPUs.
    * *Justification:* Significantly decreases training duration for multi-layered recurrent structures (LSTMs) and convolutional models.

---

## 3. Data Engineering & Analysis Libraries
* **NumPy (Numerical Python):**
    * *Role:* High-performance multidimensional matrix math structures.
    * *Justification:* Provides quick, vectorized tensor manipulations behind deep learning libraries.
* **Pandas:**
    * *Role:* High-level data transformation structures (DataFrames).
    * *Justification:* Simplifies loading large CSV benchmark files, handling missing fields, filtering columns, and manipulating tabular rows.
* **Scikit-Learn:**
    * *Role:* Machine learning utility pipeline toolkit.
    * *Justification:* Provides reliable modules for standard preprocessing operations, such as `MinMaxScaler`, `LabelEncoder`, and `train_test_split`.

---

## 4. Visual Reporting & Analytics Libraries
* **Matplotlib:**
    * *Role:* Low-level chart generation engine.
    * *Justification:* Used to render technical plots like training/validation loss arcs and model accuracy curves.
* **Seaborn:**
    * *Role:* Statistical visualization engine built on top of Matplotlib.
    * *Justification:* Generates polished, annotated confusion matrix heatmaps to analyze true positive/false alarm distributions.

---

## 5. Development Tooling & Configuration Management
* **Integrated Development Environments (IDEs):** VS Code, PyCharm, or Jupyter Notebooks.
    * *Justification:* Jupyter Notebooks enable iterative step-by-step pipeline verification and visualization, while VS Code/PyCharm support structured, modular code compilation.
* **Version Control:** Git (Optional)
    * *Justification:* Enables code history tracking and collaboration across the development team.
* **Database Engines (Optional Persistence):** MySQL (Relational storage) or MongoDB (NoSQL JSON format).
    * *Justification:* Provides permanent tracking of detection run logs and alert histories.

---

## 6. Workstation Hardware Specifications

| Component Element | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 (8th Generation or above) / AMD Ryzen 5 or higher | Intel Core i7 / AMD Ryzen 7 or above |
| **System Memory (RAM)**| 8 GB capacity | 16 GB or greater (for caching large-scale datasets) |
| **Storage Capacity** | 256 GB Solid State Drive (SSD) | 512 GB SSD (with $\ge$ 20 GB available workspace allocation) |
| **Graphics Processing (GPU)**| Not required (CPU-only execution) | Dedicated NVIDIA GPU with CUDA compatibility (e.g., RTX series) |
| **Network Infrastructure** | Stable baseline connectivity | High-speed link for downloading datasets and dependencies |
