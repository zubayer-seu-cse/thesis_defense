# InSDN TCN – Full Pipeline Documentation

**Task:** Binary network-intrusion detection — Benign (0) vs. Attack (1)  
**Dataset:** InSDN (Software-Defined Networking Intrusion Detection)  
**Model:** Temporal Convolutional Network (TCN) with dilated causal residual blocks  
**Environment:** Google Colab, NVIDIA T4 GPU, TensorFlow 2.19.0

---

## Table of Contents

- [InSDN TCN – Full Pipeline Documentation](#insdn-tcn--full-pipeline-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Dataset Overview](#1-dataset-overview)
  - [2. Preprocessing (`preprocess.py`)](#2-preprocessing-preprocesspy)
    - [Steps](#steps)
    - [Output](#output)
    - [Label Distribution (raw)](#label-distribution-raw)
  - [3. Training Pipeline (`tcn_insdn_colab.ipynb`)](#3-training-pipeline-tcn_insdn_colabipynb)
    - [3.1 Exploratory Data Analysis](#31-exploratory-data-analysis)
    - [3.2 Safety-net Cleaning \& Deduplication](#32-safety-net-cleaning--deduplication)
    - [3.3 Feature Scaling \& PCA](#33-feature-scaling--pca)
    - [3.4 Train / Validation / Test Split](#34-train--validation--test-split)
    - [3.5 TCN Architecture](#35-tcn-architecture)
      - [Residual Block](#residual-block)
      - [Full Model](#full-model)
      - [Parameter Count](#parameter-count)
    - [3.6 Training Configuration \& Callbacks](#36-training-configuration--callbacks)
  - [4. Results](#4-results)
    - [4.1 Training Curves](#41-training-curves)
      - [Loss](#loss)
      - [Accuracy](#accuracy)
      - [AUC](#auc)
      - [Precision \& Recall](#precision--recall)
    - [4.2 Test-set Evaluation](#42-test-set-evaluation)
      - [Confusion Matrix (threshold = 0.5)](#confusion-matrix-threshold--05)
      - [Per-class Metrics](#per-class-metrics)
      - [ROC Curve](#roc-curve)
    - [4.3 Security Metrics](#43-security-metrics)
  - [5. Artifacts](#5-artifacts)

---

## 1. Dataset Overview

The InSDN dataset captures network flow records from an emulated SDN environment with both benign and malicious traffic.

| Source file | Rows |
|---|---|
| `Normal_data.csv` | — |
| `OVS.csv` | — |
| `metasploitable-2.csv` | — |
| **Combined (raw)** | **343,889** |

All three files share the same 84-column schema. The `Label` column contains free-text class names (`Normal`, and various attack type strings).

---

## 2. Preprocessing (`preprocess.py`)

The script consolidates, cleans, and reduces the raw files into a single training-ready CSV.

### Steps

| Step | Action | Result |
|---|---|---|
| 1 | Load & concatenate all three CSVs | 343,889 rows × 84 cols |
| 2 | Strip whitespace from column names and string values | — |
| 3 | Drop identifier / meta columns (`Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Timestamp`) | 84 → 78 cols |
| 4 | Encode binary label: `Normal` → 0, all others → 1 | Benign: 68,424 · Attack: 275,465 |
| 5 | Replace `Inf` / `-Inf` with `NaN`, then drop affected rows | — |
| 6 | Drop remaining `NaN` rows | — |
| 7 | Remove zero-variance (constant) columns | –12 columns |
| 8 | Remove near-constant columns (>99.9 % same value) | –0 columns |
| 9 | Remove highly correlated feature pairs (Pearson \|r\| > 0.98) | –17 columns |
| 10 | Save to `insdn_consolidated.csv` | **343,889 rows × 49 cols** (48 features + Label) |

### Output

```
insdn_consolidated.csv  –  71 MB  –  343,889 rows  ×  49 columns
  Features retained : 48
  Label=0 (Benign) : 68,424   (19.9 %)
  Label=1 (Attack) : 275,465  (80.1 %)
```

### Label Distribution (raw)

| Class | Count | Share |
|---|---|---|
| Benign (0) | 68,424 | 19.9 % |
| Attack (1) | 275,465 | 80.1 % |

---

## 3. Training Pipeline (`tcn_insdn_colab.ipynb`)

### 3.1 Exploratory Data Analysis

After loading `insdn_consolidated.csv` into Colab (shape 343,889 × 49):

- Zero `NaN` cells detected before cleaning
- Zero `Inf` values detected

Class distribution chart (`eda_class_distribution.png`):

![EDA – Class Distribution](eda_class_distribution.png)

---

### 3.2 Safety-net Cleaning & Deduplication

The notebook applies a second-pass cleaning inside Colab (handles any Inf/NaN introduced after load, and removes duplicate rows):

| Metric | Value |
|---|---|
| Inf values replaced | 0 |
| NaN rows removed | 0 |
| **Duplicate rows removed** | **160,058** (343,889 → 182,831) |
| **Final shape after deduplication** | **182,831 × 49** |

> The large duplicate count reflects that the raw OVS and metasploitable-2 captures contain many identical flow records.

Post-dedup label counts:

| Class | Count |
|---|---|
| Benign (0) | 64,114 |
| Attack (1) | 118,717 |

---

### 3.3 Feature Scaling & PCA

| Setting | Value |
|---|---|
| Scaler | `StandardScaler` (zero mean, unit variance) |
| PCA variance threshold | 95 % |
| Input dimensionality | 48 features |
| **PCA output dimensionality** | **24 components** |
| Variance retained | **95.43 %** |

PCA explained-variance curve (`pca_explained_variance.png`):

![PCA – Cumulative Explained Variance](pca_explained_variance.png)

The first 24 principal components capture 95.43 % of the total variance, halving the feature space from 48 to 24 while preserving information.

---

### 3.4 Train / Validation / Test Split

Stratified splits to preserve the class ratio across all three subsets:

| Split | Samples | Attack | Benign |
|---|---|---|---|
| Train (70 %) | 127,981 | 83,101 | 44,880 |
| Validation (10 %) | 18,283 | 11,872 | 6,411 |
| Test (20 %) | 36,567 | 23,744 | 12,823 |

**Class weights** (used during training to handle imbalance):

| Class | Weight |
|---|---|
| Benign (0) | 1.4258 |
| Attack (1) | 0.7700 |

Each split is reshaped to `(N, 24, 1)` — treating the 24 PCA components as a temporal sequence with 1 channel, which is the expected input format for the TCN.

---

### 3.5 TCN Architecture

The model (`TCN_InSDN`) is a stack of dilated causal 1-D residual blocks followed by a dense classification head.

#### Residual Block

Each block applies two sequential layers:

```
Conv1D (causal, dilation_rate=d, filters=64, kernel=3)
  → BatchNormalization
  → ReLU
  → SpatialDropout1D(0.2)
```

A 1×1 projection convolution aligns the residual skip connection when channel dimensions change. The block output is `block(x) + residual`.

#### Full Model

```
Input  (24, 1)
  │
  ├─ ResidualBlock  dilation=1
  ├─ ResidualBlock  dilation=2
  ├─ ResidualBlock  dilation=4
  ├─ ResidualBlock  dilation=8
  ├─ ResidualBlock  dilation=16
  └─ ResidualBlock  dilation=32
        │
  GlobalAveragePooling1D
        │
  Dense(128) → BN → ReLU → Dropout(0.2)
  Dense(64)  → BN → ReLU → Dropout(0.2)
        │
  Dense(1, sigmoid)  ← output
```

The exponentially increasing dilation rates `[1, 2, 4, 8, 16, 32]` give the model a receptive field spanning all 24 time-steps at the deepest layer.

#### Parameter Count

| Category | Parameters |
|---|---|
| Total | 156,737 |
| Trainable | 154,817 |
| Non-trainable (BN) | 1,920 |
| Model size | ~612 KB |

---

### 3.6 Training Configuration & Callbacks

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ |
| Loss | Binary cross-entropy |
| Batch size | 2,048 |
| Max epochs | 100 |
| Class weighting | Balanced (computed) |

**Metrics monitored:** `accuracy`, `AUC`, `Precision`, `Recall`

**Callbacks:**

| Callback | Configuration |
|---|---|
| `EarlyStopping` | monitor=`val_auc`, patience=15, restore best weights |
| `ReduceLROnPlateau` | monitor=`val_loss`, factor=0.5, patience=7, min_lr=1×10⁻⁶ |
| `ModelCheckpoint` | save best model by `val_auc` → `best_tcn_insdn.keras` |
| `CSVLogger` | full epoch log → `training_log.csv` |

---

## 4. Results

### 4.1 Training Curves

Training converged in **30 epochs** (EarlyStopping triggered with patience=15).

#### Loss

![Loss](plot_loss.png)

Both train and validation loss decrease smoothly and converge, with no sign of overfitting.

#### Accuracy

![Accuracy](plot_accuracy.png)

Accuracy climbs steeply in the first 5 epochs and plateaus near **99.9 %** for both splits.

#### AUC

![AUC](plot_auc.png)

AUC reaches effectively **1.0** by epoch 10 and remains stable.

#### Precision & Recall

![Precision & Recall](plot_precision_recall.png)

Both precision and recall stabilise above **99 %** from epoch 3 onward across train and validation.

---

### 4.2 Test-set Evaluation

Evaluated on the held-out test set of **36,567 samples** using the best checkpoint (`best_tcn_insdn.keras`):

#### Confusion Matrix (threshold = 0.5)

![Confusion Matrix](plot_confusion_matrix.png)

|  | Predicted Benign | Predicted Attack |
|---|---|---|
| **True Benign** | TP = 12,776 | FP = 47 |
| **True Attack** | FN = 7 | TP = 23,737 |

#### Per-class Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Benign (0) | 99.94 % | 99.63 % | 99.79 % | 12,823 |
| Attack (1) | 99.80 % | 99.97 % | 99.89 % | 23,744 |
| **Weighted avg** | **99.85 %** | **99.85 %** | **99.85 %** | **36,567** |

| Metric | Value |
|---|---|
| **Overall Accuracy** | **99.85 %** |
| **ROC-AUC** | **0.9999** |

#### ROC Curve

![ROC Curve](plot_roc_curve.png)

---

### 4.3 Security Metrics

| Metric | Value |
|---|---|
| True Positives (detected attacks) | 23,737 |
| True Negatives (correct benign) | 12,776 |
| False Positives (false alarms) | 47 |
| False Negatives (missed attacks) | 7 |
| **Detection Rate (Recall)** | **99.97 %** |
| **Specificity** | **99.63 %** |
| **False Alarm Rate** | **0.37 %** |

The model missed only **7 attack flows** out of 23,744 in the test set, while generating only **47 false alarms** out of 12,823 benign flows.

---

## 5. Artifacts

| File | Description |
|---|---|
| `preprocess.py` | Preprocessing script — consolidates & cleans the three raw CSVs |
| `insdn_consolidated.csv` | Cleaned dataset (71 MB, 343,889 rows × 49 cols) |
| `tcn_insdn_colab.ipynb` | Training notebook (all 9 sections, single code cell) |
| `tcn_insdn_colab_res.ipynb` | Executed notebook with all outputs |
| `best_tcn_insdn.keras` | Best model checkpoint (saved by `val_auc`) |
| `training_log.csv` | Per-epoch training metrics log |
| `eda_class_distribution.png` | Class bar + pie chart |
| `pca_explained_variance.png` | PCA cumulative explained variance curve |
| `plot_loss.png` | Train / val loss curves |
| `plot_accuracy.png` | Train / val accuracy curves |
| `plot_auc.png` | Train / val AUC curves |
| `plot_precision_recall.png` | Train / val precision & recall curves |
| `plot_confusion_matrix.png` | Confusion matrix on test set |
| `plot_roc_curve.png` | ROC curve on test set |

---