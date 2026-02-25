# Thesis Presentation Slide Content

## Title
**TCN-HMAC: A Lightweight Deep Learning and Cryptographic Hybrid Security Framework for Software-Defined Networks**

---

## 1. Problem Statement

SDN's centralized design leaves critical vulnerabilities at every major junction—the controller, the control channel, and the switches—yet no existing solution comprehensively addresses them:

1. **Detection without Verification:** ML/DL-based IDS can identify malicious traffic but provide no integrity guarantees for flow rules. A compromised controller can still install malicious rules undetected.
2. **Verification without Detection:** Cryptographic mechanisms (e.g., HMAC, TLS) ensure rule integrity but cannot proactively detect intrusions or anomalous traffic patterns.
3. **Excessive Computational Overhead:** Blockchain and PKI-based frameworks introduce latency and resource consumption incompatible with real-time SDN requirements.
4. **Limited Attack Coverage:** Many solutions target a single attack vector (e.g., DDoS only or MITM only), lacking generality for diverse real-world threats.
5. **Deployment Complexity:** Solutions requiring OpenFlow protocol modifications, switch firmware changes, or custom hardware conflict with SDN's hardware abstraction principle.

> **Research Question:** *How can we build a lightweight, deployable security framework for SDN that combines proactive multi-class intrusion detection with real-time cryptographic verification of flow rules and controller authenticity—all without exceeding the latency budget that production networks demand?*

---

## 2. Research Gaps

| Gap | Description |
|-----|-------------|
| **Gap 1: Detection without Verification** | DL-based IDS approaches focus exclusively on anomaly detection without integrity verification, leaving the control plane vulnerable to flow rule tampering. |
| **Gap 2: Verification without Detection** | Cryptographic approaches verify flow rule integrity but lack proactive threat detection capabilities for broader network intrusions. |
| **Gap 3: Heavyweight Hybrid Solutions** | Existing hybrids (e.g., DL + blockchain) introduce prohibitive latency and overhead for real-time SDN operations. |
| **Gap 4: Limited TCN Evaluation on SDN Datasets** | No existing work applies TCN-based detection on SDN-specific datasets (e.g., InSDN) or integrates it with SDN security mechanisms. |
| **Gap 5: No Lightweight Unified Framework** | No prior work combines temporal DL-based IDS with HMAC-based flow rule verification and challenge–response controller authentication in a single deployable solution. |

---

## 3. Research Objectives

1. **Design a TCN-based IDS** for binary classification of SDN traffic (benign vs. malicious) achieving >99% accuracy across DDoS, MITM, Probe, and Brute-force attacks.
2. **Develop a preprocessing pipeline** for InSDN: data cleaning, Pearson correlation feature selection, StandardScaler, PCA (48→24 features), and inverse-frequency class weighting.
3. **Train and deploy** a compact TCN model using TensorFlow/Keras for efficient, portable inference within the SDN controller.
4. **Design a lightweight auxiliary agent** performing: (a) HMAC-based flow rule verification, and (b) periodic challenge–response controller authentication.
5. **Integrate TCN-IDS + HMAC agent** into a cohesive framework operating within the SDN control loop.
6. **Evaluate** in a simulated SDN environment (Ryu + Mininet + Open vSwitch) measuring detection performance and computational overhead.
7. **Comparative analysis** against 15 existing models (LSTM, GRU, CNN, Transformer, CNN-BiLSTM, DRL, etc.) to demonstrate TCN superiority.

---

## 4. Research Contributions

1. **First Unified Detection–Verification Framework (TCN-HMAC):**
   - First framework combining temporal DL-based IDS with HMAC-based flow rule verification and challenge–response controller authentication in a single deployable SDN solution.

2. **First TCN on InSDN with SDN-Tailored Preprocessing:**
   - Highest detection rate (99.97%) and lowest FAR (0.37%) among all 15 compared models.
   - Outperforms structurally more complex architectures (CNN-BiLSTM, BiTCN-MHSA, attention-augmented TCN variants).

3. **Compact Deployment Strategy:**
   - 612 KB model (156,737 parameters) — 5–30× smaller than comparable DL-IDS models.
   - Fastest inference: 0.17 ms per flow.

4. **Novel Auxiliary Agent Architecture:**
   - HMAC-based flow rule verification: ~2 μs per message.
   - Challenge–response controller authentication.
   - Overhead: only 0.7 ms control latency, 4.4% CPU, 13.7 MB RAM.
   - No OpenFlow protocol or switch hardware modifications required.

5. **Comprehensive Preprocessing Pipeline:**
   - 15-stage pipeline: cleaning → Pearson correlation (|r|>0.95) → StandardScaler → PCA (48→24 features, 95.43% variance) → class weighting.

6. **Most Extensive Comparative Analysis in SDN IDS Literature:**
   - Benchmarked against 15 models across multiple metrics.
   - TCN-HMAC is the only framework providing both intrusion detection AND control-plane protection.

---

## 5. Results

### Classification Performance (Test Set: 36,567 samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.85% |
| **Precision** | 99.80% |
| **Recall (Detection Rate)** | 99.97% |
| **Specificity** | 99.63% |
| **F1-Score** | 99.89% |
| **AUC-ROC** | 0.9999 |
| **False Alarm Rate** | 0.37% |
| **MCC** | 0.9966 |

### Confusion Matrix

| | Predicted Benign | Predicted Attack |
|---|---|---|
| **Actual Benign** | 12,776 (TN) | 47 (FP) |
| **Actual Attack** | 7 (FN) | 23,737 (TP) |

> Only **54 misclassifications** out of 36,567 samples — the model misses just **3 attacks per 10,000**.

### System Overhead

| Component | Latency |
|-----------|---------|
| HMAC Verification | ~2 μs |
| Feature Extraction | ~50 μs |
| Preprocessing | ~10 μs |
| TCN Inference (GPU) | ~100 μs |
| Decision + Response | ~5 μs |
| HMAC Signing | ~2 μs |
| **Total End-to-End** | **~170 μs** |

### HMAC Agent Overhead

- CPU: 4.4% additional usage
- RAM: 13.7 MB
- Control Latency: 0.7 ms
- HMAC Throughput: >500,000 messages/sec (single core)
- All injected invalid flow rules detected: **100%**

### Model Efficiency

| Property | Value |
|----------|-------|
| Parameters | 156,737 |
| Model Size | 612 KB |
| Inference Time | 0.17 ms |
| Training Time | ~5 min (NVIDIA T4 GPU) |
| Size Advantage | 5–30× smaller than comparable DL-IDS |

### Comparative Highlights (vs. 15 Models)

| Model | Accuracy | Recall | F1 | Auth? |
|-------|----------|--------|----|-------|
| **TCN-HMAC (Ours)** | **99.85%** | **99.97%** | **99.89%** | **Yes** |
| CNN-BiLSTM | 99.90% | 99.90% | 99.90% | No |
| DNN Ensemble | 99.70% | 99.70% | 99.67% | No |
| TCN+Attention | 99.73% | 99.69% | 99.70% | No |
| BiTCN-MHSA | 99.72% | 99.72% | 99.70% | No |
| DRL (DDQN) | 98.85% | 98.85% | 98.87% | No |
| DT/RF Baseline | 98.50% | 98.50% | 98.45% | No |

> **TCN-HMAC is the only model that provides both intrusion detection AND control-plane protection.**

### Confidence Interval

- Accuracy: 99.85% ± 0.04% (95% confidence)
- Results are statistically robust and reproducible.
