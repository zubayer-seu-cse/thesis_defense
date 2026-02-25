#!/usr/bin/env python3
"""Generate thesis defense presentation for TCN-HMAC framework."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

# ── helpers ──────────────────────────────────────────────────────────
def add_title_slide(prs, title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(2))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(16)
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(12)
    return slide

def add_content_slide(prs, title, bullet_points, font_size=16):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Title bar
    txBox = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9.4), Inches(0.7))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 51, 102)
    # Content
    txBox2 = slide.shapes.add_textbox(Inches(0.4), Inches(1.0), Inches(9.2), Inches(6.0))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, bp in enumerate(bullet_points):
        if i == 0:
            p = tf2.paragraphs[0]
        else:
            p = tf2.add_paragraph()
        # Support indentation with tuple (level, text)
        if isinstance(bp, tuple):
            level, text = bp
            p.level = level
            p.text = text
        else:
            p.text = bp
            p.level = 0
        p.font.size = Pt(font_size)
        p.space_after = Pt(4)
    return slide

def add_table_slide(prs, title, headers, rows, col_widths=None, font_size=11):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(9.4), Inches(0.6))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 51, 102)

    num_rows = len(rows) + 1
    num_cols = len(headers)
    left = Inches(0.3)
    top = Inches(0.75)
    width = Inches(9.4)
    height = Inches(0.3) * num_rows

    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table

    # Set column widths if provided
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    # Header row
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for para in cell.text_frame.paragraphs:
            para.font.size = Pt(font_size)
            para.font.bold = True
            para.font.color.rgb = RGBColor(255, 255, 255)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0, 51, 102)

    # Data rows
    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(font_size)
            # Alternate row colors
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(230, 240, 250)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255)

    return slide


# =====================================================================
# BUILD PRESENTATION
# =====================================================================
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# ── SLIDE 1: TITLE ──────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
txBox = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(9), Inches(2.5))
tf = txBox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "TCN-HMAC: A Lightweight Deep Learning and Cryptographic\nHybrid Security Framework for Software-Defined Networks"
p.font.size = Pt(26)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(0, 51, 102)

info = [
    ("Md. Mahamudul Hasan Zubayer", Pt(18), True),
    ("ID: 2021110000133", Pt(14), False),
    ("", Pt(10), False),
    ("Supervisor: Md. Maruf Hassan", Pt(16), False),
    ("", Pt(10), False),
    ("Department of Computer Science & Engineering", Pt(14), False),
    ("Southeast University, Dhaka, Bangladesh", Pt(14), False),
    ("2025", Pt(14), False),
]
txBox2 = slide.shapes.add_textbox(Inches(0.5), Inches(3.8), Inches(9), Inches(3.5))
tf2 = txBox2.text_frame
tf2.word_wrap = True
for i, (text, size, bold) in enumerate(info):
    if i == 0:
        p = tf2.paragraphs[0]
    else:
        p = tf2.add_paragraph()
    p.text = text
    p.font.size = size
    p.font.bold = bold
    p.alignment = PP_ALIGN.CENTER

# ── SLIDE 2: PROBLEM STATEMENT ──────────────────────────────────────
add_content_slide(prs, "Problem Statement", [
    "SDN centralizes control into a single programmable controller \u2014 creating critical vulnerabilities at the controller, control channel, and switches.",
    "",
    "Key Problems in Existing SDN Security Solutions:",
    "",
    (1, "Detection without Verification \u2014 ML/DL-based IDS can detect malicious traffic but provide no integrity guarantees for flow rules. A compromised controller can still install malicious rules undetected. [Ataa et al. 2024; Said et al. 2023]"),
    (1, "Verification without Detection \u2014 Cryptographic schemes (e.g., HMAC, TLS) verify rule integrity but cannot detect intrusions or anomalous traffic patterns that precede attacks. [Ahmed et al. 2023; Pradeep et al. 2023]"),
    (1, "Excessive Computational Overhead \u2014 Blockchain and PKI-based frameworks introduce seconds-level latency per transaction, incompatible with real-time SDN. [Song et al. 2023]"),
    (1, "Limited Attack Coverage \u2014 Many solutions address only one attack type (e.g., DDoS-only or MITM-only). [Khan et al. 2021; Malik et al. 2021]"),
    (1, "Deployment Complexity \u2014 Solutions requiring OpenFlow protocol modifications or custom switch firmware are impractical for production. [Buruaga et al. 2025]"),
    "",
    "Research Question: How can we build a lightweight, deployable security framework for SDN that combines proactive multi-class intrusion detection with real-time cryptographic verification \u2014 without exceeding production latency budgets?",
], font_size=14)

# ── SLIDE 3-4: LITERATURE REVIEW TABLE (split across 2 slides) ─────
lit_rows_1 = [
    ("Pradeep et al. [2023]", "EnsureS \u2014 batch hash flow rule verification", "2023", "Flow rule integrity only; no detection"),
    ("Ahmed et al. [2023]", "Modular HMAC for flow_mod verification", "2023", "Integrity only; no anomaly detection"),
    ("Buruaga et al. [2025]", "Quantum-safe TLS for SDN", "2025", "High overhead; not lightweight"),
    ("Zhou et al. [2022]", "SecureMatch \u2014 encrypted rule matching", "2022", "Requires switch hardware changes"),
    ("Song et al. [2023]", "Blockchain (IS2N) for intent-driven SDN", "2023", "Consensus latency (seconds per tx)"),
    ("Rahman et al. [2022]", "Blockchain survey for SDN security", "2022", "Storage/scalability overhead"),
    ("Poorazad et al. [2023]", "Blockchain + DL for IoT-SDN", "2023", "Very high computational overhead"),
    ("Sharma & Tyagi [2023]", "Lightweight ML-IDS for MITM/DoS", "2023", "No flow rule verification; 98.12% acc"),
    ("Ayad et al. [2025]", "ML classifiers (RF, XGBoost) for SDN-IDS", "2025", "No temporal modeling; no verification"),
    ("Said et al. [2023]", "CNN-BiLSTM on InSDN", "2023", "Bidirectional \u2014 not real-time; no verification"),
    ("Shihab et al. [2025]", "CNN-LSTM + SMOTE", "2025", "Hybrid overhead; synthetic data artifacts"),
]

lit_rows_2 = [
    ("Ataa et al. [2024]", "DNN Ensemble on InSDN", "2024", "Detection only; no deployment analysis"),
    ("Basfar [2025]", "Incremental LSTM Ensemble", "2025", "Sequential processing; high overhead"),
    ("Kanimozhi et al. [2025]", "DRL (DDQN) on InSDN", "2025", "Training instability; 98.85% acc"),
    ("Wang et al. [2025]", "TCN on InSDN (packet-length input)", "2025", "97.00% acc; 2.1ms inference; no verification"),
    ("Lopes et al. [2023]", "TCN on CIC-IDS-2017", "2023", "Not SDN-specific; no integrity mechanism"),
    ("Li & Li [2025]", "TCN-SE (squeeze-and-excitation)", "2025", "No SDN dataset; no verification"),
    ("Deng et al. [2024]", "BiTCN-MHSA", "2024", "Bidirectional \u2014 offline only; complex"),
    ("Khan et al. [2021]", "MITM-Defender for SDN", "2021", "Narrow focus \u2014 MITM only"),
    ("Malik & Habib [2021]", "Lightweight DoS agents at switches", "2021", "DoS-only; no general IDS"),
    ("Liang et al. [2021]", "Review of SDN-IDS vs. rule injection", "2021", "Identifies gap: detection \u2260 verification"),
    ("Elsayed et al. [2020]", "InSDN dataset + DT/RF baselines", "2020", "98.50% acc; foundational benchmark"),
]

add_table_slide(prs, "Literature Review (1/2)",
    ["Reference", "Approach", "Year", "Addressed Problem / Limitation"],
    lit_rows_1,
    col_widths=[1.8, 2.8, 0.6, 4.2],
    font_size=10)

add_table_slide(prs, "Literature Review (2/2)",
    ["Reference", "Approach", "Year", "Addressed Problem / Limitation"],
    lit_rows_2,
    col_widths=[1.8, 2.8, 0.6, 4.2],
    font_size=10)

# ── SLIDE 5: RESEARCH GAPS ─────────────────────────────────────────
add_content_slide(prs, "Research Gaps", [
    "Gap 1: Detection without Verification",
    (1, "DL-based IDS focuses on anomaly detection without integrity verification. Control plane remains vulnerable to flow rule tampering that bypasses the detection layer."),
    "",
    "Gap 2: Verification without Detection",
    (1, "Cryptographic schemes (HMAC, TLS) verify integrity but lack proactive threat detection \u2014 cannot spot DDoS, probes, or brute-force attacks."),
    "",
    "Gap 3: Heavyweight Hybrid Solutions",
    (1, "Existing hybrids (DL + blockchain) introduce prohibitive latency (seconds per transaction) and computational overhead for real-time SDN."),
    "",
    "Gap 4: Limited TCN Evaluation on SDN Datasets",
    (1, "TCN-based IDS shows promise but has not been evaluated on SDN-specific datasets (InSDN) or integrated with SDN security mechanisms."),
    "",
    "Gap 5: No Comprehensive Lightweight Hybrid Framework",
    (1, "No existing work combines temporal DL-based IDS + HMAC flow rule verification + challenge-response controller authentication in a unified, lightweight, deployable solution."),
], font_size=14)

# ── SLIDE 6: RESEARCH OBJECTIVES ────────────────────────────────────
add_content_slide(prs, "Research Objectives", [
    "RO1: Design a TCN-based IDS model for binary classification of SDN traffic (DDoS, MITM, Probe, Brute-force) achieving > 99% accuracy. [\u2192 Gaps 1, 4]",
    "",
    "RO2: Develop a 15-stage preprocessing pipeline for InSDN \u2014 Pearson correlation feature selection, PCA (48\u219224 features), StandardScaler, class weighting. [\u2192 Gap 4]",
    "",
    "RO3: Train and deploy a compact TCN model (TensorFlow/Keras) suitable for resource-constrained SDN controllers. [\u2192 Gap 5]",
    "",
    "RO4: Design a lightweight HMAC-based auxiliary agent for: (a) flow rule integrity verification via shadow table, (b) periodic challenge-response controller authentication. [\u2192 Gaps 2, 5]",
    "",
    "RO5: Integrate TCN-IDS + HMAC agent into a cohesive framework operating within the SDN control loop. [\u2192 Gap 5]",
    "",
    "RO6: Evaluate framework in simulated SDN environment (Ryu + Mininet + OVS) and measure detection performance + computational overhead. [\u2192 Gaps 3, 5]",
    "",
    "RO7: Conduct comprehensive comparative analysis against 15+ existing models spanning ML, DL, TCN variants, and cryptographic frameworks. [\u2192 Gaps 1\u20135]",
], font_size=13)

# ── SLIDE 7: RESEARCH CONTRIBUTIONS ────────────────────────────────
add_content_slide(prs, "Research Contributions (Top 2)", [
    "Contribution 1: A Novel Hybrid Security Framework (TCN-HMAC)",
    "",
    (1, "First framework to combine Temporal Convolutional Network-based intrusion detection with HMAC-based flow rule integrity verification and challenge-response controller authentication in a unified, deployable SDN security solution."),
    (1, "Bridges the detection-verification divide \u2014 TCN handles external threats (DDoS, probes, brute-force); HMAC agent handles internal threats (flow rule tampering, controller spoofing, replay attacks)."),
    (1, "Runs entirely as a software controller application + auxiliary agent \u2014 no OpenFlow protocol changes or custom switch firmware required."),
    "",
    "",
    "Contribution 2: First TCN Application on InSDN with State-of-the-Art Results",
    "",
    (1, "First study to apply a TCN specifically to the InSDN dataset with SDN-tailored preprocessing."),
    (1, "Achieves: 99.85% accuracy, 99.80% precision, 99.97% recall (detection rate), 99.89% F1-score, 0.9999 AUC-ROC."),
    (1, "Highest detection rate (99.97%) and lowest false alarm rate (0.37%) among all 16 compared approaches."),
    (1, "Outperforms structurally more complex architectures (CNN-BiLSTM, BiTCN-MHSA, TCN+Attention) using a clean, compact design."),
], font_size=13)

# ── SLIDE 8: SYSTEM ARCHITECTURE ────────────────────────────────────
add_content_slide(prs, "System Architecture", [
    "Dual-Boundary Defense: TCN-IDS (data plane monitoring) + HMAC Auxiliary Agent (control plane protection)",
    "",
    "TCN-IDS Module (Detection Boundary):",
    (1, "6 dilated residual blocks (dilation: 1, 2, 4, 8, 16, 32) with 64 filters, kernel size 3"),
    (1, "Classification head: GlobalAvgPooling \u2192 Dense(128) \u2192 Dense(64) \u2192 Dense(1, sigmoid)"),
    (1, "Total: 156,737 parameters | Model size: 612 KB | Inference: 0.17 ms"),
    (1, "Input: 24 PCA-transformed features \u2192 Output: benign (0) / attack (1)"),
    "",
    "HMAC Auxiliary Agent (Integrity Boundary):",
    (1, "Key Establishment: Secure key exchange between controller and agent during initialization"),
    (1, "Message Authentication: HMAC-SHA256 tag computed over flow rule fields + sequence number + timestamp \u2192 appended to every flow_mod message \u2192 verified at switch side"),
    (1, "Shadow Table Verification: Agent maintains independent copy of expected flow rules; periodic audit detects unauthorized local modifications"),
    (1, "Challenge-Response Authentication: Periodic nonce-based verification confirms controller identity and detects compromise"),
    "",
    "Overhead: HMAC ~2 \u03bcs/msg | +0.7 ms control latency | +4.4% CPU | +13.7 MB RAM",
], font_size=13)

# ── SLIDE 9: DATASET ────────────────────────────────────────────────
add_content_slide(prs, "Dataset: InSDN", [
    "InSDN (Elsayed et al., 2020) \u2014 purpose-built for SDN intrusion detection research",
    "",
    "Overview:",
    (1, "343,889 raw samples from SDN testbed (Mininet + Ryu + OVS)"),
    (1, "84 flow-level features (CICFlowMeter) covering packet counts, byte volumes, IAT statistics, flag distributions, flow duration"),
    (1, "4 attack categories: DDoS, MITM, Probe, Brute-force + Normal traffic"),
    "",
    "Why InSDN?",
    (1, "SDN-specific: Generated in a real SDN testbed \u2014 reflects actual SDN traffic patterns, unlike generic datasets (NSL-KDD, CIC-IDS-2017)"),
    (1, "Comprehensive attack coverage: Covers the four most prevalent SDN threat classes"),
    (1, "Widely adopted SDN IDS benchmark \u2014 enables direct comparison with prior work (Said et al., Ataa et al., Kanimozhi et al., Wang et al.)"),
    (1, "Well-documented feature schema with flow-level statistics suitable for temporal modeling"),
    "",
    "Class Distribution (after preprocessing):",
    (1, "Normal: 68,424 (37.4%) | Attack: 114,407 (62.6%) \u2014 addressed via inverse-frequency class weighting"),
], font_size=14)

# ── SLIDE 10: EXPERIMENTAL SETUP ────────────────────────────────────
add_content_slide(prs, "Experimental Setup", [
    "Preprocessing Pipeline (15 stages):",
    (1, "Data cleaning \u2192 duplicate removal (343,889 \u2192 182,831) \u2192 Pearson correlation thresholding (|r| > 0.95) \u2192 84 \u2192 48 features"),
    (1, "StandardScaler normalization \u2192 Label encoding \u2192 PCA dimensionality reduction (48 \u2192 24, retaining 95.43% variance)"),
    (1, "Inverse-frequency class weighting: Normal = 1.4258, Attack = 0.7700"),
    "",
    "Data Split: 70% train (127,981) / 10% validation (18,283) / 20% test (36,567)",
    "",
    "Training Configuration:",
    (1, "Framework: TensorFlow 2.19.0 / Keras"),
    (1, "Hardware: Google Colab \u2014 NVIDIA Tesla T4 GPU (16 GB), 12.7 GB RAM"),
    (1, "Optimizer: Adam (lr = 0.001, \u03b2\u2081 = 0.9, \u03b2\u2082 = 0.999)"),
    (1, "Loss: Binary Cross-Entropy with class weights"),
    (1, "Batch size: 2,048 | Epochs: 30 (EarlyStopping patience = 10)"),
    (1, "Training time: ~5 minutes total (~10 sec/epoch)"),
    "",
    "SDN Simulation: Mininet + Ryu controller + Open vSwitch (OVS)",
], font_size=13)

# ── SLIDE 11: RESULTS \u2014 PERFORMANCE METRICS ─────────────────────────
perf_rows = [
    ("Accuracy", "99.85%"),
    ("Precision", "99.80%"),
    ("Recall (Detection Rate)", "99.97%"),
    ("F1-Score", "99.89%"),
    ("AUC-ROC", "0.9999"),
    ("False Alarm Rate (FAR)", "0.37%"),
    ("Matthews Correlation Coeff.", "0.9966"),
    ("Specificity", "99.63%"),
]
slide = add_table_slide(prs, "Results: TCN-IDS Performance",
    ["Metric", "Value"],
    perf_rows,
    col_widths=[5.0, 4.4],
    font_size=14)

# Add confusion matrix + HMAC info below the table
txBox = slide.shapes.add_textbox(Inches(0.4), Inches(5.0), Inches(9.2), Inches(2.3))
tf = txBox.text_frame
tf.word_wrap = True
items = [
    "Confusion Matrix (36,567 test samples): TN = 12,776 | FP = 47 | FN = 7 | TP = 23,737",
    "   \u2192 Only 54 misclassifications (47 false alarms + 7 missed attacks)",
    "",
    "HMAC Auxiliary Agent Performance:",
    "   HMAC-SHA256: ~2 \u03bcs/msg | Throughput: ~500K msgs/sec | Tag size: 32 bytes",
    "   With Agent: +0.7 ms latency | +4.4% CPU | +13.7 MB RAM | Detected all 3 injected invalid flows",
    "   End-to-end flow processing: ~170 \u03bcs (0.17 ms) \u2014 well within SDN real-time tolerances",
]
for i, text in enumerate(items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(12)

# ── SLIDE 12: RESULTS \u2014 COMPARISON TABLE ────────────────────────────
comp_rows = [
    ("Proposed", "2025", "TCN-HMAC", "InSDN", "99.85", "99.80", "99.97", "99.89"),
    ("Said et al.", "2023", "CNN-BiLSTM", "InSDN", "99.90", "99.91", "99.90", "99.90"),
    ("Shihab et al.", "2025", "CNN-LSTM", "CIC-IDS", "99.67", "99.68", "99.67", "99.67"),
    ("Yang et al.", "2024", "CNN-GRU", "NSL-KDD", "99.35", "99.20", "99.35", "99.27"),
    ("Ataa et al.", "2024", "DNN Ensemble", "InSDN", "99.70", "99.65", "99.70", "99.67"),
    ("Basfar", "2025", "LSTM", "NSL-KDD", "99.23", "99.00", "99.23", "99.11"),
    ("Kumar et al.", "2025", "Hybrid DL", "CIC-IDS", "99.45", "99.42", "99.45", "99.43"),
    ("Kanimozhi et al.", "2025", "DRL (DDQN)", "InSDN", "98.85", "98.90", "98.85", "98.87"),
    ("Lopes et al.", "2023", "TCN", "CIC-IDS", "99.75", "99.70", "99.75", "99.72"),
    ("Li et al.", "2025", "TCN-SE", "NSL-KDD", "99.62", "99.58", "99.62", "99.60"),
    ("Wang et al.", "2025", "TCN", "InSDN", "97.00", "96.40", "96.91", "96.72"),
    ("Deng et al.", "2024", "BiTCN-MHSA", "CIC-IDS", "99.72", "99.68", "99.72", "99.70"),
    ("Ahmad et al.", "2021", "CNN", "InSDN", "99.20", "99.10", "99.20", "99.15"),
    ("Elsayed et al.", "2020", "DT/RF", "InSDN", "98.50", "98.40", "98.50", "98.45"),
]

add_table_slide(prs, "Results: Comparison with Existing Approaches",
    ["Reference", "Year", "Model", "Dataset", "Acc%", "Prec%", "Rec%", "F1%"],
    comp_rows,
    col_widths=[1.5, 0.6, 1.5, 1.0, 0.9, 0.9, 0.9, 0.9],
    font_size=9)

# ── SLIDE 13: RESULTS \u2014 ARCHITECTURAL COMPARISON ───────────────────
arch_rows = [
    ("TCN-HMAC (Proposed)", "157K", "612 KB", "0.17 ms", "Yes", "Stable", "Yes (HMAC)"),
    ("CNN-BiLSTM", "~2M", "~8 MB", "~2 ms", "Partial", "Moderate", "No"),
    ("CNN-LSTM", "~1.5M", "~6 MB", "~1.5 ms", "Partial", "Moderate", "No"),
    ("DNN Ensemble", "~5M", "~20 MB", "~3 ms", "Yes", "Stable", "No"),
    ("LSTM", "~500K", "~2 MB", "~1 ms", "No", "Unstable", "No"),
    ("DRL (DDQN)", "~3M", "~12 MB", "~5 ms", "Yes", "Stable", "No"),
    ("TCN-SE", "~300K", "~1.2 MB", "~0.3 ms", "Yes", "Stable", "No"),
    ("BiTCN-MHSA", "~500K", "~2 MB", "~0.5 ms", "Partial", "Stable", "No"),
]

add_table_slide(prs, "Architectural Comparison",
    ["Model", "Params", "Size", "Inference", "Parallel", "Gradient", "Comm. Auth."],
    arch_rows,
    col_widths=[1.9, 0.8, 1.0, 1.1, 1.0, 1.1, 1.7],
    font_size=10)

# ── SLIDE 14: RESULTS \u2014 SDN SECURITY FRAMEWORK COMPARISON ──────────
sec_rows = [
    ("TCN-HMAC (Proposed)", "TCN (DL)", "HMAC-SHA256", "Shadow Table", "Seq + TS", "Low", "Yes"),
    ("Ahmed et al. [2023]", "None", "HMAC-SHA256", "No", "Partial", "Low", "Yes"),
    ("Pradeep et al. [2023]", "EnsureS", "TLS", "No", "No", "Moderate", "Partial"),
    ("Song et al. [2023]", "Blockchain", "Consensus", "Yes", "Yes", "High", "No"),
    ("Poorazad et al. [2023]", "ML + BC", "Blockchain", "Yes", "Yes", "Very High", "No"),
    ("Zhou et al. [2022]", "Encryption", "OPE", "Partial", "No", "Moderate", "Partial"),
]

add_table_slide(prs, "Comparison with SDN Security Frameworks",
    ["Framework", "IDS", "Ctrl Auth.", "Flow Verify", "Anti-Replay", "Overhead", "Real-Time"],
    sec_rows,
    col_widths=[2.0, 1.2, 1.4, 1.3, 1.1, 1.1, 1.0],
    font_size=10)

# ── SLIDE 15: MAPPING RESULTS TO RESEARCH OBJECTIVES ───────────────
add_content_slide(prs, "Results Mapped to Research Objectives", [
    "RO1: TCN-based IDS with > 99% accuracy",
    (1, "\u2713 Achieved 99.85% accuracy, 99.97% recall, 99.89% F1 \u2014 exceeds 99% target across all metrics"),
    "",
    "RO2: Preprocessing pipeline for InSDN",
    (1, "\u2713 15-stage pipeline: 84\u219248\u219224 features (PCA retains 95.43% variance); class weighting applied"),
    "",
    "RO3: Compact deployable TCN model",
    (1, "\u2713 156,737 params | 612 KB | 0.17 ms inference \u2014 5\u201330\u00d7 smaller than comparable DL models"),
    "",
    "RO4: HMAC auxiliary agent",
    (1, "\u2713 HMAC-SHA256 (2 \u03bcs/msg) + shadow table verification (0.4 ms) + challenge-response auth"),
    (1, "\u2713 +0.7 ms latency, +4.4% CPU \u2014 negligible overhead; detected all injected invalid flows"),
    "",
    "RO5: Integrated TCN-HMAC framework",
    (1, "\u2713 Dual-boundary defense operating in the SDN control loop \u2014 software-only, no protocol changes"),
    "",
    "RO6: Evaluation in simulated SDN",
    (1, "\u2713 Ryu + Mininet + OVS testbed; comprehensive metrics; statistical confidence intervals (\u00b10.04%)"),
    "",
    "RO7: Comprehensive comparative analysis",
    (1, "\u2713 Benchmarked against 16 approaches (ML, DL, TCN variants, crypto frameworks) \u2014 most extensive in SDN IDS literature"),
], font_size=12)

# ── SLIDE 16: KEY ADVANTAGES ────────────────────────────────────────
add_content_slide(prs, "Key Advantages of TCN-HMAC", [
    "1. Highest Detection Rate: 99.97% \u2014 reduces missed attacks by 70% compared to next-best model",
    "",
    "2. Parameter Efficiency: Only 157K parameters (612 KB) \u2014 5\u201330\u00d7 smaller than comparable DL IDS",
    "",
    "3. Fastest Inference: 0.17 ms end-to-end \u2014 enables 5,000+ flow classifications/second on single GPU",
    "",
    "4. Dual Security Layer: Only framework combining DL-based IDS with cryptographic control-plane authentication",
    "",
    "5. Low Overhead: TCN inference + HMAC computation < 0.2 ms per flow (vs. seconds for blockchain)",
    "",
    "6. Training Efficiency: ~5 minutes to train \u2014 enables rapid model updates for evolving threats",
    "",
    "7. Drop-In Deployment: Software-only \u2014 runs as controller app + auxiliary agent; no protocol/hardware changes",
], font_size=14)

# ── SLIDE 17: CONCLUSION ────────────────────────────────────────────
add_content_slide(prs, "Conclusion", [
    "TCN-HMAC is a lightweight, dual-boundary security framework for Software-Defined Networks that:",
    "",
    (1, "Bridges the detection-verification divide \u2014 first unified framework combining temporal DL-based IDS with HMAC-based flow rule verification and challenge-response controller authentication"),
    "",
    (1, "Achieves state-of-the-art detection: 99.85% accuracy, 99.97% recall, 0.37% FAR, 0.9999 AUC on InSDN"),
    "",
    (1, "Operates in real time: 0.17 ms inference + 2 \u03bcs HMAC \u2014 well within SDN latency tolerances"),
    "",
    (1, "Requires no protocol modifications: runs entirely as a software controller application"),
    "",
    "Future Directions:",
    (1, "Online/incremental learning for concept drift adaptation"),
    (1, "Multi-controller SDN deployment support"),
    (1, "Cross-dataset generalization (NSL-KDD, CIC-IDS-2017, UNSW-NB15)"),
    (1, "Adversarial robustness evaluation"),
    (1, "Multi-class attack type classification (beyond binary)"),
], font_size=14)

# ── SLIDE 18: THANK YOU / Q&A ──────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
txBox = slide.shapes.add_textbox(Inches(0.5), Inches(2.0), Inches(9), Inches(3))
tf = txBox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Thank You"
p.font.size = Pt(40)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(0, 51, 102)

p2 = tf.add_paragraph()
p2.text = "Questions & Discussion"
p2.font.size = Pt(28)
p2.alignment = PP_ALIGN.CENTER
p2.space_before = Pt(24)

p3 = tf.add_paragraph()
p3.text = ""
p3.font.size = Pt(12)

p4 = tf.add_paragraph()
p4.text = "Md. Mahamudul Hasan Zubayer"
p4.font.size = Pt(16)
p4.alignment = PP_ALIGN.CENTER
p4.space_before = Pt(24)

p5 = tf.add_paragraph()
p5.text = "Department of Computer Science & Engineering"
p5.font.size = Pt(14)
p5.alignment = PP_ALIGN.CENTER

p6 = tf.add_paragraph()
p6.text = "Southeast University, Dhaka, Bangladesh"
p6.font.size = Pt(14)
p6.alignment = PP_ALIGN.CENTER


# ── SAVE ─────────────────────────────────────────────────────────────
output_path = "/home/zub/code/TCN_HMAC_Defense_Final_st/thesis_presentation.pptx"
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
print(f"Total slides: {len(prs.slides)}")
