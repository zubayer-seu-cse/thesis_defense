# 100 Defense Board Questions and Answers
## TCN-HMAC: A Hybrid Security Framework for Software-Defined Networking

---

## Section 1: Software-Defined Networking (SDN) Fundamentals (Q1–Q15)

### Q1. What is Software-Defined Networking, and why is it relevant today?
**A:** SDN is a networking paradigm that decouples the control plane (decision-making) from the data plane (packet forwarding). Instead of each switch running its own routing protocols (OSPF, BGP), a centralized software controller orchestrates all forwarding decisions. It is relevant because it gives administrators a global network view, enables policy deployment in seconds, and removes vendor lock-in — which is why SDN now underpins data centers, campus fabrics, WANs, and carrier-grade infrastructures.

### Q2. Describe the three-layer SDN architecture.
**A:**
1. **Infrastructure (Data) Layer:** Physical/virtual switches that forward packets based on flow rules received from the controller.
2. **Control Layer:** The SDN controller running on commodity servers, maintaining a global view of topology, link states, and flow tables. It provides services like topology discovery, path computation, and flow management.
3. **Application Layer:** Network applications (firewalls, IDS, load balancers, QoS managers) that communicate with the controller through northbound APIs.

### Q3. What is OpenFlow and what role does it play in SDN?
**A:** OpenFlow is the dominant southbound interface protocol between the SDN controller and data-plane switches. It allows the controller to install, modify, query, and delete flow rules on switches. It uses a match-action paradigm: each flow rule has match fields (IP addresses, ports, protocol) and actions (forward, drop, modify headers). OpenFlow messages include controller-to-switch messages (Flow-Mod, Packet-Out), asynchronous messages (Packet-In, Flow-Removed), and symmetric messages (Hello, Echo).

### Q4. Explain the Packet-In mechanism and why it's both useful and vulnerable.
**A:** When a switch receives a packet that doesn't match any flow rule, it sends a Packet-In message to the controller for processing. The controller then decides how to handle the packet and installs appropriate flow rules. This is useful because it enables reactive, dynamic forwarding. However, it's vulnerable because an attacker can generate many flows with random headers, flooding the controller with Packet-In messages, saturating the control channel, and exhausting switch flow tables — a table-flooding DDoS unique to SDN.

### Q5. Why is the SDN controller considered a single point of failure?
**A:** The controller is the centralized brain of the entire network. If it is taken down (via resource exhaustion or exploits), switches lose the ability to handle new flows, causing network-wide denial of service. If it is compromised, the attacker gains the ability to manipulate all flow rules across all switches, enabling arbitrary traffic redirection, eavesdropping, and data exfiltration.

### Q6. What are the four major attack categories your thesis focuses on?
**A:**
1. **DDoS:** Volumetric floods that exhaust bandwidth, flow tables, or controller resources. Uniquely dangerous in SDN due to the Packet-In table-flooding variant.
2. **Man-in-the-Middle (MITM):** An adversary intercepts and potentially modifies controller–switch communications, enabling eavesdropping or flow rule alteration.
3. **Probe (Reconnaissance):** Systematic scanning (SYN scans, UDP scans, ICMP sweeps) to discover hosts, ports, services, and vulnerabilities as a precursor to targeted attacks.
4. **Brute-Force:** Systematic credential enumeration against SSH, FTP, HTTP, or controller management interfaces.

### Q7. Why is TLS alone insufficient for SDN security?
**A:** TLS protects the transport layer but has several limitations: (1) It does not verify the semantic integrity of flow rules — a compromised controller can still send valid but malicious Flow-Mod messages over TLS; (2) Certificate management is complex in large-scale deployments; (3) Asymmetric cryptography imposes computational overhead that is problematic for resource-constrained environments; (4) Many deployments don't enable TLS or use weak configurations; (5) TLS can't detect unauthorized flow table modifications that occur outside the TLS-protected channel.

### Q8. What is the flow rule integrity problem in SDN?
**A:** Once a flow rule is installed in a switch's flow table, the switch executes the associated actions without re-verification. There is no built-in mechanism for switches to independently verify that a received flow rule is authentic and unaltered. A tampered or maliciously crafted flow rule will be faithfully executed, potentially redirecting traffic to an attacker, creating forwarding loops, or dropping critical packets.

### Q9. What is Open vSwitch (OVS) and why was it used in this research?
**A:** OVS is the most widely deployed software-based OpenFlow switch implementation. It supports multiple flow tables, group tables, meter tables, and extensive match field support. It operates in both kernel space (fast-path forwarding) and user space (management and control communication). It is the de facto standard for SDN research and development due to its versatility and comprehensive OpenFlow support.

### Q10. What is the Ryu SDN controller framework?
**A:** Ryu is an open-source, Python-based SDN controller platform fully compliant with OpenFlow versions 1.0–1.5. Its modular architecture allows custom network applications to be implemented as independent Python modules that register event handlers for OpenFlow messages. It was chosen for this research due to its flexibility, extensibility, and comprehensive documentation.

### Q11. Explain the difference between proactive and reactive flow rule installation.
**A:** In proactive installation, flow rules are pre-installed in switches before traffic arrives, so packets are immediately matched and forwarded without controller involvement. In reactive installation, rules are created in response to Packet-In messages triggered by new flows — the controller examines the unmatched packet, makes a decision, and then installs rules. Proactive is faster but requires predicting traffic patterns; reactive is more flexible but adds latency and makes the controller a bottleneck target.

### Q12. What is a Flow-Mod message and why is it central to your thesis?
**A:** Flow-Mod is the primary OpenFlow message through which the controller installs, modifies, or deletes flow rules on switches. It contains match fields, actions, priority, timeouts, and a 64-bit cookie field. It is central to the thesis because: (1) The HMAC-based verification leverages the cookie field to embed integrity tags without modifying the OpenFlow protocol; (2) Flow-Mod tampering is one of the key threats the HMAC mechanism addresses.

### Q13. What makes DDoS attacks particularly dangerous in SDN compared to traditional networks?
**A:** In addition to standard bandwidth exhaustion, SDN has a unique vulnerability: the table-flooding attack. Because every unmatched flow triggers a Packet-In message, an attacker generating many flows with randomized headers overwhelms the controller with Packet-In messages, saturates the control channel, and exhausts switch flow table capacity. This can render the entire network inoperable even with modest volumetric traffic.

### Q14. How do probe attacks serve as a precursor to more targeted intrusions?
**A:** Probe attacks systematically scan the network to discover active hosts, open ports, running services, and exploitable vulnerabilities. In SDN, they are particularly informative because discovering the controller's address, OpenFlow switch ports, or topology structure reveals critical management information. Detecting probes early disrupts the attack kill chain before the adversary can leverage gathered intelligence.

### Q15. What is the scalability challenge for security mechanisms in SDN?
**A:** SDN environments are highly dynamic — flow rules are installed, modified, and evicted at high rates. Any security mechanism must operate at speed commensurate with flow rule change rates. Heavyweight cryptographic operations, complex protocol negotiations, or expensive deep learning inference can introduce latency that degrades network performance. The tension between security strength and operational overhead is a central challenge.

---

## Section 2: Temporal Convolutional Networks (TCN) Theory (Q16–Q30)

### Q16. What is a Temporal Convolutional Network and how does it differ from RNNs?
**A:** A TCN processes sequences using one-dimensional causal dilated convolutions rather than recurrent connections. Unlike RNNs (LSTMs, GRUs) that step through sequences one element at a time, TCNs process the entire sequence in parallel through convolutional operations. Key differences: (1) TCNs are fully parallelizable; (2) They have stable gradient behavior; (3) Their receptive field is precisely controllable; and (4) They share convolutional filters across all time steps.

### Q17. What is a causal convolution and why is causality important for IDS?
**A:** A causal convolution ensures the output at time step *t* depends only on inputs from time steps ≤ *t*, never from future steps. Formally: (x ∗_c f)(t) = Σ f_k · x_{t-k}. Causality is critical for real-time intrusion detection because predictions must be based solely on currently available and historical data — you cannot wait for future data to decide if traffic is malicious now.

### Q18. Explain dilated convolutions and why they are essential for TCNs.
**A:** Dilated convolutions introduce a dilation factor *d* that controls spacing between kernel elements: (x ∗_d f)(t) = Σ f_k · x_{t-d·k}. Standard causal convolutions have receptive fields that grow linearly with depth, requiring impractically many layers for long-range dependencies. Dilated convolutions with exponentially increasing dilation factors (1, 2, 4, 8, ...) make the receptive field grow exponentially with depth. For our model with K=3 and L=6, the receptive field is 1 + 2×(2^6 − 1) = 127 time steps, far exceeding the input length of 24.

### Q19. What are residual connections and why are they used in the TCN?
**A:** Residual connections provide shortcut paths that add the block's input directly to its transformed output: output = F(x) + x. When dimensions don't match, a 1×1 convolution projects the input. Benefits: (1) Enable training of deeper networks by mitigating vanishing gradients; (2) Allow the network to learn incremental refinements rather than complete transformations; (3) Promote feature reuse across layers; (4) Provide direct gradient paths from loss to early layers.

### Q20. How does the receptive field calculation work in your TCN?
**A:** For L layers with dilation factors d_l = 2^(l-1) and kernel size K, the receptive field is R = 1 + (K−1)(2^L − 1). Our model uses K=3, L=6 (dilations [1,2,4,8,16,32]), giving R = 1 + 2×(64−1) = 127, but with two convolutions per block, the cumulative receptive field reaches 253. Since our input is only 24 time steps, the deepest layers can see the entire input with room to spare, providing full global context.

### Q21. What is batch normalization and how does it help TCN training?
**A:** Batch normalization normalizes layer activations to zero mean and unit variance within each mini-batch: x̂ = (x − μ_B) / √(σ²_B + ε), then scales and shifts: y = γx̂ + β. It addresses internal covariate shift, enables higher learning rates, reduces sensitivity to initialization, and acts as a regularizer. In our TCN, it's applied after each convolutional layer and before the activation function.

### Q22. Why use spatial dropout instead of standard dropout in the TCN?
**A:** Standard element-wise dropout can disrupt the spatial/temporal structure of feature maps. Spatial dropout drops entire feature map channels rather than individual elements — for a tensor of shape (N, T, C), it creates a mask of shape (N, 1, C) and broadcasts across the temporal dimension. When a channel is dropped, it's dropped for all time steps simultaneously, preserving temporal coherence of retained channels. We use spatial dropout rate of 0.2.

### Q23. What is Global Average Pooling and why was it chosen over flattening?
**A:** GAP averages activations across the temporal dimension for each channel: z_c = (1/T)Σ h_{t,c}, reducing shape from (N, T, C) to (N, C). Advantages over flattening: (1) It's parameter-free; (2) It's robust to temporal shifts since averaging is less sensitive to exact position of discriminative patterns; (3) It produces a more compact representation, reducing overfitting risk in subsequent dense layers.

### Q24. Why did you choose TCN over LSTM for intrusion detection?
**A:** Five concrete reasons: (1) **Parallelism:** TCN convolutions process all time steps simultaneously vs. LSTM's sequential one-at-a-time processing — critical for real-time IDS; (2) **Stable gradients:** Gradient path through fixed convolutional layers rather than sequence-length-dependent recurrent steps; (3) **Controllable receptive field:** Precisely set via kernel size and dilation factors vs. LSTM's implicitly learned memory; (4) **Memory efficiency:** Parameters independent of sequence length; (5) **Empirical results:** Our TCN achieves 99.85% accuracy vs. comparable LSTM's 99.23%, with 0.17ms vs. ~1ms inference.

### Q25. Could a Transformer model work for this task? Why wasn't it used?
**A:** Transformers use self-attention to capture global dependencies and could theoretically work. However: (1) Self-attention has O(n²) computational complexity w.r.t. sequence length — expensive for high-throughput deployment; (2) Transformers have many more parameters, increasing overfitting risk on our ~183K sample dataset; (3) The TCN already achieves 99.85% accuracy and 0.9999 AUC, leaving minimal room for improvement; (4) Our input sequence is only 24 steps — too short to benefit from Transformers' long-range dependency modeling.

### Q26. How does the weight-sharing property of TCNs contribute to model compactness?
**A:** Convolutional filters are reused across every temporal position — the same 3-wide kernel applied at every time step. This means the parameter count is independent of sequence length. Our model has only 156,737 parameters (612 KB), compared to ~2M for a comparable CNN-BiLSTM. The compactness enables fast deployment (millisecond loading), low memory usage, easier model updates, and edge-device deployment.

### Q27. Explain the ReLU activation function and why it was chosen.
**A:** ReLU(x) = max(0, x). It introduces nonlinearity needed for learning complex decision boundaries. Chosen over sigmoid/tanh/LeakyReLU for: (1) Computational simplicity (just a comparison); (2) Effectiveness in mitigating vanishing gradients (non-saturating for positive inputs); (3) Widespread empirical success in deep convolutional networks. Its sparse activation also promotes representational efficiency.

### Q28. What would happen if you removed the residual connections from your TCN?
**A:** Without residual connections: (1) Training would become harder due to the degradation problem — deeper layers wouldn't necessarily learn better than shallower ones; (2) Vanishing gradients would be more severe, slowing convergence; (3) The model would lose the ability to learn identity mappings (effectively "skip" unnecessary transformations); (4) Feature reuse across layers would be lost. Practically, accuracy would likely drop and training would require more epochs and careful learning rate tuning.

### Q29. Why does the exponential dilation schedule capture multi-scale patterns?
**A:** Each dilation rate captures patterns at a different temporal scale: Block 1 (d=1) captures fine-grained adjacent-component patterns; Block 3 (d=4) captures medium-scale patterns spanning 4–8 components; Block 6 (d=32) captures global patterns spanning the entire input. This multi-scale hierarchy detects both localized anomalies (single feature spikes) and distributed anomalies (correlated changes across many features).

### Q30. How do TCNs compare to standard 1D CNNs for this task?
**A:** Standard 1D CNNs lack inherent temporal awareness — they treat input as a fixed-length feature vector and only capture local spatial correlations between adjacent features. TCNs add: (1) Causal padding ensuring no future information leakage; (2) Dilated convolutions for exponentially growing receptive fields; (3) Residual blocks for deeper, more expressive networks. In the comparison, a plain CNN achieved 99.20% vs our TCN's 99.85% on InSDN.

---

## Section 3: HMAC and Security Mechanisms (Q31–Q45)

### Q31. What is an HMAC and how is it computed?
**A:** HMAC (Hash-based Message Authentication Code) is computed as: HMAC(K, M) = H((K' ⊕ opad) ‖ H((K' ⊕ ipad) ‖ M)), where H is a hash function (SHA-256), K' is the key padded to the hash block size, ipad=0x36 and opad=0x5C repeated B times. The double-hashing construction prevents length-extension attacks. It provides both message integrity (any modification invalidates the tag) and message authentication (only the key-holder can compute a valid tag).

### Q32. Why use HMAC-SHA256 instead of digital signatures (RSA/ECDSA)?
**A:** HMAC uses symmetric keys and is dramatically faster: HMAC-SHA256 completes in <1μs per message, while RSA-2048 requires 0.5–2ms and ECDSA-256 requires 0.1–0.5ms. In SDN, where flow rules are installed at hundreds/thousands per second, this 100–2000× speed advantage is critical. HMAC-SHA256 provides 256-bit authentication (2^128 security against brute-force forgery), which is more than sufficient for this application.

### Q33. How does the HMAC key establishment protocol work?
**A:** During switch connection: (1) The controller generates a random 256-bit master key; (2) The master key is transmitted over TLS; (3) Both sides derive two directional session keys using KDF: K_{C→S} = HMAC-SHA256(K_master, "controller-to-switch" ‖ nonce_C) and K_{S→C} = HMAC-SHA256(K_master, "switch-to-controller" ‖ nonce_S). Separate directional keys prevent reflection attacks. The agent secures its exchange using RSA encryption and HMAC authentication of the payload.

### Q34. How does the framework prevent replay attacks?
**A:** Two mechanisms: (1) **Sequence numbers:** Each message includes a monotonically increasing sequence number; the receiver only accepts messages with sequence numbers greater than the last accepted one; (2) **Timestamps:** Each message includes a timestamp; the receiver rejects messages outside an acceptable time window. Together, these ensure an attacker cannot capture and re-send a previously valid message.

### Q35. Explain the flow rule verification mechanism using shadow tables.
**A:** The controller maintains a shadow copy of each switch's expected flow table. Every 30 seconds (configurable), the controller sends a Flow_Stats_Request to each switch. The switch responds with its current flow rules (HMAC-authenticated). The controller compares received rules against the shadow copy. Discrepancies indicate compromise: extra rules → flow rule injection; modified rules → MITM modification; missing rules → unauthorized deletion or legitimate timeout. Unauthorized modifications are automatically corrected.

### Q36. Describe the challenge-response authentication mechanism.
**A:** For controller-authenticates-switch: The controller generates a random 128-bit nonce c_s, sends it to the switch; the switch responds with r_s = HMAC-SHA256(K_{S→C}, c_s ‖ switch_id); the controller verifies. For switch-authenticates-controller: The switch generates nonce c_c, sends to controller; controller responds with r_c = HMAC-SHA256(K_{C→S}, c_c ‖ controller_id); switch verifies. Performed during handshake and periodically (every 1800 seconds). Failed verification triggers disconnect/alert.

### Q37. How does key rotation work and why is forward secrecy important?
**A:** Keys are rotated every T_rotate seconds (default 3600): K_master^(t+1) = HMAC-SHA256(K_master^(t), r_t) where r_t is a fresh random 256-bit value. Forward secrecy means compromise of a future key doesn't compromise past keys — because the new key derivation is one-way (hash function), knowing K_master^(t+1) doesn't reveal K_master^(t). This limits the window of vulnerability if a key is ever compromised.

### Q38. Why use constant-time comparison for HMAC tag verification?
**A:** Standard string comparison returns early on the first mismatched byte, creating timing differences that leak information: an attacker can measure response times to determine how many bytes of their forged tag match the real tag, eventually forging a valid tag byte-by-byte. Constant-time comparison always takes the same time regardless of where mismatches occur, preventing this timing side-channel attack.

### Q39. What is the overhead of the HMAC mechanism?
**A:** Very low: (1) Computation: ~2μs per message; (2) Bandwidth: 32 bytes (256-bit tag) added per message — 25–50% size increase for 64–128 byte Flow_Mod messages, but trivial relative to data plane throughput; (3) Key storage: two 256-bit keys (64 bytes) per switch; for 10,000 switches, ~1.9 MB total; (4) Overall latency addition: ~0.7ms to control path; (5) CPU overhead: 4.4%; (6) RAM: 13.7 MB.

### Q40. How does the cookie field enable HMAC integration without protocol modification?
**A:** The OpenFlow Flow-Mod message contains a 64-bit cookie field that the controller can set arbitrarily and that is returned in flow statistics and Flow-Removed messages. TCN-HMAC embeds the HMAC tag (or a truncated version) in this cookie field, so switches receive and store the integrity tag alongside the flow rule without any modification to the OpenFlow protocol, switch firmware, or hardware.

### Q41. What threats does HMAC address that TLS does not?
**A:** (1) **Application-layer integrity:** If the controller software is compromised, TLS doesn't prevent it from sending valid-but-malicious Flow-Mod messages — HMAC ensures rule semantic integrity; (2) **Flow table verification:** TLS can't detect unauthorized modifications to the switch's flow table occurring outside the TLS channel; (3) **Controller authenticity:** HMAC's challenge-response proves controller identity at the application layer, independent of TLS certificate management.

### Q42. Why is HMAC preferred over blockchain-based integrity solutions?
**A:** Blockchain consensus takes seconds to minutes per transaction vs. HMAC's ~2μs — four to six orders of magnitude faster. Blockchain requires distributed consensus infrastructure, significant storage for growing chain, and high computational resources. HMAC runs as a lightweight co-located subprocess, needs only 64 bytes per switch for key storage, and adds negligible CPU/memory overhead. Both provide integrity guarantees, but only HMAC is compatible with real-time SDN requirements.

### Q43. What happens when a switch fails the challenge-response authentication?
**A:** If a switch fails authentication: the controller terminates the switch connection and generates a security alert for the network administrator. If the controller fails authentication: the switch disconnects and enters a safe mode — continuing to forward based on its last known good flow table until a verified controller becomes available. This prevents a compromised controller from continuing to issue malicious commands.

### Q44. How does the auxiliary agent architecture work?
**A:** The auxiliary agent operates as an independent co-located subprocess alongside the SDN controller. It performs two functions: (a) HMAC-based flow rule verification against a shadow table with ~2μs per-message overhead; (b) Periodic challenge-response controller authentication. It communicates with the controller via symmetric encryption with shared secrets established via one-time RSA key exchange. It is isolated from external network access to minimize exposure. It requires no OpenFlow protocol or switch hardware modifications.

### Q45. How scalable is the HMAC key management scheme?
**A:** The scheme scales linearly: O(N) for N switches. The controller stores N master keys and 2N session keys. For 10,000 switches, total key storage is ~1.9 MB — trivial for controller memory. Key rotation is done per-switch and is independent, so rotation for one switch doesn't affect others. The computational overhead per message remains constant regardless of network size. The limitation is the single-controller assumption — multi-controller architectures would need additional key coordination.

---

## Section 4: Dataset and Preprocessing (Q46–Q60)

### Q46. Why was the InSDN dataset chosen over NSL-KDD or CIC-IDS-2017?
**A:** InSDN was purpose-built for SDN intrusion detection research — generated inside a dedicated SDN testbed with OpenFlow switches, so traffic flows, feature distributions, and attack manifestations genuinely reflect SDN operation. NSL-KDD and CIC-IDS-2017 were captured in traditional network architectures, introducing a domain mismatch when applied to SDN. InSDN covers DDoS, MITM, Probe, and Brute-force attacks, exactly the threat categories relevant to SDN.

### Q47. Describe the composition of the InSDN dataset.
**A:** Three CSV files: Normal_data.csv (benign traffic), OVS.csv (Open vSwitch traces), metasploitable-2.csv (Metasploitable-2 attacks). Combined: 343,889 rows, 84 features per flow, extracted by CICFlowMeter. Raw distribution: 68,424 benign (19.9%), 275,465 attack (80.1%). After deduplication: 182,831 rows (160,058 duplicates removed — 46.5% reduction).

### Q48. Why were identifier columns (Flow ID, IP addresses, ports, timestamp) removed?
**A:** These columns contain flow-specific identifiers that don't generalize to unseen traffic. If retained, the model would memorize specific IP addresses or port numbers associated with attack labels, leading to overfitting on the training data distribution and poor generalization to different network environments. After removal: 78 columns remained (77 features + 1 label).

### Q49. Explain the Pearson correlation-based feature reduction step.
**A:** For every feature pair, we compute r_XY (Pearson correlation coefficient). Pairs with |r| > 0.98 are nearly perfectly correlated, meaning both carry essentially the same information. One from each such pair is dropped. This removed 17 features (65 → 48), primarily redundant volume metrics like forward byte count features that were nearly perfectly correlated with forward packet count features.

### Q50. Why was PCA used and how was the number of components determined?
**A:** PCA was used to remove residual redundancy not caught by Pearson correlation (sub-threshold linear and nonlinear dependencies). We set a 95% variance retention threshold, keeping the fewest principal components explaining ≥95% of total variance. Result: 48 → 24 features (50% reduction), retaining 95.43% variance. Benefits: reduced computation, reduced overfitting, decorrelated features, and interpretable variance capture.

### Q51. Why was StandardScaler chosen over MinMaxScaler?
**A:** Three reasons: (1) StandardScaler is less sensitive to outliers — it doesn't bound transformed values to [0,1] like MinMaxScaler; (2) It preserves relative distances between data points; (3) It produces distributions well-suited for gradient-based optimization in deep learning. Importantly, the scaler is fit on training data only, and the same μ and σ are applied to validation/test sets to prevent data leakage.

### Q52. How did you handle class imbalance, and why class weighting over SMOTE?
**A:** Post-deduplication ratio was 35% benign vs. 65% attack. We used inverse-frequency class weighting: w_c = N/(C × n_c), giving benign weight=1.4258 and attack weight=0.7700 (benign samples count 1.85× more in gradient computation). Chosen over SMOTE because: (1) No synthetic data that might misrepresent real traffic; (2) No data discarding like undersampling; (3) Seamless integration with binary cross-entropy loss; (4) Zero additional computational overhead.

### Q53. Why was such a large portion (46.5%) of the data removed as duplicates?
**A:** The raw OVS and metasploitable-2 captures contained many identical flow records from repeated traffic patterns during data collection. Keeping duplicates would: (1) Artificially inflate dataset size, giving disproportionate weight to certain patterns; (2) Leak information between train/test if the same flow appears in both; (3) Reduce effective training diversity, leading to overfitting. After deduplication, the class ratio shifted from 80:20 to 65:35, suggesting attack generation tools produced more repetitive patterns.

### Q54. Describe the complete 15-stage preprocessing pipeline.
**A:**
1. Concatenate 3 CSVs (343,889 × 84)
2. Strip whitespace from column names/values
3. Drop 6 identifier columns (→ 78 columns)
4. Binary label encoding (Normal→0, attacks→1)
5. Replace Inf with NaN, drop NaN rows
6. Remove 12 zero-variance features (→ 66)
7. Remove near-constant features (>99.9% single value)
8. Remove 17 correlated features (|r|>0.98) (→ 49)
9. Save consolidated CSV (71 MB)
10. Deduplicate rows (→ 182,831)
11. StandardScaler normalization
12. PCA (95% variance → 24 features)
13. Stratified 70/10/20 train/val/test split
14. Reshape to (N, 24, 1) for TCN input
15. Compute class weights [1.4258, 0.7700]

### Q55. Why use a 70/10/20 split ratio?
**A:** 70% training (127,981 samples): Provides sufficient data for learning robust feature representations. 10% validation (18,283 samples): Used for early stopping, learning rate scheduling, and model checkpoint selection — statistically reliable for generalization estimates. 20% test (36,567 samples): Enables robust final evaluation with narrow confidence intervals, never used during training or tuning, ensuring unbiased out-of-sample performance estimates.

### Q56. How does PCA interact with the TCN's causal convolution mechanism?
**A:** After PCA, the 24 components are ordered by decreasing explained variance — establishing a natural ordering from most to least important features. The TCN's causal dilated convolutions then learn patterns across this ordered representation. The residual blocks with exponentially increasing dilation factors capture dependencies between components at multiple scales of the variance hierarchy. While PCA components aren't inherently temporal, this ordering provides meaningful structure for convolutional processing.

### Q57. What zero-variance features were found and removed?
**A:** Twelve features had the same value for all samples — primarily flag-related columns (rarely-used TCP flag counts like CWE and ECE) and certain sub-flow features that didn't vary across recorded flows. These carry zero discriminative information while consuming computational resources, so they were removed.

### Q58. How do you prevent data leakage in your preprocessing?
**A:** Three key precautions: (1) StandardScaler parameters (μ, σ) are computed on training data only, then applied to validation and test sets; (2) PCA transformation is fitted on training data only, then applied to other sets using the same transformation matrix; (3) Stratified splitting ensures consistent class distribution across all subsets. Class weights are also computed from training set counts only.

### Q59. What types of features does the InSDN dataset include?
**A:** Categories include: identifier features (flow ID, IPs, ports — removed), duration features, packet count features (total, forward, backward), byte count features, packet length statistics (max/min/mean/std), inter-arrival time features (flow/fwd/bwd IAT), TCP flag features (FIN, SYN, RST, PSH, ACK, URG, CWE, ECE counts), flow-level metrics (bytes/s, packets/s, down/up ratio), sub-flow features, and active/idle period features.

### Q60. How sensitive is the model to the PCA variance threshold? What if you used 99% instead of 95%?
**A:** Using 99% would retain more components (perhaps 35–40 instead of 24), preserving more information but increasing input dimensionality. This means: larger model, slower inference, higher overfitting risk, and minimal accuracy benefit since the discarded 4.57% variance likely represents noise. The 95% threshold strikes the optimal balance — 50% dimensionality reduction with only 4.57% information loss, and our 99.85% accuracy suggests no critical information was discarded.

---

## Section 5: Model Architecture and Training (Q61–Q75)

### Q61. Describe the complete architecture of TCN_InSDN.
**A:** Input: (N, 24, 1). Six residual blocks with dilation rates [1,2,4,8,16,32], each containing two sub-layers of Conv1D(64 filters, kernel 3, causal padding) → BatchNorm → ReLU → SpatialDropout(0.2), plus a residual skip connection. Followed by GlobalAveragePooling1D → Dense(128) + BN + ReLU + Dropout(0.2) → Dense(64) + BN + ReLU + Dropout(0.2) → Dense(1, sigmoid). Total: 156,737 parameters (154,817 trainable, 1,920 non-trainable BN params), ~612 KB.

### Q62. Why 64 filters per block? Why not 32 or 128?
**A:** Determined through validation-set sweep: 32 filters underfit (accuracy plateaued at ~99.5%); 128 filters gave marginal improvement (<0.1%) while tripling model size. The 64-filter configuration achieves near-optimal performance while keeping the model compact for real-time deployment.

### Q63. Why kernel size 3 instead of 5 or 7?
**A:** Kernel size 3 is the standard TCN choice, capturing local patterns between adjacent elements. Larger kernels (5, 7) showed no significant accuracy improvement on the validation set while increasing parameters. The effectiveness of kernel size 3 is enhanced by dilated convolutions — at dilation rate d, the effective "virtual" kernel size is 3×d, meaning Block 6 (d=32) effectively has a kernel spanning 96 positions.

### Q64. Explain the training configuration details.
**A:** Optimizer: Adam (lr=10⁻³, β₁=0.9, β₂=0.999). Loss: Weighted binary cross-entropy. Batch size: 2,048 (maximizes T4 GPU utilization, ~63 iterations/epoch). Max epochs: 100. Early stopping: patience=15, monitor=val_auc, restore best weights. LR scheduler: ReduceLROnPlateau (patience=7, factor=0.5, min_lr=10⁻⁶). Checkpoint: save best by val_auc. Training converged in ~30 epochs (~5 minutes total).

### Q65. Why was Adam optimizer chosen over SGD?
**A:** (1) Adam converges faster on InSDN — near-optimal in 15–20 epochs vs. 50+ for SGD; (2) Its adaptive per-parameter learning rates handle the heterogeneous scale of PCA-transformed features effectively; (3) It combines benefits of AdaGrad (adapts to sparse gradients) and RMSProp (normalized by moving average of recent gradient magnitudes); (4) Well-demonstrated performance on convolutional architectures for classification tasks.

### Q66. Why use validation AUC for early stopping instead of validation loss?
**A:** AUC is threshold-independent — it measures the model's discriminative ability across all possible classification thresholds, not just the default 0.5. This is more robust than loss, which can fluctuate due to batch normalization, class weighting, and the inherent stochasticity of training. AUC directly measures what we care about: the model's ability to rank attacks higher than benign flows.

### Q67. What is the total FLOP count and why does it matter?
**A:** ~7.1M FLOPs per inference — roughly 250× less than typical CNN image classifiers. This matters for real-time deployment because it determines inference latency and throughput. At 7.1M FLOPs, the model can classify a flow in 0.17ms, enabling 5,000+ flow classifications per second on a single GPU, easily meeting SDN real-time requirements.

### Q68. Why batch size 2,048?
**A:** (1) Maximizes T4 GPU utilization without exceeding 16 GB GPU memory; (2) Larger batches produce more stable gradient estimates, reducing noise in parameter updates; (3) Important for batch normalization, which needs sufficiently large batches for accurate mean/variance estimates; (4) With 127,981 training samples, each epoch has ~63 iterations, enabling rapid epoch completion.

### Q69. How does the sigmoid output relate to attack probability in deployment?
**A:** The output neuron uses sigmoid: p(attack|x) = 1/(1+e^(-(w^T h + b))). The output ∈ (0,1) is the estimated probability the flow is an attack. At inference, threshold 0.5 is applied: p>0.5 → attack, p≤0.5 → benign. In deployment, tiered responses are possible: p>0.95 → automatic blocking; 0.5<p≤0.95 → block with alert; 0.05≤p≤0.5 → allow with monitoring; p<0.05 → allow freely.

### Q70. The first residual block has different parameter count — why?
**A:** Block 1 takes 1-channel input and produces 64-channel output. The channel mismatch requires a 1×1 convolutional projection on the skip connection to align dimensions (1→64 channels), adding extra parameters. Blocks 2–6 have matching input/output channels (64→64), so their skip connections are identity mappings with no extra parameters. Block 1: 1,536 params; Blocks 2–6: 25,216 params each.

### Q71. What does the learning rate schedule look like during training?
**A:** Starting at 10⁻³, the ReduceLROnPlateau scheduler halves the rate when validation loss plateaus for 7 epochs: ~epoch 12: 10⁻³ → 5×10⁻⁴ (first plateau); ~epoch 21: 5×10⁻⁴ → 2.5×10⁻⁴ (second plateau). This provides rapid initial convergence followed by increasingly fine weight adjustments. The minimum lr of 10⁻⁶ allows up to 10 reduction steps.

### Q72. How is the model saved and deployed?
**A:** The model is saved as `best_tcn_insdn.keras` (612 KB) whenever validation AUC improves. For deployment: the model file, saved scaler parameters (μ, σ), and saved PCA transformation basis are loaded by the TCN-IDS module running as a Ryu controller application. TensorFlow Lite or TensorFlow Serving can further optimize deployment. The compact size enables fast loading and efficient memory usage alongside controller operations.

### Q73. Why was binary classification chosen instead of multi-class?
**A:** For real-time deployment, the primary objective is to flag suspicious flows immediately — benign vs. malicious. Binary classification simplifies the response: detected attacks are automatically blocked. Multi-class would require more complex decision logic and might reduce per-class accuracy due to class overlap. The binary approach achieves 99.85% accuracy. Multi-class extension is identified as future work.

### Q74. What regularization techniques are used and how do they prevent overfitting?
**A:** Four mechanisms: (1) **Spatial Dropout (0.2)** in residual blocks — drops entire channels; (2) **Standard Dropout (0.2)** in dense classification head; (3) **Batch Normalization** — stabilizes training, acts as implicit regularizer; (4) **Inverse-frequency class weighting** — prevents the model from achieving high accuracy by always predicting the majority class. Evidence of effectiveness: training and validation loss curves closely track each other with minimal gap.

### Q75. Could the model be further compressed for edge deployment?
**A:** Yes, several techniques are proposed as future work: (1) **Quantization** (FP32 → INT8): could reduce size from 612 KB to ~153 KB with minimal accuracy loss; (2) **Pruning:** removing redundant weights could reduce parameters by 30–50%; (3) **Knowledge distillation:** training a smaller student model to mimic the TCN's behavior. The T4 GPU's Tensor Cores already support FP16/INT8 acceleration, enabling mixed-precision inference.

---

## Section 6: Results and Performance (Q76–Q90)

### Q76. Summarize the key test-set performance metrics.
**A:** On 36,567 held-out test samples: Accuracy 99.85%, Precision 99.80%, Recall/Detection Rate 99.97%, Specificity 99.63%, F1-Score 99.89%, AUC-ROC 0.9999, False Alarm Rate 0.37%, Matthews Correlation Coefficient 0.9966. Confusion matrix: TP=23,737, TN=12,776, FP=47, FN=7.

### Q77. What do the 7 false negatives likely represent?
**A:** These 7 missed attacks almost certainly mimic benign traffic: slow-rate probes with timing indistinguishable from normal connections, or carefully crafted packets whose feature profiles land squarely in the benign region of the model's learned representation. Even so, 7 out of 23,744 (0.03% miss rate) is strong — complementary mechanisms like firewalls, anomaly alerting, and human SOC analysts can cover this residual gap.

### Q78. What do the 47 false positives likely represent?
**A:** These benign flows likely share statistical fingerprints with genuine attacks — unusual packet sizes, atypical ports, or bursty timing the model associates with DDoS-like behavior. They sit on the decision boundary where benign and attack distributions overlap. At 0.37% FAR, roughly 1 in 270 benign flows is falsely flagged — manageable for automated whitelisting or manual review.

### Q79. Why is the error distribution asymmetric (87% FP vs. 13% FN)?
**A:** This is a desirable property — the model is conservative, preferring to flag suspicious traffic rather than let attacks through. This bias comes from: (1) Class weighting (benign samples weighted 1.85× higher, penalizing false positives less than false negatives when class weighting interacts with the natural decision boundary); (2) The binary cross-entropy loss function; (3) The inherent data distribution where attack patterns are more diverse. In security, false alarms are annoying but false negatives are dangerous.

### Q80. Explain the AUC-ROC value of 0.9999 and its practical significance.
**A:** AUC of 0.9999 (max 1.0) means near-perfect separation between benign and attack probability distributions. Practical significance: (1) The model's performance is robust to threshold selection — almost any threshold between 0.1 and 0.9 yields excellent results; (2) Attack flows receive probabilities very close to 1.0 and benign very close to 0.0, with very few ambiguous samples; (3) Enables tiered response strategies based on confidence levels.

### Q81. Why was MCC reported in addition to standard metrics?
**A:** MCC (Matthews Correlation Coefficient = 0.9966) is considered more reliable than accuracy for imbalanced datasets because it accounts for all four confusion matrix elements and produces a high score only when both classes perform well. Unlike accuracy, which can be inflated by always predicting the majority class, an MCC of 0.9966 (scale -1 to +1) confirms near-perfect correlation between predictions and ground truth for both classes.

### Q82. How fast did the model train and why?
**A:** ~5 minutes total (~30 epochs, ~10 seconds/epoch, 63 iterations/epoch). Fast convergence driven by: (1) Residual connections providing direct gradient paths; (2) Batch normalization stabilizing internal covariate shift; (3) Adam optimizer's adaptive learning rates; (4) PCA-preprocessed features providing clean, structured input; (5) Compact model (156K parameters). Peak GPU memory: ~2.5 GB on the 16 GB T4.

### Q83. How does your model compare to the CNN-BiLSTM model on InSDN?
**A:** CNN-BiLSTM (Said et al.): 99.90% accuracy vs. our 99.85% — a 0.05% gap within the ±0.04% confidence interval, not statistically significant. However, our model has key advantages: (1) Detection rate: 99.97% vs. 99.90% — a 70% reduction in missed attacks; (2) Model size: 612 KB vs. ~8 MB (13× smaller); (3) Inference: 0.17ms vs. ~2ms (12× faster); (4) No bidirectional processing needed (CNN-BiLSTM needs future time steps, limiting real-time use); (5) We add HMAC security — they provide detection only.

### Q84. Why does your simpler TCN outperform more complex TCN variants (BiTCN-MHSA, TCN-SE, TCN+Attention)?
**A:** The results demonstrate that architectural simplicity paired with rigorous preprocessing outperforms structural complexity. Our key advantages: (1) Disciplined 15-stage preprocessing produces clean, compact features; (2) PCA decorrelation eliminates multicollinearity that complex architectures must learn to disentangle; (3) Proper class weighting handles imbalance without data synthesis; (4) The clean TCN captures all necessary patterns with 64 filters and 6 blocks. Adding attention or bidirectionality adds parameters and inference time without proportional accuracy gains.

### Q85. What is the model's inference throughput for production deployment?
**A:** At 0.17ms per flow, the model can classify ~5,882 flows per second on a single GPU. On CPU, this would be slower but still adequate for many deployments. The 612 KB model loads in milliseconds, and the entire TCN+HMAC pipeline adds <0.2ms per flow. For a network processing 10,000 flows/second, this could be parallelized across 2 GPU instances.

### Q86. How do the training and validation curves confirm no overfitting?
**A:** (1) Training and validation loss curves closely track each other with a small, consistent gap — no divergence; (2) Both losses plateau at ~0.01–0.02 simultaneously; (3) Training and validation accuracy converge to ~99.8% and 99.85% respectively — the validation accuracy is marginally higher than training, confirming good generalization; (4) The model performs equally well on the unseen test set (99.85%), confirming no overfit to validation data.

### Q87. How does the TCN-HMAC framework compare to blockchain-based SDN security?
**A:** In the security frameworks comparison: TCN-HMAC provides IDS (TCN deep learning), control authentication (HMAC-SHA256), flow verification (shadow tables), anti-replay (seq+timestamps), with low overhead and real-time capability. Blockchain solutions (Song et al., Rahman et al., Poorazad et al.) provide strong integrity guarantees but with high overhead (seconds per transaction), high computational cost, and no real-time capability. TCN-HMAC is 4–6 orders of magnitude faster per message.

### Q88. What is the detection rate significance of 99.97%?
**A:** 99.97% means only 3 out of every 10,000 attack flows evade detection. This exceeds the 99% benchmark commonly cited as the minimum for production IDS. Compared to the next-best approach (CNN-BiLSTM at 99.90%), our 99.97% represents a 70% reduction in the miss rate (from 0.10% to 0.03%). In a scenario with 100,000 attack flows, we miss ~30 vs. ~100 — a meaningful improvement for security.

### Q89. How efficient is the model compared to the 15 baseline approaches?
**A:** TCN-HMAC has: the smallest parameter count (157K vs. 300K–5M), smallest model size (612 KB vs. 1.2 MB–20 MB), fastest inference (0.17ms vs. 0.3ms–5ms). It is the only model with full parallelizability, stable gradients, AND communication authentication. It achieves this while ranking #2 in accuracy (99.85% vs. 99.90%) and #1 in detection rate (99.97%) and false alarm rate (0.37%).

### Q90. What is the total overhead of the combined TCN-HMAC framework?
**A:** TCN inference: ~0.17ms + HMAC computation: ~2μs ≈ 0.17ms combined per flow. Auxiliary agent overhead: 4.4% CPU, 13.7 MB RAM. HMAC bandwidth: 32 bytes per message. Shadow table verification: every 30 seconds per switch. Key rotation: every 3600 seconds. Challenge-response: every 1800 seconds. Total: negligible impact on network performance — well within the latency budget that production networks demand.

---

## Section 7: Limitations, Future Work, and General Questions (Q91–Q100)

### Q91. What are the main limitations of your work?
**A:** (1) Single dataset (InSDN only) — generalization to other networks unconfirmed; (2) Binary classification only — no multi-class attack identification; (3) Lab environment — may not capture production complexity; (4) HMAC is theoretically designed, not implemented in production controller; (5) No adversarial robustness evaluation; (6) No concept drift study over extended periods; (7) Single-controller architecture assumption.

### Q92. How would you extend this to multi-class classification?
**A:** Replace the sigmoid output with softmax over 5 classes (Normal, DDoS, MITM, Probe, Brute-force), use categorical cross-entropy loss with per-class weights, and adjust the preprocessing to retain original attack labels instead of binary encoding. The same TCN architecture could be used with the final dense layer outputting 5 neurons. This would provide more actionable intelligence for security analysts but might reduce per-class accuracy due to inter-class feature overlap.

### Q93. How would adversarial attacks affect your model, and how could you defend?
**A:** Adversarial examples — carefully crafted inputs designed to evade detection — are a real concern. An attacker could craft traffic whose features, after PCA transformation, land in the benign region. Defenses: (1) Adversarial training — augment training data with adversarial examples created by FGSM/PGD; (2) Input gradient monitoring to detect out-of-distribution inputs; (3) Ensemble methods combining multiple models; (4) Certified defenses providing provable robustness guarantees. This is identified as a priority for future work.

### Q94. How would concept drift affect the model over time?
**A:** As network traffic patterns and attack techniques evolve, the model's learned representations may become outdated, reducing detection accuracy. Mitigation strategies: (1) Periodic retraining (feasible since training takes only 5 minutes); (2) Online/incremental learning to update the model with new traffic without full retraining; (3) Monitoring model confidence — decreasing prediction confidence over time signals concept drift; (4) Maintaining a feedback loop where confirmed misclassifications trigger retraining.

### Q95. How would you extend the framework to multi-controller SDN architectures?
**A:** Multi-controller extensions require: (1) Inter-controller key agreement protocol for HMAC key synchronization; (2) Distributed shadow tables replicated or partitioned across controllers; (3) Consensus on flow rule verification results — detecting discrepancies that might indicate one compromised controller; (4) Federated model training so each controller trains locally and shares updates; (5) Load balancing detection queries across controllers to maintain real-time performance.

### Q96. What is the novelty of your work compared to existing literature?
**A:** Five novel contributions: (1) First framework combining temporal deep learning (TCN) with HMAC-based flow rule integrity verification in a unified SDN security solution; (2) First application of TCN specifically to InSDN dataset with SDN-tailored preprocessing; (3) Lightweight deployment strategy (612 KB model, 0.17ms inference) demonstrated to outperform architecturally more complex models; (4) Auxiliary agent architecture performing both shadow-table verification and challenge-response authentication; (5) Most extensive comparative analysis in SDN IDS literature (15 models compared).

### Q97. What is defense in depth and how does TCN-HMAC implement it?
**A:** Defense in depth is a security strategy using multiple independent layers of defense so that if one fails, others compensate. TCN-HMAC implements it through: (1) **TCN layer** detects anomalous traffic (DDoS, probes, brute-force, MITM traffic patterns); (2) **HMAC layer** verifies flow rule integrity (prevents tampered rules from executing); (3) **Challenge-response** authenticates controller/switch identity (prevents spoofing); (4) **Shadow table** verification detects unauthorized flow table modifications. If attacks slip past TCN, HMAC prevents resulting rogue rules. If HMAC keys are compromised, TCN detects malicious traffic.

### Q98. Can this framework be deployed in a real production SDN network today?
**A:** Partially. The TCN-IDS component can be deployed as-is — the 612 KB model loads in a Ryu controller application to classify flows in real time. The HMAC component is designed at the protocol level but needs production implementation in a controller framework (Ryu, ONOS, or OpenDaylight) with proper key management infrastructure. No OpenFlow protocol or switch hardware modifications are needed, which is a key advantage. Full deployment validation under real production workloads is recommended.

### Q99. What would you do differently if you started this research over?
**A:** Potential improvements: (1) Evaluate on multiple datasets from the start (InSDN, CIC-IDS-2017, UNSW-NB15) for generalization confidence; (2) Implement multi-class classification alongside binary for more actionable results; (3) Build and test the HMAC agent in a real Ryu controller deployment, not just theoretically; (4) Add adversarial robustness evaluation; (5) Consider attention mechanisms or squeeze-and-excitation blocks to see if they add value with our preprocessing pipeline; (6) Test with Mininet-based SDN emulation for end-to-end performance validation.

### Q100. What is the broader impact of this research on the field of network security?
**A:** Four broader impacts: (1) **Bridges the detection–verification divide** — demonstrates that IDS and integrity verification can be integrated into a single lightweight framework, challenging the assumption that these must be separate systems; (2) **Validates TCNs for security** — provides empirical evidence that temporal convolutions match or beat heavier architectures for network IDS while offering superior latency; (3) **Demonstrates practical deployability** — shows that deep learning IDS can run inside the SDN control loop without crippling performance (612 KB, 0.17ms), countering the perception that DL-based security is too heavy for production; (4) **Provides a benchmark** — the 15-model comparative analysis on InSDN provides a reference for future SDN IDS research.

---

*Prepared for thesis defense board — covers SDN fundamentals, TCN theory, HMAC mechanisms, dataset/preprocessing, model architecture/training, results/analysis, comparative analysis, limitations, and future work.*
