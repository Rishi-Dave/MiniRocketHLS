# Plan: UCR Undergraduate Symposium Presentation

## Context
Rishi is presenting his MiniRocketHLS research at the UCR Undergraduate Research Symposium (oral presentation, 15-minute slot for completed research + 5 min Q&A). He needs a slideshow with detailed speaker notes. The project — "FPGA Acceleration of Convolution-based Classification Algorithms for Streaming Time Series" — was submitted to FCCM 2026 and represents the **first hardware accelerator for MiniRocket and HYDRA time series classifiers**.

The audience is CE/EE/CS students and faculty — technically literate but NOT HLS specialists. The presentation must be accessible, compelling, and backed by hardware-validated results.

## Deliverable
A single, self-contained **prompt for Claude cowork** that will generate:
- A complete slide deck outline (15 slides, ~12.5 min spoken time)
- **Full speaker notes for each slide** (near-verbatim script of what to say)
- Guidance on which existing figures to embed from the project

This is NOT about creating the actual PowerPoint file — it's about generating the complete content (text + speaker notes) that Rishi can paste into slides.

---

## Claude Cowork Prompt

Below is the exact prompt to paste into Claude cowork:

---

### PROMPT START

You are helping me create a 15-minute oral presentation for the UCR Undergraduate Research Symposium. I am Rishi Dave, a Computer Engineering undergraduate at UC Riverside, Bourns College of Engineering. My faculty mentor is Dr. Philip Brisk.

**Presentation requirements:**
- 15 minutes for completed research, followed by up to 5 min Q&A
- First slide must use UCR symposium template (name, major, title, college, track #)
- Audience: CE/EE/CS students and faculty — technically literate but not FPGA specialists
- Dress code: business casual
- I need to self-introduce with: name, major, faculty mentor, and presentation title

**My research:** I built the first-ever FPGA hardware accelerator for two state-of-the-art time series classification algorithms — MiniRocket and HYDRA. The paper is titled "FPGA Acceleration of Convolution-based Classification Algorithms for Streaming Time Series" and was submitted to FCCM 2026 (IEEE International Symposium on Field-Programmable Custom Computing Machines). The target FPGA is the AMD/Xilinx Alveo U280, and I used Vitis HLS 2023.2 for development.

---

**Please generate a complete 15-slide presentation with full speaker notes for each slide. For each slide, provide:**
1. **Slide title**
2. **Bullet points / visual content** (what appears on the slide)
3. **Figure suggestion** (which figure from my project to embed — I'll list them below)
4. **Speaker notes** (a near-verbatim script of what I will say — written in first person, conversational but professional, timed to the allocation below)

---

## SLIDE-BY-SLIDE SPECIFICATION

### Slide 1 — Title Slide (0:00–0:30)
**On slide:** Name: Rishi Dave | Major: Computer Engineering | Title: FPGA Acceleration of Convolution-based Classification Algorithms for Streaming Time Series | College: Bourns College of Engineering | Track: [TBD] | Faculty Mentor: Dr. Philip Brisk
**Speaker notes direction:** Self-introduction. State name, major, mentor, title. Brief one-sentence hook: "Today I'll show you how an FPGA can classify streaming sensor data up to 58 times faster than a CPU — and use 92 times less energy doing it."

### Slide 2 — Motivation: Why Time Series Classification Matters (0:30–1:30)
**On slide:** 
- Sensor data streams everywhere: medical wearables, industrial monitoring, audio classification, network security
- Classification must happen in real time — you can't batch-process a mosquito-borne disease alert
- Hook: "Can you tell a mosquito from a fruit fly by its wingbeat sound? Our accelerator can — in under a millisecond."
**Figure:** CPU-vs-FPGA network diagram showing CPU NIC path (data copied through kernel→user space) vs FPGA SmartNIC (direct processing). File: `PaperTexFiles/Figures/Introduction/cpu-vs-fpga-network.png`
**Speaker notes direction:** Make the audience care about the problem. Use the mosquito example — it's from the actual MosquitoSound dataset. Explain that time series data is generated continuously by sensors and needs real-time processing. Transition: "But the algorithms that are best at this classification have never been hardware-accelerated — until now."

### Slide 3 — The Gap: No Hardware Acceleration Exists (1:30–2:30)
**On slide:**
- MiniRocket (2021) and HYDRA (2023): state-of-the-art accuracy on 109 benchmark datasets
- Both use random convolutional kernels — fast to train, no GPU-intensive backpropagation
- Critical gap: **no FPGA or ASIC accelerator has ever been built for these algorithms**
- This work is the first
**Figure:** None needed — clean text slide with emphasis box: "First hardware accelerator for MiniRocket and HYDRA"
**Speaker notes direction:** Establish the novelty. Explain that researchers who develop these algorithms work in Python and don't do hardware optimization. Mention the 2024 "bake-off" (time series classification competition) identified MultiRocket+Hydra as best-in-class, yet nobody has built hardware for them. "That's the gap our work fills."

### Slide 4 — How MiniRocket Works (2:30–3:45)
**On slide:**
- Input: a time series (e.g., 600 audio samples from an insect wingbeat)
- Step 1: Apply 84 convolutional kernels at multiple dilation levels (~10,000 convolutions total)
- Step 2: Each kernel has 9 weights, restricted to {-1, 2} — NOT learned, randomly generated
- Step 3: For each convolution output, compute PPV (Proportion of Positive Values) = fraction of outputs above a threshold
- Step 4: Feed the ~10,000 PPV features into a simple linear classifier (ridge regression)
- Key insight: the {-1, 2} weight restriction is a huge hardware opportunity
**Figure:** Dilation illustration showing convolution at d=1 and d=2 (`PaperTexFiles/Figures/conv_d1.pdf` or a simplified version). Optionally also the PPV/reduction functions diagram (`PaperTexFiles/Figures/Reductions.pdf`).
**Speaker notes direction:** Walk through the pipeline simply. Emphasize that the kernels are NOT learned like in a neural network — they're fixed by design. The magic is in the feature extraction (PPV), not in the weights. Explain dilation intuitively: "Dilation lets the same short kernel look at the data at different zoom levels — like looking at a signal through binoculars at different magnifications." Transition: "The restriction to weights of -1 and 2 is what makes this algorithm special for FPGAs — and I'll show you why in a moment."

### Slide 5 — How HYDRA Works (3:45–4:30)
**On slide:**
- HYDRA extends MiniRocket: competing pairs of kernel groups
- Uses multiple dilation values simultaneously — captures multi-scale patterns
- More features → higher accuracy on complex datasets
- Same classification head: linear model on the feature vector
- Trade-off: more compute-intensive (learned kernels, more features)
**Figure:** `PaperTexFiles/Figures/Hydra_Arch.pdf` or `conv_d2.pdf`
**Speaker notes direction:** Keep this brief — HYDRA is an extension. Key point: "HYDRA achieves higher accuracy but requires more computation, which makes hardware acceleration even more valuable." Don't go deep into the math.

### Slide 6 — Why FPGAs? (4:30–5:15)
**On slide:**
- GPUs excel at batch processing — collect 256 samples, process together
- But streaming data arrives one sample at a time
- FPGAs are always-on pipelines: one sample in → one result out, continuously
- FPGA as SmartNIC: classify data at the network card, before it even reaches the CPU
- Low, predictable power: ~26W regardless of load
**Figure:** Re-use CPU-vs-FPGA diagram or a simple comparison graphic
**Speaker notes direction:** Address the "why not GPU?" question preemptively. "GPUs are great at batch throughput — if you can wait to collect hundreds of samples. But for real-time streaming, where each data point needs an immediate answer, FPGAs have a fundamental advantage: the data flows through the chip like water through a pipe, with no batching overhead." Mention SmartNIC use case. Transition: "Let me show you the architecture we built."

### Slide 7 — System Architecture (5:15–6:30)
**On slide:**
- Block diagram showing: CMAC (100G Ethernet) → UDP → Circular Buffers (Time Series + First-Order Difference) → Convolution Engine → Pooling Engine (PPV) → Regression Classifier → Output
- Label: "Data flows left to right — no off-chip memory bottleneck"
- Note: 1, 2, or 3 Compute Units (CUs) run in parallel on the Alveo U280
**Figure:** `PaperTexFiles/Figures/Mini-MultiRocket_Arch.pdf` — the main FPGA architecture diagram
**Speaker notes direction:** Walk through the diagram left-to-right. "Data comes in from the network through a 100-gigabit Ethernet interface. It's stored in circular buffers — think of a sliding window that always holds the most recent data points. The Convolution Engine applies all 84 kernels, the Pooling Engine extracts the PPV features, and the Regression Classifier makes the prediction. The entire pipeline runs at 300 megahertz, and we can replicate it up to 3 times on the same chip for parallel processing." Don't mention HLS pragmas.

### Slide 8 — Key Optimization 1: Dot-Product Reuse (6:30–7:45)
**On slide:**
- Naive approach: when a new data point arrives, recompute the entire convolution from scratch
- Smart approach: the sliding window shifts by 1 — most of the dot product is already computed
- Reuse: subtract the oldest contribution, add the newest one
- Result: computation per step becomes independent of time series length
- Analogy: "Like a running average — you don't re-add all the numbers each time"
**Figure:** `PaperTexFiles/Figures/Convolution Engine-a.pdf` and `Convolution Engine-b.pdf` — before/after dot product reuse, side by side
**Speaker notes direction:** This is the most intuitive optimization — spend time here. "Imagine you're computing a running average of 9 numbers. When a new number arrives, you don't add up all 9 again — you subtract the oldest and add the newest. Our Convolution Engine does exactly this with the dot products. When a new data point arrives in the stream, we reuse all the dot products we already computed and only calculate the new ones. This means that the amount of work per step doesn't grow with the length of the time series — it's constant." Use the figure to show the green (reused) vs red (new) dot products.

### Slide 9 — Key Optimizations 2 & 3: bFP Multiplier + Fused CONV+PPV (7:45–8:45)
**On slide (two-column layout):**

**Left column — Binary Floating-Point Multiplier:**
- MiniRocket weights are always {-1, 2}
- Multiply by -1: flip the sign bit (1 logic gate)
- Multiply by 2: increment the exponent (simple adder)
- Result: **zero DSP blocks** used for multiplication
- Traditional floating-point multiplier: 2-3 DSPs each

**Right column — Fused CONV+PPV:**
- Naive: store all convolution outputs, then compute PPV in a second pass
- Fused: compute PPV comparison in the same clock cycle as convolution
- Never write intermediate results to memory
- Result: **0 DSPs, 68% less memory (BRAM)**

**Figure:** Optional small diagram or equation for bFP: `XNOR(X.sign, y), X.exponent + y, X.mantissa`
**Speaker notes direction:** "The second optimization exploits a property unique to MiniRocket. Since the kernel weights can only be -1 or 2, we don't need a general-purpose multiplier. Multiplying a floating-point number by -1 is just flipping a single bit — the sign bit. Multiplying by 2 is just incrementing the exponent field. This means our entire MiniRocket convolution engine uses zero DSP blocks — the multiply is essentially free." Then briefly: "We also fuse the convolution and pooling into one pass, eliminating an entire intermediate array and cutting memory usage by 68%."

### Slide 10 — Results: Throughput and Speedup (8:45–9:45)
**On slide:**

| Algorithm | Dataset (length) | CPU C++ | FPGA 3CU | Speedup |
|-----------|-----------------|---------|----------|---------|
| MiniRocket | InsectSound (600) | 289 inf/s | 5,012 inf/s | **17.3x** |
| MiniRocket | MosquitoSound (3750) | 48 inf/s | 2,820 inf/s | **58.7x** |
| MiniRocket | FruitFlies (5000) | 258 inf/s | 2,191 inf/s | **8.5x** |
| HYDRA | InsectSound (600) | 1,372 inf/s | 10,573 inf/s | **7.7x** |
| HYDRA | MosquitoSound (3750) | 326 inf/s | 3,516 inf/s | **10.8x** |
| HYDRA | FruitFlies (5000) | 245 inf/s | 2,972 inf/s | **12.1x** |

**Figure:** `results/figures/fig1_throughput_comparison.png` — throughput bar chart
**Speaker notes direction:** "Now let's look at the results. All of these numbers are hardware-validated — measured on a real Alveo U280 FPGA, not simulated." Lead with the headline: "For MiniRocket on the MosquitoSound dataset, our FPGA achieves 2,820 inferences per second — that's 58.7 times faster than an optimized C++ baseline running on an Intel Xeon server CPU." Explain the pattern: "Notice that the speedup grows with sequence length for MiniRocket — 8.5x for short sequences up to 58.7x for long ones. That's the dot-product reuse optimization paying off — the longer the series, the more redundant work we avoid." Briefly note HYDRA achieves 7.7 to 12.1x speedup.

### Slide 11 — Results: Energy Efficiency (9:45–10:30)
**On slide:**

| Dataset | CPU (mJ/inf) | GPU T4 (mJ/inf) | FPGA 3CU (mJ/inf) | FPGA vs CPU |
|---------|-------------|-----------------|-------------------|-------------|
| InsectSound | 182.7 | 2.2 | 5.1 | **36x** |
| MosquitoSound | 833.3 | 12.9 | 9.1 | **92x** |
| FruitFlies | 193.4 | 16.7 | 11.7 | **17x** |

- FPGA power: ~25.7W (static-dominated — chip is 95% idle)
- CPU: 40–53W | GPU T4: 54–68W
**Figure:** `results/figures/fig6_energy_efficiency.png`
**Speaker notes direction:** "Energy efficiency is where the FPGA really shines. On MosquitoSound, the FPGA uses just 9.1 millijoules per inference — that's 92 times more efficient than the CPU. What's remarkable is that the FPGA draws only 25.7 watts total, and most of that is static power — the chip is 95% idle even while processing thousands of inferences per second. For always-on deployment — a sensor node running 24/7 — this difference is enormous."

### Slide 12 — Honest GPU Comparison (10:30–11:15)
**On slide:**

| Algorithm | Dataset | GPU T4 (batch=256) | FPGA 3CU | GPU advantage |
|-----------|---------|-------------------|----------|---------------|
| MiniRocket | InsectSound | 30,469 inf/s | 5,012 inf/s | 6.1x |
| MiniRocket | MosquitoSound | 4,198 inf/s | 2,820 inf/s | 1.5x |
| HYDRA | InsectSound | 13,895 inf/s | 10,573 inf/s | 1.3x |

- **But:** GPU at batch=1 (true streaming): FPGA competitive or faster
- **But:** FPGA at <5% resource utilization — massive scaling headroom
- **But:** Network latency: FPGA 4.12ms RTT vs CPU 11.0ms = **2.65x lower**
**Figure:** `results/figures/fig4_scaling.png` — GPU/FPGA throughput ratio vs sequence length
**Speaker notes direction:** Be honest and preemptive. "Now, you might be wondering — how does this compare to a GPU? Let me be upfront: a Tesla T4 GPU with batch processing beats our FPGA on raw throughput by 1.3 to 6.1 times. But there are three important caveats. First, batch processing requires collecting hundreds of samples before you can classify — in a real-time streaming scenario, you can't wait. At batch size 1, the FPGA is competitive. Second, our FPGA is using less than 5% of the available chip resources — we have enormous room to scale up. And third, when you configure the FPGA as a network SmartNIC, the round-trip latency drops to 4.1 milliseconds compared to 11 milliseconds for a CPU server — that's 2.65 times faster end-to-end."

### Slide 13 — Accuracy (11:15–11:45)
**On slide:**
- HYDRA: **exact match** between FPGA and Python reference accuracy
- MiniRocket: within 5% — difference from float64→float32 model export, not hardware error
- No accuracy-throughput tradeoff: all speedups are "free"
- Datasets: InsectSound (10-class), MosquitoSound (6-class), FruitFlies (3-class) — all from the UCR Time Series Archive (maintained here at UC Riverside!)
**Figure:** None — clean text slide
**Speaker notes direction:** "Importantly, we achieve these speedups without sacrificing classification accuracy. HYDRA results match the Python reference exactly — bit for bit. MiniRocket has a small difference, under 5%, which comes from converting the model weights from 64-bit to 32-bit floating point during export — it's a software-side precision choice, not a hardware limitation. And I should note — the benchmark datasets we use are from the UCR Time Series Archive, maintained right here at UC Riverside by Professor Keogh's lab."

### Slide 14 — Future Work (11:45–12:30)
**On slide:**
- Scale to full U280 utilization: currently at 5% → target 100K+ inf/s
- Extend to additional algorithms: ROCKET, MultiRocket+Hydra ensemble
- Explore fixed-point arithmetic to reduce HYDRA's DSP usage
- Target edge FPGAs (Zynq UltraScale+) for embedded deployment
- Integrate end-to-end on-chip classifier
**Figure:** `results/figures/cu_scaling.png` — 1→2→3 CU scaling trend, with projected arrow
**Speaker notes direction:** "Looking ahead, there's a clear path to dramatically higher performance. We're using less than 5% of the chip — by scaling up the number of compute units and optimizing further, we believe we can exceed 100,000 inferences per second on a single FPGA. We also plan to extend this work to additional algorithms in the time series classification family, and to port the design to smaller, lower-power edge FPGAs for embedded deployment in sensor nodes."

### Slide 15 — Conclusion & Acknowledgments (12:30–13:00)
**On slide:**
- **First** hardware accelerator for MiniRocket and HYDRA time series classifiers
- Three novel optimizations: dot-product reuse, binary FP multiplier, fused CONV+PPV
- Hardware-validated: up to **58.7x CPU speedup**, **92x energy efficiency**, **2.65x lower network latency**
- Under 5% FPGA utilization — this is a starting point, not a ceiling
- Submitted to FCCM 2026

**Acknowledgments:** Dr. Philip Brisk (faculty mentor), UCR Bourns College of Engineering, Open Cloud Testbed (OCT) for FPGA hardware access
**Figure:** Optional small callback to architecture diagram
**Speaker notes direction:** "To summarize: we presented the first hardware accelerator for MiniRocket and HYDRA, two state-of-the-art time series classification algorithms. Through three key optimizations — streaming dot-product reuse, a binary floating-point multiplier, and fused convolution with pooling — we achieved up to 58.7x speedup over CPU and 92x better energy efficiency. And we did this using less than 5% of the FPGA's resources, leaving enormous room to grow. I'd like to thank my faculty mentor Dr. Philip Brisk, and the Open Cloud Testbed for providing FPGA hardware access. I'm happy to take questions."

---

## ADDITIONAL INSTRUCTIONS FOR CLAUDE COWORK

**Style guidelines for speaker notes:**
- Write in first person ("I", "we", "our")
- Conversational but professional — imagine presenting to a room of 30 people
- Use simple analogies for technical concepts (running average, water through a pipe, binoculars)
- Pause after key numbers — "58.7 times faster [pause]. Let that sink in."
- Anticipate transitions between slides — end each slide's notes with a sentence that leads into the next
- Total spoken time: ~13 minutes (leaving 2-minute buffer within the 15-minute slot)

**Anticipated Q&A questions to prepare for (include brief answers):**
1. "Why not just use a GPU?" — Addressed in slide 12, but reinforce: batch vs streaming, energy, latency, utilization headroom
2. "How does the accuracy compare?" — Exact match for HYDRA, within 5% for MiniRocket due to float precision
3. "How much does the FPGA cost vs a GPU?" — Alveo U280 ~$5K, Tesla T4 ~$2K, but FPGA is reconfigurable and lower TCO for always-on edge deployment
4. "Could this work on smaller/cheaper FPGAs?" — Yes, 5% utilization means it could fit on much smaller Zynq or Kintex devices
5. "What are the limitations?" — Batch throughput gap vs GPU, float64→float32 precision loss for MiniRocket, currently single-sample dispatch (no batch mode on FPGA)
6. "What datasets did you use?" — UCR Time Series Archive (hosted at UCR!): InsectSound, MosquitoSound, FruitFlies

**Figure files available for embedding (all exist in the project):**
- `MiniRocketHLS/results/figures/fig1_throughput_comparison.png` — throughput bars
- `MiniRocketHLS/results/figures/fig2_latency_cdf.png` — latency CDF
- `MiniRocketHLS/results/figures/fig4_scaling.png` — GPU/FPGA scaling vs length
- `MiniRocketHLS/results/figures/fig5_latency_breakdown.png` — H2D/kernel/D2H breakdown
- `MiniRocketHLS/results/figures/fig6_energy_efficiency.png` — energy per inference
- `MiniRocketHLS/results/figures/cu_scaling.png` — 1 to 3 CU scaling
- `MiniRocketHLS/PaperTexFiles/Figures/Mini-MultiRocket_Arch.pdf` — FPGA architecture
- `MiniRocketHLS/PaperTexFiles/Figures/Hydra_Arch.pdf` — HYDRA architecture
- `MiniRocketHLS/PaperTexFiles/Figures/Convolution Engine-a.pdf` / `-b.pdf` — dot product reuse
- `MiniRocketHLS/PaperTexFiles/Figures/Introduction/cpu-vs-fpga-network.png` — motivation
- `MiniRocketHLS/PaperTexFiles/Figures/conv_d1.pdf` / `conv_d2.pdf` — dilation illustrations
- `MiniRocketHLS/PaperTexFiles/Figures/Reductions.pdf` — PPV/MPV/MIPV/LSPV

### PROMPT END

---

## Verification
After generating the presentation with Claude cowork:
1. Read through all speaker notes aloud — time yourself to confirm ~13 minutes
2. Check that every number cited matches the tables above
3. Verify figure file paths exist
4. Practice the Q&A answers
5. Create slides in PowerPoint/Google Slides using the UCR template, paste in content
