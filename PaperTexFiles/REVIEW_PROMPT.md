# Paper Review & Polish Prompt

Copy everything below the line into a fresh Claude conversation (or Claude Code tab) with access to the `PaperTexFiles/` directory.

---

You are acting as a senior systems-conference reviewer AND copy-editor for an FCCM 2026 resubmission. Your job is to systematically transform this paper into the strongest possible version before the April 14 deadline. The paper is at `PaperTexFiles/` and compiles via `pdflatex fccm.tex`.

## Your Background

This paper presents FPGA accelerators for MiniRocket and HYDRA time series classification on a Xilinx Alveo U280. It was previously rejected at FCCM with scores 2, 3, 3, 4. The authors (a professor, a PhD advisor, and an undergraduate) have since collected substantial new data to address every reviewer complaint. The professor and advisor wrote the original text; the undergraduate (with Claude Code assistance) collected all the new experimental data and drafted new table/narrative additions.

## FCCM Reviewer Complaints (All Must Be Addressed)

The original submission was rejected for these specific reasons:

1. **No GPU baseline** (all reviewers) — ADDED: `gpu_comparison.tex` with Tesla T4 at batch=256 and batch=1, for both MiniRocket and HYDRA
2. **Only compared vs Python CPU** (Reviewer C, strongest critic) — ADDED: MiniRocket now compared against optimized C++ (`-O3 -march=native`); HYDRA still vs Python (no C++ HYDRA implementation exists — this must be stated explicitly)
3. **No accuracy tables** (all reviewers) — ADDED: `accuracy_table.tex` with Python, C++, and FPGA accuracy for all datasets
4. **Missing resource utilization details** (Reviewers A, B) — ADDED: `resource_annotated.tex` with per-module breakdown (LUT, FF, DSP, BRAM)
5. **Limited scalability analysis** (Reviewers B, D) — ADDED: `cu_scaling.tex` (1/2/3 CU), plus HYDRA multi-CU data in `throughput_table.tex`
6. **Missing energy/power data** (Reviewer B) — ADDED: `power_table.tex` with CPU (RAPL-measured), GPU (nvidia-smi), FPGA (xbutil) power and energy-per-inference
7. **DSP count unexplained** — Fused kernel achieves 0 DSP via bFP multiplier for ternary weights; HYDRA uses 1,355 DSPs for learned kernels. Both explained in Sec 5.1.
8. **No CNN differentiation** — MiniRocket uses fixed random ternary kernels with NO backpropagation; must be clearly distinguished from CNNs in Sec 2
9. **Classifier description missing** — Ridge regression must be described and justified
10. **Power/energy terminology errors** — Must use consistent terminology (power in watts, energy in joules or mJ/inf)

**Verify that EVERY one of these 10 complaints is visibly addressed in the current draft. If any is weak or missing, flag it.**

## Paper Structure

```
fccm.tex              — Main file, IEEEtran conference format
01_introduction.tex    — Introduction
02_background.tex      — MiniRocket & HYDRA algorithm background
03_architecture.tex    — FPGA accelerator architecture
04_experimental_setup.tex — Setup, baselines, methodology
05_experimental_results.tex — All results (7 tables, 1 figure)
06_related_work.tex    — Related work
07_conclusion.tex      — Conclusion
```

## Tables Available (all in PaperTexFiles/)

**Currently included in 05_experimental_results.tex:**
| File | Table | Content |
|------|-------|---------|
| `resource_annotated.tex` | Resource utilization | Per-module LUT/FF/DSP/BRAM for MiniRocket + HYDRA |
| `accuracy_table.tex` | Accuracy | Python vs C++ vs FPGA accuracy, all datasets |
| `throughput_table.tex` | Throughput | CPU/FPGA 1CU/2CU/3CU throughput + speedup |
| `fused_comparison.tex` | Fusion impact | Before/after CONV+PPV fusion |
| `cu_scaling.tex` | CU scaling | MiniRocket 1/2/3 CU scaling factors |
| `gpu_comparison.tex` | GPU comparison | GPU T4 vs FPGA at batch=256 and batch=1 |
| `power_table.tex` | Energy efficiency | CPU/GPU/FPGA power, energy/inference, efficiency ratio |

**NOT currently included (evaluate whether they should be):**
| File | Content | Notes |
|------|---------|-------|
| `hydra_comparison_table.tex` | HYDRA design variants (monolithic vs streaming dataflow) | Complete, has accuracy + throughput for both |
| `latency_table.tex` | P50/P95/P99 latency | Mostly empty for multi-CU rows — probably skip |

**Figures available in Figures/:**
- `fig4_scaling.pdf` — GPU/FPGA throughput ratio vs time series length (INCLUDED)
- `Mini-MultiRocket_Arch.pdf` — MiniRocket architecture diagram
- `Hydra_Arch.pdf` — HYDRA architecture diagram
- `Convolution Engine-a.pdf` / `Convolution Engine-b.pdf` — Conv engine designs
- `Reductions.pdf` — Reduction operations
- `conv_d1.pdf` / `conv_d2.pdf` — Convolution with dilation
- `first_order_difference.pdf` — First-order difference operation
- `Hydra_counts.pdf` — HYDRA histogram counting

## Constraints

- **Page limit**: FCCM uses IEEE conference format, typically 8 pages (10 with references). Check the `\documentclass[conference]{IEEEtran}` and any page-limit commands.
- **Do NOT add content** that isn't supported by actual measured data. Every number must trace to a table or benchmark.
- **MultiRocket was DROPPED** from this paper. Any lingering references to MultiRocket must be removed. The paper covers MiniRocket + HYDRA only.
- **The original FCCM submission text** (sections 01-04, 06) was written by a professor and PhD advisor. Preserve their voice. New text (added to 05, and parts of the abstract/intro/conclusion) was drafted to fill data gaps. It must read as if the same senior authors wrote it — no AI-isms, no undergraduate voice.

## Your Systematic Review Process

Execute these passes IN ORDER. After each pass, report findings before moving to the next.

### Pass 1: Structural Audit
- Read every .tex file start to finish
- Map the narrative arc: what claim does each section make? Do they build logically?
- Identify any section that feels out of place, too long, or too short relative to its importance
- Check: does the abstract accurately summarize the final paper (not an older version)?
- Check: does the conclusion match the results actually presented?

### Pass 2: Reviewer Complaint Checklist
- Go through all 10 complaints listed above
- For each one, identify the EXACT location (file:line) where it is addressed
- Rate each: FULLY addressed / PARTIALLY addressed / NOT addressed
- For any not fully addressed, propose specific fixes

### Pass 3: Table & Figure Audit
- For each of the 7 included tables: Is it referenced in the text? Is it discussed? Does the text accurately describe what the table shows?
- For the 2 excluded tables: Should either be included? What narrative value would it add vs. the page cost?
- For each figure: Is placement optimal? Is it referenced before it appears? Would any figure be better as a table or vice versa?
- Cross-check: Do numbers cited in the text (abstract, intro, results, conclusion) exactly match the numbers in the tables? Flag ANY discrepancy.

### Pass 4: Number Consistency Check
- Extract every numerical claim from the abstract, introduction, and conclusion
- Verify each against the corresponding table cell
- Check that speedup calculations are correct (e.g., 10,573 / 90 = 117.5x)
- Check that "ranges" (e.g., "87.4--117.5x") use the correct min and max from the table
- Verify power/energy calculations: Energy = Power / Throughput, Efficiency ratio = CPU_energy / Platform_energy

### Pass 5: Writing Quality & Voice
- Flag any sentence that sounds like it was written by AI (hedging phrases like "it is worth noting", "importantly", "notably", filler transitions)
- Flag any sentence that sounds like an undergraduate wrote it (informal tone, imprecise language, over-explaining obvious things)
- Flag any sentence that is redundant with another part of the paper
- Check for consistent terminology: "compute unit" vs "CU", "inference" vs "classification", "time series" (not "timeseries")
- Check LaTeX: consistent use of `\times`, `\,` for units, `--` for ranges, proper `\ref{}` usage
- Fix any typos or grammar errors IN PLACE (do not just flag them — fix them)

### Pass 6: Page Budget & Placement
- Estimate current page count (IEEE 2-column, 10pt)
- If over limit: identify what to cut (in priority order)
- If under limit: identify what to add (from excluded tables, expanded discussion)
- Check figure/table placement: LaTeX `[t]` vs `[h]` vs `[tp]` — are there any that would cause awkward gaps?
- Check for widows, orphans, and sections that start at the bottom of a column

### Pass 7: Final Sweep
- Read the paper one more time as if you are Reviewer C (the harshest, who gave a 2)
- What would they still complain about?
- Make final edits to preempt those complaints
- Verify the paper compiles cleanly with no `??` references, no overfull hboxes in table environments, no LaTeX warnings

## Output Format

For each pass, produce:
1. A summary of findings (bullet list)
2. Specific edits made (file, old text → new text)
3. Any unresolved issues that need author decision

At the end, produce a final assessment: "Ready to submit" or "Needs author attention on: [list]"

## Key Data for Cross-Checking

These are the HW-validated benchmark numbers. Every table and narrative claim must be consistent with these:

**MiniRocket (Fused Kernel, Alveo U280):**
| Dataset | Length | CPU C++ | FPGA 1CU | FPGA 2CU | FPGA 3CU | Python Acc | C++ Acc | FPGA Acc |
|---------|--------|---------|----------|----------|----------|------------|---------|----------|
| GunPoint | 150 | 746 | 3,797 | 5,999 | 9,585 | 98.67% | 98.33% | 98.33% |
| InsectSound | 600 | 289 | 1,967 | 3,191 | 5,012 | 74.12% | 74.12% | 74.12% |
| MosquitoSound | 3750 | 48 | 937 | 1,497 | 2,820 | 87.88% | 87.88% | 87.88% |
| FruitFlies | 5000 | 258 | 742 | 1,121 | 2,191 | 95.82% | 95.82% | 95.82% |

**HYDRA (v2_fixed, Alveo U280):**
| Dataset | Length | CPU Python | FPGA 1CU | FPGA 2CU | FPGA 3CU | Python Acc | FPGA Acc |
|---------|--------|------------|----------|----------|----------|------------|----------|
| InsectSound | 600 | 90 | 6,326 | 8,028 | 10,573 | 69.41% | 69.41% |
| MosquitoSound | 3750 | 39 | 1,937 | 2,757 | 3,516 | 70.05% | 70.05% |
| FruitFlies | 5000 | 34 | 1,507 | 2,148 | 2,972 | 87.61% | 87.61% |

**GPU (Tesla T4, PyTorch 2.10.0+CUDA 12.8):**
| Algorithm | Dataset | GPU b=256 | GPU b=1 |
|-----------|---------|-----------|---------|
| MiniRocket | InsectSound | 30,469 | 1,091 |
| MiniRocket | MosquitoSound | 4,198 | 661 |
| MiniRocket | FruitFlies | 3,227 | 539 |
| HYDRA | InsectSound | 13,895 | 2,711 |
| HYDRA | MosquitoSound | 5,139 | 2,805 |
| HYDRA | FruitFlies | 4,267 | 1,934 |

**Power (measured):**
- FPGA: 25.7W (xbutil, static-dominated at <5% utilization)
- GPU: 54-68W (nvidia-smi under sustained inference)
- CPU RAPL: InsectSound 52.8W, MosquitoSound 40.0W, FruitFlies 49.9W

**Key derived claims that must appear consistently:**
- MiniRocket speedup range (3CU/CPU): 8.5--58.7x
- HYDRA speedup range (3CU/Python): 87.4--117.5x
- GPU/FPGA ratio for HYDRA at 3CU: 1.3--1.5x
- At batch=1, FPGA 3-CU exceeds GPU on ALL datasets for BOTH algorithms
- FPGA power: 2.1--2.6x less than GPU (25.7W vs 54-68W)
