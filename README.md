# Sharpness-Aware Minimization and Test NLL Beyond Temperature Scaling on Fashion-MNIST

<div align="center">

<a href="https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/workflow.svg">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="workflow-dark.svg">
  <img alt="Artifact workflow — how every artifact in this repo was built" src="workflow.svg">
</picture>
</a>

<sub>🖱️ <b><a href="https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/workflow.svg">Open the interactive diagram</a></b> — every card links to its artifact folder.</sub>

</div>

> **TL;DR** — On a small convolutional network trained on Fashion-MNIST, SAM has a lower test negative log-likelihood than validation-tuned SGD after both models are temperature-scaled: 0.033 lower (95% CI -0.040 to -0.026; Holm-corrected p = 0.0018). This is a single tuned estimate over five paired seeds, not specified in advance and not reproduced independently, so it is a lead rather than a confirmed result. Shuffled-label controls give larger NLL gaps than the real comparison, against raw SGD and, post hoc, against temperature-scaled SGD, so they do not show that the advantage reflects learned label information. SAM's lower raw ECE disappears once SGD is temperature-scaled, and its accuracy advantage holds at the tuned SGD learning rate but not at 0.01.

<details>
<summary>Full hypothesis</summary>

For a small two-block CNN trained on Fashion-MNIST with matched architecture, data, schedule and tuned learning rates, sharpness-aware minimization (SAM) gives lower test negative log-likelihood than plain SGD, and this NLL gain survives temperature scaling of the SGD model fitted on validation logits, so it reflects better ranking of confidence rather than a global confidence shift. SAM's raw test ECE advantage does not survive that control: against temperature-scaled SGD the ECE difference is indistinguishable from zero. Test accuracy is unchanged within seed noise. Whether the NLL gain holds at matched training loss and with learning rates away from the lower grid edge is not yet established and is the next test.

</details>

[![Read the presentation](https://img.shields.io/badge/Read-Presentation-8A2BE2?style=for-the-badge)](https://ai-inventor-papers.github.io/ai-invention-d46016-sharpness-aware-training-makes-small/) [![Explore the interactive demo](https://img.shields.io/badge/Explore-Interactive_Demo-D9480F?style=for-the-badge)](https://ai-inventor-papers.github.io/ai-invention-d46016-sharpness-aware-training-makes-small/interactive.html)

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/paper.pdf) [![Read the full report](https://img.shields.io/badge/Read-Full_Report-blue)](https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/report.pdf) [![Read the executive summary](https://img.shields.io/badge/Read-Executive_Summary-blue)](https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/exec_summary.pdf) [![Read the internal report](https://img.shields.io/badge/Read-Internal_Report-green)](https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/round-2/report.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/paper_latex)

**Round reports:** [Round 1](https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/round-1/report.pdf) · [Round 2](https://cdn.jsdelivr.net/gh/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small@main/round-2/report.pdf)

This repository contains all **5 artifacts** produced across **2 rounds** of an autonomous AI research run — round by round, exactly in the order they were invented.

## Round 1

| Artifact | Type | Demo | Source | Builds on |
|----------|------|------|--------|-----------|
| **[Does SAM Improve Calibration on Fashion-MNIST?](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-1)** | [![experiment](https://img.shields.io/badge/experiment-8b5cf6)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/blob/main/round-1/experiment-1/demo/method_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-1/src) | — |
| **[Does SAM's calibration gain track overfitting?](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-2)** | [![experiment](https://img.shields.io/badge/experiment-8b5cf6)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/blob/main/round-1/experiment-2/demo/method_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-2/src) | — |

## Round 2

| Artifact | Type | Demo | Source | Builds on |
|----------|------|------|--------|-----------|
| **[SAM vs SGD calibration on Fashion-MNIST](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/experiment-3)** | [![experiment](https://img.shields.io/badge/experiment-8b5cf6)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/experiment-3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/blob/main/round-2/experiment-3/demo/method_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/experiment-3/src) | — |
| **[SAM vs SGD Calibration at Matched Training Loss](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/experiment-4)** | [![experiment](https://img.shields.io/badge/experiment-8b5cf6)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/experiment-4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/blob/main/round-2/experiment-4/demo/method_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/experiment-4/src) | — |
| **[SAM calibration gap across training length](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/evaluation-1)** | [![evaluation](https://img.shields.io/badge/evaluation-10b981)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/evaluation-1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/blob/main/round-2/evaluation-1/demo/eval_code_demo.ipynb) | [![Source Code](https://img.shields.io/badge/Source_Code-2962FF)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-2/evaluation-1/src) | <sub><i>uses:</i><br/>[experiment‑1&nbsp;(R1)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-1)<br/>[experiment‑2&nbsp;(R1)](https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/tree/main/round-1/experiment-2)</sub> |

## Repository Structure

Artifacts are grouped by the round of invention that produced them. Each
artifact has its own folder with source code and a self-contained demo:

```
.
├── round-1/                         # One folder per round of invention
│   ├── experiment-1/
│   │   ├── README.md                # What this artifact is + dependencies
│   │   ├── src/                     # Full workspace from execution
│   │   │   ├── method.py            # Main implementation
│   │   │   ├── method_out.json      # Full output data
│   │   │   └── ...                  # All execution artifacts
│   │   └── demo/                    # Self-contained demo
│   │       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
│   ├── dataset-1/
│   │   ├── src/
│   │   └── demo/
│   └── evaluation-1/
│       ├── src/
│       └── demo/
├── round-2/                         # Later rounds build on earlier artifacts
├── paper.pdf                        # Research paper
├── paper_latex/                     # LaTeX source files
├── report.pdf                       # Full internal report — every experiment, table and dead end
├── report_latex/                    # LaTeX source of the report
├── exec_summary.pdf                 # Executive summary of the report, at most four pages
├── chat/                            # Every prompt, response and tool call, per module
├── workflow.svg                     # Artifact dependency diagram (this page's header)
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small
cd ai-invention-d46016-sharpness-aware-training-makes-small

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook <artifact_folder>/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
