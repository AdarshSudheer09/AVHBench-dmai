# Prior Dominance in Audio-Visual LLMs

[![Conference](https://img.shields.io/badge/ICML_2026-Accepted-blue)](https://icml.cc)
[![Workshops](https://img.shields.io/badge/Workshops-FoGen_|_CompLearn_|_Learning_to_Listen-success)](#)

This repository contains the code, data, and mechanistic interpretability tools for research accepted at three ICML 2026 workshops. 

**Associated Papers:**
* **FoGen:** *Prior Dominance in Audio-Visual LLMs: When Generative Models Memorize Over Reasoning Under Cross-modal Conflict*
* **CompLearn:** *Compositional Failure in Audio-Visual LLMs: Late-Layer Prior Dominance Under Cross-modal Conflict*
* **Learning to Listen:** *Prior Dominance in Audio-Visual LLMs: When Generative Models Memorize Over Reasoning Under Cross-modal Conflict*

## Overview
We investigate where autoregressive Audio-Visual LLMs (specifically VideoLLaMA 2-7B-AV) substitute memorized distributional priors for reasoning when forced to process conflicting cross-modal inputs. 

Key findings:
* **Snap Layer Discovery:** Model commitment concentrates at layer 25.5 ± 1 across all configurations, regardless of alignment intervention.
* **Early Detection, Late Overwrite:** 21 conflict-resolution heads cluster at layers 15-18, indicating the model detects conflict early but overwrites it prior to generation.
* **Behavioral Collapse:** All fine-tuned configurations and base InternVideo2 collapse to near-chance under audio-visual conflict, shifting output priors rather than improving cross-modal reasoning.

## Repository Structure

The core methodology is split across the following scripts in the root directory:

### Mechanistic Audit & Analysis
* `mech_audit.py`: Core implementation of the head shutoff ablations.
* `capture script/`, `analyze logit lens/`, `analyze trajectory/`: Modules for extracting residual streams and tracking target token probability trajectories.
* `results_logit_lens_pipeline.json` / `results_logit_lens_only.json`: Captured logit lens output data.

### Training & Alignment Pipeline
* `train_acca.py`: Main training loop for the baseline alignment functionality.
* `acca_dataloader.py`: Data loading utilities for the pretraining baseline.
* `acca_loss.py`: Loss functions for the alignment interventions.

### Evaluation & Data Formatting
* `run_eval.py` & `evaluate.py`: Standard execution scripts for the AVHBench evaluations.
* `InternVideo2_eval.py`: Behavioral baseline evaluations for off-the-shelf InternVideo2.
* `grade.py` & `score_json.py`: Utilities for grading and scoring model outputs via exact-match.
* `build_conflict_qa.py`: Scripts for curating the adversarial conflict split.
* `Counterfactual Modality Diagnostics/` & `extract_failures.py`: Tools for isolating model failure modes.

## Installation

```bash
git clone [https://github.com/AdarshSudheer09/AVHBench-dmai.git](https://github.com/AdarshSudheer09/AVHBench-dmai.git)
cd AVHBench-dmai
```

## Quick Start

*(Ensure your environment is configured with the necessary weights for VideoLLaMA 2-7B-AV before running.)*

**Mechanistic Audit**
```bash
python mech_audit.py 
```

**Training Alignment Baselines**
```bash
python train_acca.py 
```

**Evaluation**
```bash
python run_eval.py 
python InternVideo2_eval.py
```

## Citation
If you find this code or our findings useful, please cite our papers. 

**For the mechanistic audit and prior dominance findings (FoGen):**
```bibtex
@inproceedings{sudheer2026prior_fogen,
  title={Prior Dominance in Audio-Visual LLMs: When Generative Models Memorize Over Reasoning Under Cross-modal Conflict},
  author={Sudheer, Adarsh and Li, David and Elbanna, Omar and Kodarapu, Ishaan and Bahuguna, Arjun and Sharma, Vasu},
  booktitle={ICML 2026 Workshop on Foundations of Deep Generative Models (FoGen)},
  year={2026}
}
```

**For the temporal alignment and behavioral evaluations (CompLearn):**
```bibtex
@inproceedings{sudheer2026prior_complearn,
  title={[Insert your exact CompLearn paper title here]},
  author={Sudheer, Adarsh and Li, David and Elbanna, Omar and Kodarapu, Ishaan and Bahuguna, Arjun and Sharma, Vasu},
  booktitle={ICML 2026 Workshop on Compositional Learning (CompLearn)},
  year={2026}
}
```

**For the audio-specific multi-modal integration findings (Learning to Listen):**
```bibtex
@inproceedings{sudheer2026prior_listen,
  title={[Insert your exact Learning to Listen paper title here]},
  author={Sudheer, Adarsh and Li, David and Elbanna, Omar and Kodarapu, Ishaan and Bahuguna, Arjun and Sharma, Vasu},
  booktitle={Learning to Listen: ICML 2026 Workshop on Machine Learning for Audio},
  year={2026}
}
```
