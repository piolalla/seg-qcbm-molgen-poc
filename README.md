# Quantum–Classical Molecular Generation Pipeline

This repository provides three molecular generation pipelines:

1. Classical SELFIES language-model generator  
2. Non-segmented Quantum Circuit Born Machine (QCBM)  
3. Segmented QCBM generator  

A unified rule-based scoring function is used across all models.

---

## Project Structure

    project_root/
    │
    ├── data/
    │   ├── clean_kras_g12d.csv
    │   ├── clean_kras_g12d.selfies.txt
    │   ├── gen_classic_round1.csv
    │   ├── gen_classic_round1_scored.csv
    │   ├── gen_qcbm_round1.csv
    │   ├── gen_qcbm_round1_scored.csv
    │   ├── gen_qcbm_segmented_round1.csv
    │   └── gen_qcbm_segmented_round1_scored.csv
    │
    ├── models/
    │
    ├── scripts/
    │   ├── train_classic_generator.sh
    │   ├── sample_and_score_classic.sh
    │   ├── train_qcbm_latent.sh
    │   ├── sample_and_score_qcbm_latent.sh
    │   ├── run_segmented_pipeline.sh
    │   ├── analyze_classic_baseline.sh
    │   ├── analyze_qcbm.sh
    │   └── analyze_all_results.sh
    │
    ├── src/
    │   ├── classic_selfies_lm.py
    │   ├── qgen_qcbm_latent.py
    │   ├── qgen_qcbm_segmented.py
    │   ├── preprocess_segments.py
    │   ├── build_rule_based_reward.py
    │   ├── score_generated_with_rule_based.py
    │   ├── analyze_classic_baseline.py
    │   └── analyze_qcbm_with_seed.py
    │
    ├── requirements.txt
    └── README.md

---

## Environment Setup

### Install Python dependencies

    pip install -r requirements.txt

### Install RDKit (Conda)

    conda install -c conda-forge rdkit

---

## Classical Generator Pipeline

### Train the classical SELFIES language model

    bash scripts/train_classic_generator.sh

### Sample 1000 molecules and score them

    bash scripts/sample_and_score_classic.sh

### Analyze classical model performance

    bash scripts/analyze_classic_baseline.sh

---

## Non-Segmented QCBM Pipeline

### Train QCBM latent generator

    bash scripts/train_qcbm_latent.sh

### Sample and score QCBM outputs

    bash scripts/sample_and_score_qcbm_latent.sh

### Analyze QCBM results

    bash scripts/analyze_qcbm.sh

---

## Segmented QCBM Pipeline

Run the full segmented pipeline:

    bash scripts/run_segmented_pipeline.sh

Outputs will be located in:

    data/gen_qcbm_segmented_round1.csv
    data/gen_qcbm_segmented_round1_scored.csv

---

## Compare All Models

    bash scripts/analyze_all_results.sh

---

## Reproducibility Notes

Record environment information here:

    Python 3.10
    Ubuntu 22.04 (WSL2)
    Intel i7-12700H CPU
    No GPU
    Qiskit 2.2.3
    RDKit 2023.09.2

---

