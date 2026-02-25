# GPCR Coupling Leakage Benchmarking Protocol

**Addressing data leakage in GPCR–G protein coupling prediction: a benchmarking protocol with structure-aligned determinants and melanopsin case study**

Guohao Lv, Lichuan Gu — School of Information and Artificial Intelligence, Anhui Agricultural University

---

## Overview

This repository provides the complete analysis pipeline, data, and a standalone leakage diagnostic tool accompanying our manuscript submitted to *Briefings in Bioinformatics*.

We demonstrate that random train–test splitting in GPCR–G protein coupling prediction inflates AUC by ~0.25 due to phylogenetic data leakage, and propose a structure-aligned Ballesteros–Weinstein (BW) site framework that recovers genuine predictive signal under rigorous no-leak evaluation.

## Repository Structure

```
code/                           # Analysis pipeline
  fetch_gpcrdb_data.py          # Data retrieval from GPCRdb REST API
  run_paper_benchmark.py        # Leakage-aware benchmarking (6 models × 4 splits)
  run_paper_bw_enhanced.py      # BW-site chi-squared analysis & physicochemical encoding
  run_interpretability.py       # SHAP analysis, sequence logos, consensus heatmaps
  run_esm2_bwsite.py            # ESM-2 per-residue BW-site embeddings
  run_interface_v3.py           # Structural interface contact analysis (8 PDB complexes)
  run_reviewer_analyses.py      # Sequence identity gradient, feature decomposition
  run_multi_gprotein.py         # Cross-family generalization & dual-coupling analysis
  generate_figures.py           # Publication figure generation
  leakage_test.py               # Standalone leakage diagnostic tool (see below)

data/
  gpcrdb_coupling_dataset.csv   # 230-receptor physicochemical feature matrix
  gpcrdb_coupling_dataset.json  # Full receptor metadata with sequences

results/                        # Precomputed benchmark and analysis outputs (.csv)

environment.yml                 # Conda environment for full reproduction
```

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate gpcr_gq_coupling

# 2. Run the full benchmark
python code/run_paper_benchmark.py

# 3. Run BW-site analysis
python code/run_paper_bw_enhanced.py

# 4. Generate all figures
python code/generate_figures.py
```

## Leakage Diagnostic Tool

`code/leakage_test.py` is a standalone script that accepts user-supplied coupling predictions (CSV format) and evaluates performance degradation under no-leak splitting:

```bash
python code/leakage_test.py --input your_predictions.csv
```

It outputs a diagnostic report comparing AUC under random vs. subfamily splits.

## Key Results

| Split Strategy | AUC-ROC | Feature Set |
|---------------|---------|-------------|
| Random (leaky) | 0.778 | Handcrafted (99d) |
| Subfamily (no-leak) | 0.525 | Handcrafted (99d) |
| Subfamily (no-leak) | **0.738** | **BW-site (145d)** |
| Subfamily (no-leak) | 0.722 | ESM-2 (320d) |

## Data Sources

All analyses are reproducible from public data:
- [GPCRdb](https://gpcrdb.org) — receptor sequences, BW annotations, coupling labels
- [GtoPdb/IUPHAR](https://www.guidetopharmacology.org) — coupling validation
- [PDB](https://www.rcsb.org) — GPCR–G protein complex structures

## Citation

If you use this code or protocol, please cite:

> Lv G, Gu L. Addressing data leakage in GPCR–G protein coupling prediction: a benchmarking protocol with structure-aligned determinants and melanopsin case study. *Briefings in Bioinformatics*, 2025.

## License

MIT License — see [LICENSE](LICENSE) for details.
