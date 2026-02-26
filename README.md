# GPCR Coupling Leakage Benchmarking Protocol

**Addressing data leakage in GPCR-G protein coupling prediction: a benchmarking protocol with structure-aligned determinants and melanopsin case study**

Guohao Lv, Lichuan Gu — School of Information and Artificial Intelligence, Anhui Agricultural University

---

## Overview

This repository provides the complete analysis pipeline, data, and a standalone leakage diagnostic tool accompanying our manuscript.

We demonstrate that random train–test splitting in GPCR–G protein coupling prediction inflates AUC by ~0.25 due to phylogenetic data leakage, and propose a structure-aligned Ballesteros–Weinstein (BW) site framework that recovers genuine predictive signal under rigorous no-leak evaluation.

## Repository Structure

```text
code/
  fetch_gpcrdb_data.py          # Step 0a: data retrieval from GPCRdb REST API
  fetch_gpcrdb_cache.py         # Step 0b: fetch BW residue annotations (required by BW analyses)
  run_paper_benchmark.py        # Step 1:  leakage-aware benchmarking (6 models × 4 splits)
  run_paper_bw_enhanced.py      # Step 2:  BW-site chi-squared + physicochemical encoding
  run_interpretability.py       # Step 3:  SHAP analysis, sequence logos, consensus heatmaps
  run_esm2_bwsite.py            # Step 4:  ESM-2 per-residue BW-site embeddings (GPU recommended)
  run_interface_v3.py           # Step 5:  structural interface contact analysis (8 PDB complexes)
  run_reviewer_analyses.py      # Step 6:  sequence identity gradient, feature decomposition
  run_multi_gprotein.py         # Step 7:  cross-family generalization and dual-coupling analysis
  generate_figures.py           # Step 8:  publication figure generation
  leakage_test.py               # Standalone leakage diagnostic tool

data/
  gpcrdb_coupling_dataset.csv   # 230-receptor feature matrix (included)
  gpcrdb_coupling_dataset.json  # Full receptor metadata with sequences (included)

results/                        # Precomputed benchmark and analysis outputs (.csv)

environment.yml                 # Conda environment specification
requirements.txt                # pip requirements (alternative to conda)
```

## Quick Start

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate gpcr_gq_coupling
```

### Option B: pip

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Run the pipeline

```bash
# Step 0 (one-time): fetch BW residue annotations from GPCRdb (~5–10 min)
#   This creates data/gpcrdb_residues_cache.json, needed by Steps 2–7.
#   Step 1 (benchmark) does NOT require this cache and can run immediately.
python code/fetch_gpcrdb_cache.py

# Step 1: leakage-aware benchmark (no external dependencies, ~2 min)
python code/run_paper_benchmark.py

# Step 2: BW-site analysis (requires cache from Step 0)
python code/run_paper_bw_enhanced.py

# Step 3: interpretability analysis
python code/run_interpretability.py

# Step 4: ESM-2 BW-site embeddings (GPU recommended; requires fair-esm + torch)
python code/run_esm2_bwsite.py

# Step 8: generate all publication figures
python code/generate_figures.py
```

> **Note:** Step 1 (`run_paper_benchmark.py`) is fully self-contained—it uses the CSV directly and does not need the BW residue cache. You can run it immediately after environment setup.

## Leakage Diagnostic Tool

`code/leakage_test.py` accepts user-supplied coupling predictions (CSV format) and evaluates performance degradation under no-leak splitting:

```bash
# With built-in dataset
python code/leakage_test.py

# With user-supplied predictions
python code/leakage_test.py --predictions your_predictions.csv
```

The input CSV should contain columns: `entry_name`, `predicted_probability`.

## Key Results

| Split Strategy       | AUC-ROC   | Feature Set       |
|---------------------|-----------|-------------------|
| Random (leaky)       | 0.778     | Handcrafted (99d) |
| Subfamily (no-leak)  | 0.525     | Handcrafted (99d) |
| Subfamily (no-leak)  | **0.738** | **BW-site (145d)**|
| Subfamily (no-leak)  | 0.722     | ESM-2 (320d)      |

## Data Sources

All analyses are reproducible from public data:
- [GPCRdb](https://gpcrdb.org) — receptor sequences, BW annotations, coupling labels
- [GtoPdb/IUPHAR](https://www.guidetopharmacology.org) — coupling validation
- [PDB](https://www.rcsb.org) — GPCR–G protein complex structures

## Citation

If you use this code or protocol, please cite:

> Lv G, Gu L. Addressing data leakage in GPCR-G protein coupling prediction: a benchmarking protocol with structure-aligned determinants and melanopsin case study. *Briefings in Bioinformatics*, 2025.

## License

MIT License — see [LICENSE](LICENSE) for details.
