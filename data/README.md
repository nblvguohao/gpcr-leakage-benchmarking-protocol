# Data Directory — Offline Static Feature Snapshots

This directory contains **pre-fetched, version-stamped static snapshots** of all data
required to reproduce every analysis in the manuscript. These files eliminate the need
for live GPCRdb REST API access, ensuring full offline reproducibility even if the
external API changes structure or becomes temporarily unavailable.

## File Inventory

| File | Description | Records | Source | Snapshot Date |
|------|-------------|---------|--------|---------------|
| `gpcrdb_coupling_dataset.csv` | 230 human Class A/B1/C GPCRs with Gq/11 coupling labels, sequences, family assignments | 233 rows (230 after filtering) | GPCRdb v2024.1 + GtoPdb + Inoue et al. 2019 | 2025-03-03 |
| `gpcrdb_residues_cache.json` | Per-residue BW-numbered annotations for all 230 human receptors | 230 entries | GPCRdb REST API `/services/residues/` | 2025-03-04 |
| `gpcrdb_multispecies_dataset.csv` | 795 receptors across 8 species with ortholog-transferred coupling labels | 795 rows | GPCRdb REST API + ortholog mapping | 2025-03-04 |
| `gpcrdb_multispecies_residues_cache.json` | Per-residue BW annotations for all 795 multi-species receptors | 795 entries | GPCRdb REST API `/services/residues/` | 2025-03-04 |

## Data Source Versions

- **GPCRdb**: v2024.1 (https://gpcrdb.org), accessed March 2025
- **GtoPdb / IUPHAR**: Release 2024.1 (https://www.guidetopharmacology.org)
- **Inoue et al. 2019**: TGF-α shedding coupling dataset (DOI: 10.1016/j.cell.2019.06.030)

## Usage

All analysis scripts (`run_paper_benchmark.py`, `run_tta_experiments.py`,
`run_esm2_bwsite.py`, `run_multispecies_benchmark.py`, etc.) load data exclusively
from these local files by default. **No live API calls are required** for any
reported analysis.

The `fetch_gpcrdb_data.py` and `fetch_gpcrdb_cache.py` scripts are provided for
transparency and provenance documentation only — they show exactly how the cached
data were originally obtained. Re-running them is **not necessary** and may yield
slightly different results if the upstream database has been updated.

## Integrity Verification

To verify data integrity, compare SHA-256 checksums:

```bash
sha256sum data/gpcrdb_coupling_dataset.csv
sha256sum data/gpcrdb_residues_cache.json
sha256sum data/gpcrdb_multispecies_dataset.csv
sha256sum data/gpcrdb_multispecies_residues_cache.json
```

Expected checksums are recorded in `data/checksums.sha256` (generated at submission time).

## Column Descriptions

### gpcrdb_coupling_dataset.csv
- `entry_name`: GPCRdb entry name (e.g., `5ht2a_human`)
- `name`: Human-readable receptor name
- `accession`: UniProt accession
- `family`: GPCRdb family slug (e.g., `001_001_001_001`)
- `species`: Species name
- `sequence`: Full amino acid sequence
- `seq_length`: Sequence length
- `gq_coupling`: Coupling type string (`primary`, `secondary`, `none`)
- `gq_label`: Binary label (1 = Gq-coupled, 0 = non-Gq)
- `coupling_description`: Detailed coupling annotation

### gpcrdb_residues_cache.json
- Dictionary keyed by `entry_name`
- Each value is a list of residue objects with fields:
  - `sequence_number`: 1-indexed position in sequence
  - `amino_acid`: Single-letter amino acid code
  - `display_generic_number`: BW generic number (e.g., `3.50x50`)
  - `alternative_generic_numbers`: List of alternative numbering schemes
