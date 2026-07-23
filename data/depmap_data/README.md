# DepMap Data

This folder should contain raw input files downloaded from the DepMap Public data portal for the MCA target comparison.

Required downloaded input files:

- `CRISPRGeneEffect.csv`
- `CRISPRGeneDependency.csv`
- `CRISPRInferredCommonEssentials.csv`
- `Model.csv`

These files are used by `scripts/src/compare_mca_with_depmap.py` to generate:

- `mca_target_gene_depmap_summary.csv`
- `mca_target_gene_depmap_final.csv`

DepMap Public 26Q2, downloaded from https://depmap.org/portal/
