"""Compare MCA enzyme targets with DepMap essentiality resources."""

import configparser
import os
import re

import pandas as pd
from pytfa.io.json import load_json_model
from skimpy.analysis.oracle.minimum_fluxes import MinFLuxVariable  # noqa: F401


config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

base_dir = config['paths']['base_dir']
TMODEL_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_tmodel_WT']))
GENE_MAP_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_gene_to_uniprot_mapping']))
CRISPR_GENE_EFFECT_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_depmap_crispr_gene_effect']))
CRISPR_GENE_DEPENDENCY_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_depmap_crispr_gene_dependency']))
COMMON_ESSENTIALS_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_depmap_common_essentials']))
MODEL_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_depmap_model_metadata']))
GENE_SUMMARY_OUTPUT_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_depmap_gene_summary']))
FINAL_GENE_OUTPUT_PATH = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_depmap_final_gene_table']))

# Union of the significant WT and mutant MCA targets identified previously.
MCA_TARGETS = [
    "TRIOK", "MI1PP", "PPAP", "r0301", "METAT", "3DSPHR", "TMDS",
    "SERPT", "HMR_7748", "PSP_L", "NADH2_u10mi", "DGK1", "NTD1",
    "r0354", "ADSS", "r0178", "IMPD", "r0179", "PGI", "r0474",
    "ICDHyrm", "GMPS2", "UMPK2", "ICDHxm", "URIK1", "CYTK1",
    "HMR_4343", "r0426",
]


def get_depmap_column_map(file_path):
    """Return Entrez ID to DepMap gene column mappings from a wide CSV."""
    columns = pd.read_csv(file_path, nrows=0).columns
    column_map = {}
    for column in columns:
        match = re.search(r"\((\d+)\)$", column)
        if match:
            column_map[match.group(1)] = column
    return column_map


def read_depmap_gene_subset(file_path, target_columns):
    """Read only model IDs and target gene columns from a DepMap matrix."""
    usecols = ["Unnamed: 0"] + sorted(target_columns)
    data = pd.read_csv(file_path, usecols=usecols)
    data = data.rename(columns={"Unnamed: 0": "ModelID"})
    return data


def summarize_depmap_scores(data, gene_columns, value_name, threshold_pairs):
    """Summarize one DepMap matrix at gene level."""
    rows = []
    for entrez_id, column in gene_columns.items():
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        row = {
            "entrez_id": entrez_id,
            "{}_n_models".format(value_name): len(values),
            "{}_mean".format(value_name): values.mean(),
            "{}_median".format(value_name): values.median(),
        }
        for label, operator, threshold in threshold_pairs:
            if operator == "lt":
                row[label] = (values < threshold).mean()
            elif operator == "gt":
                row[label] = (values > threshold).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gene_comparison(
    comparison_level,
    model_ids,
    coverage,
    gene_effect,
    gene_dependency,
    effect_columns,
    dependency_columns,
):
    """Summarize DepMap scores for one comparison cohort."""
    model_ids = set(model_ids)
    comparison_gene_effect = gene_effect[gene_effect["ModelID"].isin(model_ids)]
    comparison_gene_dependency = gene_dependency[
        gene_dependency["ModelID"].isin(model_ids)
    ]

    gene_effect_summary = summarize_depmap_scores(
        comparison_gene_effect,
        effect_columns,
        "gene_effect",
        [
            ("fraction_gene_effect_lt_minus_0_5", "lt", -0.5),
            ("fraction_gene_effect_lt_minus_1_0", "lt", -1.0),
        ],
    )
    gene_dependency_summary = summarize_depmap_scores(
        comparison_gene_dependency,
        dependency_columns,
        "dependency_probability",
        [
            ("fraction_dependency_probability_gt_0_5", "gt", 0.5),
            ("fraction_dependency_probability_gt_0_9", "gt", 0.9),
        ],
    )

    summary = (
        coverage.merge(gene_effect_summary, on="entrez_id", how="left")
        .merge(gene_dependency_summary, on="entrez_id", how="left")
    )
    summary.insert(0, "comparison_level", comparison_level)
    summary.insert(
        1, "comparison_gene_effect_models", comparison_gene_effect["ModelID"].nunique()
    )
    summary.insert(
        2,
        "comparison_dependency_models",
        comparison_gene_dependency["ModelID"].nunique(),
    )
    return summary


def build_final_gene_table(coverage, gene_summary, comparison_labels):
    """Build the reviewer-facing final gene-level comparison table."""
    final_gene_table = (
        coverage.groupby(
            [
                "model_gene_id",
                "entrez_id",
                "gene_symbol",
                "depmap_gene_label",
                "exclusion_reason",
            ],
            as_index=False,
        )
        .agg({
            "reaction_id": lambda values: ";".join(sorted(values.unique())),
            "is_common_essential": "max",
        })
        .rename(columns={
            "model_gene_id": "Model Gene",
            "reaction_id": "Model target Enzyme name",
            "gene_symbol": "Gene",
            "entrez_id": "Entrez ID",
            "depmap_gene_label": "DepMap gene label",
            "exclusion_reason": "DepMap coverage note",
        })
    )

    for comparison_level, label in comparison_labels:
        comparison_scores = (
            gene_summary.loc[
                gene_summary["comparison_level"].eq(comparison_level),
                [
                    "model_gene_id",
                    "entrez_id",
                    "gene_effect_median",
                    "dependency_probability_median",
                    "is_common_essential",
                ],
            ]
            .drop_duplicates(subset=["model_gene_id", "entrez_id"])
            .rename(columns={
                "model_gene_id": "Model Gene",
                "entrez_id": "Entrez ID",
                "gene_effect_median": "{} gene-effect score".format(label),
                "dependency_probability_median": (
                    "{} dependency probability".format(label)
                ),
                "is_common_essential": "{} common essential status".format(label),
            })
        )
        final_gene_table = final_gene_table.merge(
            comparison_scores, on=["Model Gene", "Entrez ID"], how="left"
        )

    final_gene_table = final_gene_table.drop(columns=["is_common_essential"])
    final_columns = [
        "Model Gene",
        "Model target Enzyme name",
        "Gene",
        "Entrez ID",
        "DepMap gene label",
        "Pan-cancer gene-effect score",
        "Pan-cancer dependency probability",
        "Pan-cancer common essential status",
        "Serous ovarian cancer gene-effect score",
        "Serous ovarian cancer dependency probability",
        "Serous ovarian cancer common essential status",
        "UWB1.289 gene-effect score",
        "UWB1.289 dependency probability",
        "UWB1.289 common essential status",
        "DepMap coverage note",
    ]
    return final_gene_table[final_columns].sort_values(
        ["Model target Enzyme name", "Gene"]
    )


print("Loading thermodynamic model:", TMODEL_PATH)
tmodel = load_json_model(TMODEL_PATH)

recon_gene_map = pd.read_csv(GENE_MAP_PATH, sep="\t", dtype=str)
recon_gene_map = recon_gene_map.set_index("gene_number")

rows = []
for reaction_id in MCA_TARGETS:
    reaction = tmodel.reactions.get_by_id(reaction_id)
    model_gene_ids = sorted(gene.id for gene in reaction.genes)

    if not model_gene_ids:
        rows.append({
            "reaction_id": reaction_id,
            "gene_reaction_rule": reaction.gene_reaction_rule,
            "model_gene_id": None,
            "entrez_id": None,
            "gene_symbol": None,
            "depmap_gene_label": None,
        })
        continue

    for model_gene_id in model_gene_ids:
        entrez_id = model_gene_id.split(".")[0]
        gene_symbol = None
        if model_gene_id in recon_gene_map.index:
            gene_symbol = recon_gene_map.at[model_gene_id, "symbol"]

        rows.append({
            "reaction_id": reaction_id,
            "gene_reaction_rule": reaction.gene_reaction_rule,
            "model_gene_id": model_gene_id,
            "entrez_id": entrez_id,
            "gene_symbol": gene_symbol,
            "depmap_gene_label": (
                "{} ({})".format(gene_symbol, entrez_id)
                if pd.notna(gene_symbol) else None
            ),
        })

mca_gene_map = pd.DataFrame(rows)

# Keep one row per reaction and Entrez gene for the downstream DepMap join.
depmap_gene_map = (
    mca_gene_map.dropna(subset=["entrez_id", "gene_symbol"])
    .drop_duplicates(subset=["reaction_id", "entrez_id"])
    .reset_index(drop=True)
)

os.makedirs(os.path.dirname(GENE_SUMMARY_OUTPUT_PATH), exist_ok=True)

print("MCA targets:", len(MCA_TARGETS))
print("Unique associated Entrez genes:", mca_gene_map["entrez_id"].nunique())


print("Loading DepMap metadata and target gene columns.")
model_metadata = pd.read_csv(MODEL_PATH, usecols=[
    "ModelID",
    "CellLineName",
    "OncotreePrimaryDisease",
    "OncotreeSubtype",
])

effect_column_map = get_depmap_column_map(CRISPR_GENE_EFFECT_PATH)
dependency_column_map = get_depmap_column_map(CRISPR_GENE_DEPENDENCY_PATH)
common_essential_genes = set(
    pd.read_csv(COMMON_ESSENTIALS_PATH, dtype=str)["Essentials"].dropna()
)

coverage = depmap_gene_map.copy()
coverage["effect_column"] = coverage["entrez_id"].map(effect_column_map)
coverage["dependency_column"] = coverage["entrez_id"].map(dependency_column_map)
coverage["in_crispr_gene_effect"] = coverage["effect_column"].notna()
coverage["in_crispr_gene_dependency"] = coverage["dependency_column"].notna()
coverage["is_common_essential"] = coverage["depmap_gene_label"].isin(
    common_essential_genes
)
coverage["exclusion_reason"] = ""
coverage.loc[
    ~coverage["in_crispr_gene_effect"], "exclusion_reason"
] = "missing from CRISPRGeneEffect.csv"
coverage.loc[
    ~coverage["in_crispr_gene_dependency"], "exclusion_reason"
] = coverage.loc[
    ~coverage["in_crispr_gene_dependency"], "exclusion_reason"
].where(
    coverage.loc[~coverage["in_crispr_gene_dependency"], "exclusion_reason"].eq(""),
    coverage.loc[~coverage["in_crispr_gene_dependency"], "exclusion_reason"]
    + "; "
) + "missing from CRISPRGeneDependency.csv"

effect_columns = coverage.loc[
    coverage["in_crispr_gene_effect"], ["entrez_id", "effect_column"]
].drop_duplicates()
effect_columns = dict(zip(effect_columns["entrez_id"], effect_columns["effect_column"]))
dependency_columns = coverage.loc[
    coverage["in_crispr_gene_dependency"], ["entrez_id", "dependency_column"]
].drop_duplicates()
dependency_columns = dict(zip(
    dependency_columns["entrez_id"], dependency_columns["dependency_column"]
))

gene_effect = read_depmap_gene_subset(
    CRISPR_GENE_EFFECT_PATH, set(effect_columns.values())
)
gene_dependency = read_depmap_gene_subset(
    CRISPR_GENE_DEPENDENCY_PATH, set(dependency_columns.values())
)

crispr_model_ids = set(gene_effect["ModelID"]) & set(gene_dependency["ModelID"])
crispr_model_metadata = model_metadata[
    model_metadata["ModelID"].isin(crispr_model_ids)
].copy()

comparison_model_ids = {
    "pan_cancer": sorted(crispr_model_ids),
    "ovarian_epithelial_tumor": sorted(
        crispr_model_metadata.loc[
            crispr_model_metadata["OncotreePrimaryDisease"].eq(
                "Ovarian Epithelial Tumor"
            ),
            "ModelID",
        ]
    ),
    "serous_ovarian_cancer": sorted(
        crispr_model_metadata.loc[
            crispr_model_metadata["OncotreeSubtype"].eq("Serous Ovarian Cancer"),
            "ModelID",
        ]
    ),
    "UWB1.289": sorted(
        crispr_model_metadata.loc[
            crispr_model_metadata["CellLineName"].eq("UWB1.289"),
            "ModelID",
        ]
    ),
}
final_table_comparisons = [
    ("pan_cancer", "Pan-cancer"),
    ("serous_ovarian_cancer", "Serous ovarian cancer"),
    ("UWB1.289", "UWB1.289"),
]

gene_summary = pd.concat([
    summarize_gene_comparison(
        comparison_level,
        model_ids,
        coverage,
        gene_effect,
        gene_dependency,
        effect_columns,
        dependency_columns,
    )
    for comparison_level, model_ids in comparison_model_ids.items()
], ignore_index=True)

final_gene_table = build_final_gene_table(
    coverage, gene_summary, final_table_comparisons
)

gene_summary.to_csv(GENE_SUMMARY_OUTPUT_PATH, index=False)
final_gene_table.to_csv(FINAL_GENE_OUTPUT_PATH, index=False)

print("DepMap models with gene-effect data:", gene_effect["ModelID"].nunique())
print("DepMap models with dependency data:", gene_dependency["ModelID"].nunique())
print("DepMap models present in Model.csv:", model_metadata["ModelID"].nunique())
print("Comparison model counts:")
for comparison_level, model_ids in comparison_model_ids.items():
    print("  {}: {}".format(comparison_level, len(model_ids)))
print(
    "Target Entrez genes covered by CRISPR Gene Effect:",
    coverage.loc[coverage["in_crispr_gene_effect"], "entrez_id"].nunique(),
    "/",
    coverage["entrez_id"].nunique(),
)
print(
    "Target Entrez genes covered by CRISPR Gene Dependency:",
    coverage.loc[coverage["in_crispr_gene_dependency"], "entrez_id"].nunique(),
    "/",
    coverage["entrez_id"].nunique(),
)
print(
    "Target genes flagged as DepMap common essentials:",
    coverage.loc[coverage["is_common_essential"], "entrez_id"].nunique(),
)
print("Saved gene-level DepMap summary:", GENE_SUMMARY_OUTPUT_PATH)
print("Saved final gene-level table:", FINAL_GENE_OUTPUT_PATH)
