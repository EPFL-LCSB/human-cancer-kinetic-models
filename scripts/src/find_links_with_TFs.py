import numpy as np
import pandas as pd
from skimpy.analysis.oracle import *
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../../../NRAplus/NRAplus") # Adds higher directory to python modules path.
sys.path.append('../')

from utils.nra_save_custom_json import load_json_nra_model



# Load path from config.ini
import configparser
import os
config = configparser.ConfigParser()
config.read(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.ini')))
base_dir = config['paths']['base_dir']
path_to_nra_model = os.path.join(base_dir, config['paths']['path_to_nra_model'])
path_to_solutions = os.path.join(base_dir, config['paths']['path_to_solutions'])
path_to_tflink_database = os.path.join(base_dir, config['paths']['path_to_tflink_database'])
path_to_recon_model = os.path.join(base_dir, config['paths']['path_to_recon_model'])

nra_model = load_json_nra_model(path_to_nra_model)

# Load the names of the essential enzymes
with open('essential_enzymes.txt', 'r') as f:
    essential_enzymes_names = [line.strip() for line in f.readlines()]

subsystem_dict = {}
for rxn_id in essential_enzymes_names:
    rxn = nra_model.reactions.get_by_id(rxn_id)
    sub = rxn.subsystem
    if sub not in subsystem_dict:
        subsystem_dict[sub] = [rxn.id]
    else:
        subsystem_dict[sub].append(rxn.id)
# Print the subsystems and their reactions
for sub, rxns in subsystem_dict.items():
    print(f"Subsystem: {sub}")
    print("Reactions: ", end="")
    for rxn in rxns:
        print(rxn, end=", ")
    print('\n')

# Load the solution 
sol_original = pd.read_csv(path_to_solutions.format(250), index_col=0)

essential_enzymes = pd.DataFrame(columns=['Enzyme', 'Subsystem', 'Regulation'])

for enz in essential_enzymes_names:
    up_var = sol_original.loc['EU_'+enz]
    down_var = sol_original.loc['ED_'+enz]
    rxn = nra_model.reactions.get_by_id(enz)

    if np.isclose(down_var, 0) and not np.isclose(up_var, 0):
        regulation = 'up'
    else:
        regulation = 'down'

    essential_enzymes = essential_enzymes.append({'Enzyme': enz, 'Subsystem': subsystem_dict[rxn.subsystem], 'Regulation': regulation}, ignore_index=True)

# Set the index to the enzyme name
essential_enzymes.set_index('Enzyme', inplace=True)

# Print how many enzymes are up or down regulated 
print(f"Number of down regulated enzymes: {len(essential_enzymes[essential_enzymes['Regulation'] == 'down'])}")
print(f"Number of up regulated enzymes: {len(essential_enzymes[essential_enzymes['Regulation'] == 'up'])}")


# Load the TFLink database
tf_link = pd.read_csv(path_to_tflink_database, sep='\t')

from cobra.io.json import load_json_model
recon = load_json_model(path_to_recon_model)

# Check if TFLink returns more TF hits for the essential enzymes
essential_enzymes['TF_link'] = pd.Series(dtype=object) # Initialize with object dtype

for enz in essential_enzymes.index:
    uniprot_ids = essential_enzymes.loc[enz, 'uniprot_genes']
    if not pd.isna(uniprot_ids):
        uniprot_ids_split = uniprot_ids.split(';')
        tf_hits = []
        for uniprot_id in uniprot_ids_split:
            tf_matches = tf_link.loc[tf_link['UniprotID.Target'] == uniprot_id, 'Name.TF']
            if not tf_matches.empty:
                tf_hits.extend(tf_matches.unique())
        if tf_hits:
            essential_enzymes.at[enz, 'TF_link'] = ';'.join(tf_hits)
        else:
            essential_enzymes.at[enz, 'TF_link'] = np.nan

# Find how many unique genes are there
gene_list = []
for i in essential_enzymes['genes'].dropna():
    genes = i.split(';')
    for gene in genes:
        gene_list.append(gene)
print(f"{essential_enzymes.genes.notna().sum()}/{len(essential_enzymes)} essential enzymes have genes associated with them")
print(f"Number of genes: {len(gene_list)}")
print(f"Number of unique genes: {len(set(gene_list))}\n")

# Find how many unique TFs are there for TF_link
tfs_list_tf_link = []
for i in essential_enzymes['TF_link'].dropna():
    tfs = i.split(';')
    for tf in tfs:
        tfs_list_tf_link.append(tf)
print(f"{essential_enzymes.TF_link.notna().sum()}/{len(essential_enzymes)} essential enzymes have TF_link TFs associated with them")
print(f"Number of TFs: {len(tfs_list_tf_link)}")
print(f"Number of unique TFs: {len(set(tfs_list_tf_link))}\n")

# Find the unique genes that are associated with the TF_link TFs
genes_tfs_tf_link = []
for i in essential_enzymes[essential_enzymes.TF_link.notna()].genes:
    genes = i.split(';')
    for gene in genes:
        genes_tfs_tf_link.append(gene)
print(f"Number of genes associated with TF_link TFs: {len(genes_tfs_tf_link)}")
print(f"Number of unique genes associated with TF_link TFs: {len(set(genes_tfs_tf_link))}\n")
print('------------------------------------------------------')

print(f'In total {essential_enzymes.TF_link.notna().sum()}/{len(essential_enzymes)} essential enzymes are connected to '
      f'{len(set(genes_tfs_tf_link))} unique genes and are regulated by {len(set(tfs_list_tf_link))} unique TFs (TF_link)')