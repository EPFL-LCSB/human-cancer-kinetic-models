import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import configparser
import os
# Find the genes in the reduced model
from pytfa.io.json import load_json_model
from skimpy.analysis.oracle import *

import xml.etree.ElementTree as ET

class DrugBankAnalyzer:
    """
    A class to analyze DrugBank XML data, initialized by loading the XML file once.
    Provides methods to query drugs based on target UniProt IDs.
    """
    def __init__(self, xml_file_path: str):
        """
        Initializes the DrugBankAnalyzer by parsing the XML file.

        Args:
            xml_file_path (str): The path to the DrugBank XML file.
        """
        self.xml_file_path = xml_file_path
        self.root = None
        self.ns = {'db': 'http://www.drugbank.ca'} # Define the XML namespace

        try:
            print(f"Loading DrugBank XML file from '{self.xml_file_path}'... This may take a moment.")
            self.tree = ET.parse(self.xml_file_path)
            self.root = self.tree.getroot()
            print("DrugBank XML file loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file '{xml_file_path}' was not found. Please check the path.")
            self.root = None # Ensure root is None on error
        except ET.ParseError as e:
            print(f"Error parsing XML file '{xml_file_path}': {e}")
            self.root = None # Ensure root is None on error
        except Exception as e:
            print(f"An unexpected error occurred during XML loading: {e}")
            self.root = None # Ensure root is None on error

    def find_drugs_by_uniprot_id(self, uniprot_id: str) -> list:
        """
        Finds all drugs that target a specific UniProt ID using the pre-loaded XML data.
        Also extracts ATC codes, group categories, and drug effects for each found drug.

        Args:
            uniprot_id (str): The UniProt ID of the target protein.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'drugbank_id',
                  'name', 'atc_codes', 'groups', and 'actions' of a drug that targets
                  the given UniProt ID. Returns an empty list if the XML was not loaded
                  successfully or no drugs are found.
        """
        if self.root is None:
            print("Error: XML data not loaded. Please check the initialization of DrugBankAnalyzer.")
            return []

        targeted_drugs = []
        
        # Iterate through each <drug> element in the XML
        for drug_elem in self.root.findall('db:drug', self.ns):
            drug_name = drug_elem.find('db:name', self.ns).text if drug_elem.find('db:name', self.ns) is not None else 'Unmapped'
            
            # Get the primary DrugBank ID for the drug
            drugbank_id_elem = drug_elem.find("db:drugbank-id[@primary='true']", self.ns)
            if drugbank_id_elem is None:
                # If no primary ID, take the first available drugbank-id
                drugbank_id_elem = drug_elem.find('db:drugbank-id', self.ns)
            drugbank_id = drugbank_id_elem.text if drugbank_id_elem is not None else 'Unmapped'

            # Extract ATC codes
            atc_codes = []
            atc_codes_elem = drug_elem.find('db:atc-codes', self.ns)
            if atc_codes_elem is not None:
                for atc_code_elem in atc_codes_elem.findall('db:atc-code', self.ns):
                    # Directly get the 'code' attribute, which is the full ATC code
                    atc_full_code = atc_code_elem.attrib.get('code')
                    if atc_full_code:
                        atc_codes.append(atc_full_code)
            
            # Extract group categories
            groups = []
            groups_elem = drug_elem.find('db:groups', self.ns)
            if groups_elem is not None:
                for group_elem in groups_elem.findall('db:group', self.ns):
                    if group_elem.text:
                        groups.append(group_elem.text)

            # Define a list of interaction types to check (targets, enzymes, carriers, transporters)
            interaction_types = ['db:targets', 'db:enzymes', 'db:carriers', 'db:transporters']


            found_target_for_drug = False # Flag to stop searching for this drug once a target is found
            for interaction_type in interaction_types:
                interaction_list_elem = drug_elem.find(interaction_type, self.ns)
                if interaction_list_elem is not None:
                    # Iterate through each specific interactant (target, enzyme, etc.) within the list
                    interactant_tag = interaction_type.replace('db:', '').rstrip('s')
                    for interactant_elem in interaction_list_elem.findall(f'db:{interactant_tag}', self.ns):
                        # Find all polypeptide elements within the interactant
                        for polypeptide_elem in interactant_elem.findall('db:polypeptide', self.ns):
                            external_identifiers_elem = polypeptide_elem.find('db:external-identifiers', self.ns)
                            if external_identifiers_elem is not None:
                                # Iterate through each external identifier for the polypeptide
                                for ext_id_elem in external_identifiers_elem.findall('db:external-identifier', self.ns):
                                    resource_elem = ext_id_elem.find('db:resource', self.ns)
                                    identifier_elem = ext_id_elem.find('db:identifier', self.ns)

                                    if resource_elem is not None and identifier_elem is not None:
                                        # Check if the resource is UniProtKB and the identifier matches
                                        if resource_elem.text in ['UniProtKB', 'UniProt Accession'] and identifier_elem.text == uniprot_id:
                                            # Extract actions (drug effects) for this specific interactant
                                            actions = []
                                            actions_elem = interactant_elem.find('db:actions', self.ns)
                                            if actions_elem is not None:
                                                for action_tag in actions_elem.findall('db:action', self.ns):
                                                    if action_tag.text:
                                                        actions.append(action_tag.text)

                                            targeted_drugs.append({
                                                'drugbank_id': drugbank_id,
                                                'name': drug_name,
                                                'atc_codes': atc_codes,
                                                'groups': groups,
                                                'actions': actions
                                            })
                                            found_target_for_drug = True
                                            break # Stop searching for this UniProt ID in current polypeptide
                                if found_target_for_drug:
                                    break # Stop searching for this UniProt ID in current interactant
                        if found_target_for_drug:
                            break # Stop searching for this UniProt ID in current interaction type
                if found_target_for_drug:
                    break # Stop searching for this UniProt ID for this drug

        return targeted_drugs

# Define the path to your full DrugBank XML database file
drugbank_xml_path = "../../data/drugbank_data/full database.xml"

# Initialize the DrugBankAnalyzer. This will load the XML file once.
analyzer = DrugBankAnalyzer(drugbank_xml_path)


mca_targets = [
 'vmax_forward_TRIOK', 'vmax_forward_MI1PP', 'vmax_forward_PPAP', 'vmax_forward_r0301', 'vmax_forward_METAT',
 'vmax_forward_3DSPHR', 'vmax_forward_TMDS', 'vmax_forward_SERPT', 'vmax_forward_HMR_7748', 'vmax_forward_PSP_L',
 'vmax_forward_NADH2_u10mi', 'vmax_forward_DGK1', 'vmax_forward_NTD1', 'vmax_forward_r0354', 'vmax_forward_ADSS',
 'vmax_forward_r0178', 'vmax_forward_IMPD', 'vmax_forward_r0179', 'vmax_forward_PGI', 'vmax_forward_r0474',
 'vmax_forward_ICDHyrm', 'vmax_forward_GMPS2', 'vmax_forward_UMPK2', 'vmax_forward_ICDHxm', 'vmax_forward_URIK1',
 'vmax_forward_CYTK1', 'vmax_forward_HMR_4343', 'vmax_forward_r0426'
]

mca_targets = np.unique(mca_targets).tolist()

mca_targets = [x.replace('vmax_forward_', '') for x in mca_targets]

# Read configuration from config.ini
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# Path to tmodel from config
base_dir = config['paths']['base_dir']
path_to_tmodel_WT = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_tmodel_WT']))

# Path to ncbi_to_uniprot mapping from config
path_to_ncbi_to_uniprot = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_ncbi_to_uniprot']))

# Find the genes associated with the mca targets
tmodel = load_json_model(path_to_tmodel_WT)
mca_genes = {}
for rxn_id in mca_targets:
    rxn = tmodel.reactions.get_by_id(rxn_id)
    mca_genes[rxn_id] = [g.id for g in rxn.genes]

    
# Repeat the analysis with mapping from NCBI IDs to UniProt IDs
ncbi_to_uniprot = pd.read_csv(path_to_ncbi_to_uniprot, index_col=0)

# Keep only the reviewd columns
ncbi_to_uniprot = ncbi_to_uniprot[ncbi_to_uniprot['Reviewed'] == 'reviewed']

# Prepare a list to collect all drug-target interaction dictionaries
all_drug_target_entries_new = []

for target_name, genes in mca_genes.items():
    # Add Succinate semialdehyde dehydrogenase [NAD(P)+] (r0179) as the uniprot ID for Succinate semialdehyde dehydrogenase with NAD+
    if target_name == 'r0179':
        genes = ['7915.1']

    print(f"Reaction: {target_name}, Genes: {', '.join(genes)}")

    for gene in genes:
        gene_float = int(gene.split('.')[0])  # Convert gene to float by taking the first part before underscore

        if gene_float in ncbi_to_uniprot.index:
            target_uniprot_id_rows = ncbi_to_uniprot.loc[gene_float, 'Entry']
            if not isinstance(target_uniprot_id_rows, str):
                target_uniprot_id_rows = ncbi_to_uniprot.loc[gene_float, 'Entry'].to_list()
            else:
                target_uniprot_id_rows = [target_uniprot_id_rows]
            print(f"  Gene: {gene}, UniProt ID: {target_uniprot_id_rows}")
        else:
            print(f"  Gene: {gene} not found in ncbi to uniprot dataset.")
            continue

        print(f"\nQuerying for drugs targeting UniProt ID '{target_uniprot_id_rows}'...")
        for target_uniprot_id in target_uniprot_id_rows:
            drugs_for_target = analyzer.find_drugs_by_uniprot_id(target_uniprot_id)

            if drugs_for_target:
                print(f"\nFound {len(drugs_for_target)} drugs targeting UniProt ID '{target_uniprot_id}':")
                unique_drug_ids = set()
                drug_id_to_actions = {}

                for drug in drugs_for_target:
                    drug_id = drug['drugbank_id']

                    if drug_id not in unique_drug_ids:
                        all_drug_target_entries_new.append({
                            'Target UniProt ID': target_uniprot_id,
                            'Target Name': target_name,
                            'DrugBank ID': drug_id,
                            'Drug Name': drug['name'],
                            'ATC Codes': ", ".join(drug['atc_codes']) if drug['atc_codes'] else 'Unmapped',
                            'Groups': ", ".join(drug['groups']) if drug['groups'] else 'Unmapped',
                            'Actions (Drug Effects)': ", ".join(drug['actions']) if drug['actions'] else 'Unmapped'
                        })
                        drug_id_to_actions[drug_id] = set(drug['actions']) if drug['actions'] else set()
                        unique_drug_ids.add(drug_id)
                    else:
                        # Merge actions
                        for entry in all_drug_target_entries_new:
                            if entry['DrugBank ID'] == drug_id and entry['Target UniProt ID'] == target_uniprot_id:
                                existing_actions = set(entry['Actions (Drug Effects)'].split(', ')) if entry['Actions (Drug Effects)'] != 'Unmapped' else set()
                                new_actions = set(drug['actions']) if drug['actions'] else set()
                                merged_actions = existing_actions.union(new_actions)
                                entry['Actions (Drug Effects)'] = ", ".join(sorted(merged_actions)) if merged_actions else 'Unmapped'
                                break
            else:
                print(f"\nNo drugs found targeting UniProt ID '{target_uniprot_id}'.")

    print("\n" + "="*50 + "\n")

# Create the final DataFrame once, outside the loop
targets_to_drugs_new = pd.DataFrame(all_drug_target_entries_new)

print(len(targets_to_drugs_new.loc[:, 'DrugBank ID'].unique()), "unique drugs found in the DrugBank database for the MCA targets.")

# Save the results to a CSV file
path_to_targets_to_drugs = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_targets_to_drugs']))
targets_to_drugs_new.to_csv(path_to_targets_to_drugs, index=False)