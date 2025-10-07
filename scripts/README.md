# Workflow Overview

## Source Code

The `src` directory contains all scripts for computational modeling, simulation, and analysis. Each step below is accompanied by the exact command or code snippet required for execution.

1. **Build the pyTFA steady-state models for sampling.**  
   (`create_tfa_model.py`)
2. **Generate 5,000 steady-state samples for each model.**  
   (`create_steady_state_samples.py`)
3. **Construct the kinetic model scaffold, including reactions, species, and rate law mechanisms.**  
   (`create_kmodel.py`)
4. **Run the Metabolic Control Analysis (MCA) workflow by sampling 100 kinetic parameter sets per steady-state sample and evaluating local stability and physiological relevance.**  
   (`mca_workflow.py`)
5. **Aggregate results into a single DataFrame by computing average control coefficients for each enzyme.**  
   (`create_CCC_matrix.py` and `create_FCC_matrix.py`)
6. **Map key enzymes to existing drugs using the DrugBank database (XML format).**  
   (`map_enzymes_to_drugs.py`)
7. **Simulate cell viability and dose-response curves using saved kinetic models to propose pharmacodynamic parameters.**  
   (`cell_viability_simulations.py`)
8. **Perform drug therapy simulations to analyze time-resolved flux and concentration responses.**  
   (`drug_metabolism_simulation.py`)
9. **Build the NRA (Network Response Analysis) model using MUT steady-state and kinetic parameter samples.**  
   (`build_nra_model.py`)
10. **Optimize the NRA model with a custom flux deviation objective and analyze the effect of increasing enzyme perturbations on physiological alignment.**  
   (`optimize_nra_model.py`)
11. **Conduct variability analysis under objective constraints to identify consistently active and modifiable enzymes.**  
   (`variability_analysis_nra.py`)
12. **Link the minimal core enzyme set to their genes and regulatory transcription factors, including BRCA1 associations.**  
   (`find_links_with_TFs.py`)

---

## Utils Folder

The `utils` directory contains all supporting functions and classes required for this project. These modules provide reusable code for data processing, simulation routines, model utilities, and other helper operations used throughout the workflow.

---

## Visualization and Figure Generation

The `notebooks` directory contains all Jupyter notebooks for result visualization and figure reproduction.

1. **Reproduce all main and supplementary figure panels.**  
   *(See individual notebooks, e.g., `make_figure_2AB.ipynb`, `make_figure_3.ipynb`)*
2. **Select representative kinetic models for drug therapy simulations.**  
   *(See `stratified_sampling.ipynb`)*
3. **Identify representative pair for NRA formulation.**  
   *(See `find_representative_flux_profile_for_NRA.ipynb` and related notebooks)*
---

## Example: Running a Source Script

```bash
python src/create_tfa_model.py
```

## Example: Running a Notebook

Run jupyter notebook --ip 0.0.0.0 --no-browser --allow-root inside the docker container.
Using VScode or Jupyter connect to the provided server and run the .ipynb files: