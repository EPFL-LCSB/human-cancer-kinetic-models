# <img src="project_icon.png" alt="Human Cancer Kinetic Models project icon" width="250" align="left"> Human Cancer Kinetic Models

### Overview
A computational framework for building and using kinetic models of cancer metabolism. 
This repository contains scripts, models, and data for simulating and analyzing ovarian cancer metabolism with and without a BRCA1 mutation. It can be used to reproduce all the work presented in the research article "Multi-omics-constrained kinetic models quantify pathway rewiring and reveal metabolic vulnerabilities in BRCA1-deficient ovarian cancer".

The accompanying data and results are available in the [Zenodo repository](https://zenodo.org/records/17304777) and the links provided therein.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/EPFL-LCSB/human-cancer-kinetic-models.git
    cd human-cancer-kinetic-models
    ```
2. You will need to have Git LFS in order to properly download some binary files:
    ```sh
    git lfs install
    git lfs pull
    ```

### Recommended Docker environment
Pull the prebuilt version directly from Docker Hub:
```sh
docker pull iliastoumpe/human_kinetic_cancer_docker:v1.0
```

For strict reproducibility, the same image can be pulled using its fixed digest:
```sh
docker pull iliastoumpe/human_kinetic_cancer_docker@sha256:bdbfedd2c32cd59c9fa12b6f8f0177d29cb82591ae39c832f366dfef0271835c
```

Then rename the image:
```sh
docker tag iliastoumpe/human_kinetic_cancer_docker:v1.0 human_kinetic_cancer_docker
```

Run the Docker image:
```sh
cd docker
./run
```

Alternatively, build the Docker image locally:
```sh
cd docker
./build
./run
```

The Docker image has been successfully installed and tested on both macOS (M1 chip) and Ubuntu systems.

### Local conda environment
A local conda environment can also be created from the pinned environment file:
```sh
conda env create -f environment.yml
conda activate human-cancer-kinetic-models
```

For parts of the workflow that require optimization (in the full model-building scripts), install and license the CPLEX Python API inside the activated environment. For example:
```sh
conda activate human-cancer-kinetic-models
cd /path/to/CPLEX_Studio128/cplex/python/3.6/x86-64_linux
python setup.py install
```
We recommend using the provided Docker image, or building the Docker image from this repository, as the most complete and well-tested environment for the full workflow.

## Quick Start
After installation, run one of the smoke tests to check that the environment can load the provided files and execute a minimal analysis:
```sh
python examples/smoke_test_mca.py
python examples/smoke_test_ode.py
```

The notebook `examples/minimal_pipeline_tutorial.ipynb` provides a simplified walkthrough of the complete workflow using precomputed model and parameter subsets. It loads the WT thermodynamic and kinetic models, compiles the kinetic functions required for MCA and ODE simulation, loads one representative steady-state and kinetic-parameter sample, runs a small MCA calculation, applies a TMDS Vmax perturbation, and plots short simulated trajectories without regenerating the full model ensemble.

## Solver Requirements
This workflow uses IBM ILOG CPLEX as the primary optimization solver.
We recommend using CPLEX version 12.8, which has been extensively tested for stability and performance in this workflow.
Alternatively, Gurobi can also be used, as the scripts are solver-agnostic and compatible with both solvers. Ensure that the solver is correctly licensed and available in your environment before running the scripts.

## Usage
- Figure generation notebooks are in `scripts/notebooks/`. These provide the lightweight route for reproducing the manuscript figures from stored processed outputs.
- Core workflow scripts are located in `scripts/src/`. These are used to regenerate the full model ensemble, including steady-state samples, kinetic parameter samples, control coefficients, drug administration simulations, and steady-state, kinetic, and NRA models. The accompanying `scripts/src/config.ini` file controls important workflow settings, including sample counts, solver settings and tolerances, CPU parallelization, input/output paths, and drug-simulation parameters; drug-administration target enzymes are defined in the corresponding simulation script.
- Helper modules are located in `scripts/utils/`. These provide shared functions used by the full workflow scripts.
- `data/`: Includes gene expression, DrugBank, kinetic parameter, and steady-state sample files.
- `models/`: Contains curated model files in multiple formats (YAML, JSON, MAT).
- Additional datasets can be downloaded from the associated Zenodo repository.

## Results
- All output files, processed data, and analysis results are stored in the `results/` directory.

## Reproducing manuscript figures
The main manuscript figures can be regenerated from the processed outputs already included in the repository or available through Zenodo. This route is separate from the full kinetic-model reconstruction workflow and does not require rerunning parameter fitting, sampling, or solver-heavy optimization workflows.
