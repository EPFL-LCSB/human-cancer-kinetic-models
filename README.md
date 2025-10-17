# Human Cancer Kinetic Models

## Overview
A computational framework for building and using kinetic models of cancer metabolism. 
This repository contains scripts, models, and data for simulating and analyzing ovarian cancer metabolism with and without a BRCA1 mutation. It can be used to reproduce all the work presented in the research article "Multi-omics-driven kinetic modeling reveals metabolic vulnerabilities and differential drug-response dynamics in ovarian cancer".

## Installation
1. Clone the repository:
	```sh
	git clone https://github.com/EPFL-LCSB/human-cancer-kinetic-models.git
	```
2. You will need to have Git LFS in order to properly download some binary files:
    ```sh
    cd [path_to_git_repo]
	git lfs install
    git lfs pull
	```
3. Set up docker:
	```sh
	cd docker
	./build
	```
3. Run the docker image:
    ```sh
	./run
	```

Alternatively, you can pull the prebuilt version directly from Docker Hub without building it locally:
    ```sh
	docker pull iliastoumpe/human_kinetic_cancer_docker:latest
	```

Then rename the image:
    ```sh
	docker tag iliastoumpe/human_kinetic_cancer_docker human_kinetic_cancer_docker
	```

The Docker image has been successfully installed and tested on both macOS (M1 chip) and Ubuntu systems.

## Solver Requirements
This framework uses IBM ILOG CPLEX as the primary optimization solver.
We recommend using CPLEX version 12.8, which has been extensively tested for stability and performance in this workflow.
Alternatively, Gurobi can also be used, as the scripts are solver-agnostic and compatible with both solvers. Ensure that the solver is correctly licensed and available in your environment before running the scripts.

## Usage
- Core scripts are located in `scripts/src/`. These are used to generate steady-state samples, sample kinetic parameters, compute control coefficients, perform drug administration simulations, and build steady-state, kinetic, and NRA models.
- Figure generation notebooks are in `scripts/notebooks/`. They reproduce all main and supplementary figures from the manuscript.
- Data files are in `data/` and models in `models/`.

## Data
- `data/`: Includes gene expression, DrugBank, kinetic parameter, and steady-state sample files.
- `models/`: Contains curated model files in multiple formats (YAML, JSON, MAT).
- Additional datasets can be downloaded from the associated Zenodo repository.

## Results
- All output files, processed data, and analysis results are stored in the `results/` directory.
