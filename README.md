# begrs_sfc

This repository contains the replication files for the BEGRS application on the *Benchmark stock-flow consistent* (SFC) model of *Caiani, A., Godin, A., Caverzasi, E., Gallegati, M., Kinsella, S. and Stiglitz, J.E., 2016. Agent based-stock flow consistent macroeconomics: Towards a benchmark model. Journal of Economic Dynamics and Control, 69, pp.375-408*.

## Requirements and Installation

Running replication files require:
- The `begrs` toolbox with dependencies installed.
- Additional packages specified in `requirements_extra.txt`
- The MIC toolbox, which can be downloaded from https://github.com/Sylvain-Barde/mic-toolbox (Note, this might require compilation)

Note: the files were run using GPU-enabled and large multi-CPU HPC nodes, therefore any attempt at replication should take into account this computational requirement. This is particularly the case for the SBC analysis, which is time-consuming even on an HPC node. The files are provided for the sake of transparency and replication, and all results are provided in the associated release (see below).

## Release contents

The release provides a zipped archive containing the following folders. These contain all the configuration files for the SFC model as well as all the intermediate results of the scripts, so that the outputs of the paper (i.e. figures) can be generated directly from them, without requiring a full re-run of the entire analysis.

- `/benchmark_sfc_lib`: contains the Java libraries required to run the ABM (see below)
- `/empData`: contains the US Macroeconomic data used in the estimation
- `/figures`: contains the figures used in the paper
- `/logs` : contains run logs for the simulations and MIC analysis
- `/models`: contains the saved trained BEGRS surrogate models and their associated posterior estimates
- `/sbc`: contains the results of the SBC diagnostic
- `/scores`: contains the MIC scores on the empirical datasets
- `/setup`: contains the configuration files for running the SFC simulations (see below)
- `/simData`: contains the simulated SFC data
- `/tables`: contains the tables used in the paper

## Configuration Files for ABM

The codebase for running the ABM model can be found in the following repositories:
- The *Benchmark model* itself: https://github.com/S120/benchmark
- The Java Macroeconomic Agent-Based (JMAB) framework for running the model https://github.com/S120/jmab

This repository contains the following files/folders that were generated from this codebase:
- `bechmark_sfc.jar` - A Java Archive containing the main files for running the model in Java
- `sfc_functions.py` - A set of Python functions for parallelizing the simulations and output processing of the model

Running these require the following folders from the release: `/benchmark_sfc_lib` and `/setup`.

## Run sequence:

The various scripts should be run in the following order, as the outputs of earlier scripts for the inputs of later ones. To run a later file (e.g. output generation) without running an earlier file (e.g. estimation), use the folders provided in the release as the source of the intermediate inputs. Files within a subsection can be run in any order.

### 1. Generate simulation data

- `parallel_sfc_train_run.py` - Generate SFC simulation data (training + SBC testing). Requires multi-CPU node

### 2. Run estimations

 Run the BEGRS estimation and the SBC diagnostic on the training and testing data.

- `begrs_sfc_train.py` - Train a begrs object on the simulated VAR/VARMA data
- `begrs_sfc_est.py` - Estimate the VAR/VARMA model using BEGRS from one run of testing data
- `begrs_sfc_sbc.py` - Run a SBC diagnostic on the full set of testing data

### 3. Run MIC score analysis

- `parallel_sfc_test_run.py` - Generate SFC simulation data for BEGRS posterior estimates. Requires multi-CPU node
- `parallel_mic_compare.py` - Run MIC analysis on simulated data. Requires multi-CPU node

### 4. Generate outputs

- `begrs_sfc_est_outputs.py` - Generate outputs from BEGRS estimation for the paper
- `begrs_sfc_sbc_outputs.py` - Generate outputs from SBC diagnostic of BEGRS for the paper
- `sfc_sim_outputs.py` - Generate unconditional densities of observable variables from simulation data, used in paper.
