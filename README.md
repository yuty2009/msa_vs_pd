# Dynamic PSG Signature for MSA--PD Discrimination

This repository contains the notebooks and supporting code used to reproduce the analyses reported in the study of dynamic polysomnographic (PSG) features for distinguishing multiple system atrophy (MSA) from Parkinson's disease (PD), including the early-stage analysis.

## Contents

- `main_msa_pd_all.ipynb`: MSA versus PD analysis using the full cohort.
- `main_msa_pd_early.ipynb`: early-stage MSA versus early-stage PD analysis.
- `common/`: feature extraction, signal processing, microstate/state modeling, and evaluation utilities.
- `datasets/`: dataset readers and sleep-data utilities.
- `features_yasa_cols.csv`: feature-name definitions used by the analysis.

## Requirements

The analysis requires Python 3.10 or later and the scientific Python packages used in the notebooks, including NumPy, pandas, SciPy, scikit-learn, XGBoost, SHAP, MNE-Python, YASA, Matplotlib, and Jupyter.

Install the dependencies in an appropriate virtual environment before running the notebooks.

## Usage

1. Prepare the required PSG-derived input data according to the dataset reader configuration.
2. Open the relevant notebook in Jupyter.
3. Update local data paths and configuration variables where necessary.
4. Run the notebook cells in order.

The notebooks perform feature processing, dynamic PSG-state aggregation, model training, evaluation, and result visualization.

## Data availability

The clinical PSG data are not included in this repository because of privacy restrictions and data-use agreements. Access to the data is governed by the relevant institutional agreements.

## License

This repository is provided for research and reproducibility purposes. No license for clinical use is granted unless otherwise specified by the authors.
