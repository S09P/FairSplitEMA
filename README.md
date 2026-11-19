# FairSplitEMA

Official implementation of **FairSplitEMA**, a partition-based framework for fairness-aware machine learning. It combines subgroup balancing, EMA-based fairness-aware aggregation, and Pareto-guided model selection to improve fairness while preserving predictive performance.

## Overview

FairSplitEMA operates on a single dataset divided into multiple partitions. Its main components are:

1. **Partition-local subgroup balancing:** Uses subgroup-specific Wasserstein GANs (WGANs) to balance the joint distribution of the target label and protected attribute within each partition.
2. **EMA-based fairness-aware aggregation:** Uses exponentially smoothed disparate impact (DI) estimates alongside predictive performance to determine partition aggregation weights.
3. **Pareto-guided model selection:** Maintains candidate models across training rounds and selects a final model using accuracy, balanced accuracy, and fairness objectives.

## Repository Structure

```text
FairSplitEMA/
│
├── main.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── adult_processed.csv
│   ├── compas_processed.csv
│   ├── german_processed.csv
│   ├── bank_processed.csv
│   ├── default_processed.csv
│   └── heart_processed.csv
```

The `dataset/` directory contains the preprocessed benchmark datasets required by the implementation. The `results/` directory is created automatically when the experiments are executed.

## Requirements

- Python 3.12.3 (tested environment)
- PyTorch 2.13.0
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the required packages using:

```bash
pip install -r requirements.txt
```

**Hardware:** The implementation supports CPU execution. GPU execution is optional and can be used when a compatible CUDA-enabled PyTorch installation and GPU are available.


## Output Files

Experimental results are saved in the `results/` directory. The main output files include:

- **`master_log.csv`** – Stores the aggregated experimental results across configurations and random seeds.
- **`round_history_log.csv`** – Records training-round information, including model performance and fairness-related measurements.

These files can be used for further analysis and for generating the experimental tables and figures reported in the paper.


## Reproducibility

The default random seeds are specified in `main.py`. To reproduce a particular experiment, use the same dataset–attribute configuration, parameter settings, random seeds, and software environment.

For further methodological details, theoretical analysis, and experimental results, please refer to the corresponding FairSplitEMA research paper.

