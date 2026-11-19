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
│   └── student_processed.csv
|
└── results/                 
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

## Running the Experiments

After installing the dependencies and placing the preprocessed CSV files in the `dataset/` directory, run:

```bash
python main.py
```

The implementation loads the configured dataset–protected attribute combinations, performs the train/test split, partitions the training data, and executes the FairSplitEMA pipeline across the specified random seeds.

## Configuration

The experimental settings can be modified directly in `main.py`.

The primary configuration parameters include:

| Parameter | Default value |
|---|---:|
| Number of partitions | 10 |
| Local training epochs | 5 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Test set proportion | 0.20 |
| Communication rounds | 30 |
| Random seeds | 42, 43, 44, 45, 46 |
| EMA coefficient | 0.7 |
| Accuracy weight (α) | 1.5 |
| Fairness weight (β) | 3.0 |

The dataset and protected-attribute combinations are specified through `dataset_names` and `protected_attributes` in `main.py`. These lists are paired position-wise; therefore, each entry in `dataset_names` must correspond to the protected attribute at the same position in `protected_attributes`.

The default configuration runs the Bank dataset with the age attribute. Modify these lists to execute other dataset–attribute configurations.

## Output Files

Experimental results are saved in the `results/` directory. The main output files include:

- **`master_log.csv`** – Stores the aggregated experimental results across configurations and random seeds.
- **`round_history_log.csv`** – Records training-round information, including model performance and fairness-related measurements.

These files can be used for further analysis and for generating the experimental tables and figures reported in the paper.

## Evaluation Protocol

The implementation uses a fixed train/test split for each random seed. The held-out portion is also used during iterative fairness evaluation, aggregation, and model selection, as implemented in the experimental pipeline. Accordingly, the reported results should be interpreted in the context of this evaluation protocol rather than as performance on an independently untouched test set.

## Reproducibility

The default random seeds are specified in `main.py`. To reproduce a particular experiment, use the same dataset–attribute configuration, parameter settings, random seeds, and software environment.

For further methodological details, theoretical analysis, and experimental results, please refer to the corresponding FairSplitEMA research paper.

