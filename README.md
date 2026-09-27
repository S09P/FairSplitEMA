# FairSplitEMA
This repository contains the official implementation of FairSplitEMA, a novel decentralized approach designed to address key limitations in fairness-aware machine learning. While traditional centralized fairness methods suffer from circular dependencies in bias detection, high sensitivity to non-monotonic fairness fluctuations, and rigid model selection at fixed training endpoints, FairSplitEMA introduces a robust framework to overcome these challenges.

## Algorithm Overview

The pipeline executes the following procedural steps for each dataset:

1. **Initialization & Data Loading:** Load the dataset, normalize features, and split into training and testing sets across multiple random seeds.
2. **Subgroup Rebalancing:** Divide training data into target-label and protected-attribute subgroups, then augment smaller subgroups using WGAN-based generation.
3. **Decentralized Training:** Train across partitions incorporating momentum-smoothed fairness estimates (EMA).
4. **Pareto-Based Model Selection:** Retain candidate models across training rounds to select optimal models balancing predictive performance and group fairness.
