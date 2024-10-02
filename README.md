# DDRL: Diffusion-Driven Reinforcement Learning for Traveling Salesman Problem (TSP)

This repository contains the code for **DDRL (Diffusion-Driven Reinforcement Learning)**, a novel framework that integrates diffusion models and reinforcement learning (RL) to solve the Traveling Salesman Problem (TSP). DDRL offers scalable, stable, and efficient solutions for both standard and constraint-based TSP instances.

The code supports experiments as presented in the paper **"DDRL: A Diffusion-Driven Reinforcement Learning Approach for Enhanced TSP Solutions"** (currently under review for ICLR 2025).

## Overview

The Traveling Salesman Problem (TSP) is a fundamental challenge in combinatorial optimization, known for its NP-hard complexity. Reinforcement Learning (RL) has proven to be effective in managing larger and more complex TSP instances, yet it encounters challenges such as training instability and necessity for a substantial amount of training resources. Diffusion models, known for iteratively refining noisy inputs to generate high-quality solutions, offer scalability and exploration capabilities for TSP but may struggle with optimality in complex cases and require large, resource-intensive training datasets. To address these limitations, we propose DDRL (Diffusion-Driven Reinforcement Learning), which integrates diffusion models with RL. DDRL employs a latent vector to generate an adjacency matrix, merging image and graph learning within a unified RL framework. By utilizing a pre-trained diffusion model as a prior, DDRL exhibits strong scalability and enhanced convergence stability. We also provide theoretical analysis that training DDRL aligns with the diffusion policy gradient in the process of solving the TSP, demonstrating its effectiveness. Additionally, we introduce novel constraint datasets—obstacle, path, and cluster constraints—to evaluate DDRL's generalization capabilities. We demonstrate that DDRL offers a robust solution that outperforms existing methods in both basic and constrained TSP problems. 

## Data Requirements

To obtain the optimization results using DDRL, you will need the basic dataset:

1. Download the TSP datasets (tsp20, tsp50, tsp100, tsp200) from [here](https://github.com/chaitjo/learning-tsp).
2. Move the downloaded data to the `data/` directory of this project.

For using the three types of constraint datasets (obstacle, path, cluster), you will need to generate the datasets:

1. Use the provided scripts in `data/make_dataset_box.py`, `data/make_dataset_path.py`, and `data/make_dataset_cluster.py` to create the corresponding datasets.
2. After generation, move the datasets to the `data/` directory.

Additionally, the prior knowledge (pre-trained diffusion model) can be obtained from [this repository](https://github.com/AlexGraikos/diffusion_priors?tab=readme-ov-file). Please ensure that the prior model is correctly integrated within the DDRL framework for optimal performance.

## Code of Ethics and Ethical Considerations

All participants, including authors and reviewers, are expected to adhere to the [ICLR Code of Ethics](https://iclr.cc/public/CodeOfEthics). Please ensure that any ethical issues related to your research, such as those involving human subjects, data privacy, or potential harms, are addressed within your paper and supplementary materials.

## Reproducibility Statement

To ensure the reproducibility of our results, we encourage authors to include a reproducibility statement in their submissions, referencing all key components needed for replication, such as source code, dataset descriptions, and theoretical proofs included in the supplementary materials.

