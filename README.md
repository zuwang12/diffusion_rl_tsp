# DDRL: Diffusion-Driven Reinforcement Learning for Traveling Salesman Problem (TSP)

This repository contains the code for **DDRL (Diffusion-Driven Reinforcement Learning)**, a novel framework that integrates diffusion models and reinforcement learning (RL) to solve the Traveling Salesman Problem (TSP). DDRL offers scalable, stable, and efficient solutions for both standard and constraint-based TSP instances.

The code supports experiments as presented in the paper **"DDRL: A Diffusion-Driven Reinforcement Learning Approach for Enhanced TSP Solutions"** (currently under review for ICLR 2025).

## Overview

The Traveling Salesman Problem (TSP) is a fundamental challenge in combinatorial optimization, known for its NP-hard complexity. Reinforcement Learning (RL) has proven to be effective in managing larger and more complex TSP instances, yet it encounters challenges such as training instability and necessity for a substantial amount of training resources. Diffusion models, known for iteratively refining noisy inputs to generate high-quality solutions, offer scalability and exploration capabilities for TSP but may struggle with optimality in complex cases and require large, resource-intensive training datasets. To address these limitations, we propose DDRL (Diffusion-Driven Reinforcement Learning), which integrates diffusion models with RL. DDRL employs a latent vector to generate an adjacency matrix, merging image and graph learning within a unified RL framework. By utilizing a pre-trained diffusion model as a prior, DDRL exhibits strong scalability and enhanced convergence stability. We also provide theoretical analysis that training DDRL aligns with the diffusion policy gradient in the process of solving the TSP, demonstrating its effectiveness. Additionally, we introduce novel constraint datasets—obstacle, path, and cluster constraints—to evaluate DDRL's generalization capabilities. We demonstrate that DDRL offers a robust solution that outperforms existing methods in both basic and constrained TSP problems. 
