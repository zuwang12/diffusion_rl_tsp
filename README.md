# DDRL: Diffusion-Driven Reinforcement Learning for Traveling Salesman Problem (TSP)

This repository contains the code for **DDRL (Diffusion-Driven Reinforcement Learning)**, a novel framework that integrates diffusion models and reinforcement learning (RL) to solve the Traveling Salesman Problem (TSP). DDRL offers scalable, stable, and efficient solutions for both standard and constraint-based TSP instances.

The code supports experiments as presented in the paper **"DDRL: A Diffusion-Driven Reinforcement Learning Approach for Enhanced TSP Solutions"** (currently under review for ICLR 2025).

## Overview

The Traveling Salesman Problem (TSP) is a classic problem in combinatorial optimization where the goal is to find the shortest possible tour that visits a given set of cities exactly once and returns to the starting point. The problem is NP-hard, making it computationally infeasible for large instances.

**DDRL** addresses these challenges by:
- **Combining Diffusion Models and Reinforcement Learning**: Integrates diffusion models to iteratively refine noisy inputs, while RL optimizes the policy to navigate complex TSP instances efficiently.
- **Scalability and Stability**: Utilizes pre-trained diffusion models as priors, enabling DDRL to scale to larger instances while maintaining training stability.
- **Handling Novel Constraints**: DDRL solves both standard TSP instances and constraint-based variants (e.g., obstacle, path, and cluster constraints).

## Key Features
- **Latent Vector Optimization**: DDRL optimizes a latent vector representing the TSP graph's adjacency matrix, enabling effective merging of image and graph-based learning.
- **Diffusion as a Prior**: Uses diffusion models as priors, guiding the RL policy towards better solutions.
- **Novel Constraints**: DDRL generalizes to constraint-based TSP variants, including obstacle avoidance, path constraints, and clustering.

## Installation

First, clone the repository:

```bash
git clone https://github.com/anonymous/diffusion_rl_tsp.git
cd diffusion_rl_tsp
