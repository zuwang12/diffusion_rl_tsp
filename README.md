DDRL: Diffusion-Driven Reinforcement Learning for Traveling Salesman Problem (TSP)
This repository contains the code for DDRL (Diffusion-Driven Reinforcement Learning), a novel framework that integrates diffusion models and reinforcement learning (RL) to solve the Traveling Salesman Problem (TSP). DDRL offers scalable, stable, and efficient solutions for both standard and constraint-based TSP instances.

The code supports experiments as presented in the paper "DDRL: A Diffusion-Driven Reinforcement Learning Approach for Enhanced TSP Solutions" (currently under review for ICLR 2025).

Overview
The Traveling Salesman Problem (TSP) is a classic problem in combinatorial optimization where the goal is to find the shortest possible tour that visits a given set of cities exactly once and returns to the starting point. The problem is NP-hard, making it computationally infeasible for large instances.

DDRL addresses these challenges by:

Combining Diffusion Models and Reinforcement Learning: Integrates diffusion models to iteratively refine noisy inputs, while RL optimizes the policy to navigate complex TSP instances efficiently.
Scalability and Stability: Utilizes pre-trained diffusion models as priors, enabling DDRL to scale to larger instances while maintaining training stability.
Handling Novel Constraints: DDRL solves both standard TSP instances and constraint-based variants (e.g., obstacle, path, and cluster constraints).
Key Features
Latent Vector Optimization: DDRL optimizes a latent vector representing the TSP graph's adjacency matrix, enabling effective merging of image and graph-based learning.
Diffusion as a Prior: Uses diffusion models as priors, guiding the RL policy towards better solutions.
Novel Constraints: DDRL generalizes to constraint-based TSP variants, including obstacle avoidance, path constraints, and clustering.
Installation
First, clone the repository:

bash
코드 복사
git clone https://github.com/anonymous/diffusion_rl_tsp.git
cd diffusion_rl_tsp
Install the required dependencies:

bash
코드 복사
pip install -r requirements.txt
Repository Structure
src/: Main implementation of DDRL, including diffusion models, reinforcement learning components, and helper functions.
experiments/: Scripts for running basic and constraint-based TSP experiments.
datasets/: Tools for generating synthetic TSP datasets and constraint datasets (obstacle, path, and cluster constraints).
Results/: Directory for saving generated tours, training logs, and performance metrics.
utils/: Utility scripts for visualizing tours, analyzing results, and managing experimental settings.
Usage
Running Basic TSP Experiments
To run DDRL on a basic TSP instance with 100 cities:

bash
코드 복사
python experiments/run_ddrl_tsp.py --problem_size 100 --experiment basic
Adjust the --problem_size parameter to handle TSP instances of different sizes (e.g., 50, 200).

Running Constraint-based TSP Experiments
To run DDRL on a constraint-based TSP instance with obstacle constraints:

bash
코드 복사
python experiments/run_ddrl_tsp.py --problem_size 100 --experiment obstacle
Other supported constraints include path and cluster, which can be specified with the --experiment argument.

Visualizing Results
You can visualize the results of a generated TSP tour:

bash
코드 복사
python utils/visualize_tour.py --results_dir Results/path/tsp100
This will generate and save visual representations of the TSP tours.

Reproducing Paper Results
The code in the experiments/ directory allows for reproducing all major results from the paper, including both basic and constraint-based TSP experiments. Detailed instructions are provided in the respective scripts for setting up and running the experiments.

Citation
If you use this code in your research, please cite the following paper:

latex
코드 복사
@inproceedings{ddrl2025,
  title={DDRL: A Diffusion-Driven Reinforcement Learning Approach for Enhanced TSP Solutions},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
License
This repository is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
This work builds on prior research in diffusion models and reinforcement learning. We gratefully acknowledge [Funding Agency] for supporting this research.

Anonymous Code Repository
This repository is available for review at the following anonymous link: Anonymous Repository.

Let me know if you need any further adjustments!
