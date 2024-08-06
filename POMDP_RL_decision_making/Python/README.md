# Partially Observable Markov Decision Process Reinforcement Learning (POMDP-RL) Decision Making Model

This repository contains Python implementations of POMDP-RL model for decision-making tasks. The model simulates decision-making processes in perceptual tasks, particularly focusing on how confidence and reward prediction errors influence decision boundaries and reaction times.

## Model Overview

The model uses a POMDP framework combined with reinforcement learning based on temporal difference (TD) learning. Key aspects of this approach include:

1. **Belief State**: The agent maintains a belief state about the true state of the environment, which is updated based on observations.
2. **TD Learning**: The agent learns action values using TD learning, updating its estimates based on the difference between expected and actual rewards.
3. **Confidence-based Decision Making**: The agent decides when to stop accumulating evidence based on its current confidence and expected future confidence.
4. **Adaptive Thresholds**: The decision threshold is adjusted based on the outcome and confidence of the previous trial.

## Files

- `belief_update.py`: Updates the belief distribution (mean and standard deviation).
- `calc_belief_LR.py`: Calculates beliefs for the true state being on left and right.
- `calc_expected_conf.py`: Calculates the expected confidence for an additional observation.
- `gen_obs_opti.py`: Generates observations for the model.
- `td_w_belief.py`: Implements temporal difference learning with belief states.
- `sim_POMDP_reaction_RDM_func.py`: Main simulation function for the POMDP reaction time model.
- `main.ipynb`: Jupyter notebook for running simulations and generating plots.

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook

## Usage

1. Ensure all required packages are installed:

