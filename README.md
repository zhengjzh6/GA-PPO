# GA-PPO
# PPO-HENS: Deep Reinforcement Learning for Heat Exchanger Network Synthesis

This repository contains the official MATLAB source code for the paper: 
**"Comparison of Applications of Genetic Algorithm and Deep Reinforcement Learning in Heat Exchanger Network Synthesis"** (Submitted to *Chemical Engineering Journal*).

## Requirements
* MATLAB R2025b (or later)
* Deep Learning Toolbox & Reinforcement Learning Toolbox
* **Gurobi Optimizer** (Version 11.0 or later, required for inner-level LP solving)

## Repository Structure
* `run_hen_drl.m`: Main script to train and evaluate the PPO agent.
* `HEN_Env_Discrete.m`: Custom RL environment defining the MDP formulation and Action Masking.
* `LP_DRL.m` / `LP_GA.m`: Inner-level Linear Programming solvers utilizing Gurobi.
* `Full_GA_HEN_Solver.m`: The comparative Genetic Algorithm baseline.
* `Monte_T.m` & `plot_IQR.m`: Scripts for stochastic disturbance generation and robustness evaluation.

## How to Run
1. Ensure Gurobi is properly installed and licensed on your MATLAB path.
2. Run `run_hen_drl.m` to start training the PPO agent on the benchmark problem.
3. Run `Ternary_Search.m`.
4. To test robustness, run `Monte_T.m` after the agent is trained.
