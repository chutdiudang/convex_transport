# Convex Transportation Problem

This repository contains the implementation for the convex transportation
problem in a logistics system with quadratic (convex) transportation costs.

The problem considers multiple warehouses and multiple customers, where
transportation costs increase nonlinearly with flow to reflect congestion
and capacity effects.

## Methods
- ADMM (via OSQP solver)
- Interior Point Method (Mehrotra Predictor–Corrector)

## Project Structure
- `generate_dataset_network.py`  
  Script to generate transportation network, supply, demand and convex cost parameters.

- `Convex_Transportation_ADMM_Analysis.ipynb`  
  ADMM-based solver using OSQP and convergence analysis.

- `Interior_Point_Method.ipynb`  
  Interior Point Method (Mehrotra PC) implementation for convex QP.

- `dataset_large/`  
  Generated datasets and solution files.
