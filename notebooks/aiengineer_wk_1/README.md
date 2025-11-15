# Gradient-Based Optimization Case Study

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/tailvar/TAE/blob/master/notebook/aiengineer_wk_1/tae_capstone_wk1.ipynb
)

## Overview

This project explores the relationship between calculus-based analysis and first-order optimization methods.  
Starting from the one-dimensional function  

$f(x) = \left| \tfrac{1}{2}x^{3} - \tfrac{3}{2}x^{2} \right| + \tfrac{1}{2}x$

Notebook analyses its shape, stationary points, and global minima, then examine how various gradient-based algorithms behave when applied to it.

The goal is to understand how **learning rate (η)**, **momentum (β)**, **noise (σ)**, and **decay (γ)** influence convergence, stability, and performance near non-smooth points (such as the kink at \(x = 3\)).

---

## Contents

### 1. Gradient Descent and Heavy-Ball Momentum
- Compares standard Gradient Descent (GD) with the Heavy-Ball (HB) method.  
- Demonstrates how adding momentum accelerates convergence but may introduce oscillations if β is too high.  
- Illustrates how both methods approach the global minimum for various learning rates.

### 2. Stochastic Gradient Descent (SGD)
- Introduces randomness into gradient updates using a normally distributed noise term with standard deviation σ.  
- Explores how increasing σ affects convergence — higher noise slows progress and can destabilize the path.  
- Demonstrates that moderate noise can still reach the minimum but with more variance in the trajectory.

### 3. SGD with Decaying Learning Rate
- Implements a dynamic learning-rate schedule  
$$
\eta_k = \frac{\eta_0}{1 + \gamma k}
$$
- Shows that gradually reducing the step size improves stability over time and helps escape oscillations around non-smooth regions.  
- Compares parameter combinations such as (η₀ = 0.2, γ = 0.02) and (η₀ = 0.1, γ = 0.05).

### 4. Comparative Analysis
- Runs all four methods side-by-side:
  - Vanilla Gradient Descent  
  - Heavy-Ball Momentum  
  - Stochastic Gradient Descent  
  - SGD with Decaying η  
- Visualises trajectories for both **low (η = 0.1)** and **high (η = 0.2)** learning rates.  
- Demonstrates that higher η accelerates convergence but may overshoot near the kink, while smaller η stabilises progress but converges more slowly.

### 5. Metrics and Diagnostics
Each run computes:
- Final gap: $ f(x_K) - f(x^*) $
- Best-so-far gap: $ \min_{k \le K} f(x_k) - f(x^*) $
- Iterations to reach a tolerance threshold (e.g., $ f(x_k) - f(x^*) < 0.1 $)

These metrics quantify how efficiently each algorithm reaches the vicinity of the optimum.

---

## Protocol Summary

- **Initial points:** \(x_0 \in \{-1.0, 0.5, 2.0\}\)  
- **Methods:**  
  - GD with η ∈ {0.05, 0.10, 0.15, 0.20}  
  - SGD with constant η and with decaying ηₖ = η₀ / (1 + γk)  
- **Parameters:**  
  - η₀ ∈ {0.1, 0.2}, γ ∈ {0.02, 0.05}, β = 0.5, σ = 0.2  
- **Iterations:** K = 200  
- **Recorded outputs:** xₖ sequences and f(xₖ) values  

All stochastic runs use a **reproducible random number generator** via `np.random.default_rng(seed)`.

---

## Results Summary

- The Heavy-Ball method (β = 0.5) typically converges faster than vanilla GD for moderate η but overshoots when β or η are too large.  
- Stochastic GD shows expected instability for higher σ, with slower convergence and higher variance.  
- Decaying η schemes improve convergence smoothness, particularly in noisy settings.  
- Across all methods, higher η accelerates movement toward x*, but small η may get stuck oscillating near the kink at x = 3.

---

## Data

No external datasets are used. All results are generated from the analytical function \( f(x) \) above using Python (NumPy, Matplotlib) within a Jupyter or Colab environment.

---

## How to Run

Clone or open the notebook in JupyterLab or Google Colab.

Run all cells in order.