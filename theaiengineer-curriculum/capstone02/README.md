[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tailvar/theAIE/blob/master/theaiengineer-curriculum/capstone02/notebooks/tae_capstone_wk2.ipynb)

# Tiny One-Hidden-Layer MLP: NumPy → PyTorch (XOR)

## Overview

This project builds a complete, end-to-end walkthrough of how a tiny feed-forward neural network learns a nonlinear classification task.  
Starting with a **fully manual NumPy implementation** of forward and backward propagation, then progressively transitioning to **PyTorch**:

1. First replicating the model in tensors **without autograd**.  
2. Then enabling autograd and verifying **gradients**.  
3. Finally wrapping the network into both a custom `nn.Module` and a concise `nn.Sequential`.

The goal is to understand, from first principles, how affine layers, nonlinearities, gradients, and optimisation interact inside a minimal MLP trained on the classic **XOR** problem.

---
### Training Step Overview

Every iteration of neural-network training follows the same computational pipeline:

1. **`zero_grad()`**  
   Clears any gradients left over from the previous step.  
   (PyTorch accumulates gradients by default, so clearing them is essential.)

2. **Forward pass**  
   Inputs flow through the network to produce predictions using the current parameters:  
   
        x ──> fθ(x)


3. **Loss computation**  
   Measures how far the prediction is from the target:
    
       L = loss(fθ(x), y)


4. **Backward pass**  
   Applies the chain rule to compute gradients of the loss with respect to all parameters:  
   
       ∇θ L
           

5. **Parameter update (`step()`)**  
   The optimiser uses these gradients to tweak the parameters slightly:  
   
        θ ← θ − η · ∇θ L


Together, these operations form the core **learning loop** in both the NumPy and PyTorch implementations of the MLP.

    [ zero_grad ] ──▶ [ forward ] ──▶ [ loss ] ──▶ [ backward ] ──▶ [ step ]

---

## Contents

### 1. Manual NumPy MLP (Reference Implementation)
- Implements forward propagation for a 2-16-1 network:  
  affine → ReLU → affine → scalar output.
- Implements full backward propagation using the chain rule.  
- Trains the model using **full-batch gradient descent**.  
- Demonstrates that the model converges to ≈98% accuracy, validating the math.

### 2. PyTorch (Forward Only)
- Re-implements exactly the same computation graph using PyTorch tensors **without autograd**.  
- Confirms that the PyTorch forward pass produces the **same loss** as the NumPy model (up to floating-point tolerance).  
- Ensures we have a one-to-one correspondence between NumPy and PyTorch operations.

### 3. PyTorch Autograd (Gradient Consistency Check)
- Recreates the NumPy initial parameters as PyTorch tensors with `requires_grad=True`.  
- Runs forward and backward using autograd.  
- Compares PyTorch gradients with the manual NumPy gradients on a fixed mini-batch.  
- Shows **agreement at the 1e-8 level**, confirming correctness of the NumPy backward pass.

### 4. Training with `nn.Module` and `nn.Sequential`
Two PyTorch abstractions are used:

- **Custom `TinyMLP` (nn.Module)**:  
  Explicit parameters `W1`, `b1`, `W2`, `b2` matching the NumPy shapes.  
  Trained with SGD; tracks loss, accuracy, gradient norms, and ReLU activation fraction.

- **`nn.Sequential` implementation**:  
  A compact network using two `nn.Linear` layers and a `nn.ReLU`.  
  Achieves similar performance, illustrating that PyTorch’s built-ins reproduce the same computation graph succinctly.

### 5. Diagnostics and Visualisation
- Plots training curves for:  
  - Loss  
  - Gradient norm  
  - ReLU active fraction  
  - Accuracy  
- Visualises **decision boundaries** for all three models:  
  - NumPy MLP  
  - TinyMLP (nn.Module)  
  - nn.Sequential  
- Shows how each implementation “bends” the input space to separate the XOR classes.

---

## Results Summary

- **NumPy model** converges smoothly to ~0.031 loss and ~98% accuracy.  
- **Gradient check** confirms near-perfect alignment between NumPy and PyTorch (max error ≈ 1e-8).  
- **TinyMLP (nn.Module)** reaches ~0.036 loss and ~96% accuracy with stable gradient norms and healthy ReLU activity (~0.48–0.50 active).  
- **nn.Sequential** performs equivalently, reaching ~0.034 loss and ~95% accuracy.

Across all implementations, the network reliably learns the nonlinear XOR boundary using only one hidden layer.

---

## Data

No external data sources are required. The dataset consists of **synthetic XOR-style points** uniformly sampled from the square $[-1, 1]^2$, labelled according to the sign of $(x_1 \cdot x_2)$. All experiments run deterministically using fixed random seeds.

---

## How to Run

Open the notebook in JupyterLab or Google Colab and run all cells in order.

The notebook requires:  
- NumPy  
- PyTorch  
- Matplotlib  

All computations are CPU-friendly and complete in seconds.

---
