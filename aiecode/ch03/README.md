[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/tailvar/TheAIE/blob/master/aiecode/ch03/notebooks/tae_capstone_wk3.ipynb)
# **Tiny Transformer – Architecture, Workflow, and Reproducibility**

This project implements a compact GPT-style Transformer model alongside a fully reproducible, GPU-accelerated development and training stack. The environment is designed to be modular, cost-efficient, and seamless to work with from a local IDE while executing all computation remotely on a GPU.

---

## **Project Overview**

This repository includes:

- A minimal Transformer language model implemented in PyTorch  
- Word-level and character-level tokenisation  
- Scaled dot-product attention tests and causal mask verification  
- A single-block Transformer decomposition test  
- Sampling algorithms (greedy, temperature, top-k, top-p)  
- Training vs validation loss charting  
- Infrastructure automation for reproducible GPU-based development

The goal is to create an experimental environment that reveals *how transformers really work* and how they behave under extreme data constraints.

---

## **Transformer Experiment Scope**

### **Core Components**
- Token + positional embeddings (sinusoidal)  
- Multi-head self-attention  
- Causal masking  
- Feed-forward networks  
- LayerNorm + residual connections  
- Autoregressive LM head

### **Sampling**
- Greedy decoding  
- Temperature scaling  
- Top-k sampling  
- Top-p (nucleus) sampling  
- Combined temperature + nucleus sampling

### **Diagnostics & Unit Tests**
- Manual Q, K, V → softmax → attention output verification  
- Causal mask correctness check  
- Forward pass through a single Transformer block (with controlled weights)  
- Training vs validation loss visualisation  

---

## **Architecture**

The project runs on a **DigitalOcean GPU Droplet** using a fully containerised execution model. All machine-learning computation happens inside a **Docker container** running on the droplet, which ensures a consistent CUDA-enabled environment at all times.

### **GPU Droplet + Docker Runtime**
- Dedicated NVIDIA GPU hardware  
- Model training runs entirely inside a Docker container  
- `/workspace` inside the container is mapped to the project directory on the host  
- Guarantees reproducibility across sessions and machines  
- Ensures consistent library versions, CUDA toolkits, and Python runtime

This isolates training from the host machine and eliminates environment drift.

### **Local Development via PyCharm Remote Interpreter**
Although compute runs remotely, development occurs locally:

- PyCharm Professional connects to the droplet via SSH  
- The droplet’s Docker container is used as a **remote Python interpreter**  
- Code executes on the GPU while IDE features remain local:  
  - autocomplete  
  - linting  
  - debugging  
  - version control  
  - notebook editing  

Jupyter notebooks run in the container and are accessed through an SSH-forwarded port, giving you a local browsing experience powered by remote GPU compute.

---

## **Automated Environment Scaffolding**

Reproducibility and ease of setup are achieved using two automation layers:

### **Local Scaffolding Script**
Generates:

