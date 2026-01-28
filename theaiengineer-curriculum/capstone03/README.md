[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/tailvar/TheAIE/blob/master/theaiengineer-curriculum/capstone03/notebooks/tae_capstone_wk3.ipynb)

←**[Back to main README](../../README.md)**

# **Tiny Transformer – Architecture, Workflow, and Reproducibility**

This third capstone project implements a compact GPT-style Transformer model alongside a fully reproducible, GPU-accelerated development and training stack. The environment is designed to be modular, cost-efficient, and seamless to work with from a local IDE while executing all computation remotely on a GPU.

---

## **Project Overview**

This repository includes:

- A minimal Transformer language model implemented in PyTorch  
- Both word-level and character-level tokenisation  
- Scaled dot-product attention tests and causal mask verification  
- A single-block Transformer decomposition test  
- Sampling algorithms (greedy, temperature, top-k, top-p)  
- Training vs validation loss charting  
- Infrastructure automation for reproducible GPU-based development

The overarching goal of the project is to create an experimental environment that exposes how transformers actually work internally, and how their behaviour changes under extreme data constraints, rather than simply treating them as opaque black boxes.


## **Transformer Experiment Scope**

At the core of the project is a standard but fully transparent Transformer architecture. The model is built from token embeddings combined with sinusoidal positional encodings, followed by a multi-head self-attention mechanism with explicit causal masking. This is then passed through position-wise feed-forward networks, with LayerNorm and residual connections applied throughout, and terminated by an autoregressive language-model head. Each of these components is implemented in a way that allows intermediate activations to be inspected and tested independently.

The sampling framework is deliberately broad, covering greedy decoding, temperature-scaled sampling, top-k sampling, top-p (nucleus) sampling, and combinations of temperature with nucleus sampling. This makes it possible to study how different decoding strategies affect output diversity, stability, and coherence, and to understand the practical trade-offs involved in moving from deterministic to stochastic generation.


## **Architecture**

The project runs on a **DigitalOcean GPU Droplet** using a fully containerised execution model. All machine-learning computation happens inside a **Docker container** running on the droplet, which ensures a consistent CUDA-enabled environment at all times. ***Instructions on how to set up the droplet*** thats in a docker container are inncluded in the directory /scripts/droplet_setup/README.md with the bash scripts required to set up the digital ocean droplet so it is production ready.

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
  - debugging  
  - version control  
  - notebook editing  

Jupyter notebooks run in the container and are accessed through an SSH-forwarded port, giving you a local browsing experience powered by remote GPU compute.


## **Automated Environment Scaffolding**

Reproducibility and ease of setup are achieved using two automation layers:

### **Local Scaffolding Script (aie_full_setup.sh)**


- Installs system dependencies such as Docker, Git, Python, and virtual-environment tools.
- Pulls the latest version of the project from GitHub.
- Builds the project’s Docker image.
- Creates and runs a GPU-enabled container with the project directory mounted into it.
- Creates and activates a Python virtual environment inside the container.
- Installs all Python dependencies and optionally CUDA-enabled PyTorch.
- Drops the user into an interactive shell inside the container, ready for training.

## Data

No external data sources are required. The dataset is a text block contained in one of the notebook cells.

## Results Summary

Training and evaluation confirm the expected behaviour of a small transformer trained on an extremely limited dataset. The model quickly drives the **training loss down from ~5.6 to ~0.15**, demonstrating that it can memorise the training corpus with ease. However, the **validation loss remains high (~5.1 at best) and steadily deteriorates**, rising toward ~8 as training progresses. This widening gap between training and validation performance highlights strong overfitting: the model learns the exact sequences it sees but fails to generalise even to nearby unseen tokens from the same text.

Although we incorporated regularisation — including **dropout = 0.2** — the underlying issue is data scarcity relative to model capacity. With so few unique tokens and such short contexts, the model’s probability distribution collapses around memorised word transitions, and additional training simply reinforces these patterns.

Overall, the results are entirely consistent with expectations for a tiny GPT-style model trained on a small, repetitive corpus: fast memorisation, poor generalisation, and validation degradation over time. This makes the setup ideal for studying overfitting dynamics, sampling behaviour, and the mechanics of transformer components in a tightly controlled environment.


## How to Run

Open the notebook in JupyterLab or Google Colab and run all cells in order.

## The notebook requires:

    NumPy
    PyTorch
    Matplotlib
    random
    json
    math
    re
    dataclasses
    datetime

All computations will run on CPU, but training is **significantly slower** without hardware acceleration. While the notebooks and tests execute correctly on any machine, the full training loop for the transformer is **best run on an NVIDIA GPU**, where performance improves by one to two orders of magnitude. This project is designed with GPU execution in mind, and the recommended workflow uses a CUDA-enabled Docker environment on a cloud-based GPU droplet.




