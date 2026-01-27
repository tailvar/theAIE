# AIE GPU Project – DigitalOcean + Docker Setup

This repository defines a reproducible GPU development environment using:

- A local Linux workstation (PyCharm)
- A DigitalOcean GPU droplet
- A Docker-based CUDA container
- A Python virtual environment inside the container

The design goal is **clean separation of concerns**:

| Layer | Purpose |
|------|---------|
| Local PC | Code authoring (PyCharm) |
| Droplet host | Git + Docker orchestration |
| Container | Actual runtime environment (CUDA, Torch, Python) |

Once set up, you only ever:
- edit code locally or on the droplet,
- run experiments inside the container,
- commit and push from the droplet.

---

## Cheat Sheet

### Local PC

```bash
cd ~/PycharmProjects
./init_aie_project.sh
git config --global user.name "tailvar"
git config --global user.email "piers.watson1@bigpond.com"
cd ~/PycharmProjects/aie-gpu-project
git push
```

### Droplet Host (root@droplet)

```bash
ssh root@YOUR_DROPLET_IP
apt-get update
apt-get upgrade -y
git config --global user.name "Piers Watson"
git config --global user.email "piers.watson1@bigpond.com"
cd /root
./aie_full_setup.sh
docker exec -it aie-gpu-container bash
```

### Container (root@container:/workspace)

```bash
cd /workspace
source .venv/bin/activate
python scripts/check_gpu.py
vim scripts/train_model.py
```

You never delete any of the three layers.  
You simply respect where each one lives and what it is responsible for.

---

## Git Identity

Git identity is per machine.

You must set it:
- once on your local PC
- once on each droplet

You do NOT need it inside the container.

### Local PC

```bash
git config --global user.name "tailvar"
git config --global user.email "piers.watson1@bigpond.com"
```

### Droplet

```bash
ssh root@YOUR_DROPLET_IP
git config --global user.name "Piers Watson"
git config --global user.email "piers.watson1@bigpond.com"
```

---

## Roles of the Three Main Files

### 1. init_aie_project.sh  
(Local scaffolding script – run once on local PC only)

Creates project skeleton at:

```
~/PycharmProjects/aie-gpu-project
```

Used once at project creation.  
Never run on droplet or container.

---

### 2. Dockerfile  
(Container definition)

Defines:
- CUDA version
- Python version
- System packages (vim, git, etc.)

Used only by Docker on the droplet.  
Controls what exists inside the container.

---

### 3. aie_full_setup.sh  
(Droplet bootstrap – the “do everything” script)

Run only from:

```
root@droplet:~#
```

It performs:

1. Installs system dependencies (git, docker, python3, venv)
2. Clones or updates repo into:
   ```
   /root/aie-gpu-project
   ```
3. Builds Docker image
4. Starts GPU container:
   ```
   aie-gpu-container
   ```
5. Mounts repo into:
   ```
   /workspace
   ```
6. Creates venv:
   ```
   /workspace/.venv
   ```
7. Installs requirements and CUDA Torch
8. Drops you inside container with venv active

This script is idempotent and safe to re-run.

---

## Fresh End-to-End Setup

### Local Machine (one-off)

```bash
cd ~/PycharmProjects
./init_aie_project.sh
cd aie-gpu-project
git remote add origin https://github.com/tailvar/aie_gpu_project.git
git branch -M main
git push -u origin main
```

Open project in PyCharm.

---

### Droplet First Time Setup

```bash
ssh root@YOUR_DROPLET_IP
git config --global user.name "Piers Watson"
git config --global user.email "piers.watson1@bigpond.com"
cd /root
vim aie_full_setup.sh
chmod +x aie_full_setup.sh
./aie_full_setup.sh
```

You will now be inside:

```bash
root@CONTAINER_ID:/workspace#
```

Venv is already active.

---

## Normal Workflow

Inside container:

```bash
python scripts/train_model.py
python scripts/evaluate.py
```

Exit container:

```bash
exit
```

---

## Committing and Pushing (Always on Droplet)

```bash
cd /root/aie-gpu-project
git status
git add .
git commit -m "new experiment"
git push
```

Never commit from inside the container.

---

## Re-entering Container

```bash
docker start aie-gpu-container
docker exec -it aie-gpu-container bash
cd /workspace
source .venv/bin/activate
```

Or simply:

```bash
cd /root
./aie_full_setup.sh
```

---

## PyCharm Remote Interpreter

Configure:

```
File → Settings → Project → Python Interpreter → Add → SSH Interpreter
```

Use:

```
Host: YOUR_DROPLET_IP
Username: root
```

Start terminal:

```
Tools → Start SSH Session
```

---

## Jupyter Lab Over SSH

Inside container:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

On local PC:

```bash
ssh -L 8889:localhost:8888 root@YOUR_DROPLET_IP
```

Open browser:

```
http://localhost:8889
```






