#!/usr/bin/env bash
set -euo pipefail

# ---------- CONFIG ----------
BASE_DIR="${HOME}/PycharmProjects"
PROJECT_NAME="aie-gpu-project"

INIT_GIT=true   # set to false to disable git init
# ----------------------------

PROJECT_ROOT="${BASE_DIR}/${PROJECT_NAME}"

echo "==> Creating project at: ${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

echo "==> Creating directory structure"
mkdir -p src notebooks tests scripts

touch src/__init__.py

echo "==> Creating README.md"
cat > README.md << 'EOF'
# AIE GPU Project

This repository contains my AI Engineer (AIE) GPU experimentation project.

## Project Structure
- src/        - Python modules and training utilities
- notebooks/  - Jupyter notebooks
- tests/      - Test scripts
- scripts/    - CLI helpers and utilities
- requirements.txt
- Dockerfile (container environment)

## Local Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

EOF

echo "==> Creating requirements.txt"
cat > requirements.txt << 'EOF'
# Add Python dependencies here, for example:
# torch
# numpy
# matplotlib
# jupyterlab
EOF

echo "==> Creating .gitignore"
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]

.venv/
venv/
env/
.env/

.ipynb_checkpoints/

.idea/
.vscode/
.pycharm_helpers/

data/
datasets/
runs/
checkpoints/
*.pt
*.pth
*.h5
*.onnx

.DS_Store
EOF

echo "==> Creating placeholder Dockerfile"
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

CMD ["bash"]
EOF

if [ "${INIT_GIT}" = true ]; then
  echo "==> Initialising git repository"
  git init
  git add .
  git commit -m "Initial scaffold: src/notebooks/tests/scripts + README/requirements/Dockerfile"
  echo "==> Git repository initialised. Add your remote with:"
  echo "  git remote add origin https://github.com/<user>/<repo>.git"
  echo "  git push -u origin main"
else
  echo "==> Skipping git initialisation (INIT_GIT=false)"
fi

echo "==> Done."
echo "Project created at: ${PROJECT_ROOT}"
