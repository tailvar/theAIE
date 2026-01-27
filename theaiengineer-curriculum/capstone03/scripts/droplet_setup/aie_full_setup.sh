#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIGURE THESE VALUES
########################################

# Your GitHub repo
REPO_URL="https://github.com/tailvar/aie_gpu_project.git"

# Where to put the project on the droplet
PROJECT_ROOT="/root/aie-gpu-project"

# Docker image and container names
IMAGE_NAME="aie-project-gpu"
CONTAINER_NAME="aie-gpu-container"

# Python inside container (usually fine as 'python3' or 'python')
CONTAINER_PYTHON_BIN="python3"

########################################

# Detect sudo usage (if you ever run as non-root)
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "=== AIE FULL SETUP: system deps, git clone, docker build, venv+torch, container shell ==="

echo
echo "=== 1. Installing system dependencies (git, docker, python3, venv, pip) ==="
$SUDO apt-get update
$SUDO apt-get install -y git docker.io python3 python3-venv python3-pip

echo "Enabling and starting Docker service (if supported)..."
$SUDO systemctl enable docker || true
$SUDO systemctl start docker || true

echo
echo "=== 2. Cloning or updating project from GitHub ==="
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "Project directory not found: ${PROJECT_ROOT}"
  echo "Cloning ${REPO_URL} into ${PROJECT_ROOT}"
  git clone "${REPO_URL}" "${PROJECT_ROOT}"
else
  echo "Project directory exists: ${PROJECT_ROOT}"
  echo "Updating from remote..."
  cd "${PROJECT_ROOT}"
  git pull
  cd -
fi

cd "${PROJECT_ROOT}"

echo
echo "=== 3. Building Docker image: ${IMAGE_NAME} ==="
if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Image '${IMAGE_NAME}' already exists. Rebuilding to pick up any changes..."
fi

docker build -t "${IMAGE_NAME}" .

echo
echo "=== 4. Running or reusing container: ${CONTAINER_NAME} ==="

# Does the container already exist?
if docker ps -a --format '{{.Names}}' | grep -qw "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' already exists."

  # If it's not running, start it
  if ! docker ps --format '{{.Names}}' | grep -qw "${CONTAINER_NAME}"; then
    echo "Starting existing container..."
    docker start "${CONTAINER_NAME}"
  else
    echo "Container is already running."
  fi
else
  echo "No existing container '${CONTAINER_NAME}'. Creating a new one (detached)..."
  docker run -d --gpus all \
    --name "${CONTAINER_NAME}" \
    -v "${PROJECT_ROOT}":/workspace \
    "${IMAGE_NAME}" \
    tail -f /dev/null
fi

echo
echo "=== 5. Ensuring virtualenv and Python deps INSIDE container ==="

# Create venv if missing
docker exec "${CONTAINER_NAME}" bash -lc "
  set -e
  cd /workspace
  if [ ! -d .venv ]; then
    echo 'Creating virtual environment at /workspace/.venv'
    ${CONTAINER_PYTHON_BIN} -m venv .venv
  else
    echo 'Virtual environment already exists at /workspace/.venv'
  fi
"

# Upgrade pip and install requirements.txt if present
docker exec "${CONTAINER_NAME}" bash -lc "
  set -e
  cd /workspace
  source .venv/bin/activate
  echo 'Upgrading pip inside venv...'
  pip install --upgrade pip
  if [ -f requirements.txt ]; then
    echo 'Installing requirements.txt inside venv...'
    pip install -r requirements.txt
  else
    echo 'No requirements.txt found, skipping.'
  fi
"

# Install PyTorch with CUDA (cu121 wheel by default), non-fatal if it fails
docker exec "${CONTAINER_NAME}" bash -lc "
  set -e
  cd /workspace
  source .venv/bin/activate
  echo 'Attempting to install PyTorch with CUDA (cu121)...'
  pip install --index-url https://download.pytorch.org/whl/cu121 torch && \
    echo 'PyTorch installed successfully.' || \
    echo 'WARNING: PyTorch install failed. You may need to adjust the wheel index or version.'
"

echo
echo '=== 6. Dropping you into an interactive shell INSIDE the container, venv activated ==='
echo 'You should see a prompt like: root@<container-id>:/workspace#'
echo 'The virtualenv .venv will already be active.'

docker exec -it "${CONTAINER_NAME}" bash -lc "
  cd /workspace
  source .venv/bin/activate
  exec bash
"

echo
echo "=== Setup complete. You are now inside the container in /workspace with venv active. ==="
