#!/bin/bash
# =============================================================================
# VTOL Vision — Jetson Orin Nano Super Automated Provisioning
# =============================================================================
# Run this on the Jetson immediately after first SSH connection.
#
# What it does (fully automated, ~20 min total):
#   1. System update & essential packages
#   2. SSH key import (password-less login from dev PC)
#   3. Intel RealSense SDK (librealsense2 + pyrealsense2)
#   4. PyTorch + Ultralytics (JetPack-compatible wheels)
#   5. ROS 2 Humble
#   6. vtol-vision project files (rsync from dev PC)
#   7. TRT engine generation  [~10 min, runs in background]
#   8. ROS 2 workspace build
#   9. Systemd service for auto-start
#  10. Health check
#
# Usage:
#   bash jetson_provision.sh [OPTIONS]
#
# Options:
#   --dev-ip  <IP>      Dev-PC IP for rsync  (default: auto-detect via SSH_CLIENT)
#   --dev-user <user>   Dev-PC username       (default: dev)
#   --hostname <name>   Set Jetson hostname   (default: vtol-jetson)
#   --no-ros            Skip ROS 2 install
#   --no-trt            Skip TRT engine build
#   --no-service        Skip systemd service
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'
info()  { echo -e "${BLU}[INFO]${NC}  $*"; }
ok()    { echo -e "${GRN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YLW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
hdr()   { echo -e "\n${BOLD}══════════════════════════════════════════${NC}"; \
          echo -e "${BOLD}  $*${NC}"; \
          echo -e "${BOLD}══════════════════════════════════════════${NC}"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
DEV_IP="${DEV_IP:-$(echo "$SSH_CLIENT" | awk '{print $1}')}"
DEV_USER="${DEV_USER:-dev}"
JETSON_HOSTNAME="${JETSON_HOSTNAME:-vtol-jetson}"
SKIP_ROS=false
SKIP_TRT=false
SKIP_SERVICE=false

DEV_PUB_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH6YwCxnSFj/RiHaCY0N2de/eHlhW5nKA1zsfFEGR4mw zeetee1235@gmail.com"
MODEL_URL="https://github.com/Paichai-SkyEdge/vtol-vision/releases/download/mannequin-model-v1/best.pt"
PROJECT_DIR="$HOME/vtol-vision"
WEIGHTS_DIR="$PROJECT_DIR/weights"
ROS_WS="$HOME/ros2_ws"

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --dev-ip)     DEV_IP="$2";         shift 2 ;;
    --dev-user)   DEV_USER="$2";       shift 2 ;;
    --hostname)   JETSON_HOSTNAME="$2";shift 2 ;;
    --no-ros)     SKIP_ROS=true;       shift   ;;
    --no-trt)     SKIP_TRT=true;       shift   ;;
    --no-service) SKIP_SERVICE=true;   shift   ;;
    *) die "Unknown option: $1" ;;
  esac
done

# ── Pre-flight checks ─────────────────────────────────────────────────────────
hdr "Pre-flight"

[[ "$(uname -m)" == "aarch64" ]] || die "This script must run on the Jetson (aarch64)."

JETPACK=$(cat /etc/nv_tegra_release 2>/dev/null | grep -oP 'R\d+' | head -1 || echo "unknown")
info "JetPack release tag: $JETPACK"

CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "not found")
info "CUDA: $CUDA_VER"

TRT_VER=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "not found")
info "TensorRT: $TRT_VER"

FREE_GB=$(df -BG "$HOME" | awk 'NR==2{gsub("G","",$4); print $4}')
info "Free disk: ${FREE_GB} GB"
(( FREE_GB >= 10 )) || die "Need at least 10 GB free (found ${FREE_GB} GB)."

ok "Pre-flight passed."

# ── Step 1: Hostname & SSH key ─────────────────────────────────────────────────
hdr "Step 1 / 9 — Hostname & SSH key"

sudo hostnamectl set-hostname "$JETSON_HOSTNAME"
info "Hostname → $JETSON_HOSTNAME"

mkdir -p ~/.ssh && chmod 700 ~/.ssh
if ! grep -qF "$DEV_PUB_KEY" ~/.ssh/authorized_keys 2>/dev/null; then
  echo "$DEV_PUB_KEY" >> ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
  ok "Dev-PC SSH key added."
else
  ok "Dev-PC SSH key already present."
fi

# ── Step 2: System update ──────────────────────────────────────────────────────
hdr "Step 2 / 9 — System update"
sudo apt update -qq
sudo apt upgrade -y -qq
sudo apt install -y -qq \
  curl wget git vim htop iotop \
  build-essential cmake pkg-config \
  python3-pip python3-dev python3-venv \
  libopencv-dev \
  v4l-utils usbutils
ok "System packages installed."

# ── Step 3: Intel RealSense SDK ───────────────────────────────────────────────
hdr "Step 3 / 9 — Intel RealSense SDK"

if ! dpkg -l | grep -q librealsense2-dev; then
  # Intel librealsense2 apt repo
  sudo mkdir -p /etc/apt/keyrings
  curl -sSf https://librealsense.intel.com/Debian/apt-repo/keys/gpg_key \
    | sudo gpg --dearmor -o /etc/apt/keyrings/librealsense.gpg
  echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] \
        https://librealsense.intel.com/Debian/apt-repo \
        $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/librealsense.list
  sudo apt update -qq
  sudo apt install -y -qq librealsense2-dkms librealsense2-utils librealsense2-dev
  ok "librealsense2 apt package installed."
else
  ok "librealsense2 already installed."
fi

# pyrealsense2 — try pip, fall back to apt wheel
if ! python3 -c "import pyrealsense2" 2>/dev/null; then
  pip3 install pyrealsense2 --quiet || \
    sudo apt install -y -qq python3-pyrealsense2 || \
    warn "pyrealsense2 not installed via pip/apt — may need manual build."
fi
python3 -c "import pyrealsense2 as rs; print('  pyrealsense2', rs.__version__)"
ok "RealSense Python binding ready."

# ── Step 4: PyTorch + Ultralytics ─────────────────────────────────────────────
hdr "Step 4 / 9 — PyTorch & Ultralytics"

if ! python3 -c "import torch" 2>/dev/null; then
  # NVIDIA Jetson PyTorch wheel index (JetPack 6 / CUDA 12)
  TORCH_INDEX="https://pypi.jetson-ai-lab.dev/jp6/cu126"
  pip3 install --quiet --extra-index-url "$TORCH_INDEX" \
    torch torchvision torchaudio
fi
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())")
info "PyTorch $TORCH_VER  |  CUDA available: $CUDA_OK"
[[ "$CUDA_OK" == "True" ]] || warn "CUDA not available in PyTorch — check JetPack install."

pip3 install --quiet ultralytics
ok "Ultralytics installed."

# ── Step 5: ROS 2 Humble ──────────────────────────────────────────────────────
if $SKIP_ROS; then
  warn "Skipping ROS 2 (--no-ros)."
else
  hdr "Step 5 / 9 — ROS 2 Humble"
  if ! command -v ros2 &>/dev/null; then
    sudo apt install -y -qq software-properties-common
    sudo add-apt-repository -y universe
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) \
          signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
          http://packages.ros.org/ros2/ubuntu \
          $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
      | sudo tee /etc/apt/sources.list.d/ros2.list
    sudo apt update -qq
    sudo apt install -y -qq \
      ros-humble-desktop \
      ros-dev-tools \
      ros-humble-cv-bridge \
      ros-humble-vision-msgs
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    ok "ROS 2 Humble installed."
  else
    ok "ROS 2 Humble already present."
  fi
fi

# ── Step 6: Project files (rsync from dev PC) ─────────────────────────────────
hdr "Step 6 / 9 — vtol-vision project"

mkdir -p "$PROJECT_DIR" "$WEIGHTS_DIR"

if [[ -n "$DEV_IP" ]]; then
  info "Syncing from ${DEV_USER}@${DEV_IP}:~/vtol-vision/ ..."
  rsync -az --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='runs/' \
    --exclude='datasets/' \
    --exclude='*.pyc' \
    "${DEV_USER}@${DEV_IP}:~/vtol-vision/" \
    "$PROJECT_DIR/"
  ok "Project files synced."
else
  warn "DEV_IP not set — skipping rsync. Copy files manually to $PROJECT_DIR"
fi

# Download model weights if not present
if [[ ! -f "$WEIGHTS_DIR/best.pt" ]]; then
  info "Downloading best.pt from GitHub Releases ..."
  wget -q --show-progress -O "$WEIGHTS_DIR/best.pt" "$MODEL_URL" || \
    warn "Download failed. Place best.pt manually at $WEIGHTS_DIR/best.pt"
fi

[[ -f "$WEIGHTS_DIR/best.pt" ]] && ok "best.pt ready: $WEIGHTS_DIR/best.pt"

# ── Step 7: TRT engine generation ─────────────────────────────────────────────
if $SKIP_TRT; then
  warn "Skipping TRT engine build (--no-trt)."
elif [[ ! -f "$WEIGHTS_DIR/best.pt" ]]; then
  warn "best.pt not found — skipping TRT build."
else
  hdr "Step 7 / 9 — TensorRT FP16 engine (runs in background, ~10 min)"
  ENGINE_LOG="$WEIGHTS_DIR/trt_build.log"
  (
    cd "$WEIGHTS_DIR"
    yolo export \
      model=best.pt \
      format=engine \
      device=0 \
      imgsz=640 \
      half=True \
      > "$ENGINE_LOG" 2>&1 && \
    echo "TRT BUILD OK" >> "$ENGINE_LOG" || \
    echo "TRT BUILD FAILED" >> "$ENGINE_LOG"
  ) &
  TRT_PID=$!
  info "TRT engine build started (PID $TRT_PID)."
  info "Monitor: tail -f $ENGINE_LOG"
  info "Check when done: ls -lh $WEIGHTS_DIR/best.engine"
fi

# ── Step 8: ROS 2 workspace ────────────────────────────────────────────────────
if ! $SKIP_ROS; then
  hdr "Step 8 / 9 — ROS 2 workspace build"
  source /opt/ros/humble/setup.bash

  mkdir -p "$ROS_WS/src"
  if [[ -d "$PROJECT_DIR/src" ]]; then
    ln -sfn "$PROJECT_DIR" "$ROS_WS/src/vtol_vision" 2>/dev/null || true
    cd "$ROS_WS"
    colcon build \
      --packages-select vtol_vision \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      2>&1 | tail -5
    echo "source $ROS_WS/install/setup.bash" >> ~/.bashrc
    ok "ROS 2 workspace built."
  else
    warn "Project source not found at $PROJECT_DIR/src — skipping build."
  fi
fi

# ── Step 9: Systemd service ────────────────────────────────────────────────────
if $SKIP_SERVICE; then
  warn "Skipping systemd service (--no-service)."
else
  hdr "Step 9 / 9 — Systemd auto-start service"
  ENGINE_PATH="$WEIGHTS_DIR/best.engine"
  SERVICE_FILE=/etc/systemd/system/vtol-vision.service

  sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=VTOL Vision Node (YOLO11n TRT + ArUco)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/config/jetson.env
ExecStartPre=/bin/bash -c 'source /opt/ros/humble/setup.bash && source $ROS_WS/install/setup.bash'
ExecStart=/bin/bash -c 'source /opt/ros/humble/setup.bash && source $ROS_WS/install/setup.bash && ros2 launch vtol_vision vision.launch.py trt_engine_path:=$ENGINE_PATH'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  # Note: enabled but not started until TRT engine is ready
  sudo systemctl enable vtol-vision.service
  ok "Service installed (vtol-vision.service)."
  info "Start after TRT build: sudo systemctl start vtol-vision"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
hdr "Provisioning Complete"
echo ""
echo -e "${GRN}${BOLD}  Jetson Orin Nano Super — vtol-vision ready${NC}"
echo ""
echo "  Weights dir:  $WEIGHTS_DIR"
echo "  Project:      $PROJECT_DIR"
echo "  ROS 2 ws:     $ROS_WS"
echo ""
echo "  Next steps:"
echo "  1. Wait for TRT engine:   watch ls -lh $WEIGHTS_DIR/best.engine"
echo "  2. Start vision node:     sudo systemctl start vtol-vision"
echo "  3. Check status:          sudo systemctl status vtol-vision"
echo "  4. Stream latency bench:  python3 $PROJECT_DIR/tools/jetson_latency_bench.py \\"
echo "                              --engine $WEIGHTS_DIR/best.engine \\"
echo "                              --pt     $WEIGHTS_DIR/best.pt"
echo ""
echo "  From dev PC, once TRT done:"
echo "  scp vtol-jetson:~/latency.json paper/figures/"
echo "  python3 tools/generate_paper_figures.py --only latency \\"
echo "          --latency paper/figures/latency.json"
echo ""
