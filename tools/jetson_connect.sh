#!/bin/bash
# =============================================================================
# VTOL Vision — Jetson SSH 연결 설정 스크립트 (dev PC에서 실행)
# =============================================================================
# 1. Jetson IP 입력 → ~/.ssh/config 등록
# 2. SSH 키 복사 (비밀번호 없는 로그인)
# 3. 프로젝트 파일 전송
# 4. jetson_provision.sh 원격 실행
#
# Usage:
#   bash tools/jetson_connect.sh <JETSON_IP> [JETSON_USER]
#
# Example:
#   bash tools/jetson_connect.sh 192.168.1.42
#   bash tools/jetson_connect.sh 192.168.1.42 nvidia
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'
info() { echo -e "${BLU}[INFO]${NC}  $*"; }
ok()   { echo -e "${GRN}[ OK ]${NC}  $*"; }
die()  { echo -e "${RED}[ERR ]${NC}  $*"; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────────────
JETSON_IP="${1:-}"
JETSON_USER="${2:-nvidia}"   # JetPack default user is 'nvidia'
JETSON_HOST="vtol-jetson"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$JETSON_IP" ]]; then
  echo ""
  echo -e "${BOLD}Usage:${NC}  bash tools/jetson_connect.sh <JETSON_IP> [USER]"
  echo ""
  echo "  Jetson의 IP를 모를 경우:"
  echo "  1. Jetson에 모니터+키보드 연결 후:  hostname -I"
  echo "  2. 같은 공유기라면 라우터 관리 페이지에서 확인"
  echo "  3. mDNS가 켜져 있으면:  ping vtol-jetson.local"
  echo ""
  exit 0
fi

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  VTOL Vision — Jetson 연결 설정${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo "  Target : ${JETSON_USER}@${JETSON_IP}"
echo "  Alias  : ${JETSON_HOST}"
echo ""

# ── 1. Connectivity check ─────────────────────────────────────────────────────
info "Checking connectivity to $JETSON_IP ..."
ping -c 2 -W 2 "$JETSON_IP" > /dev/null 2>&1 \
  || die "Cannot reach $JETSON_IP — check IP and network connection."
ok "Jetson reachable."

# ── 2. SSH key generation (if needed) ─────────────────────────────────────────
if [[ ! -f ~/.ssh/id_ed25519 ]]; then
  info "Generating ED25519 SSH key ..."
  ssh-keygen -t ed25519 -C "${USER}@$(hostname)-vtol-vision" -f ~/.ssh/id_ed25519 -N ""
  ok "SSH key generated."
fi

# ── 3. Copy SSH key to Jetson ─────────────────────────────────────────────────
info "Copying SSH key to Jetson (you may be asked for password once) ..."
ssh-copy-id -i ~/.ssh/id_ed25519.pub "${JETSON_USER}@${JETSON_IP}"
ok "SSH key installed on Jetson."

# ── 4. ~/.ssh/config entry ────────────────────────────────────────────────────
SSH_CONF=~/.ssh/config
if ! grep -q "Host ${JETSON_HOST}" "$SSH_CONF" 2>/dev/null; then
  cat >> "$SSH_CONF" <<EOF

# VTOL Jetson Orin Nano Super
Host ${JETSON_HOST}
    HostName ${JETSON_IP}
    User ${JETSON_USER}
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ForwardX11 no
EOF
  chmod 600 "$SSH_CONF"
  ok "~/.ssh/config entry added: ssh ${JETSON_HOST}"
else
  # Update IP in case it changed
  sed -i "/Host ${JETSON_HOST}/{n;s/HostName .*/HostName ${JETSON_IP}/}" "$SSH_CONF"
  ok "~/.ssh/config already has '${JETSON_HOST}' entry (IP updated)."
fi

# ── 5. Transfer provision script ──────────────────────────────────────────────
info "Uploading provision script ..."
ssh "${JETSON_HOST}" "mkdir -p ~/vtol-vision/tools"
scp "$PROJECT_DIR/tools/jetson_provision.sh" \
    "$PROJECT_DIR/tools/jetson_latency_bench.py" \
    "${JETSON_HOST}:~/vtol-vision/tools/"
ssh "${JETSON_HOST}" "chmod +x ~/vtol-vision/tools/jetson_provision.sh"
ok "Scripts uploaded."

# ── 6. Confirm before full provision ─────────────────────────────────────────
echo ""
echo -e "${BOLD}연결 준비 완료.${NC} 전체 환경 설치를 지금 시작할까요?"
echo "  (약 20분 소요 / TRT 엔진 빌드는 백그라운드로 ~10분 추가)"
echo ""
read -rp "  [y/N] > " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo ""
  info "Jetson provisioning 시작 ..."
  ssh -t "${JETSON_HOST}" \
    "DEV_IP=$(hostname -I | awk '{print $1}') \
     DEV_USER=$(whoami) \
     bash ~/vtol-vision/tools/jetson_provision.sh"
else
  echo ""
  ok "연결 설정 완료. 나중에 직접 실행하려면:"
  echo ""
  echo "  ssh ${JETSON_HOST}"
  echo "  bash ~/vtol-vision/tools/jetson_provision.sh"
  echo ""
fi
