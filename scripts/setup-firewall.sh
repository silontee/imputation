#!/bin/bash
# ============================================================
# setup-firewall.sh — UFW 방화벽 설정 스크립트
#
# 허용 포트:
#   - 22  (SSH)
#   - 80  (HTTP — Let's Encrypt ACME + HTTPS 리다이렉트)
#   - 443 (HTTPS)
#
# 사용법:
#   sudo ./scripts/setup-firewall.sh
#
# 주의:
#   - 반드시 sudo 권한으로 실행하세요.
#   - SSH 접속이 끊기지 않도록 22번 포트를 먼저 허용합니다.
# ============================================================

set -euo pipefail

# root 권한 확인
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: 이 스크립트는 root 권한으로 실행해야 합니다."
    echo "  sudo $0"
    exit 1
fi

echo "==> UFW 방화벽 설정을 시작합니다..."

# ── 1단계: UFW 설치 확인 ──
if ! command -v ufw &> /dev/null; then
    echo "    UFW가 설치되어 있지 않습니다. 설치합니다..."
    apt-get update && apt-get install -y ufw
fi

# ── 2단계: 기본 정책 설정 ──
echo "==> 기본 정책: 들어오는 트래픽 차단, 나가는 트래픽 허용"
ufw default deny incoming
ufw default allow outgoing

# ── 3단계: 필요한 포트만 허용 ──
echo "==> SSH (22) 허용..."
ufw allow 22/tcp comment 'SSH'

echo "==> HTTP (80) 허용..."
ufw allow 80/tcp comment 'HTTP - ACME challenge'

echo "==> HTTPS (443) 허용..."
ufw allow 443/tcp comment 'HTTPS'

# ── 4단계: UFW 활성화 ──
echo "==> UFW 활성화..."
ufw --force enable

# ── 5단계: 상태 확인 ──
echo ""
echo "==> 현재 UFW 상태:"
ufw status verbose

# ── Docker iptables 우회 방지 안내 ──
echo ""
echo "============================================================"
echo " [중요] Docker iptables 우회 방지"
echo "============================================================"
echo ""
echo " Docker는 기본적으로 iptables를 직접 조작하여"
echo " UFW 규칙을 우회할 수 있습니다."
echo ""
echo " 이를 방지하려면 /etc/docker/daemon.json에 다음을 추가하세요:"
echo ""
echo '   {'
echo '     "iptables": false'
echo '   }'
echo ""
echo " 또는 /etc/default/docker 파일에:"
echo '   DOCKER_OPTS="--iptables=false"'
echo ""
echo " 변경 후 Docker 재시작:"
echo "   sudo systemctl restart docker"
echo ""
echo " 주의: iptables=false 설정 시 Docker 컨테이너 간"
echo " 네트워크 통신을 위한 iptables 규칙을 수동으로"
echo " 관리해야 할 수 있습니다."
echo "============================================================"
