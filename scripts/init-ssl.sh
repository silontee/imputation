#!/bin/bash
# ============================================================
# init-ssl.sh — Let's Encrypt 인증서 최초 발급 스크립트
#
# 사용법:
#   sudo ./scripts/init-ssl.sh              # 실제 발급
#   sudo ./scripts/init-ssl.sh --dry-run    # 테스트 (발급 없이 검증만)
#
# 사전 조건:
#   1. DOMAIN 환경 변수 설정 또는 .env 파일에 정의
#   2. docker-compose up nginx 실행 중 (80 포트 오픈)
#   3. DNS A 레코드가 서버 IP를 가리켜야 함
# ============================================================

set -euo pipefail

# ── 설정 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# .env 파일에서 DOMAIN 로드 (설정되어 있지 않은 경우)
if [ -z "${DOMAIN:-}" ]; then
    if [ -f "$PROJECT_DIR/backend/.env" ]; then
        DOMAIN=$(grep -E "^DOMAIN=" "$PROJECT_DIR/backend/.env" | cut -d'=' -f2 | tr -d ' "')
    fi
fi

if [ -z "${DOMAIN:-}" ]; then
    echo "ERROR: DOMAIN 환경 변수가 설정되지 않았습니다."
    echo "  export DOMAIN=your-domain.com 또는 backend/.env 파일에 DOMAIN=your-domain.com 추가"
    exit 1
fi

EMAIL="${EMAIL:-admin@$DOMAIN}"
DRY_RUN=""

# --dry-run 옵션 처리
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "==> DRY-RUN 모드: 실제 인증서가 발급되지 않습니다."
fi

echo "==> 도메인: $DOMAIN"
echo "==> 이메일: $EMAIL"
echo ""

# ── 1단계: nginx가 HTTP로 실행 중인지 확인 ──
echo "==> 1단계: nginx 컨테이너 확인..."
if ! docker compose -f "$PROJECT_DIR/docker-compose.yml" ps nginx | grep -q "Up"; then
    echo "    nginx가 실행 중이 아닙니다. HTTP 모드로 시작합니다..."
    docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d nginx
    sleep 3
fi

# ── 2단계: certbot으로 인증서 발급 ──
echo "==> 2단계: certbot으로 인증서 발급 요청..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm certbot \
    certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN" \
    $DRY_RUN

if [ -n "$DRY_RUN" ]; then
    echo ""
    echo "==> DRY-RUN 완료. 실제 발급하려면 --dry-run 없이 다시 실행하세요."
    exit 0
fi

# ── 3단계: nginx 설정을 SSL 템플릿으로 교체 ──
echo "==> 3단계: nginx 설정을 SSL 모드로 전환..."
TEMPLATE="$PROJECT_DIR/nginx/conf.d/imputex.conf.template"
OUTPUT="$PROJECT_DIR/nginx/conf.d/imputex.conf"

if [ -f "$TEMPLATE" ]; then
    # envsubst로 DOMAIN 변수만 치환 (nginx 변수 $host 등은 유지)
    export DOMAIN
    envsubst '${DOMAIN}' < "$TEMPLATE" > "$OUTPUT"
    echo "    $OUTPUT 생성 완료 (DOMAIN=$DOMAIN)"
else
    echo "ERROR: 템플릿 파일을 찾을 수 없습니다: $TEMPLATE"
    exit 1
fi

# ── 4단계: nginx 재시작 ──
echo "==> 4단계: nginx 재시작..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" restart nginx

echo ""
echo "==> SSL 인증서 설정 완료!"
echo "    https://$DOMAIN 으로 접속 가능합니다."
echo ""
echo "    인증서 자동 갱신은 certbot 컨테이너가 12시간마다 수행합니다."
echo "    수동 갱신: docker compose run --rm certbot renew"
