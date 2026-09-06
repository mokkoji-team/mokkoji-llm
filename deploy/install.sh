#!/usr/bin/env bash
# 봇(상시 실행)과 주간 sync(화 04:00 KST) systemd 유닛을 설치한다.
#   sudo bash deploy/install.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ "$(id -u)" -ne 0 ]; then
  echo "sudo로 실행해야 한다: sudo bash deploy/install.sh" >&2
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "가상환경 파이썬이 없다: $VENV_PYTHON" >&2
  echo "먼저 python3.11 -m venv .venv && pip install -e . 를 끝낼 것." >&2
  exit 1
fi

# OnCalendar의 "화 04:00"을 한국시간으로 해석시키기 위해 시간대를 맞춘다.
timedatectl set-timezone Asia/Seoul

render() {
  sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
      -e "s|__RUN_USER__|$RUN_USER|g" \
      -e "s|__VENV_PYTHON__|$VENV_PYTHON|g" \
      "$1"
}

render "$PROJECT_DIR/deploy/mokkoji-bot.service"  > /etc/systemd/system/mokkoji-bot.service
render "$PROJECT_DIR/deploy/mokkoji-sync.service" > /etc/systemd/system/mokkoji-sync.service
render "$PROJECT_DIR/deploy/mokkoji-sync.timer"   > /etc/systemd/system/mokkoji-sync.timer

systemctl daemon-reload
systemctl enable --now mokkoji-bot.service
systemctl enable --now mokkoji-sync.timer

echo
echo "== 봇 상태 =="
systemctl --no-pager --lines=0 status mokkoji-bot.service || true
echo
echo "== 다음 sync 예정 =="
systemctl list-timers mokkoji-sync.timer --no-pager
