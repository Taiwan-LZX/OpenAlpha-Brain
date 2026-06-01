#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "[setup] Installing dependencies..."
pip install -q -e . 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     OpenAlpha-Brain  Launcher                    ║"
echo "║     WorldQuant BRAIN Alpha Mining System         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  [1] Quick Run   — openalpha run --cycles 2 --no-brain"
echo "  [2] Interactive — openalpha interactive"
echo "  [3] Status      — openalpha status"
echo "  [4] Sessions    — openalpha sessions"
echo "  [0] Exit"
echo ""

read -rp "  Select mode [0-4]: " choice

case "$choice" in
    1) echo ""; echo "  ▶ Quick mining (2 cycles, no BRAIN) ..."; echo "";
       exec openalpha run --cycles 2 --no-brain ;;
    2) echo ""; echo "  ▶ Interactive REPL ..."; echo "";
       exec openalpha interactive ;;
    3) echo ""; echo "  ▶ System status ..."; echo "";
       exec openalpha status ;;
    4) echo ""; echo "  ▶ Recent sessions ..."; echo "";
       exec openalpha sessions ;;
    0|*) echo "  Goodbye!"; exit 0 ;;
esac