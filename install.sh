#!/bin/sh
# Installs AI Fuzzy Select on Linux or macOS (see README.md): run  sh install.sh  in this
# folder (on macOS you can also double-click install.command). Installer options can follow,
# e.g.  sh install.sh --model small
cd "$(dirname "$0")" || exit 1
PY=
# Prefer the newest supported Python (3.15 isn't: PyTorch crashes on it)
for v in 3.14 3.13 3.12 3.11 3.10; do
    if command -v "python$v" >/dev/null 2>&1; then
        PY="python$v"
        break
    fi
done
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1 &&
        python3 -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))'; then
    PY=python3
fi
if [ -z "$PY" ]; then
    echo "AI Fuzzy Select needs Python 3.14 (3.10 to 3.13 also work, 3.15 does not yet)."
    echo "macOS: install it from https://www.python.org/downloads/release/python-3149/"
    echo "Linux: install python3.14 (or 3.10 to 3.13) from your distribution."
    exit 1
fi
if ! "$PY" -c 'import ensurepip, venv' >/dev/null 2>&1; then
    echo "$PY can't create virtual environments: install its venv package first,"
    echo "e.g.  sudo apt install $PY-venv  on Debian and Ubuntu."
    exit 1
fi
echo "Using $PY"
exec "$PY" install_ai_select.py "$@"
