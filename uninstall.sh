#!/bin/sh
# Removes AI Fuzzy Select on Linux or macOS (see README.md): run  sh uninstall.sh  in this
# folder (on macOS you can also double-click uninstall.command). Uninstaller options can
# follow, e.g.  sh uninstall.sh --keep-checkpoints
cd "$(dirname "$0")" || exit 1
for PY in python3 python; do
    if command -v "$PY" >/dev/null 2>&1; then
        exec "$PY" uninstall_ai_select.py "$@"
    fi
done
echo "The uninstaller needs Python 3."
exit 1
