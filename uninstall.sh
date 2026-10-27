#!/bin/sh
# Removes AI Fuzzy Select on Linux or macOS (see README.md): run  sh uninstall.sh  in this
# folder (on macOS you can also double-click uninstall.command). Uninstaller options can
# follow, e.g.  sh uninstall.sh --keep-checkpoints
cd "$(dirname "$0")" || exit 1
# The system's Python, or the one install.sh downloaded (removed with the install folder)
for PY in python3 python "${XDG_DATA_HOME:-$HOME/.local/share}"/gimp-ai-select/python/cpython-3.14*/bin/python3.14; do
    if command -v "$PY" >/dev/null 2>&1; then
        exec "$PY" uninstall_ai_select.py "$@"
    fi
done
echo "The uninstaller needs Python 3."
exit 1
