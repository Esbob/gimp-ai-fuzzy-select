#!/bin/sh
# Installs AI Fuzzy Select on Linux or macOS (see README.md): run  sh install.sh  in this
# folder (on macOS you can also double-click install.command). Installer options can follow,
# e.g.  sh install.sh --model small
cd "$(dirname "$0")" || exit 1

# On Linux without a suitable Python, this offers to download one (asking first): uv, a
# Python manager (its download checked against these SHA-256s), fetches Python 3.14 into
# the install folder, where the uninstaller removes it again. No sudo needed.
UV_VERSION=0.13.5
UV_SHA256_X86_64=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f
UV_SHA256_AARCH64=4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403
PYDIR="${XDG_DATA_HOME:-$HOME/.local/share}/gimp-ai-select/python"

# A Python 3.10-3.14 that can create virtual environments (3.15 isn't supported: PyTorch
# crashes on it): the newest first, then one downloaded before. NOVENV: one that can't.
find_python() {
    PY=
    NOVENV=
    for p in python3.14 python3.13 python3.12 python3.11 python3.10 python3 \
            "$PYDIR"/cpython-3.14*/bin/python3.14; do
        command -v "$p" >/dev/null 2>&1 || continue
        "$p" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null || continue
        if "$p" -c 'import ensurepip, venv' >/dev/null 2>&1; then
            PY=$p
            return
        fi
        [ -n "$NOVENV" ] || NOVENV=$p
    done
}

# Download Python 3.14 with uv (Linux on x86_64 or ARM64 only), asking first
get_python() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) arch=x86_64 sha=$UV_SHA256_X86_64 ;;
        Linux-aarch64) arch=aarch64 sha=$UV_SHA256_AARCH64 ;;
        *) return 1 ;;
    esac
    echo "AI Fuzzy Select can download Python 3.14 for itself: about 50 MB, into"
    echo "$PYDIR, without sudo; the system's Python stays as it is."
    printf "Download Python 3.14 now? [y/N] "
    read -r answer || answer=
    case "$answer" in [yY]*) ;; *) return 1 ;; esac
    tmp=$(mktemp -d) || return 1
    url="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$arch-unknown-linux-gnu.tar.gz"
    echo "Downloading $url"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$tmp/uv.tar.gz" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$tmp/uv.tar.gz" "$url"
    else
        echo "Neither curl nor wget is installed: install one of them first."
        false
    fi || { rm -rf "$tmp"; return 1; }
    if ! echo "$sha  $tmp/uv.tar.gz" | sha256sum -c --status; then
        echo "The download is not the expected file: its SHA-256 does not match."
        rm -rf "$tmp"
        return 1
    fi
    tar -xzf "$tmp/uv.tar.gz" -C "$tmp" || { rm -rf "$tmp"; return 1; }
    uv="$tmp/uv-$arch-unknown-linux-gnu/uv"
    echo "Downloading Python 3.14..."
    # (--no-bin: no python3.14 command in ~/.local/bin; the cache goes with the uv download)
    if UV_PYTHON_INSTALL_DIR="$PYDIR" UV_CACHE_DIR="$tmp/cache" "$uv" python install --no-bin 3.14; then
        PY=$(UV_PYTHON_INSTALL_DIR="$PYDIR" UV_CACHE_DIR="$tmp/cache" "$uv" python find --managed-python 3.14)
    fi
    rm -rf "$tmp"
    [ -n "$PY" ]
}

find_python
if [ -z "$PY" ] && ! get_python; then
    echo
    if [ -n "$NOVENV" ]; then
        echo "$NOVENV can't create virtual environments: install its venv package first,"
        echo "e.g.  sudo apt install $NOVENV-venv  on Debian and Ubuntu."
    else
        echo "AI Fuzzy Select needs Python 3.14 (3.10 to 3.13 also work, 3.15 does not yet)."
        echo "macOS: install it from https://www.python.org/downloads/"
        echo "Linux: install python3.14 (or 3.10 to 3.13) from your distribution."
    fi
    exit 1
fi
echo "Using $PY"
exec "$PY" install_ai_select.py "$@"
