#!/bin/sh
# macOS: double-click to install AI Fuzzy Select. The first time, macOS may refuse a
# downloaded script: right-click it, choose Open, then Open again.
cd "$(dirname "$0")" && sh install.sh "$@"
echo
echo "Done: you can close this window."
