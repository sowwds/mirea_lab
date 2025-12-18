#!/usr/bin/env bash
set -e
echo "[pascal] compiling legacy.pas"
fpc -O2 -S2 legacy.pas
echo "[pascal] running legacy CSV generator and importer"
./legacy
