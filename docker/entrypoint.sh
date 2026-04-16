#!/usr/bin/env bash
set -euo pipefail

: "${OMNIVOICE_MODEL:=k2-fsa/OmniVoice}"
: "${OMNIVOICE_VOICES_DIR:=/voices}"
: "${OMNIVOICE_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

export OMNIVOICE_MODEL OMNIVOICE_VOICES_DIR OMNIVOICE_DEVICE HOST PORT LOG_LEVEL

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
