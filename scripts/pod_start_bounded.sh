#!/bin/bash
# EXPERIMENTAL: B200 startup did not reach a verified armed state.
# Do not rely on this script as an established billing control.
# Runpod startup override. OPENJEFF_STOP_AT is a public UTC epoch fixed before
# deployment. Existing pod-scoped credentials are never printed or exported.
set -euo pipefail
umask 077
: "${OPENJEFF_STOP_AT:?absolute stop deadline required}"
: "${RUNPOD_API_KEY:?existing pod-scoped key required}"
: "${RUNPOD_POD_ID:?existing pod ID required}"
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
now=$(date +%s)
remaining=$((OPENJEFF_STOP_AT-now))
if (( remaining < 0 )); then remaining=0; fi
if (( remaining > 2700 )); then remaining=2700; fi
(
 sleep "$remaining"
 while ! runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1; do sleep 15; done
) </dev/null >/tmp/openjeff-watchdog.log 2>&1 &
printf 'deadline_epoch=%s\nwatchdog_pid=%s\n' "$OPENJEFF_STOP_AT" "$!" > /tmp/openjeff-watchdog-status.txt
exec /start.sh
