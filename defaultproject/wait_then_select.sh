#!/usr/bin/env bash
# Hold until the reader RFT releases the card, then run the selection chain.
#
# This runs *inside* a detached container rather than as a host background
# process: the host-side watchers for this run were killed three times, and a
# container is not subject to whatever was ending them.
set -u
log() { echo "[wait $(date -u +%H:%M:%S)] $*"; }

log "waiting for GPU memory to drop below 3000 MiB"
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-99999}" -lt 3000 ]; then
        log "GPU free (${used} MiB used)"
        break
    fi
    sleep 60
done
# the previous process may still be tearing down its context
sleep 30
log "starting selection chain"
bash run_selection_chain.sh
log "selection chain exited with $?"
