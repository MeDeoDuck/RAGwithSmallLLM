#!/usr/bin/env bash
# Round 3 of the reader RFT, queued behind the selection chain.
#
# R3 was skipped earlier by leaving sentinels that satisfy run_reader_rft.sh's
# three "already done" checks. Removing them puts the round back; rounds 1 and 2
# still skip because their real artefacts exist, so only R3 runs.
#
# Waits on the selection chain's own log line rather than on GPU memory: the
# chain frees the card between evaluations, and a memory poll would fire into
# one of those gaps.
set -u
log() { echo "[r3 $(date -u +%H:%M:%S)] $*"; }

CHAIN_LOG=runs/run_selection_chain.log

log "waiting for the selection chain to finish"
while true; do
    if [ -f "$CHAIN_LOG" ] && grep -qa "DONE -- reader_selector_comparison" "$CHAIN_LOG"; then
        log "selection chain done"
        break
    fi
    sleep 60
done
sleep 30

log "removing the round 3 skip sentinels"
rm -f output/results/rollouts_r3.jsonl
rm -f output/results/prompt_rag_rft_r3_score.json
rm -rf output/reader_rft/r3

log "running round 3 (rounds 1 and 2 skip on their existing artefacts)"
bash run_reader_rft.sh
log "round 3 finished with $?"

log "re-collecting comparison facts with round 3 included"
python collect_comparison_facts.py
log "DONE"
