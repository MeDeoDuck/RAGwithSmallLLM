#!/usr/bin/env bash
# Re-run B under the corrected exemplar order, then GRPO, then C, then R3.
set -u
set -o pipefail
log() { echo "[rest $(date -u +%H:%M:%S)] $*"; }

log "=== 1. 선택 실험 ==="
bash run_selection_chain.sh
status=$?
log "selection chain exited with ${status}"
[ "$status" -ne 0 ] && { log "실패 — R3 를 돌리지 않는다"; exit "$status"; }

log "=== 2. reader RFT round 3 ==="
rm -f output/results/rollouts_r3.jsonl output/results/prompt_rag_rft_r3_score.json
rm -rf output/reader_rft/r3
bash run_reader_rft.sh
log "round 3 exited with $?"

log "=== 3. 최종 근거 수집 ==="
python collect_comparison_facts.py
log "ALL DONE"
