#!/usr/bin/env bash
# EXIT's idea without its trained classifier: sentence-level selection inside the
# token budget of the top-5 context it replaces. sent20 pays for scoring every
# sentence of the reranked top-20; the three after it reuse that cache. sent50 runs
# last because it has to score the 30 deeper passages as well.
set -u
mkdir -p runs output/results
log() { echo "[exit $(date -u +%H:%M:%S)] $*" | tee -a runs/run_exit.log; }

run() {
    local name="$1"
    if [ -s "output/results/prompt_rag_${name}_score.json" ]; then
        log "SKIP ${name} (already done)"; return 0
    fi
    log "START ${name}"
    PROMPT_RAG_SELECTED="${name}" python run_case.py prompt_rag 2>&1 | tee "runs/exit_${name}.log"
    if grep -qs "\[run_case\] DONE: prompt_rag$" "runs/exit_${name}.log"; then
        log "OK ${name}"
    else
        log "FAIL ${name} -- see runs/exit_${name}.log"
    fi
}

log "===== EXIT run start ====="
for name in sent20 sent20_last sent20_dig all_three sent50; do
    run "${name}"
done
log "===== EXIT run done ====="
