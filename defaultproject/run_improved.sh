#!/usr/bin/env bash
# InfoGain-RAG's idea (rank by answer likelihood, not query likelihood) and GRAD's
# idea (put the best evidence last) against the k_pool problem. EXIT's sentence
# compression was dropped: its cost is dominated by scoring every sentence, and it
# is the one idea whose value could not be separated from its trained classifier.
# dig_proxy pays for the answer-likelihood pass; the three after it reuse the cache.
set -u
mkdir -p runs output/results
log() { echo "[imp $(date -u +%H:%M:%S)] $*" | tee -a runs/run_improved.log; }

run() {
    local name="$1"
    if [ -s "output/results/prompt_rag_${name}_score.json" ]; then
        log "SKIP ${name} (already done)"; return 0
    fi
    log "START ${name}"
    PROMPT_RAG_SELECTED="${name}" python run_case.py prompt_rag 2>&1 | tee "runs/imp_${name}.log"
    if grep -qs "\[run_case\] DONE: prompt_rag$" "runs/imp_${name}.log"; then
        log "OK ${name}"
    else
        log "FAIL ${name} -- see runs/imp_${name}.log"
    fi
}

log "===== improved run start ====="
for name in dig_proxy dig_order dig_only dig_p20; do
    run "${name}"
done
log "===== improved run done ====="
