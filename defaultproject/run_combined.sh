#!/usr/bin/env bash
set -u
set -o pipefail
LOG=runs/run_combined.log
log() { echo "[comb $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
evaluate() {
    local name="$1"
    if [ -s "output/results/prompt_rag_${name}_score.json" ]; then
        log "SKIP ${name}"; return 0; fi
    log "EVAL ${name}"
    PROMPT_RAG_SELECTED="${name}" python run_case.py prompt_rag 2>&1 \
        | tee "runs/comb_${name}.log" | tail -12
    grep -qs "\[run_case\] DONE: prompt_rag$" "runs/comb_${name}.log" \
        && log "OK ${name}" || { log "FAIL ${name}"; return 1; }
}
# comb_all is the control: same trained reader, no selection. It should land on
# rft_r3, which checks that the wrapper did not change the reading path.
evaluate comb_all || exit 1
evaluate comb_c || exit 1
log "analysis"
python analyze_multi.py msel_a msel_b msel_c comb_all comb_c 2>&1 | tee -a "$LOG" | tail -20
python collect_comparison_facts.py 2>&1 | tail -6
log "DONE"
