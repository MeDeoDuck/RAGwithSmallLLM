#!/usr/bin/env bash
# Position study and the two oracle diagnostics, on the full 6,515-question dev set.
# slot_edge runs first: it pays for the UPR pass and caches every passage score, so
# the eight configurations after it only pay for generation.
set -u
mkdir -p runs output/results
log() { echo "[pos $(date -u +%H:%M:%S)] $*" | tee -a runs/run_position.log; }

run() {                 # run <config>
    local name="$1"
    if [ -s "output/results/prompt_rag_${name}_score.json" ]; then
        log "SKIP ${name} (already done)"; return 0
    fi
    log "START ${name}"
    PROMPT_RAG_SELECTED="${name}" python run_case.py prompt_rag 2>&1 | tee "runs/pos_${name}.log"
    if grep -qs "\[run_case\] DONE: prompt_rag$" "runs/pos_${name}.log"; then
        log "OK ${name}"
    else
        log "FAIL ${name} -- see runs/pos_${name}.log"
    fi
}

# only clear the dump when oracle_dig is going to rewrite it: on a resume the run()
# guard would skip that configuration and leave the analysis with no scores at all
[ -s output/results/prompt_rag_oracle_dig_score.json ] || rm -f output/results/oracle_dig_dump.jsonl
log "===== position study start ====="
for name in slot_edge slot_edge_last slot_asc \
            goldpos0 goldpos1 goldpos2 goldpos3 goldpos4 \
            oracle_dig; do
    run "${name}"
done
log "===== position study done ====="
