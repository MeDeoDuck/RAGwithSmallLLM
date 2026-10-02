#!/usr/bin/env bash
set -u
log() { echo "[extra $(date -u +%H:%M:%S)] $*" | tee -a runs/run_all.log; }
log "START ext_rerank_v3_p100"
PROMPT_RAG_SELECTED=ext_rerank_v3_p100 python run_case.py prompt_rag 2>&1 | tee runs/prompt_rag_p100.log
grep -q "DONE: prompt_rag" runs/prompt_rag_p100.log && log "OK ext_rerank_v3_p100" || log "FAIL ext_rerank_v3_p100"
log "START summary_scratch"
python run_case.py summary --from-scratch --batch 8 2>&1 | tee runs/summary_scratch.log
grep -q "DONE: summary-scratch" runs/summary_scratch.log && log "OK summary_scratch" || log "FAIL summary_scratch"
log "===== extra done ====="
