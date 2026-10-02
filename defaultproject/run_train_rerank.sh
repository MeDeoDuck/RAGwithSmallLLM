#!/usr/bin/env bash
# Wait for the GPU to free up, then distil DIG into GPT-small and evaluate it.
set -u
log() { echo "[rr $(date -u +%H:%M:%S)] $*" | tee -a runs/run_reranker.log; }
log "START train"
python train_reranker.py --epochs 2 2>&1 | tee runs/train_reranker.log
if [ -s output/reranker/model.safetensors ]; then
    log "OK train"
    log "START eval"
    PROMPT_RAG_SELECTED=trained_rerank python run_case.py prompt_rag 2>&1 | tee runs/rr_eval.log
    grep -qs "\[run_case\] DONE: prompt_rag$" runs/rr_eval.log && log "OK eval" || log "FAIL eval"
else
    log "FAIL train -- no checkpoint written"
fi
log "===== reranker done ====="
