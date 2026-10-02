#!/usr/bin/env bash
# Zero-shot GPT-small as the reranking scorer: the control set for a trained one.
# pretraining     -- does a plain 63M LM's query likelihood carry relevance at all?
# rag             -- does task finetuning help, or erode the language modelling it needs?
# pretraining_random -- floor: if this scores as well, the signal is an artefact.
set -u
mkdir -p runs output/results
log() { echo "[gs $(date -u +%H:%M:%S)] $*" | tee -a runs/run_gptsmall.log; }
for name in gptsmall_rerank gptsmall_rag_rerank gptsmall_random_rerank; do
    if [ -s "output/results/prompt_rag_${name}_score.json" ]; then log "SKIP ${name}"; continue; fi
    log "START ${name}"
    PROMPT_RAG_SELECTED="${name}" python run_case.py prompt_rag 2>&1 | tee "runs/gs_${name}.log"
    grep -qs "\[run_case\] DONE: prompt_rag$" "runs/gs_${name}.log" && log "OK ${name}" || log "FAIL ${name}"
done
log "===== gptsmall done ====="
