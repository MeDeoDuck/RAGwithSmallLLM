#!/usr/bin/env bash
# Unattended driver for the whole project on a 16 GB card.
# Stage 1 is fatal (everything depends on it); later stages are best-effort so one
# failure does not discard the results already on disk.
set -u
mkdir -p runs output/results

BATCH="${BATCH:-8}"
log() { echo "[run_all $(date -u +%H:%M:%S)] $*" | tee -a runs/run_all.log; }

stage() {          # stage <label> <marker> <cmd...>
    local label="$1" marker="$2"; shift 2
    if grep -qs "\[run_case\] DONE: ${marker}\$" "runs/${label}.log"; then
        log "SKIP ${label} (already done)"; return 0
    fi
    log "START ${label}"
    set -o pipefail
    "$@" 2>&1 | tee "runs/${label}.log"
    local rc=$?
    set +o pipefail
    if grep -qs "\[run_case\] DONE: ${marker}\$" "runs/${label}.log"; then
        log "OK ${label}"; return 0
    fi
    log "FAIL ${label} (rc=${rc}) -- see runs/${label}.log"
    return 1
}

log "===== driver start (batch=${BATCH}) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a runs/run_all.log

# 1) pretraining -- fatal
stage pretrain pretrain python run_case.py pretrain --batch "$BATCH" || {
    log "pretraining failed: stopping, everything else depends on it"; exit 1; }

# 2) downstream finetuning, cheapest first so results land early
stage classification classification python run_case.py classification

# dataset/rag.py converts the 7.4 GB NQ train json with pandas + from_pandas and gets
# OOM-killed under ~24 GB; build it by streaming instead, at the path that cell expects
if [ -f local_cache/rag/data/nq_open_dpr/nq_train.json ] \
   && [ ! -d local_cache/rag/data/nq_open_dpr/nq_train ]; then
    log "START prepare_nq_train"
    python prepare_nq_train.py 2>&1 | tee runs/prepare_nq_train.log
fi
stage rag           rag            python run_case.py rag --batch "$BATCH"
stage summary       summary        python run_case.py summary --batch "$BATCH"

# 3) without-pretraining controls for the report
if [ ! -f output/pretraining_random/best_model/config.json ]; then
    log "START make_random_init"
    python make_random_init.py 2>&1 | tee runs/make_random_init.log
fi
stage classification_scratch classification-scratch python run_case.py classification --from-scratch
stage rag_scratch           rag-scratch            python run_case.py rag --from-scratch --batch "$BATCH"

# 4) naive RAG reference with Llama
stage zeroshot_rag zeroshot_rag python run_case.py zeroshot_rag

# 5) prompting variants and extensions -- smoke first, then the full dev set
export PROMPT_RAG_NUM_SAMPLES=200 PROMPT_RAG_SUFFIX=_smoke200
export PROMPT_RAG_SELECTED=v0_control,v1_zeroshot,v2_fewshot,v3_json,v4_cot,ext_rerank,ext_assembly,ext_full
stage prompt_rag_smoke prompt_rag python run_case.py prompt_rag

unset PROMPT_RAG_NUM_SAMPLES
export PROMPT_RAG_SUFFIX=""
export PROMPT_RAG_SELECTED=v0_control,v1_zeroshot,v2_fewshot,v3_json,v4_cot
stage prompt_rag_prompts prompt_rag python run_case.py prompt_rag

export PROMPT_RAG_SELECTED=ext_rerank,ext_assembly,ext_full
stage prompt_rag_ext prompt_rag python run_case.py prompt_rag

# 6) retrieval diagnostics (CPU) and the conditional breakdown
if [ ! -f output/results/bm25_top100.jsonl ]; then
    log "START diagnostics pool"
    python rag_diagnostics.py pool --k 100 2>&1 | tee runs/diag_pool.log
fi
log "START diagnostics recall"
python rag_diagnostics.py recall 2>&1 | tee runs/diag_recall.log
for pred in output/results/rag_output.txt output/results/llm_rag_output.txt \
            output/results/prompt_rag_v2_fewshot_output.txt; do
    [ -f "$pred" ] && python rag_diagnostics.py breakdown --pred "$pred" 2>&1 \
        | tee "runs/diag_breakdown_$(basename "$pred" .txt).log"
done

# 7) submission archives
stage submission submission python run_case.py submission

log "===== driver done ====="
ls -la output/results/ | tee -a runs/run_all.log
