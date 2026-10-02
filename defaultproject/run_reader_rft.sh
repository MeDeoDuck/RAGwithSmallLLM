#!/usr/bin/env bash
# Reader RFT: controls first, then three rounds of sample -> train -> evaluate.
#
# Order matters. G1 and G2 come before any training because they are what the
# training result is measured against, and because a run that cannot be compared
# is not worth the hours. Every stage skips itself when its output already
# exists, so the script can be re-entered after an interruption.
#
# Run from inside the container:
#   pip install -q peft && bash run_reader_rft.sh
set -u
mkdir -p runs output/results output/reader_rft
LOG=runs/run_reader_rft.log
log() { echo "[rft $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

QUESTIONS=6000
GROUP=8
TEMP=0.9
BATCH=4

evaluate() {                    # evaluate <config>
    local name="$1"
    if [ -s "output/results/prompt_rag_${name}_score.json" ]; then
        log "SKIP eval ${name} (already done)"; return 0
    fi
    log "EVAL ${name}"
    PROMPT_RAG_SELECTED="${name}" python run_case.py prompt_rag \
        2>&1 | tee "runs/rft_${name}.log" | tail -20
    if grep -qs "\[run_case\] DONE: prompt_rag$" "runs/rft_${name}.log"; then
        log "OK eval ${name}"
    else
        log "FAIL eval ${name} -- see runs/rft_${name}.log"; return 1
    fi
}

# ---------------------------------------------------------------- 학습 데이터
if [ ! -s output/results/reader_rft_train.jsonl ]; then
    log "prepare training contexts"
    python prepare_reader_rft.py --questions "$QUESTIONS" 2>&1 | tee -a "$LOG"
else
    log "SKIP prepare (reader_rft_train.jsonl exists)"
fi

# ------------------------------------------------------------------ G1 통제군
# the untrained model under the new output contract: every later number is read
# against this one, not against dig_order, because the contract moves it too
evaluate rft_g1 || exit 1

# ------------------------------------------------------------------ G2 pass@8
if [ ! -s output/results/rft_pass8.json ]; then
    log "G2 pass@8"
    python pass_at_k.py --group "$GROUP" --temperature "$TEMP" \
        --out output/results/rft_pass8.json 2>&1 | tee -a "$LOG"
else
    log "SKIP G2 (rft_pass8.json exists)"
fi

# ------------------------------------------------------------------ C1 통제군
if [ ! -s output/results/rollouts_c1.jsonl ]; then
    python prepare_c1.py 2>&1 | tee -a "$LOG"
fi
if [ ! -d output/reader_rft/c1 ]; then
    log "TRAIN c1 (gold SFT)"
    python train_reader_rft.py --rollouts output/results/rollouts_c1.jsonl \
        --out output/reader_rft/c1 2>&1 | tee -a "$LOG"
else
    log "SKIP train c1"
fi
evaluate rft_c1 || exit 1

# ------------------------------------------------------------------ RFT 회전
prev=""
all_rolls=""
for round in 1 2 3; do
    roll="output/results/rollouts_r${round}.jsonl"
    out="output/reader_rft/r${round}"
    if [ ! -s "$roll" ]; then
        log "SAMPLE round ${round}"
        # round 1 samples from the base policy; later rounds from the previous adapter
        if [ -z "$prev" ]; then
            python sample_rollouts.py --out "$roll" --group "$GROUP" \
                --temperature "$TEMP" --batch "$BATCH" 2>&1 | tee -a "$LOG"
        else
            python sample_rollouts.py --adapter "$prev" --out "$roll" --group "$GROUP" \
                --temperature "$TEMP" --batch "$BATCH" 2>&1 | tee -a "$LOG"
        fi
    else
        log "SKIP sample round ${round}"
    fi
    all_rolls="$all_rolls $roll"
    if [ ! -d "$out" ]; then
        log "TRAIN round ${round} on:${all_rolls}"
        # trained from the base weights on every round collected so far, so a
        # round that samples badly dilutes the data rather than becoming the
        # only thing the next adapter sees
        # shellcheck disable=SC2086
        python train_reader_rft.py --rollouts $all_rolls --out "$out" 2>&1 | tee -a "$LOG"
    else
        log "SKIP train round ${round}"
    fi
    evaluate "rft_r${round}" || exit 1
    prev="$out"
done

# -------------------------------------------------------------------- 보고
log "analysis"
python analyze_rft.py rft_g1 rft_c1 rft_r1 rft_r2 rft_r3 2>&1 | tee -a "$LOG"
log "DONE"
