#!/usr/bin/env bash
# Single-passage selection, then multi-passage selection. Run after the reader
# RFT pipeline has released the GPU.
#
# Nothing here writes into output/reader_rft or the rft_* result files. Every
# stage skips itself when its output exists, so the script can be re-entered.
#
#   pip install -q peft trl && bash run_selection_chain.sh
set -u
# `cmd | tee` reports tee's exit status, so a failing stage piped into the log
# returned 0 and the chain walked straight past it -- that is how the GRPO run
# started without its training file. pipefail makes the pipeline carry the
# failure that actually happened.
set -o pipefail
mkdir -p runs output/selector output/multi_select output/results
LOG=runs/run_selection_chain.log
log() { echo "[sel $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

evaluate() {
    local name="$1"
    if [ -s "output/results/prompt_rag_${name}_score.json" ]; then
        log "SKIP eval ${name}"; return 0
    fi
    log "EVAL ${name}"
    PROMPT_RAG_SELECTED="${name}" python run_case.py prompt_rag \
        2>&1 | tee "runs/sel_${name}.log" | tail -16
    if grep -qs "\[run_case\] DONE: prompt_rag$" "runs/sel_${name}.log"; then
        log "OK ${name}"
    else
        log "FAIL ${name} -- runs/sel_${name}.log"; return 1
    fi
}

# ===================================================================== 공통 검증
# the claim that the answer generator stays frozen is checked before anything
# is trained, not asserted afterwards
if [ ! -s output/multi_select/two_call_check.json ]; then
    log "VERIFY two-call adapter isolation"
    python verify_two_call.py 2>&1 | tee -a "$LOG" || exit 1
fi

# ===================================================== 1. 단일 선택은 진단 하나만
# The single-selection training arm was dropped: `sel_abstain` is bit-identical
# to `msel_d`, and conditions B and C would cost about two and a half hours to
# answer a question the multi-selection arm already answers.
#
# `sel_oracle1` stays because it is not a duplicate. It hands the reader the one
# answer-bearing passage; `msel_oracle` hands it all n of them. They agree on the
# 57.8% of questions with n <= 1 and differ on the 42.2% with n >= 2 -- and that
# difference is the measurement the Dice reward's design rests on. The reward
# pays 1.0 for [2,4] and 0.67 for [2]; if one passage reads better than all of
# them, the reward is teaching the reader something harmful and this pair of
# evaluations is what shows it.
evaluate sel_oracle1 || exit 1

# ============================================================= 2. 복수 선택 GRPO
# training inputs are the pipeline's own five passages on the train split, so
# this stage pays for a UPR pass, a first-pass answer and an ALR pass
if [ ! -s output/results/multi_train.jsonl ]; then
    log "PREPARE multi-select train inputs (real pipeline on train split)"
    python prepare_multi_train.py --questions 6000 2>&1 | tee -a "$LOG" || {
        log "FAIL prepare -- 학습 입력이 없으면 GRPO 를 돌릴 수 없다"; exit 1; }
fi

evaluate msel_a || exit 1          # = dig_order, wrapper sanity check
evaluate msel_d || exit 1          # always empty
evaluate msel_e || exit 1          # always passage 5
evaluate msel_oracle || exit 1     # exactly the label set
evaluate msel_b || exit 1          # untrained selector

if [ ! -s output/multi_select/grpo_r1/adapter_model.safetensors ]; then
    # 6,000 questions at the measured 27 s/step was an 11-hour run on this card.
    # The budget is cut rather than the method changed, and the question count
    # is reported with the result.
    log "TRAIN multi-select GRPO (1,500 questions)"
    python train_multi_grpo.py --out output/multi_select/grpo_r1 --limit 1500 \
        2>&1 | tee -a "$LOG" || { log "FAIL GRPO -- 어댑터 없이 msel_c 를 돌리지 않는다"; exit 1; }
    cp -f output/multi_select/grpo_r1/grpo_stats.json \
          output/multi_select/msel_c_grpo_stats.json 2>/dev/null || true
fi
evaluate msel_c || exit 1

# ====================================================================== 보고
log "analysis"
python analyze_multi.py msel_a msel_b msel_c msel_d msel_e msel_oracle \
    2>&1 | tee -a "$LOG"

# everything the three-way comparison report is allowed to cite, with the path
# each number came from and an explicit list of what was never run
log "collect comparison facts"
python collect_comparison_facts.py 2>&1 | tee -a "$LOG"
log "DONE -- reader_selector_comparison.md 작성 대기"
