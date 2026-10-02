# 보고서 뼈대 (ACL 양식, 6~8쪽)

> 핸드아웃 Submission 1번: `defaultproject_report.pdf`, ACL 스타일(https://github.com/acl-org/acl-style-files), 본문 6~8쪽(references·appendix 제외).
> **(Required)** 표시 절은 반드시 들어가야 한다. 공간이 부족하면 핸드아웃 지시대로 **baseline RAG 개선·확장(4.2.1·4.3)에 지면을 몰아준다.**
> 이 문서는 구조와 "어느 파일의 어느 숫자를 어디에 넣는지"만 정한다. **수치는 실행 전이므로 비워 둔다. 추정치를 표에 적지 않는다.**

## 0. 준비물

- ACL 템플릿: `git clone https://github.com/acl-org/acl-style-files` → `latex/acl.sty`, `acl_natbib.bst`, `acl_latex.tex`
- Overleaf를 쓴다면 "ACL 2023 Proceedings Template" 검색해서 그대로 사용

---

## 1. Title / Abstract (Required) — 0.2쪽

- 제목에 "GPT-small from scratch"와 "retrieval-augmented QA on NQ-open" 둘 다 들어가게.
- Abstract에 넣을 것: 63.58M 파라미터 GPT-small을 밑바닥부터 사전학습, 요약·분류·NQ RAG로 파인튜닝, Llama-3.2-1B-Instruct 프롬프트 기반 baseline RAG 구축, 그 위에 **질의 우도 재정렬 + 적응형 컨텍스트 조립** 확장. 최종 accuracy 델타 한 줄.

## 2. Introduction (Required) — 0.5쪽

- LLM의 지식 갱신 문제 → RAG. 핸드아웃 §1을 그대로 옮기지 말고 압축.
- 이 보고서의 기여 3줄: (i) 구조 구현 + 사전학습 재현, (ii) naive→baseline 프롬프팅 ablation, (iii) 재정렬·조립 확장과 **retrieval-level 지표를 병기한 분석**.

## 3. Related Work (Required) — 0.5쪽

인용할 것 (핸드아웃 참고문헌 번호 그대로 써도 됨):
- Transformer [14], RoPE [13], GQA [1], RMSNorm [16]
- RAG [9], DPR [7], NQ [8], Pyserini [11]
- 4.2.1 근거: CoT [15]
- 4.3 근거: **UPR (Sachan et al. 2022, "Improving Passage Retrieval with Zero-Shot Question Generation")** ← 재정렬의 직접 출처, 참고문헌에 추가할 것
- 기각한 것들도 한 문단: PCW [12], context compression [6], Self-RAG [3], RAFT [17] — **왜 안 썼는지**를 쓰는 게 §4.3 서사의 핵심이다(아래 §6 참고).

## 4. Approach (Required) — 1.5~2쪽

### 4.1 GPT-small
`TODO_GUIDE.md` §1의 내용을 수식 중심으로 압축. 그림이나 수식으로 쓸 것:
- RoPE: `[x1', x2'] = [x1·cos − x2·sin, x1·sin + x2·cos]`, 회전 후 내적이 상대 위치에만 의존
- RMSNorm: `x / sqrt(mean(x²)+ε) ⊙ γ`, 통계만 fp32
- GQA: query head 16 / KV head 4, `repeat_kv`
- SwiGLU: `W_down(SiLU(W_gate x) ⊙ W_up x)`
- Pre-Norm 잔차, additive causal+padding 마스크(`finfo.min`을 쓰는 이유: 전부 막힌 행의 NaN 방지)
- 한 칸 shift cross-entropy, `ignore_index=-100`
- **검증 문장 한 줄**: 파라미터 수 63.58M이 핸드아웃 로그와 일치.

### 4.2 Task별 데이터 구성
- 요약: `[pad][bos] "Context: …\n Summary:\n" 요약문 [eos]`, 길이 1024 고정, 요약문에만 loss
- 분류: 기존 패딩 제거 후 배치 최장으로 **left padding** (분류 헤드가 마지막 위치를 읽으므로)
- RAG 학습: `positive_ctxs`에서 최대 5개 샘플링, 추론과 동일한 `Title/Passage/Question/Answer:` 템플릿, context 768토큰 예산

### 4.3 Baseline RAG (핸드아웃 4.2.1) — **여기부터가 본론**
`prompt_rag.py`. 변인 하나씩만 움직인 4개 변형 + 통제군:

| 이름 | 바뀐 것 |
|---|---|
| `v0_control` | naive 프롬프트 그대로, greedy·`max_new_tokens=16`·동일 파싱만 적용 |
| `v1_zeroshot` | + Llama chat template, 간결 답변 지시 |
| `v2_fewshot` | + few-shot 2개 (인물형·연도형) ← **baseline RAG** |
| `v3_json` | + `{"answer": ...}` 구조화 출력 |
| `v4_cot` | + 길이 제약 2줄 CoT |

쓸 논점 (전부 구현에 반영돼 있음):
1. **메트릭이 설계를 결정한다.** `best_subspan_exact_match`는 containment라 길게 쓰면 accuracy엔 유리하고 ROUGE엔 치명적이다. 빈 문자열은 무조건 오답이다. → 최적점은 "짧은 정답 span만". **기권(I don't know)은 항상 손해**이므로 system prompt에 "모르면 최선의 추측"을 명시했다.
2. **파싱 체인**(`rag_parsing.py`): 특수토큰 잔재 제거 → 환각된 다음 턴 절단 → 라벨/마크다운 제거 → 근거절 제거 → 12단어 상한. 전부 실패해도 **빈 문자열을 내지 않는다**(빈 문자열은 확정 오답, 깨진 문자열은 정답을 품을 확률이 0보다 크므로 약지배). 단위 테스트 22개.
3. **통제**: 전 구간 greedy(`do_sample=False`). naive 참고값 0.2528은 샘플링 설정이라 재현되지 않는다는 각주 필수. chat template이 프롬프트에 오늘 날짜를 삽입하므로 `date_string`을 고정했다.
4. **고친 버그 2개**: `apply_chat_template` 결과를 `add_special_tokens=True`로 토크나이즈하면 BOS 중복 → `False`로. 제공 `eval_for_rag`가 GPT-2의 `eos_token_id`(50256)를 Llama의 `pad_token_id`로 넘겨 답 뒤에 일반 텍스트 토큰이 붙는다 → Llama 토크나이저 값으로 덮어씀. **노트북 원본 셀은 건드리지 않고 서브클래스에서 덮어썼다**(naive 수치의 재현성 보존).

### 4.4 확장 (핸드아웃 4.3)
`enhanced_rag.py`.
- **Primary — `UPRRerankRAG`**: BM25 top-20으로 풀을 넓히고, **생성기 자신**이 계산한 `log P(question | passage)`로 재정렬해 top-5만 프롬프트에 넣는다. 새 모델 0개, 학습 0, 생성 호출 1.0×(추가 비용은 디코딩 없는 prefill). 같은 질문 안에서는 채점 토큰이 동일하므로 길이 정규화가 불필요하다는 점을 명시.
- **Secondary — `AdaptiveAssemblyRAG`**: 그 점수를 재활용해 (a) 최고 점수를 질문에 인접 배치, (b) softmax 누적질량 `tau_mass`까지만 포함, (c) `max softmax < tau_low`면 컨텍스트를 버리고 closed-book. (c)는 Self-RAG의 "Retrieve?" 반성을 **critic 학습 없이** 임계값으로 근사한 것임을 정직하게 표기.

## 4-A. 환경 일탈 기록 (Experiments 또는 Appendix에 반드시 명시)

실제 실행 환경이 핸드아웃 명세와 두 군데 다르다. **감추지 말고 사유와 함께 적는다.**

| 항목 | 핸드아웃 | 실제 | 사유 |
|---|---|---|---|
| PyTorch | 2.6.0 (cu124) | **2.7.1 (cu128)** | 실행 GPU가 RTX 5080 = Blackwell **sm_120**. 2.6.0+cu124 휠은 sm_50~sm_90만 빌드돼 있어 `torch.cuda.get_arch_list()`에 sm_120이 없고, bf16 matmul 한 줄에서 `CUDA error: no kernel image is available for execution on the device`로 즉시 실패한다(실측). RTX 50 시리즈는 CUDA 12.8 + PyTorch 2.7 이상이 필요하다. **핸드아웃 버전으로는 이 하드웨어에서 실행 자체가 불가능**하다 |
| micro-batch | 16 (사전학습·요약·RAG) | **8** | VRAM 16GB. 실측 스윕에서 batch 8 = peak 10.95 GiB, **batch 12는 OOM**. `gradient_accumulation_steps`를 2배로 올려 **실효 배치 32를 유지**했고, `global_steps`가 micro-batch 단위로 증가하는 구조이므로 `eval_interval`(2500→5000)과 `logging_interval`(10→20)도 같은 배수로 조정해 **평가가 핸드아웃과 동일한 토큰 위치에서 찍히도록** 맞췄다. 즉 학습 수식·LR 스케줄·실효 배치·평가 지점이 모두 동일하다 |

전자는 재현성에 영향이 있을 수 있으므로(커널·cuBLAS 버전 차이) 한 문장으로 짚어 두고, 후자는 **수학적으로 동등**하다는 점을 위 근거와 함께 적으면 감점 요소가 아니라 오히려 이해도를 보이는 대목이 된다.

`transformers`는 4.52.4로 유지했다(4.54+는 `lm_head`를 임베딩에 자동 tying해 모델이 39.04M이 되고 `save_pretrained`가 깨진다). torch 2.7.1에서도 **파라미터 수 63.58M과 loss shift·causal 마스크 동치를 재검증**했다.

### 환경 구성에서 고친 것 하나 더 (Appendix 소재)
`pyserini 0.44.0`의 `pyserini/encode/_openai.py`가 **import 시점에 `openai.OpenAI()` 클라이언트를 생성**한다. `OPENAI_API_KEY`가 없으면 여기서 예외가 나고, `from pyserini.search.lucene import LuceneSearcher`가 이를 전이적으로 import하므로 **BM25만 쓰는데도 RAG 케이스가 OpenAI 에러로 죽는다.** 컨테이너에 더미 `OPENAI_API_KEY`를 넣어 우회했다(해당 인코더는 호출되지 않는다).

## 5. Experiments (Required) — 0.5쪽

- 데이터·지표는 공통이라 상세 생략 가능(핸드아웃이 허용). NQ-open dev 6,515.
- 환경: Python 3.12 / PyTorch 2.6.0 / transformers 4.52.4 / bf16 / GPU 종류 / greedy decoding / variant별 `max_new_tokens`.
- **subset(N=1,000, seed 42) 결과와 full 결과를 같은 열에 섞지 않는다.** 표 캡션과 열 머리에 둘 다 표기.
- config 간 비교는 같은 문항이므로 paired다 → McNemar 또는 paired bootstrap을 쓴다고 명시.

## 6. Results (Required) — 1.5~2쪽

### 표 1 — GPT-small 사전학습 곡선
출처 `logs/pretraining/eval.jsonl`. step / loss / ppl / token_acc. 핸드아웃 참고값(2500: 4.47/87.5/0.309 → 15000: 3.22/25.25/0.419)과 나란히.
→ **그림 1**: `logs/pretraining/train.jsonl`의 loss 곡선 (핸드아웃이 "training progress curve"를 명시적으로 요구).

### 표 2 — downstream, 사전학습 유무 비교
핸드아웃 Part 1이 **"comparing the results with and without pretraining"**을 요구한다. 즉 **랜덤 초기화에서 바로 파인튜닝한 대조군이 필요하다.**
| task | w/ pretraining | w/o pretraining |
|---|---|---|
| summarization (ROUGE-1/2/L) | `output/results/summary_score.json` | 별도 실행 필요 |
| classification (acc) | `output/results/classification_score.json` | 별도 실행 필요 |
| RAG finetune (acc, ROUGE) | `output/results/rag_score.json` | 별도 실행 필요 |
> 대조군 실행법: `main.ipynb` 해당 셀의 `TransformerForCausalLM.from_pretrained(...)`를 `TransformerForCausalLM(model_config)`로 바꿔 1회. 분류(수 분)와 RAG(20~30분)만 해도 충분하고, 요약(1시간 45분)은 여력이 없으면 생략하고 그 사실을 적는다.

### 표 3 — 4.2.1 프롬프팅 ablation (**메인 표 1**)
출처 `output/results/prompt_rag_*_score.json`.
| config | acc | R-1 | R-2 | R-L | hasanswer@5 | mean pred words | 비고 |
|---|---|---|---|---|---|---|---|
| naive (핸드아웃 참고값 0.2528) | | | | | | | 샘플링 설정 |
| naive (greedy 재측정) | | | | | | | |
| v0_control | | | | | | | 프롬프트는 그대로, 디코딩/파싱만 |
| v1_zeroshot | | | | | | | |
| **v2_fewshot (baseline RAG)** | | | | | | | |
| v3_json | | | | | | | JSON 엄격 파싱 성공률 같이 |
| v4_cot | | | | | | | `Answer:` 미출현률 같이 |
기준선은 0.2528이 아니라 **v0_control**이다. 그래야 v1~v4의 델타를 "프롬프트 효과"라고 말할 수 있다.

### 표 4 — 4.3 확장 (**메인 표 2**, 2×2 factorial)
| config | acc | R-1 | hasanswer@5 | avg #passages | gate 발동률 |
|---|---|---|---|---|---|
| v2_fewshot (baseline) | | | | 5 | – |
| + rerank (`ext_rerank`) | | | | 5 | – |
| + assembly만 (`ext_assembly`) | | | | | |
| + 둘 다 (`ext_full`) | | | | | |

### 표 5 — retrieval-only (GPU 0)
출처 `python rag_diagnostics.py recall`. k=1/5/10/20/50/100의 gold recall@k, hasanswer@k, 정답 보유 passage의 rank 분포, 재정렬 후 hasanswer@5.
`hasanswer@k_pool`이 재정렬의 oracle 상한이므로 **"oracle 대비 몇 % 메웠나"**를 한 줄 보고.

## 7. Analysis and Discussion — 1쪽

핵심은 **모든 결과를 해석 가능하게 만드는 분해**다:
- `rag_diagnostics.py breakdown` 출력으로 `acc | hasanswer@5=True` vs `False`를 나눠 제시. 전자가 높으면 병목은 retrieval, 낮으면 프롬프트/추출.
- accuracy가 안 올랐다면 **hasanswer@5가 올랐는지**로 "재정렬이 나빴다"와 "1B 생성기가 못 썼다"를 분리한다. 후자면 그 자체가 결론이다.
- containment 메트릭 함정: 컨텍스트가 깨끗해질수록 답이 짧아지고, 짧아지면 containment accuracy가 손해를 볼 수 있다. `mean pred words` 열로 이걸 짚는다.
- **왜 PCW를 안 썼는가**: Llama-3.2-1B는 128k 컨텍스트이고 top-5 × 100단어는 약 700토큰이다. PCW가 푸는 문제(입력 길이 제한)가 이 세팅엔 존재하지 않는다. 같은 이유로 context compression도 기각(압축 단위가 이미 100단어). 핸드아웃이 같은 문장에서 제시한 "질문과의 관련도로 세그먼트 선택 후 재구성"이 압축을 지배한다. RAFT·Self-RAG는 학습 예산 밖.

## 8. Conclusion (Required) — 0.3쪽
## 9. References (Required)
## 10. Appendix (optional)
프롬프트 전문, 하이퍼파라미터 표, 파싱 규칙 전체, 실패 사례 몇 개.

---

## 실행 체크리스트 (보고서를 쓰려면 이 숫자들이 필요하다)

- [ ] ① pretrain → `logs/pretraining/*.jsonl`, `output/pretraining/best_model`
- [ ] ② summary ③ classification ④ rag → `output/results/*_score.json`
- [ ] **대조군**: 사전학습 없이 ③ classification, ④ rag 각 1회 (표 2)
- [ ] ⑤ zeroshot_rag (naive 참고값)
- [ ] `rag_diagnostics.py pool --k 100` → `recall` (표 5, GPU 불필요, **가장 먼저 해도 된다**)
- [ ] ⑥ prompt_rag: `PROMPT_RAG_NUM_SAMPLES=1000`으로 v0~v4 탐색 → 최종 5개 config를 `None`으로 full 실행
- [ ] `rag_diagnostics.py breakdown --pred output/results/prompt_rag_v2_fewshot_output.txt` (§7)
- [ ] ⑦ submission
