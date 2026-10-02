# 변수 설명서

> 처음이면 [00_START_HERE.md](00_START_HERE.md)에서 읽는 순서와 폴더 경로를 먼저 확인한다.
> 각 파일의 설정값, 클래스 속성, 함수 인자, 그리고 **빈칸(TODO)에 새로 만든 지역 변수**를 설명한다.
> 표기: `B` = batch size, `S` = 이번에 넣은 토큰 수(query 길이), `T` = attention이 보는 전체 key 길이(= cache 길이 + S),
> `H` = attention head 수, `H_kv` = key/value head 수, `D` = head_dim, `V` = vocab 크기.
> ★ 표시 = 빈칸을 채우며 새로 만든 변수. TODO별 해설은 [TODO_GUIDE.md](TODO_GUIDE.md) 참고.

---

## 1. `model.py`

### 1-1. `TransformerConfig` (모델 하이퍼파라미터)
노트북 셀 7의 "About 70M parameters" 설정값을 함께 적었다(실제 파라미터 63.58M).

| 인자 | 기본값 | 노트북 값 | 의미 |
|---|---|---|---|
| `vocab_size` | 50000 | `len(tokenizer)` = **50258** | 어휘 크기 (GPT-2 50257 + `<\|padding\|>` 1개) |
| `hidden_size` | 512 | 512 | 토큰 임베딩/은닉 벡터 차원 (d_model) |
| `intermediate_size` | 2048 | 2048 | FFN 중간 차원 (보통 hidden의 4배) |
| `num_hidden_layers` | 6 | **4** | Transformer 층 수 |
| `num_attention_heads` | 16 | 16 | query head 수 `H` |
| `num_key_value_heads` | 4 | 4 | key/value head 수 `H_kv` (GQA). `H`의 약수여야 함 |
| `head_dim` | None → `hidden/heads` | 32 | head 하나의 차원 `D`. `H × D = hidden_size`여야 함 (assert) |
| `max_postion_embeddings` | 2048 | 1024 | 최대 길이 정보 (원본 인자 이름의 오타 `postion`). `self.max_position_embeddings`로 저장만 되고 RoPE 계산에는 쓰이지 않음 |
| `initializer_range` | 0.02 | 0.02 | `nn.Linear`/`nn.Embedding` 가중치 초기화 표준편차 |
| `rms_norm_eps` | 1e-6 | 1e-6 | RMSNorm 분모의 ε (0으로 나누기 방지) |
| `pad_token_id` | 0 | 50257 | 패딩 토큰 id (`<\|padding\|>`), 임베딩의 `padding_idx` |
| `bos_token_id` / `eos_token_id` | 1 / 2 | 50256 / 50256 | 문장 시작/끝 토큰 id (GPT-2는 둘 다 `<\|endoftext\|>`) |
| `rope_theta` | 10000.0 | `ROPE_THETA` = 20000.0 | RoPE 주파수의 base. 클수록 긴 문맥에서 각도가 천천히 돈다 |
| `attention_dropout` | 0.1 | 0.1 | attention 확률에 거는 dropout 비율 |
| `ffn_dropout` | 0.0 | 0.05 | FFN 출력에 거는 dropout 비율 |

### 1-2. `apply_rotary_emb(x, position_embeddings)` — TODO 1
| 변수 | shape | 의미 |
|---|---|---|
| `x` | `(B, H, S, D)` 또는 `(B, H_kv, S, D)` | 회전시킬 query 또는 key |
| `position_embeddings` | `(cos, sin)` 튜플 | `RotaryEmbedding.forward`가 만든 값 |
| `cos`, `sin` | 입력 `(B, S, D/2)` → ★ `unsqueeze(1)` 후 `(B, 1, S, D/2)` | 위치 m, 주파수 i별 `cos(m·θ_i)`, `sin(m·θ_i)`. head 축으로 broadcast |
| ★ `x1`, `x2` | `(B, H, S, D/2)` 각각 | `x`의 앞 절반/뒤 절반. `(x1[i], x2[i])`가 함께 회전하는 2D 쌍 |
| 반환 `x` | `(B, H, S, D)` | 회전된 벡터 `[x1·cos − x2·sin, x1·sin + x2·cos]` |

### 1-3. `repeat_kv(hidden_states, n_rep)` (원본)
| 변수 | 의미 |
|---|---|
| `hidden_states` | `(B, H_kv, T, D)` key 또는 value |
| `n_rep` | 복제 횟수 = `H / H_kv` (= 4). GQA에서 K/V head 하나를 query head 4개가 공유 |
| 반환 | `(B, H, T, D)` |

### 1-4. `RMSNorm` — TODO 2
| 변수 | 의미 |
|---|---|
| `self.eps` | 분모 안정화 상수 (`config.rms_norm_eps`) |
| `self.weight` | `(hidden_size,)` 학습 가능한 스케일 γ, 1로 초기화 |
| ★ `input_dtype` | 입력 dtype(bf16 등). 계산 후 이 dtype으로 되돌림 |
| ★ `variance` | `(…, 1)` 마지막 차원의 제곱 평균 `mean(x²)` (fp32) |
| ★ `output` | `weight × x / sqrt(variance + eps)` |

### 1-5. `RotaryEmbedding` — TODO 3, 4
| 변수 | shape | 의미 |
|---|---|---|
| `self.config` | | 모델 설정 |
| ★ `base` | 스칼라 | `config.rope_theta` |
| ★ `dim` | 스칼라 | `config.head_dim` (= 32) |
| ★ `inv_freq` → 버퍼 `self.inv_freq` | `(D/2,)` | 주파수 `θ_i = base^(−2i/D)`. `persistent=False`라 체크포인트에 저장되지 않고 생성 시 다시 계산 |
| `position_ids` | `(B 또는 1, S)` | 각 토큰의 위치 번호 m |
| `inv_freq_expanded` | `(B, D/2, 1)` | 원본: 배치만큼 복제한 `θ` |
| `position_ids_expanded` | `(B, 1, S)` | 원본: 행렬곱용으로 차원을 맞춘 위치 |
| ★ `freqs` | `(B, S, D/2)` | 각도 `m·θ_i` |
| ★ `cos`, `sin` | `(B, S, D/2)` | 반환값. 입력 `x`의 dtype으로 변환되어 나감 |

### 1-6. `MultiHeadAttention` — TODO 5, 6
| 변수 | shape / 값 | 의미 |
|---|---|---|
| `self.head_dim` | 32 | `hidden_size / num_attention_heads` |
| `self.num_key_value_groups` | 4 | query head 몇 개가 K/V head 하나를 공유하는지 (`H / H_kv`) |
| `self.scale` | `32^-0.5` | `1/√D`. 내적 값이 커져 softmax가 포화되는 것을 막음 |
| `self.dropout` | 0.1 | attention dropout 확률 (float) |
| ★ `self.q_proj` | Linear 512→512 | query projection (`H·D`) |
| ★ `self.k_proj` | Linear 512→128 | key projection (`H_kv·D`, GQA라 작음) |
| ★ `self.v_proj` | Linear 512→128 | value projection |
| ★ `self.o_proj` | Linear 512→512 | head들을 합친 뒤의 출력 projection |
| `hidden_states` | `(B, S, 512)` | 층 입력 (RMSNorm 통과 후) |
| `hidden_shape` | `(B, S, -1, D)` | head 단위로 나누기 위한 view shape |
| `query_states` | `(B, H, S, D)` | RoPE 적용된 query |
| `key_states`, `value_states` | `(B, H_kv, S, D)` → cache 연결 `(B, H_kv, T, D)` → `repeat_kv` 후 `(B, H, T, D)` | key/value |
| `past_key_value` | `(key_cache, value_cache)` 또는 None | 생성 시 이전 토큰들의 K/V. 갱신된 값을 반환 |
| `attention_mask` | `(B, 1, S, T)` | TODO 9가 만든 additive 마스크 (0 / 최솟값) |
| ★ `attention_weights` | `(B, H, S, T)` | `softmax(QKᵀ·scale + mask)`, dropout 적용 |
| ★ `attn_output` | `(B, H, S, D)` → `(B, S, 512)` | 가중합 `weights·V` → head 합치기 → `o_proj` |

### 1-7. `FeedForwardNetwork` — TODO 7
| 변수 | 의미 |
|---|---|
| `self.gate_proj` | Linear 512→2048, SiLU를 통과할 "게이트" |
| `self.up_proj` | Linear 512→2048, 게이트와 곱해질 값 |
| `self.down_proj` | Linear 2048→512, 원래 차원으로 복귀 |
| `self.act_fn` | `SiLU(x) = x·sigmoid(x)` |
| `self.dropout` | `nn.Dropout(ffn_dropout)` |
| ★ `outputs` | `(B, S, 512)` = `dropout(down(SiLU(gate(x)) ⊙ up(x)))` |

### 1-8. `TransformerLayer` — TODO 8
| 변수 | 의미 |
|---|---|
| `self.self_attn` | `MultiHeadAttention` |
| `self.feed_forward` | `FeedForwardNetwork` |
| `self.input_layernorm` | attention **앞**의 RMSNorm (Pre-Norm) |
| `self.post_attention_layernorm` | FFN **앞**의 RMSNorm |
| `self.layer_idx` | 층 번호 (0~3) |
| ★ `residual` | 잔차 연결용으로 저장한 블록 입력 |
| `hidden_states` | `(B, S, 512)` 층 입출력 |
| `position_embeddings` | `(cos, sin)` — 모든 층이 공유 |

### 1-9. `TransformerPreTrainedModel` / `TransformerModel` — TODO 9
| 변수 | 의미 |
|---|---|
| `config_class`, `base_model_prefix="model"` | HF `from_pretrained`/`save_pretrained` 연동용 |
| `_init_weights` | Linear·Embedding을 `N(0, 0.02)`로 초기화, 패딩 임베딩은 0 |
| `self.padding_idx` | `pad_token_id` |
| `self.embed_tokens` | `nn.Embedding(V, 512)` 토큰 임베딩 |
| `self.layers` | `TransformerLayer` × `num_hidden_layers` |
| `self.norm` | 마지막 RMSNorm |
| `self.rotary_emb` | `RotaryEmbedding` (cos/sin을 한 번 계산해 모든 층이 공유) |
| `self.gradient_checkpointing` | True면 메모리 절약을 위해 층 활성값을 재계산 (기본 False) |
| `inputs_embeds` | `(B, S, 512)` 임베딩 결과 |
| `position_ids` | 없으면 `arange(0, S)`. 생성할 때는 `attention_mask.cumsum − 1`(left padding 보정) |
| `seen_token_length` | KV cache에 이미 들어 있는 토큰 수 |
| `target_length` | `S + seen_token_length` = 마스크의 key 길이 `T` |
| `position_embed` | `(cos, sin)` |
| `kv_cache_new` | 층별 갱신된 `(key, value)` 리스트 |
| ★ `min_dtype` | `torch.finfo(dtype).min` — "차단" 위치에 더할 아주 작은 값 (−inf 대신 사용해 NaN 방지) |
| ★ `query_positions` | `(S,)` query의 절대 위치 = `arange(S) + seen_token_length` |
| ★ `key_positions` | `(T,)` key의 위치 `arange(T)` |
| ★ `allowed` | `(B, 1, S, T)` bool. causal(`key ≤ query`) AND padding(`attention_mask == 1`) |
| ★ `mask` | `(B, 1, S, T)` 허용 0, 차단 `min_dtype`인 additive 마스크 (반환값) |

### 1-10. `TransformerForCausalLM` — TODO 10
| 변수 | 의미 |
|---|---|
| `self.model` | `TransformerModel` 본체 |
| `self.lm_head` | Linear 512→V, 은닉 벡터를 어휘 logits로 (임베딩과 가중치 공유 안 함) |
| `labels` | `(B, S)` 정답 토큰 (shift 전). `-100`은 loss 제외 |
| `use_cache` | True면 빈 KV cache(`dummy`, 길이 0)를 만들어 생성용 cache를 쌓기 시작 |
| `logits` | `(B, S, V)` (loss 계산 시 fp32로 변환) |
| ★ `shift_logits` | `(B, S−1, V)` 위치 0 ~ S−2의 예측 |
| ★ `shift_labels` | `(B, S−1)` 위치 1 ~ S−1의 정답 |
| ★ `loss` | 스칼라, `-100`을 제외한 평균 cross-entropy |
| 반환 | labels가 있으면 `(loss, logits)`, 없으면 `(logits, past_key_values)` |

`generate()` (원본, greedy 디코딩):
| 변수 | 의미 |
|---|---|
| `max_new_tokens` | 최대 생성 토큰 수 |
| `return_response_only` | True면 프롬프트를 뺀 생성 부분만 반환 |
| `unfinish_flag` | `(B,)` 아직 EOS를 내지 않은 샘플이면 1 |
| `position_ids` | `cumsum(attention_mask) − 1`, 패딩 위치는 −1 |
| `next_tokens` | `argmax(logits[:, -1])` (끝난 샘플은 EOS로 채움) |

### 1-11. `TransformerForSequenceClassification` (원본)
| 변수 | 의미 |
|---|---|
| `self.num_labels` | 클래스 수 (20 Newsgroups → 20) |
| `self.classifier` | Linear 512→num_labels. **마지막 위치** 은닉 벡터 `hidden_states[:, -1]`로 분류 → 입력은 left padding이어야 함 |

---

## 2. `model_rag.py` (`ModelRAG`) — TODO 11, 12, 13

| 변수 | 의미 |
|---|---|
| `self.model` | 생성 모델: GPT-small(`TransformerForCausalLM`) 또는 HF `Llama-3.2-1B-Instruct` |
| `self.retriever` | pyserini `LuceneSearcher` (BM25, 인덱스 `wikipedia-dpr-100w`, 약 2100만 passage) |
| `self.tokenizer` | `self.model`에 맞는 토크나이저 |
| `queries` | 질문 문자열 리스트 (배치) |
| `qids` | 질문 id 리스트 (`eval_0000` 등), `batch_search`의 결과 key |
| `k` | 질문당 검색할 passage 수 (평가 기본 5) |
| ★ `batch_hits` | `{qid: [hit, …]}` BM25 결과. `hit.docid`, `hit.score`, `hit.lucene_document` |
| ★ `doc` | `hit2docdict(hit)` = `{"id": …, "contents": '"제목"\n본문'}` |
| ★ `title`, `text` | `contents`의 첫 줄(따옴표 제거) / 나머지 |
| ★ `passages`, `scores` | 한 질문의 `[{id, title, text}, …]`, `[BM25 점수, …]` |
| `list_passages`, `list_scores` | 배치 전체 (`search`의 반환값) |
| ★ `input_text_ctx` | `"Title: …\nPassage: …"` × k를 줄바꿈으로 이은 문자열 |
| ★ `ctx_ids` | `input_text_ctx`의 토큰 id. 768개를 넘으면 잘라서 다시 디코딩 |
| `list_input_text_without_answer` | 최종 프롬프트 `…\nQuestion: q\nAnswer:` 리스트 |
| ★ `input_texts` | 위 프롬프트 리스트 (`retrieval_augmented_generate` 안) |
| ★ `inputs` | `{"input_ids": (B, L), "attention_mask": (B, L)}` left-padding 텐서 |
| `kwargs` | `generate`에 그대로 넘길 인자 (`max_new_tokens=10`, HF 모델이면 `pad_token_id`) |
| `outputs` | `(B, ≤10)` 프롬프트를 제외한 생성 토큰 |

---

## 3. `dataset/`

### 3-1. `pretrain.py` (원본, 참고)
| 변수 | 의미 |
|---|---|
| `dataset_name_or_path`, `dataset_subset` | `chengjunyan1/smollm-12.5-corpus`, `cosmopedia-v2` |
| `max_length` / `block_length` | 1024. 문서들을 `bos+텍스트+eos`로 이어 붙인 뒤 1024토큰 블록으로 자름(packing) |
| `train_sample_size`, `eval_sample_size` | 사용할 문서 수 (노트북: 1,048,576 / 8,096) |
| `cache_path` | 전처리 결과 저장 위치 (`DRIVE_CACHE_PATH/pretrain`) |
| `collate_fn_for_pretrain` | 1024로 패딩, `labels = input_ids` (패딩은 −100) |

### 3-2. `summary.py` — TODO 16
| 변수 | 값 / shape | 의미 |
|---|---|---|
| `dataset_name_or_path`, `dataset_subset` | `abisee/cnn_dailymail`, `3.0.0` | 뉴스 기사 → 하이라이트 요약 |
| `context_column_name`, `target_column_name` | `article`, `highlights` | 입력/정답 컬럼 |
| `context_max_length` | 896 | 프롬프트(기사) 최대 토큰 |
| `target_max_length` | 128 | 요약 최대 토큰 (eos 포함) |
| `prompt` | `"Context: {context}\n Summary:\n"` | 입력 템플릿 |
| `contexts`, `targets` | 문자열 리스트 | 토큰 기준으로 먼저 잘라낸 기사/요약 (원본 코드) |
| ★ `max_length` | 1024 | `context_max_length + target_max_length`, 모든 샘플의 고정 길이 |
| ★ `prompt_ids` | 리스트 | `[bos] + tokenize(prompt.format(context))`, 최대 896 |
| ★ `target_ids` | 리스트 | `tokenize(target)[:127] + [eos]` |
| ★ `input_ids` / `labels` / `attention_mask` | 길이 1024 리스트 | 입력 / 프롬프트·패딩은 −100인 정답 / 실제 토큰 1 |
| ★ `pad_length` | int | 1024에서 모자란 길이 (`padding_side` 방향으로 채움) |
| ★ `inputs` | dict of lists | `map(batched=True)`에 돌려줄 결과 |
| `pad_to_multiple_of` (collate) | 1024 | 배치 길이를 1024 배수로 맞춤 |

### 3-3. `classification.py` — TODO 14
| 변수 | 의미 |
|---|---|
| `dataset_name_or_path` | `SetFit/20_newsgroups` (20개 주제 분류) |
| `text_column_name`, `label_column_name` | `text`, `label` |
| `_preprocessing` (원본) | map-batch 단위로 `padding="longest"`, `max_length=512` 토크나이즈 |
| `batch` | DataLoader가 넘긴 샘플 dict 리스트 |
| ★ `sequences` | 패딩을 걷어낸 실제 토큰 리스트들 |
| ★ `max_length` | 이 배치의 최장 길이 (최소 1) |
| ★ `input_ids`, `attention_mask` | `(B, max_length)` left padding 텐서 |
| ★ `inputs` | `{"input_ids", "attention_mask", "labels": (B,)}` |

### 3-4. `rag.py` — TODO 15
| 변수 | 의미 |
|---|---|
| `URL_*`, `DIR_*`, `FILE_NAME_*` | DPR의 NQ train/dev json 다운로드 주소와 저장 경로 (`local_cache/rag/data/nq_open_dpr/…`) |
| `RAGDataset.prefix_title` / `prefix_passage` | `'Title: '` / `'Passage: '` 프롬프트 접두어 |
| `is_train` | True면 학습 텍스트 생성, False면 질문/정답만 |
| `num_samples` | 앞에서부터 일부만 쓸 때 개수 |
| 샘플 필드 (NQ-DPR) | `question`, `answers`(정답 여러 표현), `positive_ctxs`(정답이 든 passage: `title/text/score/title_score/passage_id`), `negative_ctxs`, `hard_negative_ctxs`(DPR 학습용, 여기선 미사용) |
| `uid` | 샘플 id. 없으면 `train_0000`/`eval_0000` 형식으로 생성 |
| `answer` | `answers[0]` (학습 정답) |
| ★ `positive_ctxs` | 이 질문의 gold passage 리스트 |
| ★ `ctxs` | 그중 최대 5개를 랜덤 샘플링 (평가 top-5와 맞춤) |
| ★ `ctx_ids` | context 토큰 id (768 초과 시 절단용) |
| ★ `input_text_ctx` | passage 부분 문자열 |
| ★ `input_text_without_answer` | `…\nQuestion: q\nAnswer:` |
| `input_text` | 학습 전체 문장 `… Answer: 정답<eos>` (원본 코드) |
| `RAGCollator.max_len` | 992. 학습 시 (context + 정답) 총 길이 |
| `inputs_only_answer` | `' '+정답+eos`를 오른쪽 패딩(최대 64) |
| `inputs_without_answer` | context를 `992 − 정답길이`로 왼쪽 패딩 + **오른쪽 절단** |
| `labels` | context·패딩은 −100, 정답 토큰만 학습 |

---

## 4. `main.ipynb`

### 4-1. 전역 설정 (셀 0~3)
| 변수 | 값 | 의미 |
|---|---|---|
| `PROJ_PATH` | Colab: Drive 경로 / 로컬·Docker: `"."` | 프로젝트 루트. import 경로이자 cache/logs/output의 기준 |
| `os.environ["JAVA_HOME"]` | `/usr/lib/jvm/java-21-openjdk-amd64` | pyserini(Lucene)가 쓸 Java 21 경로 (Docker 이미지와 일치) |
| `MODEL_CONTEXT_LENGTH` | 1024 | 모델 최대 문맥 길이, 사전학습 블록 길이 |
| `ROPE_THETA` | 20000.0 | RoPE base (주석: 1024 문맥 20k, 2048 문맥 50k 권장) |
| `DRIVE_CACHE_PATH` | `PROJ_PATH/cache` | 전처리된 사전학습 데이터 저장 |
| `LOCAL_CACHE_PATH` | `local_cache` | NQ 데이터·BM25 인덱스(약 8.5GB) 저장 |
| `LOG_PATH` | `PROJ_PATH/logs` | `train.jsonl`/`eval.jsonl` 로그 (학습 곡선용) |
| `OUTPUT_PATH` | `PROJ_PATH/output` | 체크포인트, `best_model` |
| `RESULTS_PATH` | `output/results` | 최종 점수 json, 예측 txt |
| `SEED` | 42 | random/numpy/torch 시드 |
| `DO_PRETRAIN` … `DO_SUBMISSION` | bool | 실행할 케이스 스위치 ([RUN_GUIDE.md](RUN_GUIDE.md)) |
| `rouge` | `evaluate.load("rouge")` | ROUGE 지표 |
| `tokenizer` | GPT-2 + `<\|padding\|>`, `padding_side="left"`, `model_max_length=1024` | GPT-small용 토크나이저 |
| `additional_special_tokens` | dict | 없는 special token(pad/eos/bos)을 추가하기 위한 임시 dict |

### 4-2. `TrainingConfig` 필드 (셀 4)
| 필드 | 의미 |
|---|---|
| `device` | `"cuda"` |
| `model_dtype` | `"cast_bf16"`(모델 전체를 bf16으로, Ampere 이상 GPU) 또는 `"amp_fp16"`(fp16 autocast) |
| `batch_size`, `eval_batch_size` | 학습/평가 배치 크기 |
| `gradient_accumulation_steps` | 몇 스텝마다 optimizer를 갱신할지 (실효 배치 = batch × 이 값) |
| `num_train_epochs`, `max_steps` | 에폭 수 / 최대 스텝 (지정하면 우선) |
| `optimizer_type` | 기록용 (실제로는 항상 AdamW) |
| `learning_rate`, `weight_decay` | AdamW 학습률 / 가중치 감쇠 |
| `warmup_steps` | 학습률을 0에서 올리는 스텝 수 |
| `max_grad_norm` | gradient clipping 임계값 |
| `lr_scheduler_type` | `"linear"` / `"cosine"` |
| `metric_for_best_model`, `metric_greater_is_better` | best 체크포인트 선정 기준 (예: `ppl` 낮을수록, `accuracy` 높을수록) |
| `eval_interval`, `logging_interval` | 평가/로그 주기(스텝) |
| `save_total_limit` | 보관할 상위 체크포인트 수 |
| `logging_path`, `output_path` | 케이스별 로그/출력 폴더 |

### 4-3. 케이스별 학습 설정 (셀 7~12의 값)
| 케이스 | 모델 | batch | accum | epoch | lr | scheduler | best 기준 | 결과 파일 |
|---|---|---|---|---|---|---|---|---|
| 사전학습 | `TransformerForCausalLM`(새로 생성) | 16 | 2 | 2 | 3e-4 | cosine | `ppl` ↓ | `output/pretraining/best_model` |
| 요약 | 사전학습 best에서 시작 | 16 | 2 | 3 | 5e-5 | cosine | `loss` ↓ | `results/summary_score.json` |
| 분류 | 사전학습 best + `classifier`(20) | 32 | 1 | 5 | 5e-5 | cosine | `loss` ↓ | `results/classification_score.json` |
| RAG 파인튜닝 | 사전학습 best | 16 | 1 | 3 | 5e-5 | cosine | `accuracy` ↑ | `results/rag_score.json` |
| Zero-shot RAG | `Llama-3.2-1B-Instruct`(bf16) | 4 | – | – | – | – | – | `results/llm_rag_score.json` |

### 4-4. `train()` / 평가 함수 내부 변수 (셀 5~6)
| 변수 | 의미 |
|---|---|
| `train_model` | 실제로 학습하는 `nn.Module` (RAG면 `model.model`) |
| `num_training_steps` | `len(train_loader) × epochs` (또는 `max_steps`) |
| `global_steps` | 지금까지의 배치 스텝 수 |
| `loss_window` | 최근 100 스텝 loss. 진행바 `loss=현재::평균` 표시에 사용 |
| `best_models` | `(체크포인트명, 지표)` 상위 N개 |
| `eval_loop` | 케이스별 평가 함수 (`eval_loop_for_pretraining`, `eval_loop_for_loss`, `eval_for_rag`) |
| `loss`, `ppl`, `token_acc` | 사전학습 평가: 평균 NLL, `exp(NLL)`, 다음 토큰 정확도 |
| `rouge1/2/L/Lsum` | ROUGE F-score |
| `accuracy` (RAG) | `best_subspan_exact_match`: 정규화한 정답이 예측 안에 **포함**되면 정답 |
| `predictions` (`eval_for_rag`) | 디코딩한 생성문 → TODO 18에서 첫 줄만 남김 |
| `extra_kw_args` | 모델이 HF `GenerationMixin`이면 `pad_token_id`를 넘김 (Llama용) |
| `llm_model`, `llm_tokenizer`, `llm_batch_size` | zero-shot RAG용 Llama 모델/토크나이저/배치(4) |
| `searcher_bm25` | pyserini BM25 검색기 |
| `submission_code_list`, `submission_directory_list` | 제출 zip에 넣을 파일/폴더 목록 |
