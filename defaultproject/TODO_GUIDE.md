# TODO 설명서 (빈칸 채우기 해설)

> 처음이면 [00_START_HERE.md](00_START_HERE.md)에서 읽는 순서와 폴더 경로를 먼저 확인한다.
> 기준 문서: `defaultproject_handout.pdf` (CSE402 Default Final Project)
> 원칙: **`#### YOUR CODES; TODO` 빈칸과 노트북의 빈칸만 채움.** 마커(`######`, `#### YOUR CODES; TODO`)와 힌트 주석은 그대로 두고, 마커 바로 아래의 빈 줄 자리에 코드를 넣었다.
> 빈칸이 아닌 곳을 고친 것은 **딱 1군데**이며, 이유는 [§4 빈칸 외 수정 사항](#4-빈칸-외-수정-사항-이유)에 적었다.
> 변수 하나하나에 대한 설명은 [VARIABLES.md](VARIABLES.md), 환경 설정과 실행 명령은 [RUN_GUIDE.md](RUN_GUIDE.md)를 참고한다.

---

## 0. 한눈에 보기

| # | 파일:줄 | 위치 | 핸드아웃 항목 | 한 줄 요약 |
|---|---|---|---|---|
| 1 | `model.py:66` | `apply_rotary_emb` | RoPE | 앞/뒤 절반을 (x1, x2) 쌍으로 묶어 각도 `m·θ_i`만큼 2D 회전 |
| 2 | `model.py:91` | `RMSNorm.forward` | RMSNorm | fp32에서 `x / sqrt(mean(x²)+eps)` 후 `weight`를 곱함 |
| 3 | `model.py:112` | `RotaryEmbedding.__init__` | RoPE | `inv_freq = θ_i = base^(-2i/d)` 계산 |
| 4 | `model.py:132` | `RotaryEmbedding.forward` | RoPE | `freqs = position × θ_i` → `cos`, `sin` |
| 5 | `model.py:153` | `MultiHeadAttention.__init__` | MHSA+GQA | `q/k/v/o_proj` 선언 (K·V는 `num_key_value_heads`개만) |
| 6 | `model.py:196` | `MultiHeadAttention.forward` | MHSA+GQA | `softmax(QKᵀ/√d + mask)·V` → head 합치기 → `o_proj` |
| 7 | `model.py:224` | `FeedForwardNetwork.forward` | FFN | SwiGLU: `down(SiLU(gate(x)) * up(x))` |
| 8 | `model.py:254` | `TransformerLayer.forward` | Transformer Layer | Pre-Norm + residual (Attn → FFN) |
| 9 | `model.py:385` | `TransformerModel._prepare_attention_mask` | Attention mask | causal + padding 마스크를 더하기용(additive) 4D 마스크로 |
| 10 | `model.py:441` | `TransformerForCausalLM.forward` | Causal LM loss | 한 칸 shift한 cross-entropy (`ignore_index=-100`) |
| 11 | `model_rag.py:26` | `ModelRAG.search` | BM25 retrieval | `batch_search` → top-k 문서를 `{title, text}`로 파싱 |
| 12 | `model_rag.py:55` | `ModelRAG.make_augmented_inputs_for_generate` | Prompt formation | `Title/Passage × k + Question + Answer:` 프롬프트 |
| 13 | `model_rag.py:72` | `ModelRAG.retrieval_augmented_generate` | RAG inference | 프롬프트를 left-padding으로 토크나이즈해 `inputs` 생성 |
| 14 | `dataset/classification.py:76` | `collate_fn_for_classification` | Classification | 기존 padding 제거 → 배치 최장 길이로 left-padding |
| 15 | `dataset/rag.py:123` | `RAGDataset.__getitem__` (train) | RAG finetune | `positive_ctxs`로 학습 프롬프트 생성 (추론과 같은 형식) |
| 16 | `dataset/summary.py:80` | `_preprocess` | Summarization | `[pad][bos]prompt target[eos]`, 요약 부분에만 loss |
| 17 | `main.ipynb` 셀 0 | `PROJ_PATH =` | (Colab 설정) | Colab Drive 경로 기입 (예시 경로) |
| 18 | `main.ipynb` 셀 5 | `eval_for_rag` `# fill here` | 4.2.1 Parsing | 생성 결과의 첫 줄만 정답 후보로 사용 |

**검증 결과 (핸드아웃 버전: Python 3.12 + PyTorch 2.6.0 + transformers 4.52.4, CPU)**
- 70M 설정의 파라미터 수 = **63.58M** → 핸드아웃 로그 `Training Total size=63.58M params`와 정확히 일치
- 같은 가중치를 넣은 HuggingFace `LlamaForCausalLM`(RoPE·GQA·RMSNorm·SwiGLU 구조가 같음)과 비교: logits 차이 **0.0**, loss 동일, KV-cache greedy `generate` 결과 동일
- left-padding 배치, bf16 forward, 역전파(모든 파라미터에 gradient 흐름), 작은 배치 overfit(4.60 → 0.03), `save_pretrained`/`from_pretrained` 왕복 재현 모두 통과
- 데이터 쪽(summary/classification/RAG dataset·collator, 가짜 BM25 retriever를 붙인 ModelRAG) 25개 검사 통과, 노트북 전 셀 컴파일 통과

---

## 1. `model.py` — GPT-small 본체

### TODO 1. `apply_rotary_emb` (RoPE 적용)
```python
cos = cos.unsqueeze(1)          # (B, 1, S, D/2): head 차원으로 broadcast
sin = sin.unsqueeze(1)
x1, x2 = x.chunk(2, dim=-1)     # 앞 절반 / 뒤 절반
x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```
- **무엇을:** 위치 `m`의 query/key 벡터에서 i번째 쌍 `(x1[i], x2[i])`를 각도 `m·θ_i`만큼 2D 회전시킨다.
  `[x1', x2'] = [x1·cos − x2·sin, x1·sin + x2·cos]`
- **왜 이렇게 되나:** 회전 후 내적은 `⟨R_m q, R_n k⟩ = f(q, k, m−n)`, 즉 **상대 위치에만 의존**한다(RoPE 논문의 핵심 성질). 테스트에서 `(m,n)=(7,3)`과 `(27,23)`의 내적이 같은 것을 확인했다.
- **shape:** 힌트대로 `cos/sin: (B, S, D/2)`, `x: (B, H, S, D)`. `unsqueeze(1)`로 head 축을 만들어 모든 head에 같은 각도를 적용한다.
- **설계 선택:** 논문 원형은 인접 원소 `(x0,x1),(x2,x3)…`를 쌍으로 묶는다(interleaved). 여기서는 HuggingFace Llama와 같은 "앞 절반/뒤 절반" 방식을 썼다. q와 k에 같은 순열을 적용하는 것이라 내적 값은 동일하고, 처음부터 학습하는 모델이라 표현력 차이가 없다. 이 방식이 slicing 비용이 더 적고, HF Llama와 수치 비교로 검증할 수 있다.

### TODO 2. `RMSNorm.forward`
```python
input_dtype = x.dtype
x = x.to(torch.float32)
variance = x.pow(2).mean(-1, keepdim=True)
x = x * torch.rsqrt(variance + self.eps)
output = self.weight * x.to(input_dtype)
```
- **수식:** `RMSNorm(x) = x / sqrt(mean(x²) + ε) ⊙ γ` (RMSNorm 논문). LayerNorm과 달리 평균을 빼지 않고 bias도 없다.
- **fp32 계산 이유:** 노트북이 모델을 `bfloat16`으로 바꿔 학습(`cast_bf16`)하므로, 제곱 평균을 bf16으로 계산하면 정밀도가 떨어진다. 통계만 fp32로 계산하고 원래 dtype으로 되돌린다.

### TODO 3. `RotaryEmbedding.__init__` (주파수 초기화)
```python
base = config.rope_theta
dim = config.head_dim
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
```
- **수식:** `θ_i = base^(−2i/d)`, `i = 0 … d/2−1`. `base = rope_theta`(노트북에서 `ROPE_THETA = 20000.0`)
- **주석 shape 힌트와의 차이:** 힌트는 `inv_freq: (1, head_dim // 2)`이지만, 빈칸 아래 원본 `forward`가 `self.inv_freq[None, :, None]` 형태로 **1차원 텐서를 전제**한다. 2차원이면 `expand`에서 에러가 나므로 `(head_dim // 2,)` 1차원으로 만들었다. 원소 개수(`head_dim // 2`)는 힌트와 같다.
- `persistent=False` 버퍼(원본 코드)라서 체크포인트에 저장되지 않고 로드 때마다 다시 계산된다. transformers 버전에 따라 이 재계산이 깨질 수 있어 버전을 고정했다([§5-1](#5-1-transformers-버전-의존성--4524로-고정한-이유)).

### TODO 4. `RotaryEmbedding.forward` (cos/sin 계산)
```python
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # m * theta_i
cos = freqs.cos()
sin = freqs.sin()
```
- `(B, D/2, 1) @ (B, 1, S)` → `(B, D/2, S)` → transpose → `(B, S, D/2)`. 각 원소가 `m·θ_i`이다.
- 원본 코드가 이 블록을 `autocast(enabled=False)`로 감싸고 있어 fp32로 계산된다. 위치가 크면 bf16으로는 각도 오차가 커지기 때문이다.

### TODO 5. `MultiHeadAttention.__init__` (Q/K/V/O projection)
```python
self.q_proj = nn.Linear(hidden, num_attention_heads * head_dim, bias=False)
self.k_proj = nn.Linear(hidden, num_key_value_heads * head_dim, bias=False)
self.v_proj = nn.Linear(hidden, num_key_value_heads * head_dim, bias=False)
self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden, bias=False)
```
- **GQA:** query head는 16개, key/value head는 4개(`num_key_value_heads`)만 만들고, forward의 `repeat_kv`가 K/V를 4배로 복제해 16개 query head와 짝을 맞춘다. KV cache와 파라미터가 줄어든다.
- **이름:** `q_proj`는 원본 코드(`TransformerForCausalLM.forward`의 `self_attn.q_proj.weight`)에서 참조하고, `k_proj`, `v_proj`는 forward에서 참조한다. 출력 projection 이름은 Llama 관례대로 `o_proj`로 했다.
- **`bias=False` 근거:** bias를 넣으면 파라미터가 63.59M이 되고, 빼면 핸드아웃 로그의 **63.58M과 정확히 일치**한다. `FeedForwardNetwork`(원본)도 `bias=False`이다.

### TODO 6. `MultiHeadAttention.forward` (attention 계산)
```python
attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale   # (B,H,S,S+cache)
if attention_mask is not None:
    attention_weights = attention_weights + attention_mask       # 0 = 허용, dtype 최솟값 = 차단
attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
attn_output = torch.matmul(attention_weights, value_states)      # (B,H,S,D)
attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)   # (B,S,H·D)
attn_output = self.o_proj(attn_output)
```
- **수식:** `Attention(Q,K,V) = softmax(QKᵀ/√d_k + M)·V` (Transformer 논문). `self.scale = head_dim^-0.5`
- **마스크를 더하는 이유:** TODO 9에서 만드는 마스크는 허용 위치 0, 차단 위치 `finfo(dtype).min`인 additive 마스크다. `-inf` 대신 최솟값을 쓰기 때문에, left-padding의 pad 행처럼 모든 칸이 막힌 행이 있어도 softmax가 NaN을 내지 않는다.
- **softmax를 fp32로 하는 이유:** bf16 학습에서 exp/합산 정밀도를 확보하기 위해서다.
- **dropout:** `config.attention_dropout`(노트북 0.1)을 attention 확률에 적용하며, `eval()` 모드에서는 꺼진다.
- **KV cache:** 빈칸 위 원본 코드가 이전 key/value를 이어 붙이므로 key 길이는 `S + cache_len`이다(힌트의 shape 그대로).

### TODO 7. `FeedForwardNetwork.forward` (SwiGLU FFN)
```python
outputs = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
outputs = self.dropout(outputs)
```
- 원본 `__init__`이 `gate_proj`, `up_proj`, `down_proj`, `SiLU`를 선언해 두었다. 그대로 조합하면 SwiGLU `W_down(SiLU(W_gate x) ⊙ W_up x)`가 된다.
- dropout(`ffn_dropout`, 노트북 0.05)은 FFN 출력(residual에 더해지기 직전)에 적용했다.

### TODO 8. `TransformerLayer.forward` (Pre-Norm 블록)
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, past_key_value = self.self_attn(hidden_states=..., position_embeddings=..., attention_mask=..., past_key_value=...)
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.feed_forward(hidden_states)
hidden_states = residual + hidden_states
```
- **Pre-Norm 구조:** `h = x + Attn(Norm(x))`, `out = h + FFN(Norm(h))`. 원본 `__init__`의 `input_layernorm`/`post_attention_layernorm` 이름이 Llama와 같은 Pre-Norm 배치를 뜻한다.
- 힌트("return layer hidden states and past key values with output of attention")대로, attention이 돌려준 `past_key_value`(갱신된 KV cache)를 그대로 반환한다.

### TODO 9. `TransformerModel._prepare_attention_mask`
```python
min_dtype = torch.finfo(dtype).min
query_positions = torch.arange(sequence_length) + seen_token_length    # query의 절대 위치
key_positions = torch.arange(target_length)
allowed = key_positions[None, :] <= query_positions[:, None]            # causal (S, T)
allowed = allowed[None, None].expand(batch_size, 1, S, T)
if attention_mask is not None:
    allowed = allowed & attention_mask[:, None, None, :].bool()         # pad key 차단
mask = torch.zeros((B, 1, S, T), dtype=dtype).masked_fill(~allowed, min_dtype)
```
- **두 가지를 합친다.**
  1. **causal:** query i는 자기 자신과 이전 토큰(key j ≤ i)만 본다. KV cache가 있으면 query의 절대 위치가 `seen_token_length + i`이므로 그만큼 이동한다(생성 단계에서 `S=1`, `T=지금까지 길이`).
  2. **padding:** `attention_mask == 0`인 key(패딩 토큰)는 어떤 query도 보지 못하게 한다.
- **결과 형태:** 힌트대로 `(batch_size, 1, sequence_length, target_length)`. head 축이 1이라 16개 head에 broadcast된다.
- `attention_mask`가 없으면(None) causal 마스크만 쓴다.

### TODO 10. `TransformerForCausalLM.forward` loss
```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100)
```
- **다음 토큰 예측:** 위치 t의 logits가 t+1의 토큰을 맞혀야 하므로 한 칸 어긋나게 맞춘다. 데이터 쪽(`collate_fn_for_pretrain` 등)은 `labels = input_ids.clone()`으로 shift 없이 넘기므로 모델 안에서 shift하는 것이 맞다. 노트북 `eval_loop_for_pretraining`도 같은 방식으로 shift해서 계산한다.
- `-100`(패딩, 프롬프트 부분)은 loss에서 제외되고, 결과는 평균을 낸 스칼라다(힌트 "loss must be scalar").

---

## 2. `model_rag.py` — RAG (retrieve → augment → generate)

### TODO 11. `ModelRAG.search` (BM25 검색)
```python
batch_hits = self.retriever.batch_search(queries=list(queries), qids=qids, k=k, threads=8)
for qid in qids:
    for hit in batch_hits[qid]:
        doc = hit2docdict(hit)                          # {"id":..., "contents": '"제목"\n본문'}
        title, sep, text = doc["contents"].partition("\n")
        passages.append({"id":..., "title": title.strip().strip('"'), "text": text.strip()})
        scores.append(hit.score)
```
- **retriever:** 노트북이 넣어 주는 pyserini `LuceneSearcher`(인덱스 `wikipedia-dpr-100w`, BM25)다. 배치 전체를 `batch_search`로 한 번에 멀티스레드 검색한다.
- **문서 파싱:** 이 인덱스의 원문은 `"제목"\n본문` 형식이라, 첫 줄을 제목(따옴표 제거)으로, 나머지를 본문으로 나눈다. 이미 import되어 있던 `utils.etc.hit2docdict`를 사용했다.
- **반환:** 질문마다 `[{id, title, text}, …]`(BM25 점수 내림차순)와 점수 리스트. `{title, text}` 형식은 학습 데이터 `positive_ctxs`와 같아서 프롬프트 코드를 공유할 수 있다.

### TODO 12. `ModelRAG.make_augmented_inputs_for_generate` (프롬프트 구성)
```text
Title: <제목1>
Passage: <본문1>
...                (k개, 기본 5)
Question: <질문>
Answer:
```
- **학습 형식(TODO 15)과 정확히 같은 템플릿**이다. GPT-small은 zero-shot 능력이 없어서 finetuning 형식과 추론 형식이 같아야 한다.
- **context 768토큰 예산:** GPT-small의 최대 길이는 1024이고, 학습 collator는 `max_len=992`에서 **오른쪽을 잘라낸다**. 그래서 passage 부분을 768토큰(해당 모델 토크나이저 기준) 이하로 줄여 `Question/Answer:`가 절대 잘리지 않게 했다. 추론에서도 같은 예산을 써서 학습과 입력 분포를 맞췄다.
- Llama-3.2-1B-Instruct에 이 프롬프트를 그대로 쓰면 핸드아웃이 말하는 **naive RAG**가 된다. 4.2.1의 프롬프트 개선(chat template, few-shot, CoT)은 `ModelRAG`를 상속한 새 클래스에서 하라는 것이 핸드아웃의 요구다.

### TODO 13. `ModelRAG.retrieval_augmented_generate`
```python
input_texts = self.make_augmented_inputs_for_generate(queries, qids, k=k)
self.tokenizer.padding_side = "left"
inputs = self.tokenizer(input_texts, padding="longest", return_tensors="pt", return_token_type_ids=False)
```
- 빈칸 아래 원본 코드가 `inputs`(dict)를 device로 옮기고 `self.model.generate(**inputs, **kwargs)`를 호출한 뒤 프롬프트 길이만큼 잘라 **답변 토큰만** 반환한다. 빈칸이 할 일은 그 `inputs`를 만드는 것이다.
- **left padding:** decoder-only 모델은 오른쪽 끝에서 이어서 생성하므로 패딩을 왼쪽에 둬야 한다. 커스텀 `generate`도 `attention_mask.cumsum`으로 position을 계산하므로 left padding을 전제한다.
- `return_token_type_ids=False`: 커스텀 `generate`는 `token_type_ids` 인자를 받지 않기 때문에 넣었다(GPT-2/Llama 토크나이저는 원래 반환하지 않지만 안전장치).
- GPT-2 토크나이저는 special token을 붙이지 않고, Llama 토크나이저는 BOS를 자동으로 붙인다. 각 모델의 기본 동작을 그대로 따른다.

---

## 3. `dataset/*.py` — task별 데이터

### TODO 14. `collate_fn_for_classification` (20 Newsgroups)
- **문제:** `_preprocessing`(원본)은 `map(batched=True)`의 **1000개 단위 map-batch마다** `padding="longest"`로 패딩한다. 그래서 저장된 샘플들의 길이가 map-batch마다 다르고, 이미 pad 토큰이 섞여 있다.
- **처리:**
  1. `attention_mask == 1`인 실제 토큰만 남겨 기존 패딩을 제거한다.
  2. 현재 DataLoader 배치의 최장 길이로 **왼쪽 패딩**한다.
  3. `labels`는 `(B,)` long 텐서로 만든다.
- **left padding인 이유:** `TransformerForSequenceClassification`은 `hidden_states[:, -1, :]`(마지막 위치)로 분류하므로, 마지막 위치가 반드시 실제 토큰이어야 한다.
- 빈 텍스트 샘플이 섞여 있으면 길이 0이라 전부 마스크된 행이 된다. 이 경우에도 에러 없이 처리되도록 했다.
- RoPE는 상대 위치만 쓰므로 left padding으로 절대 위치가 밀려도 결과가 같다(테스트로 확인).

### TODO 15. `RAGDataset.__getitem__` (학습 샘플)
```python
positive_ctxs = inputs['positive_ctxs']
ctxs = random.sample(positive_ctxs, k=min(5, len(positive_ctxs)))
input_text_ctx = "\n".join(f"Title: {title}\nPassage: {text}" ...)   # prefix_title / prefix_passage 사용
(768토큰 초과 시 잘라냄)
input_text_without_answer = f"{input_text_ctx}\nQuestion: {question}\nAnswer:"
```
- 핸드아웃 4.1.2 / Part 2-2: "Training samples use positive contexts (positive_ctxs)". 평가 샘플은 원본 코드대로 질문/정답/uid만 넘기고, 검색은 `ModelRAG`가 한다.
- **최대 5개를 랜덤 샘플링하는 이유:** 평가 때 `eval_for_rag`가 BM25 top-**5**를 넣으므로 학습 입력의 passage 개수와 길이를 비슷하게 맞춘다. 순서를 섞으면 "정답은 항상 첫 passage"라는 편향도 줄어든다. 파일 상단에 원래 `import random`이 있었던 것도 이런 용도로 보인다.
- 정답이 붙는 형식은 원본 코드가 정한다: `input_text = f"{input_text_without_answer} {answer}<eos>"` → `…\nAnswer: 정답<eos>`
- **768토큰 예산:** `RAGCollator`(원본)가 `max_length = 992 − 정답길이`로 **오른쪽을 잘라내므로**, context가 길면 `Question/Answer:`가 잘려 나간다. 테스트에서 400단어짜리 passage 8개를 넣어도 `Answer:`가 보존되고 loss는 정답 토큰에만 걸리는 것을 확인했다.

### TODO 16. `_preprocess` (CNN/DailyMail 요약)
```text
input_ids : [pad … pad][bos] "Context: …\n Summary:\n" 요약문 [eos]    ← 전체 길이 = 896 + 128 = 1024 고정
labels    : [-100 …………………………………………… -100] 요약문 [eos]    ← 요약문에만 loss
attention : [0 … 0][1 ……………………………………………… 1]
```
- **길이를 1024로 고정하는 이유:** 원본 `collate_fn_for_summary`가 `torch.tensor(list_of_lists)`로 바로 텐서를 만들기 때문에 모든 샘플 길이가 같아야 한다. 이후 1024의 배수로 잘라 쓴다.
- **padding 방향:** `tokenizer.padding_side`를 따른다(노트북은 `"left"`). collate도 두 방향을 모두 처리한다.
- **라벨:** 프롬프트와 패딩은 `-100`, 요약문과 `eos`만 학습한다. `eos`를 넣어야 생성할 때 멈출 줄 안다.
- **평가 호환:** 노트북 `eval_for_summary`는 `labels != -100`인 위치를 지워 프롬프트만 남긴 뒤 생성하고, 라벨을 디코드해서 정답으로 쓴다. 이 구조에 맞췄다(테스트에서 프롬프트가 `Summary:\n`로 끝나는 것 확인).
- `bos`로 시작하는 이유: 사전학습 데이터가 `bos + 텍스트 + eos` 형식이라 같은 분포를 유지한다.
- 경계 처리: 재토크나이즈로 길이가 약간 늘면 프롬프트 앞쪽을 잘라 `896`을 넘지 않게 하고, 요약문은 `eos`를 포함해 `128`토큰 이내로 한다.

---

## 3-1. `main.ipynb`의 빈칸

### TODO 17. 셀 0 `PROJ_PATH =` (Colab 전용 분기)
- 원본은 `PROJ_PATH =`(값 없음)라서 **셀 전체가 SyntaxError**다. 로컬/Docker에서도 셀 0이 실행되지 않아 이후 모든 셀이 실패한다.
- 주석의 예시 경로 `'/content/drive/MyDrive/Colab Notebooks/repository/default_proj'`로 채웠다.
- **Docker/로컬 실행에서는 이 줄이 쓰이지 않는다**(`import google.colab` 실패 → `except` 분기에서 `PROJ_PATH = "."`). Colab에서 돌린다면 본인 Drive 경로로 바꿔야 한다.

### TODO 18. 셀 5 `eval_for_rag`의 `# fill here` (예측 후처리)
```python
# predictions = [pred.split("\n")[0] for pred in predictions]      ← 원래 있던 힌트 (그대로 둠)
predictions = [pred.strip().split("\n")[0].strip() for pred in predictions]
```
- 힌트대로 **생성문의 첫 줄만** 답으로 쓴다. 다만 앞에 공백이나 줄바꿈이 있으면 첫 줄이 빈 문자열이 되어 오답 처리되므로 먼저 `strip()`했다(Llama가 `"\n\nParis …"`처럼 시작하는 경우).
- GPT-small은 `" 정답<eos>"`를 생성하도록 학습되므로 결과는 `"정답"`이다. Llama는 `"Paris\nQuestion: …"`처럼 이어서 쓰는 경우가 많아 첫 줄만 남긴다.
- JSON 파싱처럼 더 정교한 후처리는 핸드아웃 4.2.1(baseline RAG 개선)에서 직접 할 부분이다.

---

## 4. 빈칸 외 수정 사항 (이유)

**수정한 곳은 딱 1군데다.**

| 파일 | 위치 | 원본 | 수정 | 이유 |
|---|---|---|---|---|
| `main.ipynb` | 셀 10 (`if DO_FINETUNE_RAG:`) 마지막 부분 | `print(f"======================"` | `print(f"======================")` | **닫는 괄호 누락으로 SyntaxError.** 파이썬은 셀 전체를 먼저 컴파일하므로 `DO_FINETUNE_RAG = False`여도 이 셀은 항상 에러가 나고, "Run All"이 여기서 멈춰 뒤의 zero-shot RAG·제출 셀까지 실행되지 않는다. 핸드아웃이 요구하는 "RAG 파인튜닝(Listing 5·6)"과 "DO_SUBMISSION으로 zip 생성"을 하려면 반드시 고쳐야 한다. 같은 문장이 셀 12에는 괄호가 제대로 있어 오타로 판단했다. |

- 수정 전후 확인: 원본 노트북은 셀 0(빈칸)과 셀 10(괄호) 두 곳에서 컴파일 에러가 났고, 지금은 모든 코드 셀이 컴파일된다.
- `.py` 파일들은 빈칸 줄 외에 **한 글자도 바뀌지 않았다**(원본과 diff하면 삭제된 줄은 공백뿐인 빈 줄 16개).

### 새로 추가한 파일 (기존 코드 수정 아님)
| 파일 | 용도 |
|---|---|
| `Dockerfile` | 핸드아웃 요구 환경(Python 3.12 / PyTorch 2.6.0 / Java 21) 컨테이너 |
| `run_case.py` | `main.ipynb`를 **수정하지 않고** 케이스별(DO_* 플래그)로 실행하는 헬퍼 |
| `check_env.py` | 최초 세팅(버전·GPU·Java·HF 토큰·Llama 권한 등)을 한 번에 점검 |
| `00_START_HERE.md`, `RUN_GUIDE.md`, `TODO_GUIDE.md`, `VARIABLES.md` | 설명 문서 (읽는 순서는 `00_START_HERE.md`) |

노트북의 제출 셀(`DO_SUBMISSION`)은 `submission_code_list`에 명시된 파일만 zip에 넣으므로, 위 파일들은 제출물에 포함되지 않는다.

---

## 5. 고치지 않았지만 알아둘 점

### 5-1. transformers 버전 의존성 → 4.52.4로 고정한 이유
핸드아웃은 PyTorch 2.6.0과 Python 3.12만 명시하고 transformers 버전은 정하지 않았다. 그런데 **빈칸 밖 원본 코드**(`TransformerPreTrainedModel._init_weights`, `eval_for_rag`의 `GenerationMixin` 분기, weight tying 정책)가 버전마다 다르게 동작한다. 직접 측정한 결과:

| transformers | 파라미터 수 | 결과 |
|---|---|---|
| 4.51.3 | 63.58M ✅ | 커스텀 모델이 `GenerationMixin`으로 판정 → `eval_for_rag`가 커스텀 `generate()`에 `pad_token_id`를 넘김 → **RAG 평가 TypeError** |
| **4.52.4** | **63.58M ✅** | **전체 테스트 통과** (핸드아웃 로그와 일치) |
| 4.53.3 | 63.58M ✅ | 위 세 항목(파라미터 수·GenerationMixin·저장/로드)만 확인, 문제 없음 |
| 4.54 ~ 4.57 | 39.04M ❌ | `lm_head`와 임베딩이 자동 weight tying → 핸드아웃과 모델이 달라지고 `save_pretrained`가 RuntimeError |
| 5.x (2026년 기본값) | 63.58M | `from_pretrained` 시 non-persistent 버퍼 `inv_freq`가 초기화되지 않은 메모리로 채워짐 → **사전학습 모델을 불러오는 finetune 단계에서 RoPE가 깨짐** |

→ 과제 기간(마감 2025-06-07) 당시 최신이던 **4.52.4**로 고정했다(Dockerfile).
Colab처럼 버전을 지정하지 않고 `pip install transformers`만 하는 환경에서는 최신 버전(5.x)이 설치될 수 있으므로 **반드시 버전을 고정**해야 한다.

### 5-2. zero-shot RAG의 `pad_token_id` (셀 12, 수정 안 함)
`eval_for_rag(model_rag, eval_loader, rag_config, tokenizer)`에 Llama 토크나이저가 아닌 **GPT-2 토크나이저**가 넘어간다. 그래서 HF `generate`의 `pad_token_id`가 50256(GPT-2 EOS)이 되는데, Llama 어휘에서는 다른 일반 토큰이다. 문장이 EOS로 일찍 끝나면 그 뒤를 이 토큰으로 채우고, 디코딩하면 답 뒤에 엉뚱한 글자가 붙을 수 있다. 정확도 지표는 "정답이 예측 안에 포함되는가"(부분 일치)라 영향이 작지만 ROUGE는 약간 낮아진다. baseline 개선(4.2.1) 때 `llm_tokenizer`를 넘기도록 바꾸면 해결된다.

### 5-3. Llama 생성의 무작위성
`meta-llama/Llama-3.2-1B-Instruct`의 기본 `generation_config`가 샘플링(`do_sample=True`, temperature 0.6, top_p 0.9)이라 zero-shot 점수가 실행마다 조금씩 다를 수 있다. 재현성이 필요하면 개선 버전에서 `do_sample=False`를 넘기면 된다.

### 5-4. 제출 셀의 `README`
`submission_code_list`에 `"README"`(확장자 없음)가 있지만 폴더에는 `README.txt`만 있다. 그대로 두면 zip 안에 `README not exist`라는 내용의 파일이 들어간다. 핸드아웃이 "including README file"을 요구하므로 제출 전에 `README` 파일을 직접 작성해 두면 된다.

### 5-5. pyserini는 모든 케이스에서 필요
`utils/etc.py`가 모듈 최상단에서 `pyserini`를 import하고 노트북 셀 1이 이를 import한다. 그래서 사전학습만 돌려도 Java와 pyserini가 설치돼 있어야 한다(Docker 이미지에 포함).
