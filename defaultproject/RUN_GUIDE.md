# 실행 가이드 (최초 세팅 + 환경 + 케이스별 명령어)

> 처음이면 [00_START_HERE.md](00_START_HERE.md)에서 읽는 순서와 폴더 경로를 먼저 확인한다.
> 이 폴더(`main.ipynb`가 있는 곳)를 GPU 서버로 복사한 뒤, `Dockerfile`로 핸드아웃이 요구하는 환경을 만들어 실행한다.
> 빈칸 해설은 [TODO_GUIDE.md](TODO_GUIDE.md), 변수 설명은 [VARIABLES.md](VARIABLES.md).

---

## 0. 최초 1회 세팅 (여기만 끝내면 이후는 실행 명령어만 치면 된다)

### 0-1. 미리 준비할 것 (계정·토큰·서버)

| # | 준비물 | 어디서 / 어떻게 | 필요한 케이스 |
|---|---|---|---|
| 1 | HuggingFace 계정 | https://huggingface.co/join | ⑤ zero-shot RAG |
| 2 | **Llama-3.2 라이선스 동의** | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct 에서 약관 동의 → 승인 메일 대기(수 분~수 시간 걸릴 수 있어 **가장 먼저** 신청) | ⑤ |
| 3 | **HF 액세스 토큰 (`HF_TOKEN`)** | https://huggingface.co/settings/tokens → Create new token → **Classic + Read** 권장. Fine-grained를 쓴다면 Repositories에서 **"Read access to contents of all public gated repos you can access"**를 직접 체크해야 meta-llama에 접근된다 | ⑤⑥ (다른 케이스도 다운로드 속도 제한 완화에 도움) |
| 4 | GPU 서버 | NVIDIA GPU **Ampere 이상** = compute capability 8.0 이상 (A100·A40·A10·L4·L40S·H100·RTX 30/40). **V100(sm_70)·T4(sm_75)는 안 된다** — 노트북 셀 2의 주석이 V100을 Ampere 예시로 잘못 적어 놓았으니 무시한다. VRAM **16GB 이상 권장**(12GB는 ①②④에서 빠듯), 드라이버 550 이상 | 전체 |
| 5 | Docker + NVIDIA Container Toolkit | `docker run --rm --gpus all ubuntu nvidia-smi`가 GPU 표를 출력하면 OK | 전체 |
| 6 | 디스크 여유 **100~150GB** | 사전학습 코퍼스(cosmopedia-v2 약 35GB) + BM25 인덱스(8.55GB, 전개 후 더 큼) + DPR NQ + 체크포인트 + Docker 이미지. **`/workspace`에 50~70GB, `/var/lib/docker`에 50~60GB**로 나뉘니 `df -h /var/lib/docker .`로 두 파티션을 각각 확인한다 | ①④⑤ |
| 7 | 호스트 RAM **64GB 권장** | `dataset/rag.py`가 `pd.read_json`으로 DPR NQ train을 통째로 읽는다. 32GB에서는 ④⑤가 OOM-kill될 수 있다 | ④⑤⑥ |
| 8 | 프로젝트 폴더를 서버로 복사 | [00_START_HERE.md §1](00_START_HERE.md#1-폴더-경로) 의 `scp` 명령 | 전체 |

> GPU 아키텍처는 컨테이너 안에서 이 한 줄로 확정한다. `(8, 0)` 이상이어야 한다.
> ```bash
> python -c "import torch; print(torch.cuda.get_device_capability())"
> ```
> `torch.cuda.is_bf16_supported()`는 Ampere 미만에서도 에뮬레이션을 이유로 `True`를 돌려주므로 믿으면 안 된다.

### 0-2. 한 번에 세팅 (서버에서 순서대로 복사·실행)

```bash
# (1) 프로젝트 폴더로 이동 — 서버에 복사해 둔 경로
cd ~/defaultproject

# (2) 비밀값 파일 .env 작성 — 토큰은 여기 한 곳에만 적고, 이후 모든 docker run이 --env-file .env로 읽는다
cat > .env <<'EOF'
# docker --env-file 형식: KEY=VALUE (따옴표·공백 없이)
HF_TOKEN=hf_여기에_발급받은_토큰
JUPYTER_TOKEN=jupyter_접속용_비밀번호_아무거나
EOF
chmod 600 .env

# (3) Docker 이미지 빌드 (최초 1회, Dockerfile이 파일을 COPY하지 않아 빌드 컨텍스트 없이 빌드)
docker build -t cse402 - < Dockerfile

# (4) 다운로드 캐시 볼륨 (HF 모델·데이터셋, pyserini 인덱스를 보관해 재다운로드 방지)
docker volume create cse402-cache

# (5) 한 번에 점검: Python·PyTorch·GPU(bf16)·패키지 버전·Java·pyserini·마운트·디스크·HF 토큰·Llama 권한·README
docker run --gpus all --rm --env-file .env \
  -v "$PWD":/workspace -v cse402-cache:/root/.cache cse402 python check_env.py
```

| `.env` 항목 | 용도 | 필수 여부 |
|---|---|---|
| `HF_TOKEN` | gated 모델 `meta-llama/Llama-3.2-1B-Instruct` 다운로드 (노트북 셀 11의 `notebook_login()` 대신 사용) | ⑤⑥에 필수 |
| `JUPYTER_TOKEN` | Jupyter 접속 비밀번호를 고정 (없으면 매번 로그에서 토큰을 찾아야 함) | **Jupyter 방식을 쓸 때만. `run_case.py`로만 작업하면 아예 없어도 된다** |

- `.env`는 비밀값이다. 공유하거나 git에 올리지 않는다. 노트북 제출 셀은 정해진 파일만 zip에 넣으므로 제출물에는 포함되지 않는다.
- 토큰을 바꾸면 `.env`만 고치면 된다(이미지 재빌드 불필요).

### 0-3. 점검 결과 보는 법 (`check_env.py`)
정상이면 모두 `[OK]`이고 마지막 줄이 `READY: training cases can run`이다. `FAIL`이 하나라도 있으면 `NOT READY`가 나오며 학습 전에 반드시 고쳐야 한다. `WARN`은 해당 케이스 전에만 해결하면 된다.

| 항목 | FAIL/WARN일 때 조치 |
|---|---|
| `PyTorch 2.6.0 + GPU (bf16)` | `docker run --rm --gpus all ubuntu nvidia-smi`로 GPU가 보이는지 확인하고 NVIDIA Container Toolkit을 설치한다. `bf16=False`면 Ampere 미만 GPU이므로 A100급 서버를 사용한다 |
| `transformers 4.52.4 / datasets 3.6.0`, `Java 21`, `pyserini (JVM)` | 이미지가 잘못 빌드된 것이다. `docker build --no-cache -t cse402 - < Dockerfile`로 다시 빌드한다 |
| `project files mounted` | `cd ~/defaultproject`(main.ipynb가 있는 폴더)에서 실행했는지, `-v "$PWD":/workspace`를 넣었는지 확인한다 |
| `disk space` (WARN) | 50GB 이상 확보한다 |
| `HF_TOKEN` (WARN) | `.env`의 토큰 값(`hf_`로 시작, 따옴표 없이)을 확인하고, 필요하면 Read 권한으로 재발급한다 |
| `Llama-3.2 license / access` (WARN) | 모델 페이지에서 라이선스에 동의했는지 확인한다. 승인 전이면 ①~④를 먼저 돌리고 ⑤는 나중에 한다 |
| `README for submission` (WARN) | 제출(⑥) 전에 확장자 없는 `README` 파일을 작성한다 ([TODO_GUIDE §5-4](TODO_GUIDE.md#5-4-제출-셀의-readme)) |

---

## 1. 환경 (핸드아웃 요구 버전)

| 항목 | 버전 | 근거 |
|---|---|---|
| Python | **3.12** | 핸드아웃 3장 "install PyTorch 2.6.0 under Python 3.12" |
| PyTorch | **2.6.0** (CUDA 12.4 휠) | 핸드아웃 3장 |
| Java | **OpenJDK 21** | 핸드아웃 3장 "Installing Java is also required", 노트북 `JAVA_HOME=java-21`, 로그의 "Java 21" |
| transformers | **4.52.4** | 핸드아웃에 버전 명시 없음 → 원본 코드가 정상 동작하는 과제 기간 버전으로 고정 ([TODO_GUIDE §5-1](TODO_GUIDE.md#5-1-transformers-버전-의존성--4524로-고정한-이유)) |
| 그 외 | datasets 3.6.0 · accelerate 1.7.0 · evaluate 0.4.3 · faiss-cpu 1.11.0 · pyserini 0.44.0 · nltk 3.9.1 · rouge_score 0.1.2 · wget 3.2 · absl-py 2.3.0 · numpy 2.2.6 · pandas 2.3.0 · notebook 7.4.3 | `README.txt`의 패키지 목록을 과제 기간(마감 2025-06-07) 버전으로 고정 |

노트북은 `model_dtype="cast_bf16"`(bf16)으로 설정돼 있어 **Ampere 이상 GPU**가 필요하다. 노트북 메타데이터 기준 원 실행 환경은 Colab A100이다.

> 이 조합(Python 3.12.14 + torch 2.6.0 + transformers 4.52.4 + datasets 3.6.0)에서 채운 코드의 테스트 42개가 모두 통과했다(CPU 검증). Linux/Python 3.12 대상 pip 의존성 해석도 충돌 없이 끝났다.

---

## 2. 컨테이너 실행 (매번)

§0을 끝냈다면 이후에는 아래 둘 중 하나로 컨테이너를 연다. 항상 **프로젝트 폴더에서** 실행한다.

**A. Jupyter로 쓰기** (브라우저에서 `main.ipynb` 열기)
```bash
cd ~/defaultproject
docker run --gpus all -it --rm --ipc=host -p 8888:8888 --env-file .env \
  -v "$PWD":/workspace -v cse402-cache:/root/.cache cse402
# 브라우저: http://<서버주소>:8888  → .env의 JUPYTER_TOKEN 값을 입력
```

**B. 터미널로 쓰기** (케이스별 명령어 실행용)
```bash
cd ~/defaultproject
docker run --gpus all -it --rm --ipc=host --env-file .env \
  -v "$PWD":/workspace -v cse402-cache:/root/.cache cse402 bash
```

- `--env-file .env` : §0에서 만든 `HF_TOKEN`·`JUPYTER_TOKEN`을 컨테이너 환경변수로 넣는다.
- `-v "$PWD":/workspace` : 프로젝트 폴더를 마운트한다. 학습 결과(`logs/`, `output/`, `cache/`, `local_cache/`, `runs/`)가 서버 폴더에 그대로 남는다.
- `-v cse402-cache:/root/.cache` : HuggingFace·pyserini 다운로드 캐시(§0에서 만든 볼륨)를 재사용한다.
- `--ipc=host` : 데이터 전처리 멀티프로세싱의 공유 메모리 부족을 막는다.
- Windows PowerShell에서 실행할 경우 빌드는 `Get-Content Dockerfile -Raw | docker build -t cse402 -`, 마운트는 `-v "${PWD}:/workspace"`로 쓴다.

---

## 3. 케이스별 학습 명령어

### 3-1. 터미널 방식 (`run_case.py`) — 권장
`run_case.py`는 `main.ipynb`를 **수정하지 않고**, 메모리에서 `DO_*` 플래그 하나만 켠 복사본(`runs/main_<case>.ipy`)을 만들어 IPython으로 실행한다. 진행바와 평가 로그가 터미널에 실시간으로 나온다.

컨테이너 안(`/workspace`)에서:
```bash
mkdir -p runs

python run_case.py pretrain        2>&1 | tee runs/pretrain.log        # ① GPT-small 사전학습 (가장 먼저)
python run_case.py summary         2>&1 | tee runs/summary.log         # ② 요약 파인튜닝        (① 필요)
python run_case.py classification  2>&1 | tee runs/classification.log  # ③ 분류 파인튜닝        (① 필요)
python run_case.py rag             2>&1 | tee runs/rag.log             # ④ RAG 파인튜닝 (NQ)    (① 필요)
python run_case.py zeroshot_rag    2>&1 | tee runs/zeroshot_rag.log    # ⑤ Llama naive RAG      (HF_TOKEN 필요)
python run_case.py prompt_rag      2>&1 | tee runs/prompt_rag.log      # ⑥ baseline RAG 4.2.1 + 확장 4.3 (HF_TOKEN 필요)
python run_case.py submission      2>&1 | tee runs/submission.log      # ⑦ 제출 zip 생성        (마지막)
```
- 정상 종료되면 마지막 줄에 `[run_case] DONE: <case>`가 찍힌다. 이 줄이 없으면 위쪽 traceback을 확인한다.
- **`| tee`를 쓰면 `$?`가 항상 0이다**(파이프라인 종료코드는 `tee`의 것). 종료코드로 판정하려면 `set -o pipefail`을 먼저 실행하거나 `${PIPESTATUS[0]}`을 보고, 아니면 아래 루프처럼 `DONE:` 줄로 판정한다.
- ⑥ `prompt_rag`가 어떤 설정을 돌릴지는 노트북 `DO_PROMPT_RAG` 셀의 `PROMPT_RAG_SELECTED` 리스트로 고른다(기본값 `["v2_fewshot"]`). 탐색 단계에서는 `PROMPT_RAG_NUM_SAMPLES = 1000`으로 줄여 쓰고, 최종 표만 `None`(전체 6,515문항)으로 돌린다.
- ②③④는 `output/pretraining/best_model`이 없으면 바로 멈추고 ①을 먼저 하라고 안내한다.

**①~⑤를 순서대로 한 번에** (실패하면 멈춤):
```bash
mkdir -p runs
for c in pretrain summary classification rag zeroshot_rag; do
  python run_case.py $c 2>&1 | tee runs/$c.log
  grep -q "\[run_case\] DONE: $c" runs/$c.log || { echo "FAILED: $c"; break; }
done
```

**컨테이너를 백그라운드로 띄워서 한 케이스 돌리기** (SSH가 끊겨도 계속 실행, 서버의 프로젝트 폴더에서):
```bash
docker run --gpus all -d --name cse402-pretrain --ipc=host --env-file .env \
  -v "$PWD":/workspace -v cse402-cache:/root/.cache cse402 \
  bash -c "mkdir -p runs && python run_case.py pretrain 2>&1 | tee runs/pretrain.log"
docker logs -f cse402-pretrain        # 진행 상황 보기 (Ctrl+C는 보기만 중단)
docker rm cse402-pretrain             # 끝난 뒤 컨테이너 정리 (이름 재사용 전)
```
(`pretrain` 두 곳을 다른 케이스 이름으로 바꾸면 된다.)

### 3-2. Jupyter 방식 (핸드아웃 Listing 1·3·5·7 그대로)
`main.ipynb` 셀 2의 플래그를 아래처럼 바꾸고 **Kernel → Restart Kernel and Run All Cells**.

| 케이스 | `DO_PRETRAIN` | `DO_FINETUNE_SM` | `DO_FINETUNE_CF` | `DO_FINETUNE_RAG` | `DO_ZEROSHOT_RAG` | `DO_SUBMISSION` |
|---|---|---|---|---|---|---|
| ① 사전학습 (Listing 1) | **True** | False | False | False | False | False |
| ② 요약 (Listing 3) | False | **True** | False | False | False | False |
| ③ 분류 | False | False | **True** | False | False | False |
| ④ RAG 파인튜닝 (Listing 5) | False | False | False | **True** | False | False |
| ⑤ Zero-shot RAG (Listing 7) | False | False | False | False | **True** | False |
| ⑥ 제출 | False | False | False | False | False | **True** |

> 제출 파일의 `main.ipynb` 플래그는 사용자가 정할 값이라 원본 상태(`DO_FINETUNE_SM = True`)를 그대로 두었다. 셀 11의 `notebook_login()` 위젯은 `.env`로 `HF_TOKEN`을 넣었다면 무시해도 된다.

---

## 4. 순서 · 소요 시간 · 결과 위치

```
① pretrain ──┬──> ② summary
             ├──> ③ classification
             └──> ④ rag
⑤ zeroshot_rag  (독립, 언제든 가능)
⑥ submission    (모두 끝난 뒤)
```

| 케이스 | 스텝 수 / 규모 | 예상 시간 (핸드아웃 로그 기준 ~8.5 it/s, GPU에 따라 다름) | 주요 결과 |
|---|---|---|---|
| ① 사전학습 | 93,844 스텝 (2 epoch) | 약 3시간 + 최초 코퍼스 다운로드·전처리 | `output/pretraining/best_model`, `logs/pretraining/*.jsonl` |
| ② 요약 | 53,835 스텝 (3 epoch) | 약 1시간 45분 + 검증셋 생성 평가 | `output/results/summary_score.json`, `summary_output.txt` |
| ③ 분류 | 약 1,770 스텝 (5 epoch) | 수 분 | `output/results/classification_score.json` |
| ④ RAG 파인튜닝 | 11,040 스텝 (3 epoch) | 약 20~30분 + BM25 인덱스 8.55GB 최초 다운로드 + 평가(dev 6,515문항) | `output/results/rag_score.json`, `rag_output.txt` |
| ⑤ Zero-shot RAG | dev 6,515문항 | 검색+생성 시간 | `output/results/llm_rag_score.json`, `llm_rag_output.txt` |
| ⑥ 제출 | – | 수 초 | `defaultproject_code.zip`, `defaultproject_supplementaries.zip` |

**핸드아웃 로그의 참고 값** (정상 구현이면 비슷한 추세가 나와야 한다)
- 사전학습 `Training Total size=63.58M params` → eval@2500: loss 4.47 / ppl 87.5 / token_acc 0.309 → eval@15000: loss 3.22 / ppl 25.25 / token_acc 0.419
- 요약 eval@2500: loss ≈ 4.29
- RAG 파인튜닝 eval@2500: accuracy ≈ 0.033 (초반 값)
- Zero-shot RAG(naive): accuracy ≈ 0.253, rouge1 ≈ 0.153 (Llama 기본 설정이 샘플링이고 후처리 방식에 따라 조금 다를 수 있음)

**학습 곡선(보고서용):** `logs/<case>/train.jsonl`(10스텝마다 loss·lr), `logs/<case>/eval.jsonl`(2500스텝마다 평가 지표).

---

## 5. 주의사항

- **캐시 재사용:** 전처리된 데이터는 `cache/pretrain`, `cache/summary`, `cache/classification`에 저장되고 다음 실행 때 그대로 재사용된다. 이전에 다른 코드로 전처리한 캐시가 있으면 해당 폴더를 지우고 다시 실행한다.
- **로그 덮어쓰기:** 같은 케이스를 다시 실행하면 `Logger(remove_existing=True)` 때문에 `logs/<case>/`의 이전 로그가 지워진다. 보존하려면 먼저 복사해 둔다.
- **메모리 부족(OOM):** 노트북은 A100 기준 설정이다. 메모리가 작은 GPU라면 해당 셀의 `batch_size`를 줄이고 `gradient_accumulation_steps`를 늘린다(실효 배치 유지).
- **제출 전 `README`:** 제출 셀은 확장자 없는 `README` 파일을 찾는다(현재 폴더에는 `README.txt`만 있음). [TODO_GUIDE §5-4](TODO_GUIDE.md#5-4-제출-셀의-readme) 참고.
- **Colab에서 돌릴 경우:** ① 셀 0의 `PROJ_PATH`를 본인 Drive 경로로 바꾸고, ② 맨 처음에 `%pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 transformers==4.52.4 datasets==3.6.0 pyserini==0.44.0`을 실행한 뒤 런타임을 재시작한다. Colab 기본 설치 버전은 핸드아웃 버전과 다를 수 있다. `HF_TOKEN`은 Colab 왼쪽 🔑(Secrets)에 등록하거나 셀 11의 로그인 위젯을 쓴다.
