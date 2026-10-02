# 여기서 시작 — 읽는 순서 · 폴더 경로

> CSE402 Default Final Project (GPT-small + RAG). 빈칸은 모두 채워져 있고, 핸드아웃 버전 환경은 `Dockerfile`로 만든다.
> 이 파일 → `RUN_GUIDE.md` → `TODO_GUIDE.md` → `VARIABLES.md` 순서로 보면서 진행한다.

---

## 1. 폴더 경로

| 위치 | 경로 | 설명 |
|---|---|---|
| **이 PC** (빈칸 채운 원본) | `C:\Users\seank\Downloads\defaultproject\defaultproject` | 문서(.md)·코드가 모두 이 폴더 바로 아래에 있다 |
| **GPU 서버** (복사할 곳, 예시) | `~/defaultproject` (= `/home/<계정>/defaultproject`) | 모든 `docker` 명령을 이 폴더에서 실행한다. 다른 경로를 써도 된다 |
| **Docker 컨테이너 안** | `/workspace` | 서버 폴더가 그대로 마운트된다. `run_case.py`를 여기서 실행한다 |

이 PC → 서버 복사 (이 PC의 PowerShell에서):
```powershell
scp -r "C:\Users\seank\Downloads\defaultproject\defaultproject" <계정>@<서버주소>:~/defaultproject
```
- 서버에 `~/defaultproject`가 **없을 때** 실행한다. 이미 있으면 그 안에 `defaultproject/defaultproject`로 한 단계 더 들어가므로 먼저 지우거나 이름을 바꾼다.
- 학습 결과(`logs/`, `output/` 등)는 **서버 폴더**에 생긴다. 이 PC 폴더에는 생기지 않는다.

---

## 2. 읽는 순서 (= 진행 순서)

모든 문서는 위 폴더 바로 아래에 있다. 경로는 이 PC 기준이며, 서버에서는 `~/defaultproject/<파일명>`이다.

| 순서 | 문서 (볼 부분) | 경로 | 이때 할 일 |
|---|---|---|---|
| **0** | `00_START_HERE.md` (이 파일) | `C:\Users\seank\Downloads\defaultproject\defaultproject\00_START_HERE.md` | 전체 흐름과 폴더 경로 파악, 서버로 폴더 복사 |
| **1** | `RUN_GUIDE.md` **§0** | `C:\Users\seank\Downloads\defaultproject\defaultproject\RUN_GUIDE.md` | **최초 1회 세팅:** Llama 라이선스 신청, HF 토큰 발급, `.env` 작성, 이미지 빌드, `check_env.py`로 `READY` 확인 |
| **2** | `RUN_GUIDE.md` §1~§3 | (위와 같음) | 환경 버전 확인 → 컨테이너 열기 → **① 사전학습 시작** (약 3시간으로 가장 오래 걸리니 먼저 돌려 둔다) |
| **3** | `TODO_GUIDE.md` | `C:\Users\seank\Downloads\defaultproject\defaultproject\TODO_GUIDE.md` | 학습이 도는 동안 읽기: 빈칸을 무엇으로 왜 채웠는지, 빈칸 외 수정 1곳의 사유, 알아둘 점 → 보고서 Approach 재료 |
| **4** | `VARIABLES.md` | `C:\Users\seank\Downloads\defaultproject\defaultproject\VARIABLES.md` | 코드·노트북을 읽을 때 옆에 두는 변수 사전 (설정값, 텐서 shape, 학습 설정) |
| **5** | `RUN_GUIDE.md` §3 → §4 | (1번과 같음) | ① 완료 후 ②요약 ③분류 ④RAG ⑤zero-shot RAG 실행, 결과 파일과 핸드아웃 참고값 비교, 학습 곡선 확인 |
| **6** | `defaultproject_handout.pdf` 4.2.1 · 4.3 · Submission | `C:\Users\seank\Downloads\defaultproject\defaultproject\defaultproject_handout.pdf` | **직접 해야 하는 파트:** baseline RAG 프롬프트 개선(4.2.1), RAG 확장 아이디어(4.3), 보고서(ACL 양식 6~8쪽) |
| **7** | `RUN_GUIDE.md` §5 + `TODO_GUIDE.md` §5-4 | (1·3번과 같음) | 제출 전 `README` 작성 → ⑥ `submission` 실행 → zip 2개 + 보고서 PDF 제출 |

---

## 3. 폴더 구조

```
defaultproject/                       이 PC: C:\Users\seank\Downloads\defaultproject\defaultproject
│                                     서버:  ~/defaultproject      컨테이너: /workspace
├── 00_START_HERE.md                  ← 지금 이 파일 (읽는 순서·경로)
├── RUN_GUIDE.md                      ← 최초 세팅(§0) + 환경 + 케이스별 실행 명령어
├── TODO_GUIDE.md                     ← 채운 빈칸 해설 + 빈칸 외 수정 사유 + 주의사항
├── VARIABLES.md                      ← 변수 설명서
├── Dockerfile                        ← Python 3.12 / PyTorch 2.6.0 / Java 21 환경 (새로 추가)
├── check_env.py                      ← 세팅 점검 스크립트 (새로 추가)
├── run_case.py                       ← 케이스별 실행 스크립트, main.ipynb는 수정 안 함 (새로 추가)
├── .env                              ← (서버에서 직접 생성) HF_TOKEN·JUPYTER_TOKEN — 공유·제출 금지
│
├── defaultproject_handout.pdf        과제 설명서
├── README.txt                        과제 제공 설치 안내
├── main.ipynb                        실행 노트북 (빈칸 2개 + 괄호 1개 수정)
├── model.py                          GPT-small (빈칸 10개)
├── model_rag.py                      RAG (빈칸 3개)
├── dataset/
│   ├── pretrain.py                   사전학습 데이터 (원본 그대로)
│   ├── summary.py                    요약 데이터 (빈칸 1개)
│   ├── classification.py             분류 데이터 (빈칸 1개)
│   └── rag.py                        NQ RAG 데이터 (빈칸 1개)
├── utils/                            etc.py · logger.py · metrics.py (원본 그대로)
│
└── (실행하면 서버 폴더에 생기는 것)
    ├── runs/                         run_case.py가 만든 실행 스크립트·로그 (<case>.log)
    ├── cache/                        전처리된 데이터셋 (pretrain·summary·classification)
    ├── local_cache/rag/              NQ 데이터 + BM25 인덱스 (약 8.55GB)
    ├── logs/<case>/                  train.jsonl · eval.jsonl (학습 곡선)
    ├── output/<case>/best_model/     체크포인트 (pretraining·summary·classification·rag)
    ├── output/results/               *_score.json · *_output.txt (최종 점수·예측)
    └── defaultproject_code.zip / defaultproject_supplementaries.zip   (⑥ 제출 실행 후)
```

---

## 4. 진행 체크리스트

- [ ] Llama-3.2-1B-Instruct 라이선스 동의 신청 (승인 대기가 있어 **가장 먼저**) — RUN_GUIDE §0-1
- [ ] HF Read 토큰 발급 — RUN_GUIDE §0-1
- [ ] 폴더를 서버 `~/defaultproject`로 복사 — 이 파일 §1
- [ ] `.env` 작성 → 이미지 빌드 → 캐시 볼륨 → `check_env.py`가 `READY` — RUN_GUIDE §0-2, §0-3
- [ ] ① `pretrain` 완료 (`[run_case] DONE: pretrain`) — RUN_GUIDE §3
- [ ] ② `summary` · ③ `classification` · ④ `rag` 완료
- [ ] ⑤ `zeroshot_rag` 완료 (naive RAG 점수)
- [ ] 결과·학습 곡선 정리 (`output/results/`, `logs/`) — RUN_GUIDE §4
- [ ] 4.2.1 baseline RAG 개선 · 4.3 확장 (직접 구현, `ModelRAG`를 상속한 새 클래스로)
- [ ] `README` 작성 → ⑥ `submission` → `defaultproject_code.zip` · `defaultproject_supplementaries.zip` · 보고서 PDF 제출
