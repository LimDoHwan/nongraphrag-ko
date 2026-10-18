# nongraphRAG-ko

**관계 없는 그래프 RAG의 한국어 엔티티 추출 방식별 비용-효과 분석 — LLM 색인 투자는 성능으로 이어지는가**의 재현 저장소입니다.

관계 추출 없이 엔티티만으로 그래프를 만드는 LinearRAG(Zhuang et al., 2026)를 한국어 코퍼스 두 개(영어 위키 기원 `wiki_ko`, AI-Hub 행정문서 `admin`)에 적용했습니다. 이 저장소로 다음 세 가지를 비교·재현할 수 있습니다.

- 엔티티 추출 방식(고유명사·명사 전체·복합명사 결합·통계 사전·LLM 증류 사전)
- LLM 관계 그래프(HippoRAG2)와의 성능·색인 비용
- 검색 조정(dense-seed, PRF)의 효과

## 재현 수준

| 수준 | 무엇을 | 필요한 것 | 방법 |
|:--|:--|:--|:--|
| ① 표 재생성 | 논문의 결과표, 가설 검정(p값·신뢰구간), 비용 표 | Python만 (GPU·LLM·원천 데이터 불필요) | [표 재생성](#1-표-재생성) |
| ② 검색 재실행 | R@5와 검색 결과 | 임베딩 모델(bge-m3), admin은 AI-Hub 원천 | [검색 재실행](#2-검색-재실행) |
| ③ 전체 재실행 | 응답 생성·판정, HippoRAG2 색인, 증류 사전 | vLLM 서버(gemma-4-31B-it, gemma-4-E4B-it), OpenAI API | [전체 재실행](#3-전체-재실행) |

같은 코드와 설정으로 다시 실행했을 때 검색 결과는 동일하게 재현되었습니다(admin 24개 실행의 R@5 전부 일치, wiki_ko 3개 실행의 1,000문항 검색 결과 전부 일치). LLM 생성(질의 합성, 번역, 응답)은 temperature 0이어도 서빙 환경에 따라 달라질 수 있으므로, 합성 질의·번역본·문항별 결과를 파일로 함께 제공합니다.

## 저장소 구성

```
analysis/        표 재생성 스크립트, 시스템 목록(systems.csv), 논문 수록값(reference/),
                 탐색 실험 목록(systems_explore.csv)·분석(explore_compsp.py)
data/
  2wiki_ko/      wiki_ko: 2WikiMultiHopQA 한국어 번역본 (Apache-2.0, LICENSE·NOTICE 참조)
  wiki_ko/       wiki_ko 엔티티 사전
  admin_ko/      admin: 합성 질의, 엔티티 사전, 코퍼스 문서 ID 목록, 체크섬 (원문 없음)
results/track_02/  시스템×데이터셋별 eval.json, predictions.json (41개 시스템 × 4개 집합 + hard2k 5개)
                 LinearRAG 계열의 wiki_ko 결과는 {시스템}_wiki_ko_v2/ (아래 "wiki_ko_v2" 참조)
scripts/         데이터 구축·복원, 시스템 실행, 보조 분석
src/track_02/    실험 코드 (LinearRAG 재구현, HippoRAG2 재구현, Vanilla RAG, 데이터 합성)
src/track_01/    공용 모듈 (임베딩, LLM 클라이언트, 평가)
```

## 환경

- Python 3.12, Ubuntu 22.04에서 실행했습니다. 패키지 버전은 `requirements.lock.txt`에 있습니다.
- 형태소 분석기 kiwipiepy 0.23.1(모델 0.23.0), 임베딩 BAAI/bge-m3, LLM 서빙 vLLM 0.27.1을 썼습니다.

```bash
pip install -r requirements.lock.txt
cp .env.example .env            # 수준 ②·③에서 필요한 값만 채운다
```

## 1. 표 재생성

GPU와 원천 데이터 없이, 저장소에 든 문항별 결과만으로 논문의 표를 다시 만듭니다.

```bash
cd analysis
python master_table.py        # 부록 B <표 46>: 41개 시스템 × 4개 집합, 논문 수록값(수정본)과 자동 대조
python hypothesis_tests.py    # 제4장 <표 22>·<표 23>, <표 25>~<표 27>, <표 35>·<표 36>: 정확 McNemar 검정, 95% 신뢰구간
python cost.py                # 제4장 <표 30>, <표 31>, <표 33>, <표 34>: 토큰, GPU-시간, ICER
```

- `analysis/systems.csv`: 논문의 시스템 이름과 결과 폴더, 실행 설정, 사용한 엔티티 사전 파일을 잇는 목록입니다. `wiki_dataset` 열은 wiki_ko 결과 폴더의 접미사(`wiki_ko` 또는 `wiki_ko_v2`)입니다. `main10 = 1`은 가설 검정에 쓴 10개 비교 시스템(응답 생성 조건 동일)입니다.
- `analysis/reference/appendix_b.csv`: 원고 부록 B의 수록값입니다. wiki_ko 재실행으로 바뀐 LinearRAG 계열 74칸과 원고의 이중 반올림 6칸을 수정한 값(2026-10-15)이며, `master_table.py`는 이 값과 모두 일치해야 합니다.
- `results/track_02/graph_connectivity.csv`: 제4장 <표 24>의 추출 방식별 그래프 연결성 지표(고유·다리 엔티티 수, 허브 끈 비율, 끈 없는 언급 비율, 고립 문서 비율). `scripts/graph_connectivity.py`의 출력이며, 입력인 색인 캐시는 원문 조각이 들어 있어 공개하지 않으므로 결과를 파일로 제공합니다. wiki_ko는 `wiki_ko_v2` 색인 기준입니다.
- 비용 표의 입력(`analysis/cost_inputs.csv`)에 대한 설명은 다음과 같습니다.
  - 입력 토큰: 실제 요청을 재구성해 센 정확값
  - 출력 토큰: 저장된 추출 결과로 계산한 하한값
  - 처리 시간: 동시 16요청 기준의 잠정치(논문 제3장 제4절)

## 2. 검색 재실행

### wiki_ko

번역본이 저장소에 있으므로 바로 실행할 수 있습니다. LinearRAG 계열 38개 시스템은 `wiki_ko_v2`(아래)로 검색합니다.

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
python scripts/rerun_wiki_v2.py retrieve        # LinearRAG 계열 38개, 상위 5개 → cache/track_02/rerun_wiki_v2/retrieved/
python scripts/run_system.py --system "HippoRAG2 (31B 그래프)" --dataset wiki_ko   # HippoRAG2·Vanilla RAG
```

**wiki_ko_v2.** wiki_ko 문서는 원래 "번호:본문" 형태입니다. 원 LinearRAG 구현은 이 번호로 앞뒤 문서를 인접 문단 간선(무게 1.0)으로 잇고, 번호가 붙은 채로 문서를 임베딩합니다. 처음 실험한 로더(`wiki_ko`)는 번호를 떼어 이 간선이 생기지 않았기 때문에, 번호를 유지하는 `wiki_ko_v2`로 LinearRAG 계열을 다시 실행했고 논문의 wiki_ko 결과는 이것입니다(2026-10-15). 응답 생성에 넣는 지문은 두 판 모두 번호 없는 본문이므로 바뀌는 것은 검색뿐입니다. `scripts/run_system.py`로 LinearRAG를 wiki_ko에서 돌리면 이전 로더가 쓰이므로 `--legacy-wiki`를 붙여야만 실행됩니다.

### admin (AI-Hub 원천 필요)

행정문서 원문은 AI-Hub 이용정책상 재배포할 수 없어, 이 저장소에는 **문서 ID 목록과 체크섬만** 있습니다. 다음 순서로 같은 코퍼스를 복원합니다.

1. [AI-Hub](https://aihub.or.kr)에서 「행정 문서 대상 기계독해 데이터」(데이터셋 번호 569)를 신청해 내려받습니다. AI-Hub 정책상 내국인만 신청할 수 있습니다.
2. span_extraction 라벨링 데이터 두 파일을 아래 위치에 둡니다. 두 분할은 같은 문단 10,654개를 서로 다른 질의응답과 함께 중복 수록하고 있어, 문서 레코드는 75,917건이지만 고유 문단은 65,263개입니다. 복원 스크립트는 Training → Validation 순으로 읽어 같은 ID는 Validation 레코드를 씁니다(실험과 동일).

   | 경로 | MD5 |
   |:--|:--|
   | `data/admin_ko/raw/1.Training/라벨링데이터/TL_span_extraction.zip` | `74213aa4441d402aa5436686f2ba20cf` |
   | `data/admin_ko/raw/2.Validation/라벨링데이터/VL_span_extraction.zip` | `8824dd82d8a9c29e7c913bb2c592a903` |

3. 원천을 문서 단위로 풀고, 문서 ID 목록으로 코퍼스와 admin single 질의를 복원합니다.

   ```bash
   python -m src.track_02.data.load_raw
   python scripts/build_corpus_from_ids.py     # 체크섬이 모두 "일치"여야 한다
   ```

4. 시스템을 실행합니다.

   ```bash
   python scripts/run_system.py --system "comp +dense0.5 +PRF(문장)" --dataset admin_hard
   ```

검색 지표만 볼 때는 `src.track_02.linearrag.engine`에 `--retrieval-only`를 붙이면 응답 생성 없이 R@5를 계산합니다.

## 3. 전체 재실행

응답 생성, HippoRAG2의 OpenIE, 증류 사전 구축에는 LLM 서버가 필요합니다(`.env`).

| 단계 | 명령 | 비고 |
|:--|:--|:--|
| 응답 생성·판정 | `scripts/run_system.py` 실행 결과에 대해 `python -m src.track_02.common.unify_qa` → `python -m src.track_02.common.rejudge_gpt4o` | 10개 비교 시스템은 공통 프롬프트·출력 한도 2,000토큰으로 응답을 통일. 판정은 gpt-4o |
| wiki_ko LinearRAG 계열 | `python scripts/rerun_wiki_v2.py retrieve` → `qa --urls <31B 서버 주소…>` → `judge` → `assemble` | 생성 조건은 다른 시스템과 같음(◆ 출력 한도 2,000, 나머지 256, temperature 0). 결과는 `results/track_02/{시스템}_wiki_ko_v2/`. `validate`(원 저장소 코드 결과 대조)는 그 결과 파일이 저장소에 없어 건너뜀 |
| HippoRAG2 (31B) | `python scripts/run_system.py --system "HippoRAG2 (31B 그래프)" --dataset admin_full` | OpenIE를 문서당 2회 호출 (admin full 21,108회) |
| HippoRAG2 (4B) | `python -m src.track_02.graphrag.openie_4b --dataset admin_full` 후 run_system | wiki_ko는 `openie_4b_rerun.py`로 다시 추출한 판(`4bv2`) 사용 |
| LLM 증류 사전 | `python -m src.track_02.data.distill_glossary --openie <OpenIE 캐시> --mode nerre --out <사전>` | HippoRAG2의 OpenIE 결과에서 엔티티 표층형만 회수. 검색 코퍼스 전체(정답 문서 포함)의 OpenIE 결과를 쓰며, 질의·정답 정보는 사용하지 않음 |
| 통계(PMI) 사전 | `python -m src.track_02.data.build_glossary` | 평가 코퍼스 본문으로 구축. 질의·정답 정보는 사용하지 않음(구축 풀에는 평가 집합의 정답 문서도 포함됨) |

### admin full·hard 질의 합성 (참고)

합성 질의는 `data/admin_ko/`에 파일로 들어 있으므로 다시 만들 필요가 없습니다. 만든 절차는 다음과 같습니다.

```bash
python -m src.track_02.data.build_bridge_index     # 연속 명사 결합으로 연결어 후보 색인
python -m src.track_02.data.synthesize_anchored     # 문서쌍 선택(규칙) + 질의 작성(gemma-4-31B-it) + 규칙 검증
python -m src.track_02.data.filter_adversarial --distractors 10000 --topk 5   # 밀집 검색이 정답 문서를 못 찾는 질의만 → hard
```

- 연결어 후보는 실험 조건 comp와 같은 규칙(연속 명사 결합)으로 뽑았습니다. 이 점이 결과에 주는 영향은 `scripts/bridge_strata.py`로 확인할 수 있습니다(논문 제4장 추가 분석).
- 방해 문서는 원천 풀에서 무작위로 뽑았습니다(시드 42).

## 4. 탐색적 추가 실험: compsp·gazcompsp·compsp + 검색 조정

논문의 가설 검정을 마친 뒤 결과를 보고 설계한 **탐색적 실험**입니다(2026-10-17~18). 부록 B의 41개 시스템과 분리해 `analysis/systems_explore.csv`에 따로 두었고, 논문 표 재생성(`master_table.py`)에는 들어가지 않습니다.

**동기.** comp는 연속 명사를 띄어쓰기와 상관없이 묶습니다. 행정문서의 붙여 쓴 복합명사("농업재해보험심의회")에는 유리하지만, 위키의 띄어 쓴 이름("미망인 마로지아")을 "미망인마로지아"로 붙여 다른 문서의 "마로지아"와 끊어 버립니다. 그래서 wiki_ko에서 comp가 noun보다 크게 낮았습니다.

**추출 규칙.**

| 이름 | 규칙 | 예 |
|:--|:--|:--|
| compsp | comp 엔티티 + 공백으로 나뉜 어절 조각 | "미망인 마로지아" → 미망인마로지아, 미망인, 마로지아 / "농업재해보험심의회" → 그대로 하나 |
| gazcompsp | gazcomp(사전 최장 일치 우선) + 사전에 안 잡힌 명사에 compsp 규칙 | 사전은 논문의 PMI·4B 증류·31B 증류 사전 |
| compsp + 조정 | compsp에 comp와 같은 검색 조정 6종(dense-seed 0.5/1.0 × PRF 없음/문서/문장, β = 0.6) | — |

**조건.** `engine.py`는 바꾸지 않고 추출기만 바꿔 끼웠습니다. 같은 경로로 comp와 gazcomp(PMI)를 다시 돌려 저장된 결과와 상위 5개가 103/103, 102/103 일치함을 확인했습니다(다른 1문항도 R@5는 같음). 검색 설정은 논문 기본값이고, 응답 생성과 판정은 10개 비교 시스템(◆)과 같습니다(gemma-4-31B-it, QA_SYS, 출력 한도 2,000, gpt-4o). compsp + 조정 6종의 출력 한도는 짝이 되는 comp + 조정 행과 같게 했습니다(dense0.5 + PRF(문장)만 2,000, 나머지 256). 그래서 이 5개 행의 GPT-Acc는 같은 조정의 comp와만 비교하고, compsp 기본과는 R@5로 비교합니다. 조정 실행 전에 같은 경로로 comp + dense1.0 + PRF(문서)를 다시 돌려 admin_hard 상위 5개가 103/103 일치함을 확인했습니다.

**결과 재생성.**

```bash
cd analysis
python explore_compsp.py      # GPT-Acc·Contain·R@5 표, 정확 McNemar 짝 비교, 검색 조정 6종 비교, 연결성 지표
```

**요약 (GPT-Acc).**

| 시스템 | wiki_ko | admin single | admin full | admin hard |
|:--|--:|--:|--:|--:|
| compsp | 0.785 | 0.923 | 0.710 | 0.427 |
| gazcompsp(PMI) | 0.784 | 0.910 | 0.706 | 0.437 |
| compsp + dense0.5 + PRF(문장) | 0.786 | 0.937 | 0.720 | 0.417 |
| comp + dense0.5 + PRF(문장) | 0.625 | 0.907 | 0.710 | 0.427 |
| comp | 0.632 | 0.877 | 0.674 | 0.369 |
| noun | 0.796 | 0.873 | 0.602 | 0.330 |
| HippoRAG2 (31B 그래프) | 0.793 | 0.933 | 0.652 | 0.233 |

- compsp는 wiki_ko에서 comp보다 15.3%p 높고(p = 1.2×10⁻²⁹) noun·HippoRAG2와 차이가 없습니다. admin에서는 comp보다 3.6~5.8%p 높습니다(single만 유의).
- 공백 조각의 효과는 사전이 약할수록 큽니다. 같은 사전의 gazcomp 대비 wiki_ko에서는 세 사전 모두 유의하게 높지만, admin hard에서는 PMI +4.9%p, 4B·31B 증류 −3.9%p입니다(모두 유의하지 않음).
- gazcompsp에서는 통계 사전과 증류 사전의 차이가 wiki_ko에서도 유의하지 않습니다(−0.8%p).
- 검색 조정 6종 모두에서 compsp는 같은 조정의 comp보다 wiki_ko GPT-Acc가 16~18%p 높습니다(모두 p < 10⁻³⁰). admin에서는 single이 조금 높고(dense0.5 + PRF(문장)만 +3.0%p로 유의) full·hard는 차이가 유의하지 않습니다.
- 조정은 compsp의 단일 도약 R@5만 올립니다(기본 대비 +3.3~4.3%p, 모두 유의). hard R@5는 모두 내려가고, dense1.0 계열에서 크게 내려갑니다(−7.8~−8.7%p).
- compsp + dense0.5 + PRF(문장)은 compsp 기본과 네 집합 모두 차이가 유의하지 않으며(−1.0~+1.3%p), admin full GPT-Acc 0.720은 전체 시스템 중 가장 높습니다.

**다시 실행하기** (`scripts/explore_compsp/`)

| 순서 | 명령 | 하는 일 |
|:--|:--|:--|
| 1 | `python scripts/explore_compsp/conn_pieces.py wiki_ko_v2 admin_full` | comp 대 compsp 연결성 지표 (CPU, Kiwi만) |
| 2 | `python scripts/explore_compsp/run_compsp.py` | 검증(comp 재실행) 후 compsp admin 3개 집합 색인·검색 |
| 3 | `python scripts/explore_compsp/run_compsp_wiki.py` | compsp wiki_ko_v2 색인·검색 |
| 4 | `python scripts/explore_compsp/run_compsp_stack_all.py` | 검증(comp + 조정 재실행) 후 compsp + 검색 조정 6종 × 4개 집합 검색 (`run_compsp_stack.py`는 첫 1종만 하던 이전 판) |
| 5 | `python scripts/explore_compsp/run_gazcompsp.py` | 검증(gazcomp 재실행) 후 gazcompsp 사전 3종 × 4개 집합 색인·검색 |
| 6 | `LLM_URLS=<31B 서버 주소들> SYSN=<compsp\|gazcompsp\|gazcompsp_llm4b\|gazcompsp_llm> [STACK=<pnw0.5-prf0.6sent 등> MAXTOK=<2000\|256>] python scripts/explore_compsp/compsp_eval.py qa <집합...>` | 응답 생성 (조정 행은 STACK·MAXTOK 지정) |
| 7 | `SYSN=… python scripts/explore_compsp/compsp_eval.py judge <집합...>` → `report <집합...>` | gpt-4o 판정, 결과 폴더 작성 |

2~5는 bge-m3 임베딩이 필요합니다(GPU 권장). 색인은 기존 comp·gazcomp 캐시의 문장 임베딩을 재사용하므로 그 캐시가 먼저 있어야 합니다(수준 ② 검색 재실행으로 생성). 6은 기존 결과와 질문·상위 5개가 같은 문항의 답을 재사용하므로, 공개 저장소에서 처음부터 돌리면 모든 답을 새로 생성합니다.

**주의.** 논문 가설 검정 이후 결과를 보고 설계한 실험이라 사전에 정한 검정이 아니며, 비교 수가 많아 다중 비교 보정을 하지 않았습니다. 논문에 인용할 때는 탐색적 결과로 표기해야 합니다.

## 공개하지 않은 것

| 항목 | 이유 | 대신 제공하는 것 |
|:--|:--|:--|
| AI-Hub 원문, 원문이 든 검색 코퍼스 | AI-Hub 이용정책: 원본·단순 전처리본 재배포 금지 | 코퍼스별 문서 ID 목록 + 체크섬, 복원 스크립트 |
| admin single 질의·정답 텍스트와 모델 출력 | AI-Hub 원본 질의응답의 부분집합. 맞힌 모델 출력은 원본 정답과 같은 문자열이라, 문항 ID 목록과 합치면 원본 정답이 복원됨 | (문항 ID, 문서 ID, 원본 QA 순번) 목록. 결과 파일의 정오 판정(`gpt4o`, `contain`)과 검색 결과는 그대로. 출력 12,300건은 `prediction: null`, `prediction_withheld: true` |
| admin full·hard·hard2k 결과 중 추론 과정이 든 긴 모델 출력 (1,965건) | 출력 한도에 걸려 최종 답 없이 추론 과정만 남은 출력으로, 원문 인용이 들어 있을 수 있음. 기준: 200자 초과 또는 "Thought" 포함 | 정오 판정은 그대로. 해당 항목은 `prediction: null`, `prediction_withheld: true`. full·hard·hard2k의 나머지 단답 14,212건과 wiki_ko 출력 41,000건은 공개 |
| 색인 캐시(OpenIE 결과, 임베딩) | 원문 조각 포함, 용량 | 재실행으로 생성 |

## 알려진 제약

- **admin single의 선별 스크립트가 없습니다.** 300문항의 목록(`questions_admin_single_ids.json`)으로 정확히 복원할 수는 있지만, 고른 규칙은 코드로 남아 있지 않습니다. 데이터로 확인되는 사실은 다음과 같습니다. 300문항 모두 Validation 분할의 원본 질의응답이며, 문서당 1문항입니다. 카테고리 구성은 Validation 분할의 구성 비율을 따릅니다. 방해 문서 10,000개는 정답 문서를 뺀 원천 풀에서 뽑았습니다.
- **증류 사전 구축 스크립트(`distill_glossary.py`)는 복원본입니다.** 원 스크립트가 남아 있지 않아 산출물을 대조해 규칙을 복원했습니다. wiki_ko 4B 사전은 (낱말, 빈도) 집합까지 같게 재현됩니다.
- **실행 시 어떤 사전을 썼는지는 기록되지 않았습니다.** `systems.csv`의 사전 파일은 색인 캐시를 각 후보 사전으로 다시 추출해 정확히 일치하는 것을 찾아 정했습니다. compdict(admin)만은 표본 문서에서 `glossary.json`과 거의 같은 `glossary_clean.json`이 구별되지 않아, 다른 PMI 조건과 같은 `glossary.json`으로 두었습니다.
- **HippoRAG2는 공식 패키지가 아니라 원논문을 따라 재구현했습니다**(`src/track_02/graphrag/hipporag2.py`, `src/track_01/graph_rag/hipporag2_korean.py`).
- **LinearRAG도 원 저장소 코드가 아니라 원 저장소(DEEP-PolyU/LinearRAG) 구현을 따라 재구현했습니다**(`src/track_02/linearrag/engine.py`).
  - 같은 점: 씨앗 엔티티 선택, 문장 단위 활성화 전파, 문서 점수(정규화 유사도 × passage_ratio + log(1 + 엔티티 보너스)), 문서–엔티티 끈 무게(등장 횟수 비율), 개인화 페이지랭크(prpack)의 계산식이 원 저장소와 같습니다. 설정값 7개(top-5, 반복 3, 문장 1, passage_ratio 1.5, passage_node_weight 0.05, damping 0.5, 임계값 0.5)는 원 저장소 `config.py`의 기본값입니다. 원 저장소의 2wiki 실행 스크립트는 임계값 0.4·passage_ratio 0.05를 쓰는데, 이 연구는 두 코퍼스 모두 기본값을 썼습니다.
  - 입력 처리: 처음 실험한 wiki_ko 로더는 문서 앞의 번호를 떼어 원 구현의 인접 문단 간선이 생기지 않았습니다. 이를 바로잡은 `wiki_ko_v2`로 LinearRAG 계열을 다시 실행했고, 논문과 이 저장소의 wiki_ko 결과는 모두 `wiki_ko_v2` 결과입니다(위 "wiki_ko_v2"). admin 문서는 번호로 시작하지 않아 원 저장소에서도 이 간선이 생기지 않으므로 영향이 없습니다.
  - 남은 차이: 원 저장소는 엔티티 등장 횟수를 셀 때 소문자로 바꿔 셉니다(한국어에는 영향이 거의 없음). 2wiki 전용 실행값(임계값 0.4·passage_ratio 0.05)은 쓰지 않았습니다.
  - 대조: 원 저장소 코드로 낸 wiki_ko 검색 결과(같은 bge-m3·고유명사 추출·2wiki 실행값)와 상위 5개 문서를 1,000문항에서 비교했습니다. `wiki_ko_v2`에서 849문항이 순서까지 같았고 평균 겹침은 96.8%입니다. 번호를 뗀 이전 로더에서는 64문항(평균 겹침 70%)이었습니다. 계산식은 원 저장소와 같고, 차이는 입력 처리에서 왔다는 뜻입니다.
- **gpt-4o 판정자는 모델 별칭으로 호출했습니다.** 10개 비교 시스템을 판정한 2026-09-15에 응답한 모델은 `gpt-4o-2024-08-06`입니다.

## 원고 부록 E의 파일 해시와 공개본

논문 부록 E <표 67>의 파일과 이 저장소의 대응입니다. 해시는 SHA-256 앞 12자리입니다.

| 부록 E 파일 | 공개본 경로 | 부록 E 해시 | 공개본 해시 | 차이 |
|:--|:--|:--|:--|:--|
| `hipporag2_korean.py` | `src/track_01/graph_rag/hipporag2_korean.py` | `9d2fd188b65e` | `9d2fd188b65e` | 같음 |
| `graphrag/hipporag2.py` | `src/track_02/graphrag/hipporag2.py` | `50f83806b21c` | `efdb985ef65e` | 부록 E의 해시를 기록한 뒤(2026-09-14), wiki_ko의 4B 재추출판을 돌리려고 실행 옵션 `--variant 4bv2`를 추가했다. 바뀐 곳은 캐시·결과 경로, 결과 폴더 이름 표기, 옵션 선택지, OpenIE 단계 실행 조건 4곳뿐이다. 이 4곳을 되돌리면 부록 E의 해시 `50f83806b21c`가 그대로 나온다. 검색 함수(`run_retrieval_qa`)는 바뀌지 않았으므로 31B·4B 그래프 결과는 변경 전 코드와 같다. 공개본은 실험 작업본과 같은 파일이다 |
| `linearrag/engine.py` | `src/track_02/linearrag/engine.py` | `74c3c47ee54f` | `74c3c47ee54f` | 같음 |
| `vanillarag/run.py` | `src/track_02/vanillarag/run.py` | `351787976145` | `351787976145` | 같음 |
| `openie_4b.py` | `src/track_02/graphrag/openie_4b.py` | `f841466e0c45` | `abf34607b72a` | 공개 정제: 내부 서버 주소를 환경변수 `OPENIE_4B_URL`로 바꾸고 `import os` 추가. 그 밖에는 같음 |
| `rejudge_gpt4o.py` | `src/track_02/common/rejudge_gpt4o.py` | `43287b60c14c` | `43287b60c14c` | 같음 |
| BAAI/bge-m3 `pytorch_model.bin` | (포함하지 않음, Hugging Face에서 내려받음) | `b5e0ce3470ab` | — | — |

부록 E에 없는 파일 중 공개 정제로 바뀐 것은 두 개입니다.

| 파일 | 실험 작업본 해시 | 공개본 해시 | 바뀐 것 |
|:--|:--|:--|:--|
| `src/track_02/graphrag/openie_4b_rerun.py` | `e756567b8ec7` | `dcac9e661cda` | 내부 서버 주소를 환경변수 `OPENIE_4B_BASE`로, 서버 실행 옵션에서 호스트 주소 제거, `import os` 추가 |
| `scripts/translate_dataset.py` | — | `5ad23c372a16` | 원 영어 데이터 다운로드 경로를 `resolve/main`에서 취득 시점 커밋 `resolve/f7971bb84fe205d0c080df5f48b1393d9f580c68`로 고정 |

## 인용

이 저장소를 사용하면 다음 논문을 인용해 주세요. `CITATION.cff`에도 같은 정보가 있습니다.

> 임도환 (2027). 관계 없는 그래프 RAG의 한국어 엔티티 추출 방식별 비용-효과 분석 — LLM 색인 투자는 성능으로 이어지는가. 경희대학교 경영학 석사학위논문.
>
> Lim, D. (2027). *A Cost-Effectiveness Analysis of Entity Extraction Methods for Korean in Relation-Free Graph RAG: Does Investment in LLM-Based Indexing Translate into Performance?* Master's thesis, Kyung Hee University.

```bibtex
@mastersthesis{lim2027nongraphrag,
  author = {임도환},
  title  = {관계 없는 그래프 RAG의 한국어 엔티티 추출 방식별 비용-효과 분석 — LLM 색인 투자는 성능으로 이어지는가},
  school = {경희대학교},
  type   = {경영학 석사학위논문},
  year   = {2027},
  url    = {https://github.com/LimDoHwan/nongraphrag-ko}
}
```

## 데이터 출처와 라이선스

- **wiki_ko**: 2WikiMultiHopQA(Ho et al., 2020, Apache-2.0)의 1,000문항 서브셋을 LinearRAG 배포본(`Zly0523/linear-rag`, 커밋 `f7971bb8`)에서 받아 한국어로 기계 번역했습니다. `data/2wiki_ko/LICENSE`와 `NOTICE`를 참조하세요.
- **admin**: 이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 '행정 문서 대상 기계독해 데이터'를 활용하여 수행된 연구입니다. 본 연구에 활용된 데이터는 AI 허브(aihub.or.kr)에서 다운로드할 수 있습니다.
  This research used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'. All data information can be accessed through 'AI-Hub (www.aihub.or.kr)'.
- **코드**: Apache License 2.0 (`LICENSE`). 데이터의 라이선스는 위 출처별 조건을 따릅니다. 특히 admin 관련 파일(합성 질의, 사전, 결과)을 쓸 때는 AI-Hub 출처 표기가 필요합니다.
