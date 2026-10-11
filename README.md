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
analysis/        표 재생성 스크립트, 시스템 목록(systems.csv), 논문 수록값(reference/)
data/
  2wiki_ko/      wiki_ko: 2WikiMultiHopQA 한국어 번역본 (Apache-2.0, LICENSE·NOTICE 참조)
  wiki_ko/       wiki_ko 엔티티 사전
  admin_ko/      admin: 합성 질의, 엔티티 사전, 코퍼스 문서 ID 목록, 체크섬 (원문 없음)
results/track_02/  시스템×데이터셋별 eval.json, predictions.json (41개 시스템 × 4개 집합 + hard2k 5개)
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
python master_table.py        # 부록 B <표 41>: 41개 시스템 × 4개 집합, 논문 수록값과 자동 대조
python hypothesis_tests.py    # 제4장 <표 20>~<표 24>, <표 31>: 정확 McNemar 검정, 95% 신뢰구간
python cost.py                # 제4장 <표 26>, <표 27>, <표 29>, <표 30>: 토큰, GPU-시간, ICER
```

- `analysis/systems.csv`: 논문의 시스템 이름과 결과 폴더, 실행 설정, 사용한 엔티티 사전 파일을 잇는 목록입니다. `main10 = 1`은 가설 검정에 쓴 10개 비교 시스템(응답 생성 조건 동일)입니다.
- 비용 표의 입력(`analysis/cost_inputs.csv`)에 대한 설명은 다음과 같습니다.
  - 입력 토큰: 실제 요청을 재구성해 센 정확값
  - 출력 토큰: 저장된 추출 결과로 계산한 하한값
  - 처리 시간: 동시 16요청 기준의 잠정치(논문 제3장 제4절)

## 2. 검색 재실행

### wiki_ko

번역본이 저장소에 있으므로 바로 실행할 수 있습니다.

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
python scripts/run_system.py --list
python scripts/run_system.py --system "noun" --dataset wiki_ko
```

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
| HippoRAG2 (31B) | `python scripts/run_system.py --system "HippoRAG2 (31B 그래프)" --dataset admin_full` | OpenIE를 문서당 2회 호출 (admin full 21,108회) |
| HippoRAG2 (4B) | `python -m src.track_02.graphrag.openie_4b --dataset admin_full` 후 run_system | wiki_ko는 `openie_4b_rerun.py`로 다시 추출한 판(`4bv2`) 사용 |
| LLM 증류 사전 | `python -m src.track_02.data.distill_glossary --openie <OpenIE 캐시> --mode nerre --out <사전>` | HippoRAG2의 OpenIE 결과에서 엔티티 표층형만 회수 |
| 통계(PMI) 사전 | `python -m src.track_02.data.build_glossary` | 평가 집합의 정답 문서를 뺀 문서에서 구축 |

### admin full·hard 질의 합성 (참고)

합성 질의는 `data/admin_ko/`에 파일로 들어 있으므로 다시 만들 필요가 없습니다. 만든 절차는 다음과 같습니다.

```bash
python -m src.track_02.data.build_bridge_index     # 연속 명사 결합으로 연결어 후보 색인
python -m src.track_02.data.synthesize_anchored     # 문서쌍 선택(규칙) + 질의 작성(gemma-4-31B-it) + 규칙 검증
python -m src.track_02.data.filter_adversarial --distractors 10000 --topk 5   # 밀집 검색이 정답 문서를 못 찾는 질의만 → hard
```

- 연결어 후보는 실험 조건 comp와 같은 규칙(연속 명사 결합)으로 뽑았습니다. 이 점이 결과에 주는 영향은 `scripts/bridge_strata.py`로 확인할 수 있습니다(논문 제4장 추가 분석).
- 방해 문서는 원천 풀에서 무작위로 뽑았습니다(시드 42).

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
- **LinearRAG도 원 저장소 코드가 아니라 원논문 구현을 따라 재구현했습니다**(`src/track_02/linearrag/engine.py`).
- **gpt-4o 판정자는 모델 별칭으로 호출했습니다.** 10개 비교 시스템을 판정한 2026-09-15에 응답한 모델은 `gpt-4o-2024-08-06`입니다.

## 원고 부록 E의 파일 해시와 공개본

논문 부록 E <표 59>의 파일과 이 저장소의 대응입니다. 해시는 SHA-256 앞 12자리입니다.

| 부록 E 파일 | 공개본 경로 | 부록 E 해시 | 공개본 해시 | 차이 |
|:--|:--|:--|:--|:--|
| `hipporag2_korean.py` | `src/track_01/graph_rag/hipporag2_korean.py` | `9d2fd188b65e` | `9d2fd188b65e` | 같음 |
| `graphrag/hipporag2.py` | `src/track_02/graphrag/hipporag2.py` | `50f83806b21c` | `efdb985ef65e` | 부록 E의 해시를 기록한 뒤인 2026-09-14에 질문 임베딩을 한 번에 미리 계산해 두는 방식(`compute_query_vecs`, `qvecs.npz`)으로 실행 방식만 바꿨다. 같은 bge-m3 모델·정규화를 쓰며 검색 알고리즘은 같다. 공개본은 실험 작업본과 같은 파일이다 |
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

## 데이터 출처와 라이선스

- **wiki_ko**: 2WikiMultiHopQA(Ho et al., 2020, Apache-2.0)의 1,000문항 서브셋을 LinearRAG 배포본(`Zly0523/linear-rag`, 커밋 `f7971bb8`)에서 받아 한국어로 기계 번역했습니다. `data/2wiki_ko/LICENSE`와 `NOTICE`를 참조하세요.
- **admin**: 이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 '행정 문서 대상 기계독해 데이터'를 활용하여 수행된 연구입니다. 본 연구에 활용된 데이터는 AI 허브(aihub.or.kr)에서 다운로드할 수 있습니다.
  This research used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'. All data information can be accessed through 'AI-Hub (www.aihub.or.kr)'.
- **코드**: Apache License 2.0 (`LICENSE`). 데이터의 라이선스는 위 출처별 조건을 따릅니다. 특히 admin 관련 파일(합성 질의, 사전, 결과)을 쓸 때는 AI-Hub 출처 표기가 필요합니다.
