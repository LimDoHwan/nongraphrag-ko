# 데이터 딕셔너리: 문항별 결과 CSV

작성 2026-11-02. 제출 자료의 "데이터 파일(CSV)"과 "변수 코드북(Data Dictionary)"에 해당한다. CSV는 공개 저장소 `nongraphRAG-ko`의 `analysis/item_level_csv.py`로 만들며, 원천은 같은 저장소의 문항별 결과 파일이다(새 실행 없음).

## 1. 파일

| 파일 | 행 수 | 한 행의 단위 | 원고 대응 |
|:--|--:|:--|:--|
| `문항별결과_시스템.csv` | 85,782 | 집합 × 시스템 × 문항 (51개 시스템 × 1,682문항) | 제4장 <표 21>, 부록 B <표 54>. 이 시스템들 사이의 짝지은 검정(정확 McNemar)도 이 파일로 다시 계산할 수 있다 |
| `문항별결과_스윕.csv` | 60,016 | 집합 × 계열 × 문단 가중 계수 × 문항 (8개 계열 × 11단계 × 682문항) | 제4장 <표 39>·<표 40>·<그림 4>, 부록 F <표 81>~<표 83> |
| `문항별결과_스윕확인용.csv` | 2,728 | 집합 × 계열 × 계수 × 문항 (4개 점 × 682문항) | 제4장 제5절 본문의 확인용 GPT-Acc |

- 인코딩은 UTF-8(BOM 포함), 구분자는 쉼표이다. 첫 줄은 열 이름이다.
- 평가 집합의 문항 수: wiki_ko 1,000 / admin_single 300 / admin_full 279 / admin_hard 103.
- **질의·정답 문장과 모델 출력은 넣지 않았다.** admin single의 질의·정답은 AI 허브 원본 질의응답이라 재배포할 수 없으므로, 모든 파일에서 문항은 ID로만 식별한다. 문장이 필요하면 아래 3절의 파일로 연결한다.
- 시스템별 정답률은 각 파일을 `dataset`·`system`(또는 `series`·`passage_node_weight`)으로 묶어 지표 열을 평균하면 된다. R@5는 빈칸을 뺀 평균이다.

## 2. 열 정의

### 2.1 `문항별결과_시스템.csv`

| 열 | 자료형 | 정의 | 코딩·단위 | 빈칸 |
|:--|:--|:--|:--|:--|
| `dataset` | 문자 | 평가 집합 | `wiki_ko`, `admin_single`, `admin_full`, `admin_hard` (원고의 wiki_ko, 단일 도약 집합, 다중 도약 혼합 집합, 탐색 필수 집합) | 없음 |
| `system_no` | 문자 | 시스템 번호 | `1`~`41` = `analysis/systems.csv`의 `no`(부록 B <표 54>의 행 순서), `E1`~`E10` = `analysis/systems_explore.csv`의 행 순서(☆ 탐색 조건) | 없음 |
| `system` | 문자 | 시스템 이름 | 부록 B <표 54>의 행 이름에서 ◆·☆ 표시를 뺀 것. 원고 이름과의 대응은 4절 | 없음 |
| `system_group` | 문자 | 시스템 구분 | `비교 시스템(◆)` = 원고의 10개 비교 시스템 / `기타 시스템` = 부록 B의 나머지 31개 / `탐색 조건(☆)` | 없음 |
| `answer_max_tokens` | 정수 | 응답 생성의 출력 한도(토큰) | `2000` = ◆와 같은 응답 조건 / `256` = 일부 답이 잘린 조건이라 GPT-Acc·Contain은 참고값(부록 B <표 54> 주) | 없음 |
| `question_id` | 문자 | 문항 ID | 3절 | 없음 |
| `gpt_acc` | 정수(0/1) | GPT-Acc: gpt-4o(별칭 호출. 세부 버전은 2026-09-14~15 판정분만 `gpt-4o-2024-08-06`으로 확인)가 응답이 정답과 의미상 일치한다고 판정했는지. 판정 시점은 2026년 6월부터 10월 사이로 문항마다 다를 수 있다 | 1 = 정답, 0 = 오답 | 없음 |
| `contain` | 정수(0/1) | Contain: 조사를 떼고 정규화한 정답 문자열이 응답 안에 들어 있는지 | 1 = 포함, 0 = 미포함 | 없음 |
| `r5` | 정수(0/1) | R@5: 문항의 정답 근거 문서가 **모두** 검색 상위 5개 안에 있는지 | 1 = 모두 포함, 0 = 아님 | **wiki_ko는 모든 행이 빈칸.** 번역 코퍼스에서 정답 근거 문서 대응을 산출할 수 없다(제3장 제2절). admin_hard의 Vanilla RAG는 적대적 필터의 정의상 모두 0이며 빈칸이 아니다 |

### 2.2 `문항별결과_스윕.csv` (검색만 실행, LLM 응답 없음)

| 열 | 자료형 | 정의 | 코딩·단위 | 빈칸 |
|:--|:--|:--|:--|:--|
| `dataset` | 문자 | 평가 집합 | `admin_single`, `admin_full`, `admin_hard` | 없음 |
| `series` | 문자 | 계열 | 부록 F <표 81>~<표 83>의 열 이름 그대로(5절) | 없음 |
| `base_system` | 문자 | 계열이 쓰는 추출 방식의 시스템 이름 | 2.1의 `system` 값. 계수 0.05가 이 시스템의 기본 설정이다 | 없음 |
| `prf_sentence` | 정수(0/1) | PRF(문장) 사용 여부 | 1 = 사용, 0 = 사용하지 않음 | 없음 |
| `passage_node_weight` | 실수 | 문단 가중 계수 | 0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0 (원논문 기본값 0.05) | 없음 |
| `question_id` | 문자 | 문항 ID | 3절 | 없음 |
| `r5` | 정수(0/1) | R@5 (2.1과 같은 정의) | 1 = 모두 포함, 0 = 아님 | 없음 |

계수 0.05의 행은 스윕 실행에서 함께 낸 값이다. 그래서 `문항별결과_시스템.csv`의 기본 행과 최대 1문항 다를 수 있다(부록 F <표 81> 주).

### 2.3 `문항별결과_스윕확인용.csv`

열은 `dataset`, `series`, `passage_node_weight`, `question_id`, `gpt_acc`, `contain`, `r5`이며 정의는 2.1·2.2와 같다. 응답 조건은 ◆와 같다(출력 한도 2,000). 빈칸은 없다.

## 3. 문항 ID

| 집합 | ID 형식 | 문항 정보가 있는 파일(공개 저장소) |
|:--|:--|:--|
| wiki_ko | 2WikiMultiHopQA의 문항 ID(32자리 16진수) | `data/2wiki_ko/questions.json`의 `id` (한국어 번역 질의·정답 포함) |
| admin_single | `sgl_0`~`sgl_299` | `data/admin_ko/questions_admin_single_ids.json`: (문항 ID, AI 허브 문서 ID, 원본 질의응답 순번). 질의·정답 문장은 AI 허브 원천 데이터에서 복원한다(README) |
| admin_full | `anc_0`~`anc_464` 사이의 비연속 279개 | `data/admin_ko/questions_anchored.json`의 `id` (합성 질의) |
| admin_hard | `admh_0`~`admh_102` | `data/admin_ko/questions_admin_hard.json`의 `id` (합성 질의) |

## 4. 시스템 이름과 원고 조건 이름의 대응, 원천 파일

원천 파일은 공개 저장소의 `results/track_02/{폴더}/predictions.json`이다. 열의 대응: `gpt4o` → `gpt_acc`, `contain` → `contain`, `supporting`·`retrieved` → `r5`. `{집합}`은 `admin_single`, `admin_full`, `admin_hard`이다.

| system_no | system | system_group | answer_max_tokens | 부록 B <표 54> 행 이름 | 제4장 <표 21> 행 이름 | wiki_ko 폴더 | admin 폴더 |
|:--|:--|:--|--:|:--|:--|:--|:--|
| 1 | Vanilla RAG | 비교 시스템(◆) | 2000 | Vanilla RAG ◆ | Vanilla RAG | `vanillarag_wiki_ko` | `vanillarag_{집합}` |
| 2 | HippoRAG2 (31B 그래프) | 비교 시스템(◆) | 2000 | HippoRAG2 (31B 그래프) ◆ | HippoRAG2 (31B 그래프) | `graphrag-org_wiki_ko` | `graphrag-org_{집합}` |
| 3 | HippoRAG2 (4B 그래프) | 비교 시스템(◆) | 2000 | HippoRAG2 (4B 그래프) ◆ | HippoRAG2 (4B 그래프) | `graphrag-4bv2_wiki_ko` | `graphrag-4b_{집합}` |
| 4 | nnp (원논문 대응) | 비교 시스템(◆) | 2000 | nnp (원논문 대응) ◆ | nnp (원논문 대응) | `linearrag-org-nnp_wiki_ko_v2` | `linearrag-org-nnp_{집합}` |
| 5 | noun | 비교 시스템(◆) | 2000 | noun ◆ | noun | `linearrag-new-noun_wiki_ko_v2` | `linearrag-new-noun_{집합}` |
| 6 | comp | 비교 시스템(◆) | 2000 | comp ◆ | comp | `linearrag-new-comp_wiki_ko_v2` | `linearrag-new-comp_{집합}` |
| 7 | dict | 기타 시스템 | 256 | dict | — | `linearrag-new-dict_wiki_ko_v2` | `linearrag-new-dict_{집합}` |
| 8 | compdict | 기타 시스템 | 256 | compdict | — | `linearrag-new-compdict_wiki_ko_v2` | `linearrag-new-compdict_{집합}` |
| 9 | gazcomp(PMI) | 비교 시스템(◆) | 2000 | gazcomp(PMI) ◆ | gazcomp (PMI) | `linearrag-new-gazcomp_wiki_ko_v2` | `linearrag-new-gazcomp_{집합}` |
| 10 | gazcomp-sparse(PMI) | 기타 시스템 | 256 | gazcomp-sparse(PMI) | — | `linearrag-new-gazcomp-sparse_wiki_ko_v2` | `linearrag-new-gazcomp-sparse_{집합}` |
| 11 | gazet(PMI) | 기타 시스템 | 256 | gazet(PMI) | — | `linearrag-new-gazet_wiki_ko_v2` | `linearrag-new-gazet_{집합}` |
| 12 | gazcomp(31B 증류 NER) | 기타 시스템 | 256 | gazcomp(31B 증류 NER) | — | `linearrag-new-gazcomp-llmner_wiki_ko_v2` | `linearrag-new-gazcomp-llmner_{집합}` |
| 13 | gazcomp(31B 증류 NER+RE) | 비교 시스템(◆) | 2000 | gazcomp(31B 증류 NER+RE) ◆ | gazcomp (31B 증류) | `linearrag-new-gazcomp-llm_wiki_ko_v2` | `linearrag-new-gazcomp-llm_{집합}` |
| 14 | gazcomp(4B 증류 NER) | 기타 시스템 | 256 | gazcomp(4B 증류 NER) | — | `linearrag-new-gazcomp-llm4bner_wiki_ko_v2` | `linearrag-new-gazcomp-llm4bner_{집합}` |
| 15 | gazcomp(4B 증류 NER+RE) | 비교 시스템(◆) | 2000 | gazcomp(4B 증류 NER+RE) ◆ | gazcomp (4B 증류) | `linearrag-new-gazcomp-llm4bv2_wiki_ko_v2` | `linearrag-new-gazcomp-llm4b_{집합}` |
| 16 | kw(TF-IDF×TextRank k5) | 기타 시스템 | 256 | kw(TF-IDF×TextRank k5) | — | `linearrag-new-kw-k5_wiki_ko_v2` | `linearrag-new-kw-k5_{집합}` |
| 17 | kw(TF-IDF×TextRank k10) | 기타 시스템 | 256 | kw(TF-IDF×TextRank k10) | — | `linearrag-new-kw-k10_wiki_ko_v2` | `linearrag-new-kw-k10_{집합}` |
| 18 | kw(TF-IDF×TextRank k20) | 기타 시스템 | 256 | kw(TF-IDF×TextRank k20) | — | `linearrag-new-kw-k20_wiki_ko_v2` | `linearrag-new-kw-k20_{집합}` |
| 19 | kw(TF-IDF×TextRank k30) | 기타 시스템 | 256 | kw(TF-IDF×TextRank k30) | — | `linearrag-new-kw-k30_wiki_ko_v2` | `linearrag-new-kw-k30_{집합}` |
| 20 | gazcomp(병합 PMI∪31B) | 기타 시스템 | 256 | gazcomp(병합 PMI∪31B) | — | `linearrag-new-gazcomp-m31_wiki_ko_v2` | `linearrag-new-gazcomp-m31_{집합}` |
| 21 | gazcomp(병합 PMI∪4B) | 기타 시스템 | 256 | gazcomp(병합 PMI∪4B) | — | `linearrag-new-gazcomp-m4_wiki_ko_v2` | `linearrag-new-gazcomp-m4_{집합}` |
| 22 | gazcomp(PMI) +dense0.5 | 기타 시스템 | 256 | gazcomp(PMI) +dense0.5 | — | `linearrag-new-gazcomp-pnw0.5_wiki_ko_v2` | `linearrag-new-gazcomp-pnw0.5_{집합}` |
| 23 | gazcomp(PMI) +dense1.0 | 기타 시스템 | 256 | gazcomp(PMI) +dense1.0 | — | `linearrag-new-gazcomp-pnw1_wiki_ko_v2` | `linearrag-new-gazcomp-pnw1_{집합}` |
| 24 | comp +dense0.5 | 기타 시스템 | 256 | comp +dense0.5 | — | `linearrag-new-comp-pnw0.5_wiki_ko_v2` | `linearrag-new-comp-pnw0.5_{집합}` |
| 25 | comp +dense1.0 | 기타 시스템 | 256 | comp +dense1.0 | — | `linearrag-new-comp-pnw1_wiki_ko_v2` | `linearrag-new-comp-pnw1_{집합}` |
| 26 | comp +dense0.5 +PRF(문서) | 기타 시스템 | 256 | comp +dense0.5 +PRF(문서) | — | `linearrag-new-comp-pnw0.5-prf0.6_wiki_ko_v2` | `linearrag-new-comp-pnw0.5-prf0.6_{집합}` |
| 27 | comp +dense1.0 +PRF(문서) | 기타 시스템 | 256 | comp +dense1.0 +PRF(문서) | — | `linearrag-new-comp-pnw1-prf0.6_wiki_ko_v2` | `linearrag-new-comp-pnw1-prf0.6_{집합}` |
| 28 | comp +dense0.5 +PRF(문장) | 비교 시스템(◆) | 2000 | comp +dense0.5 +PRF(문장) ◆ | comp + dense0.5 + PRF(문장) | `linearrag-new-comp-pnw0.5-prf0.6sent_wiki_ko_v2` | `linearrag-new-comp-pnw0.5-prf0.6sent_{집합}` |
| 29 | comp +dense1.0 +PRF(문장) | 기타 시스템 | 256 | comp +dense1.0 +PRF(문장) | — | `linearrag-new-comp-pnw1-prf0.6sent_wiki_ko_v2` | `linearrag-new-comp-pnw1-prf0.6sent_{집합}` |
| 30 | gazcomp(PMI) +dense0.5 +PRF(문장) | 기타 시스템 | 256 | gazcomp(PMI) +dense0.5 +PRF(문장) | — | `linearrag-new-gazcomp-pnw0.5-prf0.6sent_wiki_ko_v2` | `linearrag-new-gazcomp-pnw0.5-prf0.6sent_{집합}` |
| 31 | gazcomp(PMI) +dense1.0 +PRF(문장) | 기타 시스템 | 256 | gazcomp(PMI) +dense1.0 +PRF(문장) | — | `linearrag-new-gazcomp-pnw1-prf0.6sent_wiki_ko_v2` | `linearrag-new-gazcomp-pnw1-prf0.6sent_{집합}` |
| 32 | gazcomp(4B 증류 NER+RE) +dense0.5 +PRF(문장) | 기타 시스템 | 256 | gazcomp(4B 증류 NER+RE) +dense0.5 +PRF(문장) | — | `linearrag-new-gazcomp-llm4bv2-pnw0.5-prf0.6sent_wiki_ko_v2` | `linearrag-new-gazcomp-llm4b-pnw0.5-prf0.6sent_{집합}` |
| 33 | gazcomp(31B 증류 NER+RE) +dense0.5 +PRF(문장) | 기타 시스템 | 256 | gazcomp(31B 증류 NER+RE) +dense0.5 +PRF(문장) | — | `linearrag-new-gazcomp-llm-pnw0.5-prf0.6sent_wiki_ko_v2` | `linearrag-new-gazcomp-llm-pnw0.5-prf0.6sent_{집합}` |
| 34 | gazcomp(PMI) +dense0.5 +PRF(문서) | 기타 시스템 | 256 | gazcomp(PMI) +dense0.5 +PRF(문서) | — | `linearrag-new-gazcomp-pnw0.5-prf0.6_wiki_ko_v2` | `linearrag-new-gazcomp-pnw0.5-prf0.6_{집합}` |
| 35 | gazcomp(PMI) +dense1.0 +PRF(문서) | 기타 시스템 | 256 | gazcomp(PMI) +dense1.0 +PRF(문서) | — | `linearrag-new-gazcomp-pnw1-prf0.6_wiki_ko_v2` | `linearrag-new-gazcomp-pnw1-prf0.6_{집합}` |
| 36 | gazcomp(4B 증류 NER+RE) +dense0.5 +PRF(문서) | 기타 시스템 | 256 | gazcomp(4B 증류 NER+RE) +dense0.5 +PRF(문서) | — | `linearrag-new-gazcomp-llm4bv2-pnw0.5-prf0.6_wiki_ko_v2` | `linearrag-new-gazcomp-llm4b-pnw0.5-prf0.6_{집합}` |
| 37 | gazcomp(31B 증류 NER+RE) +dense0.5 +PRF(문서) | 기타 시스템 | 256 | gazcomp(31B 증류 NER+RE) +dense0.5 +PRF(문서) | — | `linearrag-new-gazcomp-llm-pnw0.5-prf0.6_wiki_ko_v2` | `linearrag-new-gazcomp-llm-pnw0.5-prf0.6_{집합}` |
| 38 | gazcomp(4B 증류 NER+RE) +dense0.5 | 기타 시스템 | 256 | gazcomp(4B 증류 NER+RE) +dense0.5 | — | `linearrag-new-gazcomp-llm4bv2-pnw0.5_wiki_ko_v2` | `linearrag-new-gazcomp-llm4b-pnw0.5_{집합}` |
| 39 | gazcomp(4B 증류 NER+RE) +dense1.0 | 기타 시스템 | 256 | gazcomp(4B 증류 NER+RE) +dense1.0 | — | `linearrag-new-gazcomp-llm4bv2-pnw1_wiki_ko_v2` | `linearrag-new-gazcomp-llm4b-pnw1_{집합}` |
| 40 | gazcomp(31B 증류 NER+RE) +dense0.5 | 기타 시스템 | 256 | gazcomp(31B 증류 NER+RE) +dense0.5 | — | `linearrag-new-gazcomp-llm-pnw0.5_wiki_ko_v2` | `linearrag-new-gazcomp-llm-pnw0.5_{집합}` |
| 41 | gazcomp(31B 증류 NER+RE) +dense1.0 | 기타 시스템 | 256 | gazcomp(31B 증류 NER+RE) +dense1.0 | — | `linearrag-new-gazcomp-llm-pnw1_wiki_ko_v2` | `linearrag-new-gazcomp-llm-pnw1_{집합}` |
| E1 | compsp | 탐색 조건(☆) | 2000 | ☆ compsp | (탐색 조건) compsp | `linearrag-new-compsp_wiki_ko_v2` | `linearrag-new-compsp_{집합}` |
| E2 | gazcompsp(PMI) | 탐색 조건(☆) | 2000 | ☆ gazcompsp(PMI) | — | `linearrag-new-gazcompsp_wiki_ko_v2` | `linearrag-new-gazcompsp_{집합}` |
| E3 | gazcompsp(4B 증류 NER+RE) | 탐색 조건(☆) | 2000 | ☆ gazcompsp(4B 증류 NER+RE) | — | `linearrag-new-gazcompsp_llm4b_wiki_ko_v2` | `linearrag-new-gazcompsp_llm4b_{집합}` |
| E4 | gazcompsp(31B 증류 NER+RE) | 탐색 조건(☆) | 2000 | ☆ gazcompsp(31B 증류 NER+RE) | — | `linearrag-new-gazcompsp_llm_wiki_ko_v2` | `linearrag-new-gazcompsp_llm_{집합}` |
| E5 | compsp +dense0.5 | 탐색 조건(☆) | 256 | ☆ compsp +dense0.5 | — | `linearrag-new-compsp-pnw0.5_wiki_ko_v2` | `linearrag-new-compsp-pnw0.5_{집합}` |
| E6 | compsp +dense1.0 | 탐색 조건(☆) | 256 | ☆ compsp +dense1.0 | — | `linearrag-new-compsp-pnw1_wiki_ko_v2` | `linearrag-new-compsp-pnw1_{집합}` |
| E7 | compsp +dense0.5 +PRF(문서) | 탐색 조건(☆) | 256 | ☆ compsp +dense0.5 +PRF(문서) | — | `linearrag-new-compsp-pnw0.5-prf0.6_wiki_ko_v2` | `linearrag-new-compsp-pnw0.5-prf0.6_{집합}` |
| E8 | compsp +dense1.0 +PRF(문서) | 탐색 조건(☆) | 256 | ☆ compsp +dense1.0 +PRF(문서) | — | `linearrag-new-compsp-pnw1-prf0.6_wiki_ko_v2` | `linearrag-new-compsp-pnw1-prf0.6_{집합}` |
| E9 | compsp +dense0.5 +PRF(문장) | 탐색 조건(☆) | 2000 | ☆ compsp +dense0.5 +PRF(문장) | (탐색 조건) compsp + dense0.5 + PRF(문장) | `linearrag-new-compsp-pnw0.5-prf0.6sent_wiki_ko_v2` | `linearrag-new-compsp-pnw0.5-prf0.6sent_{집합}` |
| E10 | compsp +dense1.0 +PRF(문장) | 탐색 조건(☆) | 256 | ☆ compsp +dense1.0 +PRF(문장) | — | `linearrag-new-compsp-pnw1-prf0.6sent_wiki_ko_v2` | `linearrag-new-compsp-pnw1-prf0.6sent_{집합}` |

원고 본문의 약칭: "4B 증류 사전" = gazcomp(4B 증류 NER+RE), "31B 증류 사전" = gazcomp(31B 증류 NER+RE), "대표 구성" = comp +dense0.5 +PRF(문장). 시스템 이름의 `+dense0.5`·`+dense1.0`은 문단 가중 계수 0.5·1.0을 뜻한다.

## 5. 스윕 계열과 원천 파일

| series (부록 F 열 이름) | prf_sentence | base_system | 원천 파일 |
|:--|--:|:--|:--|
| comp | 0 | comp | `results/track_02/pnw_sweep/comp_{집합}.json` |
| noun | 0 | noun | `results/track_02/pnw_sweep/noun_{집합}.json` |
| gazcomp(PMI) | 0 | gazcomp(PMI) | `results/track_02/pnw_sweep/gazcomp_{집합}.json` |
| 4B 증류 사전 | 0 | gazcomp(4B 증류 NER+RE) | `results/track_02/pnw_sweep/gazcomp_llm4b_{집합}.json` |
| 31B 증류 사전 | 0 | gazcomp(31B 증류 NER+RE) | `results/track_02/pnw_sweep/gazcomp_llm_{집합}.json` |
| compsp (탐색 조건) | 0 | compsp | `results/track_02/pnw_sweep/compsp_{집합}.json` |
| comp + PRF(문장) | 1 | comp | `results/track_02/pnw_sweep/comp_prf_{집합}.json` |
| 4B 증류 사전 + PRF(문장) | 1 | gazcomp(4B 증류 NER+RE) | `results/track_02/pnw_sweep/gazcomp_llm4b_prf_{집합}.json` |

R@5의 정답 근거 문서는 같은 집합의 Vanilla RAG 결과(`results/track_02/vanillarag_{집합}/predictions.json`의 `supporting`)에서 가져온다(`analysis/pnw_sweep.py`와 같음).

스윕 확인용 GPT-Acc의 원천 파일:

| series | passage_node_weight | 원천 파일 |
|:--|--:|:--|
| comp | 0.5 | `results/track_02/linearrag-new-comp-sweep-pnw0.5_{집합}/predictions.json` |
| comp | 1.0 | `results/track_02/linearrag-new-comp-sweep-pnw1.0_{집합}/predictions.json` |
| 4B 증류 사전 | 0.2 | `results/track_02/linearrag-new-gazcomp_llm4b-sweep-pnw0.2_{집합}/predictions.json` |
| 4B 증류 사전 | 1.0 | `results/track_02/linearrag-new-gazcomp_llm4b-sweep-pnw1.0_{집합}/predictions.json` |

## 6. 검증 (2026-11-02)

CSV를 다시 읽어 집계한 값을 원고 표와 대조하였다.

| 대조 대상 | 칸 수 | 불일치 |
|:--|--:|--:|
| 제4장 <표 21> (12행 × 7열) | 84 | 0 |
| 부록 B <표 54> (51행 × 11열) | 561 | 0 |
| 부록 F <표 81>~<표 83> (11단계 × 8계열 × 3표) | 264 | 0 |
| 공개 저장소 `analysis/reference/appendix_b.csv` (41행 × 11열) | 451 | 0 |

처음 대조할 때는 <표 54>의 ☆ 행 admin full Contain 4칸이 원고의 반올림 차이로 달랐다(172/279 = 0.6165 → 0.616, 160/279 = 0.5735 → 0.573). 원고를 고친 뒤 다시 대조하여 0칸이 되었다.

## 7. 다시 만들기

```
cd nongraphRAG-ko/analysis
python item_level_csv.py --out <출력 폴더>
```

필요한 것은 공개 저장소의 `results/track_02/`뿐이다. 스크립트는 쓴 CSV를 다시 읽어 `analysis/reference/appendix_b.csv`와 대조한 결과를 출력한다.
