# 제3자 고지 (Third-Party Notices)

이 저장소의 코드는 Apache License 2.0(`LICENSE`)으로 배포합니다. 다만 아래 부분은 제3자 저작물을 바탕으로 하므로 해당 라이선스 고지를 함께 둡니다. 데이터의 출처와 라이선스는 README의 「데이터 출처와 라이선스」와 `data/2wiki_ko/LICENSE`·`NOTICE`를 참조하세요.

## HippoRAG (OSU-NLP-Group/HippoRAG, MIT License)

- 원 저장소: https://github.com/OSU-NLP-Group/HippoRAG (라이선스 확인 시점 커밋 `d5c8329422e0a0b834a15874545cb6a74b4f9b26`)
- 이 저장소의 HippoRAG2는 공식 패키지를 쓰지 않고 원논문과 원 저장소를 따라 재구현했습니다. 그 가운데 원 저장소에서 가져온 부분은 다음과 같습니다.
  - `src/track_01/graph_rag/hipporag2_korean.py`의 OpenIE 프롬프트(`NER_SYSTEM`, `ONE_SHOT_PARA`, `ONE_SHOT_NER_OUT`, `RE_SYSTEM`, `RE_FRAME`, `ONE_SHOT_RE_OUT`): 원 저장소 `src/hipporag/prompts/templates/ner.py`와 `triple_extraction.py`를 한국어로 번역했습니다(one-shot 예시와 JSON 출력 형식 포함).
  - 질의응답 시스템 프롬프트(`src/track_01/graph_rag/hipporag2_korean.py`와 `src/track_02/linearrag/engine.py`의 `QA_SYS`): 원 저장소 `src/hipporag/prompts/templates/rag_qa_musique.py`의 시스템 프롬프트("As an advanced reading comprehension assistant …", Thought/Answer 형식)를 한국어로 번역했습니다. LinearRAG 원 저장소도 같은 영어 프롬프트를 씁니다.
  - `hipporag2_korean.py`의 기본 설정값(동의어 간선 임계값 0.8, `passage_node_weight` 0.05, damping 0.5 등)은 원 저장소의 기본값을 따랐습니다.
- OpenIE 프롬프트는 `src/track_02/graphrag/hipporag2.py`, `openie_4b.py`, `openie_4b_rerun.py`가, `engine.py`의 `QA_SYS`는 `src/track_02/common/`의 응답 재생성 스크립트와 `scripts/rerun_wiki_v2.py`, `scripts/explore_compsp/compsp_eval.py`가 가져다 씁니다.
- 사실 필터 프롬프트(`FILTER_PROMPT`)는 원 저장소의 문구를 옮긴 것이 아니라 새로 작성했습니다.
- 원고 부록 E에 기록한 파일 해시를 유지하려고 소스 파일 안에는 고지를 넣지 않고 이 파일에 둡니다.

원 저장소의 라이선스 전문은 다음과 같습니다.

```
MIT License

Copyright (c) 2025 OSU Natural Language Processing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## LinearRAG (DEEP-PolyU/LinearRAG, GPL-3.0)

- 원 저장소: https://github.com/DEEP-PolyU/LinearRAG (커밋 `5da5262231fff421f05fa8654528987cc252ac8d`)
- `src/track_02/linearrag/engine.py`: LinearRAG 원 저장소(DEEP-PolyU/LinearRAG, 커밋 5da5262, 2026-06-11 취득, GPL-3.0)의 코드를 참고해 계산식과 기본값을 맞추었으나, 원 코드를 복사하지 않고 새로 작성했습니다. 원 저장소 코드는 이 저장소에 포함되어 있지 않습니다.
- 질의응답 프롬프트(`QA_SYS`)는 LinearRAG 원 저장소도 쓰는 HippoRAG 프롬프트의 번역이며, 위 HippoRAG 절에 적었습니다.
