"""통합 결과표 생성: 부록 B의 41개 시스템 + 탐색 실험 10개 행 → RESULTS.md

문항별 결과(results/track_02/*/predictions.json)에서 GPT-Acc·R@5·Contain을 다시 계산해 표 하나로 만든다.
값은 analysis/master_table.py(부록 B 대조)·analysis/explore_compsp.py와 같은 계산이다.

실행: python analysis/integrated_table.py        # 저장소 루트의 RESULTS.md를 다시 쓴다
"""
import csv

from common import DATASETS, ROOT, load, rate, systems

COLS = [("wiki_ko", "acc"), ("wiki_ko", "contain"),
        ("admin_single", "acc"), ("admin_single", "r5"), ("admin_single", "contain"),
        ("admin_full", "acc"), ("admin_full", "r5"), ("admin_full", "contain"),
        ("admin_hard", "acc"), ("admin_hard", "r5"), ("admin_hard", "contain")]
HEAD = ["wiki GPT-Acc", "wiki Contain", "single GPT-Acc", "single R@5", "single Contain",
        "full GPT-Acc", "full R@5", "full Contain", "hard GPT-Acc", "hard R@5", "hard Contain"]
# 탐색 행 중 ◆와 같은 응답 조건(출력 한도 2,000)으로 생성한 행
EXPLORE_LONG = {"compsp", "gazcompsp(PMI)", "gazcompsp(4B 증류 NER+RE)", "gazcompsp(31B 증류 NER+RE)",
                "compsp +dense0.5 +PRF(문장)"}
# 2026-10-19 수정한 행(full·hard: 질문 엔티티 추출 사전을 바로잡아 다시 냄)
CORRECTED = {"gazcomp(4B 증류 NER+RE) +dense0.5", "gazcomp(4B 증류 NER+RE) +dense1.0",
             "gazcomp(31B 증류 NER+RE) +dense0.5", "gazcomp(31B 증류 NER+RE) +dense1.0"}

INTRO = """# 통합 결과표

이 파일은 `python analysis/integrated_table.py`로 만든다(직접 고치지 않는다). 모든 값은 `results/track_02/` 문항별 결과에서 다시 계산한 것이다.

- **행:** 논문 부록 B <표 53>의 41개 시스템(원고 순서) + 논문 가설 검정 이후의 탐색 실험 10개(☆, README 4절).
- **지표:** GPT-Acc(gpt-4o 판정), R@5(정답 근거 문서가 모두 상위 5개 안에 있으면 1. wiki_ko는 산출하지 않음), Contain(정답 문자열 포함).
- **문항 수:** wiki_ko 1,000 / admin single 300 / full 279 / hard 103.
- **◆ = 응답 생성 조건을 통일한 행**(QA 프롬프트 LinearRAG `QA_SYS`, 출력 한도 2,000토큰, temperature 0, 생성기 `gemma-4-31B-it`). 논문 비교에 쓰는 10개 시스템이다. **◆가 없는 행은 출력 한도 256토큰**으로 응답해 일부 답이 잘렸으므로 GPT-Acc·Contain은 참고값이다. R@5는 응답과 무관해 모든 행을 직접 비교할 수 있다.
- **☆** = 탐색 실험. `(2000)`은 ◆와 같은 응답 조건, `(256)`은 같은 검색 조정의 comp 행과 같은 256토큰 조건이다.
- **✎** = 2026-10-19 수정 행. admin full·hard의 질문 엔티티를 single용 증류 사전으로 뽑았던 처음 실행을 바로잡아 다시 냈다(README 6절).
- **굵게** = 열별 최고(☆ 행 포함). wiki_ko의 LinearRAG 계열은 원 구현과 같은 입력 처리(`wiki_ko_v2`)로 낸 값이다.
"""


def main():
    rows = []
    for s in systems():
        mark = " ◆" if s["main10"] == "1" else ""
        mark += " ✎" if s["name"] in CORRECTED else ""
        rows.append((s["name"] + mark, s))
    with open(ROOT / "analysis" / "systems_explore.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cond = "2000" if r["name"] in EXPLORE_LONG else "256"
            rows.append((f"☆ {r['name']} ({cond})", r))
    vals = [[rate(load(r, d), m) for d, m in COLS] for _, r in rows]
    best = [max(v[j] for v in vals) for j in range(len(COLS))]
    out = [INTRO, "| 시스템 | " + " | ".join(HEAD) + " |", "|:--|" + "--:|" * len(COLS)]
    for (name, _), v in zip(rows, vals):
        cells = [f"**{x:.3f}**" if abs(x - best[j]) < 1e-12 else f"{x:.3f}" for j, x in enumerate(v)]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    (ROOT / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"RESULTS.md: {len(rows)}행 × {len(COLS)}열")


if __name__ == "__main__":
    main()
