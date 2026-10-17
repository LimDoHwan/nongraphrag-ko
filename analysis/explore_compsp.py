"""탐색적 추가 실험(논문 가설 검정 이후, 2026-10-17): compsp·gazcompsp 결과표 재생성.

compsp    : comp 엔티티(연속 명사 결합) + 공백으로 나뉜 어절 조각.
            예) "미망인 마로지아" → 미망인마로지아, 미망인, 마로지아 / "농업재해보험심의회" → 그대로 하나
gazcompsp : gazcomp(사전 최장 일치 우선) + 사전에 안 잡힌 명사에 compsp 규칙. 사전은 논문의 PMI·4B·31B 증류 사전.

부록 B 41개 시스템과 분리된 목록(analysis/systems_explore.csv)을 쓴다. 응답 생성·판정 조건은 논문의
10개 비교 시스템(◆)과 같다(gemma-4-31B-it, QA_SYS, 출력 한도 2,000, gpt-4o 판정).
실험 코드: scripts/explore_compsp/ (README 참조).

실행: python analysis/explore_compsp.py
"""
import csv
import json

from common import RESULTS, ROOT, load, mcnemar, fmt_p, rate, system

DATASETS = ["wiki_ko", "admin_single", "admin_full", "admin_hard"]
EXPLORE = list(csv.DictReader(open(ROOT / "analysis" / "systems_explore.csv", encoding="utf-8")))
EX = {r["name"]: r for r in EXPLORE}
BASE = ["comp", "gazcomp(PMI)", "gazcomp(4B 증류 NER+RE)", "gazcomp(31B 증류 NER+RE)", "noun",
        "HippoRAG2 (31B 그래프)", "comp +dense0.5 +PRF(문장)"]
SAME_DICT = {"gazcompsp(PMI)": "gazcomp(PMI)", "gazcompsp(4B 증류 NER+RE)": "gazcomp(4B 증류 NER+RE)",
             "gazcompsp(31B 증류 NER+RE)": "gazcomp(31B 증류 NER+RE)"}


def row(name):
    return EX[name] if name in EX else system(name)


def table(metric, datasets):
    print(f"\n### {metric}\n")
    print("| 시스템 | " + " | ".join(datasets) + " |")
    print("|:--|" + "--:|" * len(datasets))
    for name in [r["name"] for r in EXPLORE] + BASE:
        print(f"| {name} | " + " | ".join(f"{rate(load(row(name), d), metric):.3f}" for d in datasets) + " |")


def compare(a, b, metric, datasets):
    cells = []
    for d in datasets:
        _, _, bb, cc, diff, _, p = mcnemar(load(row(a), d), load(row(b), d), metric)
        cells.append(f"{diff:+.1f}%p ({bb}/{cc}, p={fmt_p(p)})")
    print(f"| {a} 대 {b} | " + " | ".join(cells) + " |")


def main():
    print("## 결과표 (GPT-Acc = gpt-4o 판정, R@5 는 admin 만)")
    table("acc", DATASETS)
    table("contain", DATASETS)
    table("r5", DATASETS[1:])

    for metric, ds in (("acc", DATASETS), ("r5", DATASETS[1:])):
        print(f"\n## 짝 비교 ({metric}, 정확 McNemar, 괄호 안은 b/c)\n")
        print("| 비교 | " + " | ".join(ds) + " |")
        print("|:--|" + ":--|" * len(ds))
        for b in ("comp", "noun", "HippoRAG2 (31B 그래프)", "comp +dense0.5 +PRF(문장)"):
            compare("compsp", b, metric, ds)
        for a, b in SAME_DICT.items():                      # 공백 조각의 효과(같은 사전)
            compare(a, b, metric, ds)
        for b in ("gazcompsp(4B 증류 NER+RE)", "gazcompsp(31B 증류 NER+RE)"):   # compsp 위의 H2-b
            compare("gazcompsp(PMI)", b, metric, ds)

    print("\n## 검색 조정(dense0.5 + PRF(문장)) R@5 — 검색만 실행\n")
    print("| 집합 | compsp + 조정 | compsp | comp + 조정 | comp |")
    print("|:--|--:|--:|--:|--:|")
    for d in DATASETS[1:]:
        rows = json.load(open(RESULTS / f"linearrag-new-compsp-pnw0.5-prf0.6sent_{d}" / "predictions.json",
                              encoding="utf-8"))
        r5 = sum(set(map(str, r["supporting"])) <= set(map(str, r["retrieved"][:5])) for r in rows) / len(rows)
        vals = [rate(load(row(n), d), "r5") for n in ("compsp", "comp +dense0.5 +PRF(문장)", "comp")]
        print(f"| {d} | {r5:.3f} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |")

    print("\n## 그래프 연결성 (comp 대 compsp)\n")
    rows = list(csv.DictReader(open(RESULTS / "explore_compsp_connectivity.csv", encoding="utf-8")))
    print("| 집합 | 추출 | 고유 엔티티 | 다리(문서 층) | 다리(문장 층) | 허브 끈 | 끈 없는 언급 | 문서당 엔티티 |")
    print("|:--|:--|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        print(f"| {r['dataset']} | {r['extract']} | {int(r['unique_entities']):,} | {int(r['bridge_entities']):,} | "
              f"{int(r['bridge_entities_sentence_layer']):,} | {float(r['hub_edge_ratio']):.1%} | "
              f"{float(r['unlinked_mention_ratio']):.1%} | {float(r['entities_per_doc']):.1f} |")


if __name__ == "__main__":
    main()
