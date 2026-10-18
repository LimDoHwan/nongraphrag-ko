"""탐색적 추가 실험(논문 가설 검정 이후, 2026-10-17~18): compsp·gazcompsp·compsp+검색 조정 결과표 재생성.

compsp    : comp 엔티티(연속 명사 결합) + 공백으로 나뉜 어절 조각.
            예) "미망인 마로지아" → 미망인마로지아, 미망인, 마로지아 / "농업재해보험심의회" → 그대로 하나
gazcompsp : gazcomp(사전 최장 일치 우선) + 사전에 안 잡힌 명사에 compsp 규칙. 사전은 논문의 PMI·4B·31B 증류 사전.
compsp +조정 : comp와 같은 검색 조정 6종(dense-seed 0.5/1.0 × PRF 없음/문서/문장, β=0.6).

부록 B 41개 시스템과 분리된 목록(analysis/systems_explore.csv)을 쓴다. 응답 생성·판정 조건은 논문의
10개 비교 시스템(◆)과 같다(gemma-4-31B-it, QA_SYS, 출력 한도 2,000, gpt-4o 판정). 단 compsp +조정 6종의 출력 한도는
짝이 되는 comp +조정 행과 같게 했다(dense0.5 +PRF(문장)만 2,000, 나머지 256). 그래서 이 5개 행의 GPT-Acc·Contain은
같은 조정의 comp 행과만 비교하고, compsp 기본과는 R@5(응답과 무관)로 비교한다.
실험 코드: scripts/explore_compsp/ (README 참조).

원고 대응(2026-10-19 번호): <표 25>, 부록 B <표 53> ☆행, 부록 C <표 65>, <표 67>.

실행: python analysis/explore_compsp.py
"""
import csv

from common import RESULTS, ROOT, load, mcnemar, fmt_p, rate, system

DATASETS = ["wiki_ko", "admin_single", "admin_full", "admin_hard"]
EXPLORE = list(csv.DictReader(open(ROOT / "analysis" / "systems_explore.csv", encoding="utf-8")))
EX = {r["name"]: r for r in EXPLORE}
BASE = ["comp", "gazcomp(PMI)", "gazcomp(4B 증류 NER+RE)", "gazcomp(31B 증류 NER+RE)", "noun",
        "HippoRAG2 (31B 그래프)"]
STACKS = ["+dense0.5", "+dense1.0", "+dense0.5 +PRF(문서)", "+dense1.0 +PRF(문서)", "+dense0.5 +PRF(문장)",
          "+dense1.0 +PRF(문장)"]
BASE += [f"comp {t}" for t in STACKS]
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
        for b in ("gazcompsp(4B 증류 NER+RE)", "gazcompsp(31B 증류 NER+RE)"):   # compsp 위의 H3 (가) 사전 출처 대응쌍
            compare("gazcompsp(PMI)", b, metric, ds)

    print("\n## 검색 조정 6종: compsp +조정 대 같은 조정의 comp (GPT-Acc, 정확 McNemar)\n")
    print("| 비교 | " + " | ".join(DATASETS) + " |")
    print("|:--|" + ":--|" * len(DATASETS))
    for t in STACKS:
        compare(f"compsp {t}", f"comp {t}", "acc", DATASETS)
    for metric in ("r5",):
        print(f"\n## 검색 조정 6종 ({metric}): 같은 조정의 comp, compsp 기본과 비교\n")
        print("| 비교 | " + " | ".join(DATASETS[1:]) + " |")
        print("|:--|" + ":--|" * 3)
        for t in STACKS:
            compare(f"compsp {t}", f"comp {t}", metric, DATASETS[1:])
            compare(f"compsp {t}", "compsp", metric, DATASETS[1:])
    print("\n(◆ 조건이 같은 dense0.5 +PRF(문장)만 compsp 기본과 GPT-Acc를 직접 비교)\n")
    print("| 비교 | " + " | ".join(DATASETS) + " |")
    print("|:--|" + ":--|" * len(DATASETS))
    compare("compsp +dense0.5 +PRF(문장)", "compsp", "acc", DATASETS)

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
