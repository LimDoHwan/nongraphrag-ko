"""H1(새) 대 Vanilla RAG: 형태소 기반 추출 조건 대 Vanilla RAG (2026-10-31 원고 요청 R12).

H1(새): 형태소 분석기로 추출한 엔티티만으로도, 관계 없는 그래프 RAG는 그래프 없는 밀집 검색보다
다중 도약 질의의 정확도가 높을 것이다.

A 조건: 호출 0회 후보 nnp, noun, comp, gazcomp(PMI) + 보고용 compsp(탐색), comp +dense0.5 +PRF(문장)(대표 구성)
B 조건: Vanilla RAG
집합·지표: wiki_ko GPT-Acc / admin_full GPT-Acc·R@5 / admin_single GPT-Acc·R@5(단일 도약 대조, 보고용).
admin_hard는 제외(적대적 필터의 정의상 Vanilla RAG의 R@5가 0).
응답 조건: 모든 조건이 ◆ 또는 ◆와 같은 응답 생성 조건(출력 한도 2,000)이다.

검정은 문항 단위 정확 McNemar(양측), 95% 신뢰구간은 짝지은 비율 차이의 정규 근사.
다중 비교 보정은 하지 않고 원 p값을 낸다. 판정(comp·gazcomp(PMI) × wiki_ko·admin_full GPT-Acc 4개,
Bonferroni α = 0.05/4)은 원고 쪽이 한다.

원고 대응: 제4장 제2절 "H1(새) 대 Vanilla RAG"(표 번호 미정).

실행: python analysis/h1_vanilla_tests.py
"""
import csv

from common import ROOT, fmt_p, load, mcnemar, system

EXPLORE = {r["name"]: r for r in csv.DictReader(open(ROOT / "analysis" / "systems_explore.csv", encoding="utf-8"))}

B = "Vanilla RAG"
PRIMARY = ["comp", "gazcomp(PMI)"]
ZERO_CALL = ["nnp (원논문 대응)", "noun", "comp", "gazcomp(PMI)"]
REPORT = ["compsp", "comp +dense0.5 +PRF(문장)"]
PLAN = [("wiki_ko", "acc"), ("admin_full", "acc"), ("admin_full", "r5"),
        ("admin_single", "acc"), ("admin_single", "r5")]

HEAD = "| A | 집합 | 지표 | n | A | B | b/c | 차이(%p) | 95% CI | p |\n" \
       "|:--|:--|:--|--:|--:|--:|--:|--:|:--|--:|"


def row(name):
    return EXPLORE[name] if name in EXPLORE else system(name)


def line(a, d, metric):
    A, Bv = load(row(a), d), load(row(B), d)
    ra, rb, bb, cc, diff, (lo, hi), p = mcnemar(A, Bv, metric)
    label = {"acc": "GPT-Acc", "r5": "R@5"}[metric]
    return (f"| {a} | {d} | {label} | {len(A)} | {ra:.3f} | {rb:.3f} | {bb}/{cc} | "
            f"{diff:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {fmt_p(p)} |")


def block(title, names, plan):
    print(f"\n## {title}\n")
    print(HEAD)
    for a in names:
        for d, m in plan:
            print(line(a, d, m))


def main():
    print(f"# H1(새) 대 Vanilla RAG (정확 McNemar 양측, 보정 없는 원 p값, B = {B})")
    block("판정용 4개 비교 (comp·gazcomp(PMI) × wiki_ko·admin_full GPT-Acc, 판정은 원고 쪽, α = 0.05/4)",
          PRIMARY, [("wiki_ko", "acc"), ("admin_full", "acc")])
    block("호출 0회 후보 4개 조건 대 Vanilla RAG (전체)", ZERO_CALL, PLAN)
    block("보고용: compsp(탐색), 대표 구성(검색 조정 포함) 대 Vanilla RAG", REPORT, PLAN)


if __name__ == "__main__":
    main()
