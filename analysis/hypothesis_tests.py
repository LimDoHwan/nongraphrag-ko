"""제4장 가설 검정표 재생성 (정확 McNemar, 양측).

  <표 22> H1  : noun 대 nnp
  <표 23> 추가 분석: comp 대 noun, gazcomp(PMI) 대 comp
  <표 26> H3 (가) 대표 조건: comp·gazcomp(PMI) 대 HippoRAG2(31B)
  <표 27> H3 (가) 사전 출처 대응쌍: gazcomp(PMI) 대 4B·31B 증류 사전
  <표 30> 구축 LLM 규모: 4B 대 31B (증류 사전, HippoRAG2 그래프)
  <표 43> H5 보조: 검색 조정 6개 구성 대 comp (다중 도약 R@5)
  <표 44> 관련: 검색 조정 6개 구성 대 comp (단일 도약 R@5)

표 번호는 2026-10-29 원고 14단계 기준. 판정 유의수준: H1 0.05/3, H5 보조(<표 43>) 0.05/2 (Bonferroni), H3 (가) 0.05 (보정 없음, 제3장 제5절).
표의 "b/c"는 A만 맞힌 문항 수 / B만 맞힌 문항 수, 차이는 A − B (%p)와 95% 신뢰구간.

실행: python analysis/hypothesis_tests.py
"""
from common import fmt_p, load, mcnemar

ADMIN = ["admin_full", "admin_hard"]
MULTI = ["wiki_ko"] + ADMIN


def row(a, b, ds, key):
    A, B = load(a, ds), load(b, ds)
    ra, rb, bb, cc, d, (lo, hi), p = mcnemar(A, B, key)
    name = {"acc": "GPT-Acc", "r5": "R@5"}[key]
    return f"| {ds} | {name} | {ra:.3f} | {rb:.3f} | {bb}/{cc} | {d:+.1f} [{lo:+.1f}, {hi:+.1f}] | {fmt_p(p)} |"


def table(title, a, b, plan):
    print(f"\n### {title}\nA = {a}, B = {b}\n")
    print("| 집합 | 지표 | A | B | b/c | 차이 [95% CI] | p |")
    print("|:--|:--|--:|--:|:--:|:--|:--|")
    for ds, key in plan:
        print(row(a, b, ds, key))


def full_plan(r5_single=True):
    plan = [(d, "acc") for d in MULTI] + [(d, "r5") for d in ADMIN]
    plan += [("admin_single", "acc")] + ([("admin_single", "r5")] if r5_single else [])
    return plan


def main():
    table("<표 22> H1", "noun", "nnp (원논문 대응)", full_plan())

    print("\n## <표 23> 추가 분석 (탐색적)")
    table("comp 대 noun", "comp", "noun", [(d, "acc") for d in MULTI] + [(d, "r5") for d in ADMIN])
    table("gazcomp(PMI) 대 comp", "gazcomp(PMI)", "comp", [(d, "acc") for d in MULTI] + [(d, "r5") for d in ADMIN])

    print("\n## <표 26> H3 (가) 대표 조건")
    for a in ("comp", "gazcomp(PMI)"):
        table(f"{a} 대 HippoRAG2 (31B 그래프)", a, "HippoRAG2 (31B 그래프)", full_plan())

    print("\n## <표 27> H3 (가) 사전 출처 대응쌍")
    for b in ("gazcomp(4B 증류 NER+RE)", "gazcomp(31B 증류 NER+RE)"):
        table(f"gazcomp(PMI) 대 {b}", "gazcomp(PMI)", b, full_plan())

    print("\n## <표 30> 구축 LLM 규모 (A = 4B, B = 31B)")
    plan = [("wiki_ko", "acc")] + [(d, k) for d in ("admin_full", "admin_hard", "admin_single") for k in ("acc", "r5")]
    table("증류 사전", "gazcomp(4B 증류 NER+RE)", "gazcomp(31B 증류 NER+RE)", plan)
    table("HippoRAG2 그래프", "HippoRAG2 (4B 그래프)", "HippoRAG2 (31B 그래프)", plan)

    configs = ["comp +dense0.5 +PRF(문장)", "comp +dense0.5", "comp +dense1.0",
               "comp +dense0.5 +PRF(문서)", "comp +dense1.0 +PRF(문서)", "comp +dense1.0 +PRF(문장)"]
    print("\n## <표 43> H5 보조: 검색 조정 구성 대 comp (R@5)")
    for c in configs:
        table(c, c, "comp", [(d, "r5") for d in ADMIN])
    print("\n## <표 44> 관련: 단일 도약 R@5, 검색 조정 구성 대 comp")
    for c in configs:
        table(c, c, "comp", [("admin_single", "r5")])


if __name__ == "__main__":
    main()
