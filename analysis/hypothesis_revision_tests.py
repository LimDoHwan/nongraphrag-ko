"""가설 재정리(2026-10-18)에 필요한 짝 검정: H3·H4·H5.

요청 1 (H3) compsp·noun 대 LLM 증류 사전
요청 2 (H4) 증류 사전 대 HippoRAG2 관계 그래프
요청 4 (H5) 검색 조정 대 조정 없음 (부록 C <표 60>의 구성)
추가 (H5) comp + 검색 조정 6종 대 HippoRAG2 31B·4B 그래프 (admin)

검정은 문항 단위 정확 McNemar(양측), 95% 신뢰구간은 짝지은 비율 차이의 정규 근사
(차이 ± 1.96 × √(b + c − (b − c)²/n) / n). 지표는 admin R@5·GPT-Acc, wiki_ko GPT-Acc.
GPT-Acc는 두 조건의 응답 조건이 모두 출력 한도 2,000(◆ 또는 ◆와 같은 조건의 탐색 행)일 때만 낸다.
한쪽이라도 256토큰 행이면 "비교 불가"로 표시한다(R@5는 응답과 무관해 항상 비교 가능).
다중 비교 보정은 하지 않았다(원고의 다른 검정과 같음).

원고 대응(2026-10-19 번호): <표 28>(wiki noun 행), <표 35>, <표 39>, <표 40>, 부록 C <표 64>, <표 66>.

실행: python analysis/hypothesis_revision_tests.py
"""
import csv

from common import DATASETS, ROOT, load, mcnemar, fmt_p, system

EXPLORE = {r["name"]: r for r in csv.DictReader(open(ROOT / "analysis" / "systems_explore.csv", encoding="utf-8"))}
# 탐색 행 중 ◆와 같은 응답 조건(출력 한도 2,000)으로 생성한 행
EXPLORE_LONG = {"compsp", "gazcompsp(PMI)", "gazcompsp(4B 증류 NER+RE)", "gazcompsp(31B 증류 NER+RE)",
                "compsp +dense0.5 +PRF(문장)"}
ADMIN = DATASETS[1:]

D4B, D31B = "gazcomp(4B 증류 NER+RE)", "gazcomp(31B 증류 NER+RE)"
H31B, H4B = "HippoRAG2 (31B 그래프)", "HippoRAG2 (4B 그래프)"
STACK4 = ["+dense0.5", "+dense1.0", "+dense0.5 +PRF(문서)", "+dense0.5 +PRF(문장)"]
STACK_PMI = STACK4[:2] + ["+dense0.5 +PRF(문서)", "+dense0.5 +PRF(문장)", "+dense1.0 +PRF(문장)"]
STACK_COMP = ["+dense0.5", "+dense1.0", "+dense0.5 +PRF(문서)", "+dense1.0 +PRF(문서)", "+dense0.5 +PRF(문장)",
              "+dense1.0 +PRF(문장)"]


def row(name):
    return EXPLORE[name] if name in EXPLORE else system(name)


def long_answer(name):
    return name in EXPLORE_LONG or (name not in EXPLORE and system(name)["main10"] == "1")


HEAD = "| 비교 (A 대 B) | 집합 | 지표 | n | A | B | b/c | 차이(%p) | 95% CI | p |\n" \
       "|:--|:--|:--|--:|--:|--:|--:|--:|:--|--:|"


def line(a, b, d, metric):
    label = {"acc": "GPT-Acc", "r5": "R@5"}[metric]
    if metric == "acc" and not (long_answer(a) and long_answer(b)):
        return f"| {a} 대 {b} | {d} | {label} | | | | | 비교 불가 (256토큰 행 포함) | | |"
    ra, rb, bb, cc, diff, (lo, hi), p = mcnemar(load(row(a), d), load(row(b), d), metric)
    star = "**" if p < 0.05 else ""
    return (f"| {a} 대 {b} | {d} | {label} | {len(load(row(a), d))} | {ra:.3f} | {rb:.3f} | {bb}/{cc} | "
            f"{star}{diff:+.1f}{star} | [{lo:+.1f}, {hi:+.1f}] | {star}{fmt_p(p)}{star} |")


def block(title, pairs, datasets, metrics, skip_incomparable=False):
    print(f"\n## {title}\n")
    print(HEAD)
    for a, b in pairs:
        for d in datasets:
            for m in metrics:
                if m == "r5" and d == "wiki_ko":
                    continue
                if skip_incomparable and m == "acc" and not (long_answer(a) and long_answer(b)):
                    continue
                print(line(a, b, d, m))


def main():
    print("# 가설 재정리용 짝 검정 (정확 McNemar, 굵은 글씨 = p < 0.05)")
    block("요청 1 (H3). compsp 대 LLM 증류 사전", [("compsp", D4B), ("compsp", D31B)], DATASETS, ["r5", "acc"])
    block("요청 1 (H3). noun 대 LLM 증류 사전 (wiki_ko)", [("noun", D4B), ("noun", D31B)], ["wiki_ko"], ["acc"])
    block("요청 2 (H4). 증류 사전 대 HippoRAG2 관계 그래프",
          [(D4B, H31B), (D4B, H4B), (D31B, H31B)], DATASETS, ["r5", "acc"])
    pairs = ([(f"{D4B} {s}", D4B) for s in STACK4] + [(f"{D31B} {s}", D31B) for s in STACK4]
             + [(f"gazcomp(PMI) {s}", "gazcomp(PMI)") for s in STACK_PMI])
    block("요청 4 (H5). 검색 조정 대 조정 없음 — 사전 계열 (admin)", pairs, ADMIN, ["r5"])
    block("요청 4 (H5). 검색 조정 대 조정 없음 — comp (wiki_ko)",
          [(f"comp {s}", "comp") for s in STACK_COMP], ["wiki_ko"], ["acc"])
    # 추가 요청(2026-10-18): GPT-Acc는 ◆ 쌍(comp +dense0.5 +PRF(문장))만 내고, 256토큰 행은 생략
    for g in (H31B, H4B):
        block(f"추가 (H5). comp + 검색 조정 6종 대 {g} (admin)",
              [(f"comp {s}", g) for s in STACK_COMP], ADMIN, ["r5", "acc"], skip_incomparable=True)


if __name__ == "__main__":
    main()
