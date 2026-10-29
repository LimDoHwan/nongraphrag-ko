"""H5: dense 가중치(pnw) 스윕 분석 (2026-10-18, admin R@5, 검색만).

입력: results/track_02/pnw_sweep/{추출}[_prf]_{집합}.json  (scripts/pnw_sweep/pnw_sweep.py 출력, pnw 별 문항 상위 5개)
- R@5 곡선(추출 방식 × pnw × single·full·hard)과 Vanilla 기준선
- 추출 방식별 full R@5 최고 pnw (같은 값이면 가장 작은 pnw)
- 정확 McNemar(양측) + 95% CI: ① 최고점 대비 그보다 큰 pnw (최고점을 결과로 골랐으므로 탐색적)
                               ② 미리 정한 기준점 0.05(논문 기본값) 대비
- 질문 엔티티가 없어 밀집 검색으로 대체된 문항 수, pnw 0에서 시작 벡터가 비어 전역 PageRank가 된 문항 수
다중 비교 보정 없음.

원고 대응(2026-10-29 14단계 번호): <표 39>의 최고 계수·①·③ 열, <그림 4>. 2026-10-28 가설 재정의로 H5의 주 근거(대 Vanilla 검정·판정표는 h5_tests.py).

실행: python analysis/pnw_sweep.py
"""
import json

from common import RESULTS, fmt_p, load, mcnemar

SWEEP = RESULTS / "pnw_sweep"
SETS = ["admin_single", "admin_full", "admin_hard"]
NAMES = {"comp": "comp", "noun": "noun", "gazcomp": "gazcomp(PMI)", "gazcomp_llm4b": "4B 증류",
         "gazcomp_llm": "31B 증류", "compsp": "compsp (탐색)"}
BASE_P = "0.05"


def supporting(ds):
    rows = json.load(open(RESULTS / f"vanillarag_{ds}" / "predictions.json", encoding="utf-8"))
    return {r["id"]: set(map(str, r["supporting"])) for r in rows}


SUP = {d: supporting(d) for d in SETS}


def r5map(top5, ds):
    return {i: {"r5": SUP[ds][i] <= set(t[:5])} for i, t in top5.items()}


def load_sweep(extract, prf):
    out, meta = {}, {}
    for d in SETS:
        f = SWEEP / f"{extract}{'_prf' if prf else ''}_{d}.json"
        if not f.exists():
            return None, None
        j = json.load(open(f, encoding="utf-8"))
        meta[d] = j["meta"]
        out[d] = {p: r5map(t, d) for p, t in j["top5"].items()}
    return out, meta


def rate(m):
    return sum(v["r5"] for v in m.values()) / len(m)


def cell(a, b):
    ra, rb, bb, cc, diff, (lo, hi), p = mcnemar(a, b, "r5")
    s = "**" if p < 0.05 else ""
    return f"{s}{diff:+.1f}{s} ({bb}/{cc}, p={fmt_p(p)})"


def report(prf):
    tag = "PRF(문장) 켬" if prf else "PRF 끔"
    vanilla = {d: rate(load("Vanilla RAG", d)) for d in SETS}
    print(f"\n# H5: dense 가중치 스윕 — {tag} (R@5)\n")
    best_tbl = []
    for x, name in NAMES.items():
        S, meta = load_sweep(x, prf)
        if S is None:
            continue
        pnws = list(S[SETS[0]].keys())
        full = {p: rate(S["admin_full"][p]) for p in pnws}
        top = max(full.values())
        ties = [p for p in pnws if abs(full[p] - top) < 1e-12]
        best = min(ties, key=float)
        best_tbl.append((name, best, ties, S, pnws))
        print(f"\n## {name}\n")
        print(f"질문 엔티티 없음(밀집 검색 대체): " + ", ".join(f"{d} {meta[d]['no_seed_questions']}/{meta[d]['n']}" for d in SETS)
              + " · pnw 0에서 전역 PageRank: " + ", ".join(f"{d} {meta[d]['empty_reset_global_pagerank']['0']}" for d in SETS))
        print(f"\nfull 최고 pnw = **{best}**" + (f" (같은 값: {', '.join(ties)})" if len(ties) > 1 else "") + "\n")
        print("| pnw | single | full | hard | single 대 0.05 | full 대 0.05 | hard 대 0.05 |")
        print("|--:|--:|--:|--:|:--|:--|:--|")
        for p in pnws:
            mark = " ★" if p == best else ""
            vals = " | ".join(f"{rate(S[d][p]):.3f}" for d in SETS)
            vs = " | ".join("—" if p == BASE_P else cell(S[d][p], S[d][BASE_P]) for d in SETS)
            print(f"| {p}{mark} | {vals} | {vs} |")
        print(f"| Vanilla | " + " | ".join(f"{vanilla[d]:.3f}" for d in SETS) + " | | | |")
        higher = [p for p in pnws if float(p) > float(best)]
        if higher:
            print(f"\n최고점({best}) 대비 pnw를 올렸을 때 (탐색적):\n")
            print("| pnw | single | full | hard |")
            print("|--:|:--|:--|:--|")
            for p in higher:
                print(f"| {p} | " + " | ".join(cell(S[d][p], S[d][best]) for d in SETS) + " |")
    print(f"\n## 요약 — 추출 방식별 full 최고 pnw ({tag})\n")
    print("| 추출 | full 최고 pnw | single | full | hard |")
    print("|:--|--:|--:|--:|--:|")
    for name, best, ties, S, pnws in best_tbl:
        print(f"| {name} | {best}{' (동률 ' + ','.join(ties) + ')' if len(ties) > 1 else ''} | "
              + " | ".join(f"{rate(S[d][best]):.3f}" for d in SETS) + " |")


CONFIRM = {  # 확인용 GPT-Acc 점(◆ 조건: 31B, QA_SYS, 출력 한도 2,000, gpt-4o). 0.05 는 논문 ◆ 행
    "comp": {"0.05": "linearrag-new-comp", "0.5": "linearrag-new-comp-sweep-pnw0.5",
             "1.0": "linearrag-new-comp-sweep-pnw1.0"},
    "4B 증류": {"0.05": "linearrag-new-gazcomp-llm4b", "0.2": "linearrag-new-gazcomp_llm4b-sweep-pnw0.2",
               "1.0": "linearrag-new-gazcomp_llm4b-sweep-pnw1.0"},
}


def confirm():
    """확인용 점의 GPT-Acc: 0.05 대비 정확 McNemar. 최고점은 full R@5 를 보고 고른 점이다."""
    print("\n# H5 확인용 GPT-Acc (◆ 조건, PRF 끔)\n")
    print("| 추출 | pnw | single | full | hard | single 대 0.05 | full 대 0.05 | hard 대 0.05 |")
    print("|:--|--:|--:|--:|--:|:--|:--|:--|")
    for name, pts in CONFIRM.items():
        res = {p: {d: load({"admin_folder": f, "wiki_folder": f}, d) for d in SETS} for p, f in pts.items()}
        for p in pts:
            vals = " | ".join(f"{sum(v['acc'] for v in res[p][d].values()) / len(res[p][d]):.3f}" for d in SETS)
            vs = []
            for d in SETS:
                if p == BASE_P:
                    vs.append("—")
                    continue
                ra, rb, bb, cc, diff, (lo, hi), pv = mcnemar(res[p][d], res[BASE_P][d], "acc")
                s = "**" if pv < 0.05 else ""
                vs.append(f"{s}{diff:+.1f}{s} ({bb}/{cc}, p={fmt_p(pv)})")
            print(f"| {name} | {p} | {vals} | " + " | ".join(vs) + " |")


if __name__ == "__main__":
    report(prf=False)
    report(prf=True)
    confirm()
