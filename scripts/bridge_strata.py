"""연결어 보유 여부로 층화한 성능 비교 (탐색적 분석, 2026-10-10).

admin full·hard 질의의 연결어(bridge)는 comp와 같은 규칙(extract_compounds)으로 뽑혔다.
각 방식이 연결어를 "두 정답 문서 모두에서" 엔티티로 가졌는지로 질의를 나누고,
층마다 두 조건의 R@5·GPT-Acc를 정확 McNemar 검정으로 비교한다.

  - LinearRAG 계열: cache/track_02/linrag_{mode}_admin_full/ner.json 의 p2e
    (hard 코퍼스는 full과 문서 집합이 같으므로 full 추출 결과를 그대로 쓴다)
  - HippoRAG2(31B): cache/track_02/hipporag2_admin_full/openie/{idx}.json 의
    named_entities + 트리플 주어·목적어. 정확 일치와 부분 포함 두 기준을 함께 본다.

실행: python scripts/bridge_strata.py
"""
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset  # noqa: E402

CACHE = ROOT / "cache" / "track_02"
RES = ROOT / "results" / "track_02"
DATA = ROOT / "data" / "admin_ko"

SYSTEMS = {  # 원고 표기 → (결과 폴더 접두, 엔티티 캐시)
    "comp": ("linearrag-new-comp", "linrag_comp_admin_full"),
    "gazcomp(PMI)": ("linearrag-new-gazcomp", "linrag_gazcomp_admin_full"),
    "31B 증류": ("linearrag-new-gazcomp-llm", "linrag_gazcomp_llm_admin_full"),
    "4B 증류": ("linearrag-new-gazcomp-llm4b", "linrag_gazcomp_llm4b_admin_full"),
    "HippoRAG2(31B)": ("graphrag-org", None),
}
PAIRS = [  # (A, B, 층을 나누는 기준이 되는 조건)
    ("comp", "HippoRAG2(31B)"),
    ("gazcomp(PMI)", "HippoRAG2(31B)"),
    ("gazcomp(PMI)", "31B 증류"),
    ("gazcomp(PMI)", "4B 증류"),
]


def load_preds(prefix, ds):
    p = json.load(open(RES / f"{prefix}_{ds}" / "predictions.json", encoding="utf-8"))
    out = {}
    for r in p:
        r5 = set(map(str, r["supporting"])) <= set(map(str, r["retrieved"][:5]))
        out[r["id"]] = {"r5": r5, "acc": bool(r["gpt4o"]), "q": r["question"]}
    return out


def bridge_flags():
    qs = json.load(open(DATA / "questions_anchored.json", encoding="utf-8"))
    by_q = {q["question"]: q for q in qs}
    passages = load_dataset("admin_full")[0]
    idx = {str(p["id"]): i for i, p in enumerate(passages)}
    flags = {}
    for name, (_, cache) in SYSTEMS.items():
        if cache:
            p2e = {k: set(v) for k, v in json.load(open(CACHE / cache / "ner.json"))["p2e"].items()}
            flags[name] = {q["question"]: all(q["bridge"] in p2e.get(str(p), ()) for p in q["supporting_facts"])
                           for q in qs}
        else:
            def ents(pid):
                o = json.load(open(CACHE / "hipporag2_admin_full" / "openie" / f"{idx[str(pid)]}.json"))
                s = set(o.get("named_entities", []))
                for t in o.get("triples", []):
                    if len(t) == 3:
                        s.update((t[0], t[2]))
                return s
            E = {str(p): ents(p) for q in qs for p in q["supporting_facts"]}
            flags[name] = {q["question"]: all(q["bridge"] in E[str(p)] for p in q["supporting_facts"]) for q in qs}
            flags[name + "~부분포함"] = {
                q["question"]: all(any(q["bridge"] in e for e in E[str(p)]) for p in q["supporting_facts"])
                for q in qs}
    return by_q, flags


def mcnemar(a, b, ids, key):
    bb = sum(1 for i in ids if a[i][key] and not b[i][key])
    cc = sum(1 for i in ids if b[i][key] and not a[i][key])
    p = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
    n = len(ids)
    ra = sum(a[i][key] for i in ids) / n if n else float("nan")
    rb = sum(b[i][key] for i in ids) / n if n else float("nan")
    return ra, rb, bb, cc, p


def main():
    _, flags = bridge_flags()
    for ds in ("admin_full", "admin_hard"):
        preds = {k: load_preds(v[0], ds) for k, v in SYSTEMS.items()}
        ids = sorted(preds["comp"])
        print(f"\n===== {ds} (n={len(ids)}) =====")
        for A, B in PAIRS:
            a, b = preds[A], preds[B]
            keys = [B] + ([B + "~부분포함"] if B + "~부분포함" in flags else [])
            for fk in keys:
                fa, fb = flags[A], flags[fk]
                strata = {
                    "둘 다 보유": [i for i in ids if fa[a[i]["q"]] and fb[a[i]["q"]]],
                    "A만 보유": [i for i in ids if fa[a[i]["q"]] and not fb[a[i]["q"]]],
                    "B만 보유": [i for i in ids if not fa[a[i]["q"]] and fb[a[i]["q"]]],
                    "둘 다 없음": [i for i in ids if not fa[a[i]["q"]] and not fb[a[i]["q"]]],
                }
                print(f"\n[{A} 대 {B}]  층 기준: {fk}")
                for metric in ("r5", "acc"):
                    ra, rb, bb, cc, p = mcnemar(a, b, ids, metric)
                    print(f"  전체 {metric:3s}: A {ra:.3f} / B {rb:.3f}  b/c {bb}/{cc}  p={p:.3g}")
                for s, sid in strata.items():
                    if not sid:
                        continue
                    for metric in ("r5", "acc"):
                        ra, rb, bb, cc, p = mcnemar(a, b, sid, metric)
                        print(f"  {s:6s} n={len(sid):3d} {metric:3s}: A {ra:.3f} / B {rb:.3f}  "
                              f"차이 {100 * (ra - rb):+5.1f}%p  b/c {bb}/{cc}  p={p:.3g}")


if __name__ == "__main__":
    main()
