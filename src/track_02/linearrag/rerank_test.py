"""번외 — bge-m3 ColBERT 리랭킹 테스트.

top-20 검색 → bge-m3 ColBERT(멀티벡터) 리랭킹 → top-5. R@5 orig vs rerank 비교.
대상: gazcomp(4b증류 NER+RE) / Vanilla RAG.
실행: python -m src.track_02.linearrag.rerank_test --dataset admin_full
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset          # noqa: E402
import src.track_02.linearrag.engine as E                      # noqa: E402


def r5(preds_top5, sups):
    return np.mean([set(map(str, s)) <= set(map(str, p))
                    for p, s in zip(preds_top5, sups)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ncand", type=int, default=20)
    args = ap.parse_args()
    passages, questions = load_dataset(args.dataset)
    pid2text = {str(p["id"]): p["text"] for p in passages}
    sups = [q["supporting"] for q in questions]
    qtexts = [q["question"] for q in questions]

    from FlagEmbedding import BGEM3FlagModel
    rr = BGEM3FlagModel("./models/bge-m3", use_fp16=True)

    def rerank(cands_per_q):
        """cands_per_q: 질문별 후보 pid 리스트 → ColBERT로 top-5 재정렬."""
        out = []
        for q, cands in zip(qtexts, cands_per_q):
            if not cands:
                out.append([]); continue
            pairs = [[q, pid2text[c]] for c in cands]
            sc = rr.compute_score(pairs, weights_for_different_modes=[0, 0, 1])["colbert"]
            order = np.argsort(-np.asarray(sc))[:5]
            out.append([cands[i] for i in order])
        return out

    # top-N 후보 확보 (TOPK 임시 확장)
    E.TOPK = args.ncand
    results = {}

    # 1) gazcomp(4b증류 NER+RE)
    rag = E.LinearRAG(args.dataset, "gazcomp",
                      glossary_path=str(ROOT / "data/admin_ko/glossary_llm4b_full.json"),
                      tag="llm4b")
    rag.index(passages)
    cand_g = [list(map(str, rag.retrieve_one(q))) for q in qtexts]
    results["gazcomp-4b orig"] = r5([c[:5] for c in cand_g], sups)
    results["gazcomp-4b rerank"] = r5(rerank(cand_g), sups)

    # 2) Vanilla (dense top-N)
    pe = rag.pemb; pid = rag.pid
    qv = np.asarray(rag.emb.encode(qtexts, batch_size=64), dtype=np.float32)
    cand_v = [[str(pid[i]) for i in np.argsort(-(pe @ qv[k]))[:args.ncand]]
              for k in range(len(qtexts))]
    results["vanilla orig"] = r5([c[:5] for c in cand_v], sups)
    results["vanilla rerank"] = r5(rerank(cand_v), sups)

    print(f"\n=== [{args.dataset}] ColBERT 리랭킹 (top-{args.ncand}→5) R@5 ===")
    for k, v in results.items():
        print(f"  {k:22} {v:.3f}")


if __name__ == "__main__":
    main()
