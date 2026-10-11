"""dense-seeded PPR 검증(고속판): gazcomp을 1회 색인 후 pnw(passage dense seed 강도)만
스윕하며 single R@5(both@5)를 측정.

최적화: ① 질문당 무거운 계산(엔티티 활성·passage bonus)을 1회만 수행해 캐시,
        ② passage bonus의 text.count 루프를 log(1+count) 희소행렬 sparse matmul로 대체,
        ③ pnw별로는 reset 재구성 + PPR만 반복(가벼움).
pnw는 reset의 passage-node 가중치만 바꾸므로 그래프/엔티티 reset 불변.

실행: python -m src.track_02.linearrag.dseed_sweep --dataset admin_single \
        --extract gazcomp --glossary data/admin_ko/glossary.json
"""
import argparse
import math
import time

import numpy as np
import scipy.sparse as sp

from src.track_02.common.datasets import load_dataset
from src.track_02.linearrag.engine import LinearRAG, PASSAGE_RATIO, DAMPING


def build_logcount(rag):
    """passages × entities 희소행렬, data = log(1+count). bonus를 matmul로."""
    rows, cols, data = [], [], []
    for pi, p in enumerate(rag.passages):
        text = rag.ptext[pi]
        for e in rag.p2e[p["id"]]:
            ei = rag.eidx.get(e)
            if ei is None:
                continue
            c = text.count(e)
            if c > 0:
                rows.append(pi); cols.append(ei); data.append(math.log(1 + c))
    return sp.csr_matrix((data, (rows, cols)),
                         shape=(len(rag.passages), len(rag.entities)))


def precompute_query(rag, q, LC):
    """질문당 1회: 엔티티 reset(ent_w, 전 노드) + passage_score(per passage, pnw 미적용)."""
    qv = np.asarray(rag.emb.encode(q, batch_size=1), dtype=np.float32)
    seed_idx, seed_scores = rag._seed_entities(q)
    if not seed_idx:
        sims = rag.pemb @ qv          # seed 없으면 dense 순위로 폴백
        return None, sims
    ent_w, actived = rag._activate(qv, seed_idx, seed_scores)
    # dense passage 유사도 정규화(0~1)
    sims = rag.pemb @ qv
    lo, hi = sims.min(), sims.max()
    norm = (sims - lo) / ((hi - lo) or 1.0)
    # bonus = Σ_actived (sc/tier)·log(1+count)  → LC[:,actived] @ w  (sparse matmul)
    if actived:
        eis = np.fromiter(actived.keys(), dtype=int)
        w = np.array([sc / (tier if tier >= 1 else 1) for sc, tier in actived.values()],
                     dtype=np.float32)
        bonus = np.asarray(LC[:, eis].dot(w)).ravel()
    else:
        bonus = np.zeros(len(rag.passages), dtype=np.float32)
    passage_score = PASSAGE_RATIO * norm + np.log1p(bonus)   # pnw 미적용
    return ent_w, passage_score


def ppr_rank(rag, ent_w, passage_score, pnw):
    reset = ent_w.copy()
    reset[rag.p_vidx] += passage_score * pnw
    reset = np.where((np.isnan(reset)) | (reset < 0), 0, reset)
    if reset.sum() == 0:
        reset = None
    pr = rag.g.personalized_pagerank(damping=DAMPING, directed=False,
                                     weights="weight", reset=reset,
                                     implementation="prpack")
    dsc = np.array([pr[v] for v in rag.p_vidx])
    order = np.argsort(-dsc)[:5]
    return [rag.pid[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extract", default="gazcomp")
    ap.add_argument("--tag", default="")
    ap.add_argument("--glossary", default="data/admin_ko/glossary.json")
    ap.add_argument("--pnws", default="0.05,0.2,0.5,1.0,2.0,5.0")
    args = ap.parse_args()

    passages, questions = load_dataset(args.dataset)
    rag = LinearRAG(args.dataset, args.extract, glossary_path=args.glossary, tag=args.tag)
    t0 = time.time()
    rag.index(passages)
    print(f"색인 완료 ({time.time()-t0:.0f}s) V={rag.g.vcount()} E={rag.g.ecount()} "
          f"q={len(questions)}", flush=True)

    LC = build_logcount(rag)
    print(f"logcount 행렬 {LC.shape} nnz={LC.nnz} ({time.time()-t0:.0f}s)", flush=True)

    # 질문당 1회 precompute (캐시)
    cache, dense_fallback = [], []
    for i, q in enumerate(questions):
        ent_w, ps = precompute_query(rag, q["question"], LC)
        if ent_w is None:                       # seed 없음 → dense 순위 폴백 고정
            order = np.argsort(-ps)[:5]
            cache.append(("dense", [rag.pid[j] for j in order]))
        else:
            cache.append(("ppr", ent_w, ps))
        if (i + 1) % 100 == 0:
            print(f"  precompute {i+1}/{len(questions)} ({time.time()-t0:.0f}s)", flush=True)

    pnws = [float(x) for x in args.pnws.split(",")]
    print(f"\n{'pnw':>6} | {'R@5(both@5)':>12} | {'any@5':>7}   (vanilla=0.917, RRF=0.950, baseline pnw0.05)")
    print("-" * 60)
    for pnw in pnws:
        both = any1 = 0
        for c, q in zip(cache, questions):
            if c[0] == "dense":
                retr = set(map(str, c[1]))
            else:
                retr = set(map(str, ppr_rank(rag, c[1], c[2], pnw)))
            sup = set(map(str, q["supporting"]))
            both += sup <= retr
            any1 += len(sup & retr) > 0
        n = len(questions)
        print(f"{pnw:>6} | {both/n:>12.3f} | {any1/n:>7.3f}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
