"""Track 2 - 적대적 필터: 진짜 traversal-필수 multi-hop 만 선별 (MuSiQue 방식).

anchored 후보 중, **dense(vanilla)가 docB(답 문서)를 직접 못 찾는 질문만** 채택.
→ 이 셋에선 docB가 anchor→bridge 경로로만 도달 가능 = 진짜 2-hop.
공정성: 같은 hard 셋에서 vanilla / LinearRAG(nnp,파편화) / LinearRAG(+사전,보존) 비교.
핵심 가설: 사전이 bridge(한국어 복합명사) 보존 → traversal 성공 → vanilla·nnp 능가.

산출:
  data/admin_ko/questions_admin_hard.json   (hard 질문)
  data/admin_ko/chunks_hard.json            (hard gold + distractor 코퍼스)
실행: python -m src.track_02.data.filter_adversarial --distractors 10000 --topk 5
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"
CACHE = ROOT / "cache" / "track_02"

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.track_01.common.embedding import Embedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", type=int, default=10000)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cand = json.load(open(DATA / "questions_anchored.json", encoding="utf-8"))
    # train+val 풀링 (새 질문이 train+val 문서 사용)
    docs = {d["passage_id"]: d
            for split in ("docs_train", "docs_val")
            for d in json.load(open(DATA / f"{split}.json", encoding="utf-8"))}

    gold = sorted({str(p) for q in cand for p in q["supporting_facts"]})
    rest = [p for p in docs if p not in set(gold)]
    rng.shuffle(rest)
    corpus_ids = gold + rest[:args.distractors]
    print(f"후보 {len(cand)}, gold {len(gold)}, 코퍼스 {len(corpus_ids)}")

    emb = Embedder()
    C = np.asarray(emb.encode([docs[p]["text"] for p in corpus_ids], batch_size=64),
                   dtype=np.float32)
    Q = np.asarray(emb.encode([q["question"] for q in cand], batch_size=64), dtype=np.float32)
    idx = {p: i for i, p in enumerate(corpus_ids)}

    hard = []
    vboth = 0
    for k, q in enumerate(cand):
        sup = [str(p) for p in q["supporting_facts"]]
        if len(sup) < 2 or sup[1] not in idx:
            continue
        docB = sup[1]
        sims = C @ Q[k]
        top = set(int(i) for i in np.argsort(-sims)[:args.topk])
        # hard = dense 가 docB(답 문서) 를 top-k 에서 못 찾음
        if idx[docB] not in top:
            hard.append(q)
        if all(idx[s] in top for s in sup):
            vboth += 1
    print(f"vanilla both@{args.topk}(후보 전체) = {vboth/len(cand):.3f}")
    print(f"→ hard(=dense가 docB 못 찾음): {len(hard)}/{len(cand)}")

    # hard 질문 저장
    for i, q in enumerate(hard):
        q["id"] = f"admh_{i}"
    json.dump(hard, open(DATA / "questions_admin_hard.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # hard 코퍼스 = hard gold + distractor (코퍼스 재사용)
    hard_gold = sorted({str(p) for q in hard for p in q["supporting_facts"]})
    hg = set(hard_gold)
    chunks = ([{"passage_id": p, "text": docs[p]["text"], "role": "gold"} for p in hard_gold]
              + [{"passage_id": p, "text": docs[p]["text"], "role": "distractor"}
                 for p in corpus_ids if p not in hg])
    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    json.dump(chunks, open(DATA / "chunks_hard.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"hard 코퍼스 {len(chunks)} (gold {len(hard_gold)}) → chunks_hard.json")
    print(f"hard 질문 {len(hard)} → questions_admin_hard.json")


if __name__ == "__main__":
    main()
