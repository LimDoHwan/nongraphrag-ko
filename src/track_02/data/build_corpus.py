"""Track 2 - distractor 주입 + 코퍼스 생성 (3단계 / §11.3 [5]).

평가셋(questions_admin.json)의 정답경로 문서(gold) + distractor 로 색인 코퍼스를 만든다.
  - gold   : 240쌍의 supporting_facts 문서 (반드시 포함)
  - hard   : gold 와 같은 doc_class 이면서 bge-m3 코사인 상위 (혼동 유발)
  - easy   : 전체 풀에서 무관 랜덤
distractor 비율(hard/easy)은 변인화 가능(--hard-ratio) → 난이도-성능 곡선(§11.3[5]).
파일럿 2천 먼저(§11.6), 이후 --distractors 20000 로 확장.

산출:
  data/admin_ko/chunks.json              # [{chunk_id, passage_id, text, doc_class, doc_source, role}]
  cache/track_02/chunk_emb.npy + chunk_ids.json   # 코퍼스 임베딩(baseline 재사용)

실행: python -m src.track_02.data.build_corpus --distractors 2000 --hard-ratio 0.5
"""
import argparse
import json
import random
from collections import Counter, defaultdict
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
    ap.add_argument("--distractors", type=int, default=2000)
    ap.add_argument("--hard-ratio", type=float, default=0.5)
    ap.add_argument("--cand-per-class", type=int, default=1500,
                    help="hard 후보 임베딩 상한(클래스당)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    CACHE.mkdir(parents=True, exist_ok=True)

    docs = (json.load(open(DATA / "docs_train.json", encoding="utf-8"))
            + json.load(open(DATA / "docs_val.json", encoding="utf-8")))
    by_pid = {d["passage_id"]: d for d in docs}
    qs = json.load(open(DATA / "questions_admin.json", encoding="utf-8"))

    # 1) gold 문서 (정답경로)
    gold_pids = []
    seen = set()
    for q in qs:
        for p in q["supporting_facts"]:
            if p not in seen:
                seen.add(p)
                gold_pids.append(p)
    gold_classes = {by_pid[p]["doc_class"] for p in gold_pids}
    print(f"gold 문서 {len(gold_pids)} (클래스 {len(gold_classes)}종)")

    n_hard = int(args.distractors * args.hard_ratio)
    n_easy = args.distractors - n_hard

    emb = Embedder()
    print(f"임베딩 device={emb.device}")

    # 2) hard: 같은 doc_class 후보를 임베딩 → gold(동일클래스) 코사인 max 상위
    by_class = defaultdict(list)
    for d in docs:
        if d["passage_id"] in seen or d["doc_class"] not in gold_classes:
            continue
        by_class[d["doc_class"]].append(d["passage_id"])

    gold_vec = emb.encode([by_pid[p]["text"] for p in gold_pids], batch_size=64)
    gold_vec = np.asarray(gold_vec)
    gold_cls_arr = np.array([by_pid[p]["doc_class"] for p in gold_pids])

    cand_pids = []
    for c, pids in by_class.items():
        rng.shuffle(pids)
        cand_pids.extend(pids[:args.cand_per_class])
    print(f"hard 후보 {len(cand_pids)} 임베딩 중...")
    cand_vec = np.asarray(emb.encode([by_pid[p]["text"] for p in cand_pids],
                                     batch_size=64))

    # 후보별 동일클래스 gold 와의 최대 코사인
    scores = []
    for i, p in enumerate(cand_pids):
        mask = gold_cls_arr == by_pid[p]["doc_class"]
        sim = float((cand_vec[i] @ gold_vec[mask].T).max()) if mask.any() else -1.0
        scores.append(sim)
    order = sorted(range(len(cand_pids)), key=lambda i: -scores[i])
    hard_pids = [cand_pids[i] for i in order[:n_hard]]
    hard_set = set(hard_pids)

    # 3) easy: 전체 풀에서 무관 랜덤 (gold/hard 제외, 어떤 클래스든)
    rest = [d["passage_id"] for d in docs
            if d["passage_id"] not in seen and d["passage_id"] not in hard_set]
    rng.shuffle(rest)
    easy_pids = rest[:n_easy]

    # 4) 코퍼스 조립
    chunks = []
    for role, pids in [("gold", gold_pids), ("hard", hard_pids), ("easy", easy_pids)]:
        for p in pids:
            d = by_pid[p]
            chunks.append({"chunk_id": len(chunks), "passage_id": p,
                           "text": d["text"], "doc_class": d["doc_class"],
                           "doc_source": d["doc_source"], "role": role})
    rng.shuffle(chunks)  # 역할 순서 섞기
    for i, ch in enumerate(chunks):
        ch["chunk_id"] = i

    json.dump(chunks, open(DATA / "chunks.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    # 5) 코퍼스 임베딩 캐시 (baseline 재사용)
    print(f"코퍼스 {len(chunks)} 임베딩 캐싱...")
    cv = np.asarray(emb.encode([c["text"] for c in chunks], batch_size=64))
    np.save(CACHE / "chunk_emb.npy", cv.astype(np.float16))
    json.dump([c["passage_id"] for c in chunks],
              open(CACHE / "chunk_ids.json", "w", encoding="utf-8"))

    roles = Counter(c["role"] for c in chunks)
    print(f"\n코퍼스 {len(chunks)}개  {dict(roles)}")
    print(f"임베딩 {cv.shape} float16 ~{cv.nbytes//1024//1024}MB")
    print("saved data/admin_ko/chunks.json + cache/track_02/chunk_emb.npy")


if __name__ == "__main__":
    main()
