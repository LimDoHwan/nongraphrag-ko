"""Track 2 - Vanilla RAG baseline (§3.1, 두 데이터셋 공통).

dense 검색(bge-m3 코사인 top-k) → Gemma QA → Contain-Acc / Gemma-Acc 평가.
그래프 없음 — baseline 하한선. wiki_ko / admin_ko 동일 코드로 실행.

임베딩 캐시: cache/track_02/emb_{dataset}.npy (+ ids). 색인 LLM 호출 0 (§9.3).
산출: results/track_02/vanilla_{dataset}/predictions.json + eval.json

실행: python -m src.track_02.vanillarag.run --dataset admin_ko --topk 5 --workers 8
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache" / "track_02"
RESULTS = ROOT / "results" / "track_02"

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset            # noqa: E402
from src.track_01.common.embedding import Embedder        # noqa: E402
from src.track_01.common.llm_client import LLMClient      # noqa: E402
from src.track_01.common.evaluation import contain_acc, gpt_acc  # noqa: E402

QA_SYSTEM = "주어진 문서들만 근거로 질문에 답하세요. 답은 핵심 어구만 간결하게 적습니다."
QA_USER = "[문서]\n{ctx}\n\n질문: {q}\n답:"


def get_corpus_emb(dataset, passages, emb):
    f = CACHE / f"emb_{dataset}.npy"
    fi = CACHE / f"emb_{dataset}_ids.json"
    if f.exists() and fi.exists():
        ids = json.load(open(fi, encoding="utf-8"))
        if ids == [p["id"] for p in passages]:
            return np.load(f).astype(np.float32)
    print(f"코퍼스 {len(passages)} 임베딩...", flush=True)
    v = np.asarray(emb.encode([p["text"] for p in passages], batch_size=64))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, v.astype(np.float16))
    json.dump([p["id"] for p in passages], open(fi, "w", encoding="utf-8"))
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["wiki_ko", "admin_ko", "admin_hard", "admin_full", "admin_single", "admin_hard2k"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true", help="Gemma-Acc 생략")
    args = ap.parse_args()

    passages, questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[:args.limit]
    print(f"[{args.dataset}] passages={len(passages)} questions={len(questions)}")

    emb = Embedder()
    P = get_corpus_emb(args.dataset, passages, emb)          # (N,1024) f32
    ptext = [p["text"] for p in passages]
    pid = [p["id"] for p in passages]

    # 1) 검색 + 2) QA (동시)
    llm = LLMClient()
    qvecs = np.asarray(emb.encode([q["question"] for q in questions], batch_size=64))

    def answer(idx):
        sims = P @ qvecs[idx]
        top = np.argsort(-sims)[:args.topk]
        ctx = "\n".join(f"- {ptext[j]}" for j in top)
        out = llm.chat([{"role": "system", "content": QA_SYSTEM},
                        {"role": "user", "content": QA_USER.format(
                            ctx=ctx, q=questions[idx]["question"])}],
                       temperature=0.0, max_tokens=128)
        return [pid[j] for j in top], out.strip()

    t0 = time.time()
    preds = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (retr, pred) in enumerate(ex.map(answer, range(len(questions)))):
            q = questions[i]
            preds.append({"id": q["id"], "question": q["question"],
                          "answer": q["answer"], "prediction": pred,
                          "retrieved": retr, "supporting": q["supporting"]})
            if (i + 1) % 50 == 0:
                print(f"  QA {i+1}/{len(questions)} ({time.time()-t0:.0f}s)", flush=True)

    # 3) 평가
    for p in preds:
        p["contain"] = contain_acc(p["prediction"], p["answer"])
    if not args.no_judge:
        def judge(p):
            return gpt_acc(p["question"], p["answer"], p["prediction"], llm)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for p, g in zip(preds, ex.map(judge, preds)):
                p["gpt"] = g

    outdir = RESULTS / f"vanillarag_{args.dataset}"
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(outdir / "predictions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n = len(preds)
    ca = sum(p["contain"] for p in preds) / n
    summary = {"dataset": args.dataset, "system": "vanillarag", "n": n,
               "topk": args.topk, "contain_acc": round(ca, 4)}
    if not args.no_judge:
        summary["gpt_acc"] = round(sum(p["gpt"] for p in preds) / n, 4)
    json.dump(summary, open(outdir / "eval.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[{args.dataset}] Vanilla RAG  n={n}  {summary}")


if __name__ == "__main__":
    main()
