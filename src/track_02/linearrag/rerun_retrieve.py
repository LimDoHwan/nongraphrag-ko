"""Track 2 - LinearRAG 재실험: 색인 + 검색만 수행하고 검색 결과(top-5)를 저장 (2026-09).

engine.py는 검색과 QA(31B)를 한 번에 실행하므로, 31B 서버가 없는 동안 색인·검색만 먼저
돌리기 위한 드라이버. QA는 rerun_qa.py가 저장된 검색 결과로 수행한다(검색 결과가 기존과 같은
문항은 기존 답변 재사용).

- 설정 인자·시스템 ID 규칙은 engine.py와 동일(--extract gazcomp --tag --glossary --pnw --prf-beta --prf-sent).
- 임베딩은 DEVICE 환경변수를 따른다(로컬 GPU 점유 시 DEVICE=cpu).
- 대조 모드(--control): 기존 태그·캐시(6월 GPU 임베딩)로 검색해 기존 predictions의 retrieved와
  일치율을 보고 → CPU 질의 임베딩·코드 경로가 검색을 바꾸지 않는지 확인.

산출: cache/track_02/rerun_v2/retrieved/{system_id}_{dataset}.json

실행:
  DEVICE=cpu python -m src.track_02.linearrag.rerun_retrieve --dataset wiki_ko --tag llm4bv2 \
      --glossary data/wiki_ko/glossary_llm4b_wiki_v2.json
  DEVICE=cpu python -m src.track_02.linearrag.rerun_retrieve --dataset wiki_ko --tag llm4b \
      --glossary data/wiki_ko/glossary_llm4b_wiki.json --control
"""
import argparse
import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset              # noqa: E402
from src.track_02.linearrag import engine as E                     # noqa: E402

OUT = ROOT / "cache" / "track_02" / "rerun_v2" / "retrieved"
# (pnw, prf_beta, prf_sent) — 기존 wiki llm4b 5개 런과 동일
CONFIGS = [(E.PASSAGE_NODE_WEIGHT, 0.0, False), (0.5, 0.0, False), (1.0, 0.0, False),
           (0.5, 0.6, False), (0.5, 0.6, True)]


def system_id(extract, tag, pnw, prf_beta, prf_sent):
    """engine.main()의 ID 규칙과 동일(weight=b0, rrf 없음)."""
    wsuf = ""
    if abs(pnw - E.PASSAGE_NODE_WEIGHT) > 1e-9:
        wsuf += f"-pnw{pnw:g}"
    if prf_beta > 0:
        wsuf += f"-prf{prf_beta:g}" + ("sent" if prf_sent else "")
    return f"linearrag-new-{extract}" + (f"-{tag}" if tag else "") + wsuf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extract", default="gazcomp")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--glossary", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--control", action="store_true",
                    help="기존 캐시로 검색 후 기존 predictions와 retrieved 일치율만 보고(저장 안 함)")
    ap.add_argument("--only-base", action="store_true", help="기본 설정(pnw 기본값) 하나만")
    args = ap.parse_args()

    passages, questions = load_dataset(args.dataset)
    OUT.mkdir(parents=True, exist_ok=True)
    configs = CONFIGS[:1] if args.only_base else CONFIGS
    # 메모리 보호: LinearRAG가 설정마다 bge-m3(약 2.3GB)를 새로 올리지 않도록 임베더 1개를 공유
    shared_emb = E.Embedder()
    E.Embedder = lambda *a, **k: shared_emb
    for pnw, beta, sent in configs:
        sid = system_id(args.extract, args.tag, pnw, beta, sent)
        t0 = time.time()
        rag = E.LinearRAG(args.dataset, args.extract, glossary_path=args.glossary, tag=args.tag,
                          pnw=pnw, prf_beta=beta, prf_sent=sent)
        rag.index(passages)
        print(f"[{sid}] 색인 {time.time()-t0:.0f}s 엔티티 {len(rag.entities)} 문장 {len(rag.sentences)} "
              f"device={rag.emb.device}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            retr = list(ex.map(lambda q: [str(r) for r in rag.retrieve_one(q["question"])], questions))
        rows = [{"id": q["id"], "question": q["question"], "answer": q["answer"],
                 "supporting": q["supporting"], "retrieved": r} for q, r in zip(questions, retr)]
        if args.control:
            old = {x["id"]: [str(r) for r in x["retrieved"]] for x in json.load(open(
                ROOT / "results/track_02" / f"{sid}_{args.dataset}" / "predictions.json", encoding="utf-8"))}
            same = sum(1 for x in rows if old.get(x["id"]) == x["retrieved"])
            same_set = sum(1 for x in rows if set(old.get(x["id"], [])) == set(x["retrieved"]))
            print(f"[control] {sid}: 기존과 top-5 순서까지 일치 {same}/{len(rows)}, 집합 일치 {same_set}/{len(rows)}",
                  flush=True)
            continue
        f = OUT / f"{sid}_{args.dataset}.json"
        json.dump({"system": sid, "dataset": args.dataset, "glossary": args.glossary,
                   "device": rag.emb.device, "elapsed_sec": round(time.time() - t0, 1), "rows": rows},
                  open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{sid}] 검색 저장 → {f.relative_to(ROOT)} ({time.time()-t0:.0f}s)", flush=True)
        del rag
        gc.collect()


if __name__ == "__main__":
    main()
