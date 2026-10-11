"""Track 2 - HippoRAG2 baseline (그래프 RAG, 공식 재구현).

track_01 의 한국어 HippoRAG2 재구현을 track_02 데이터셋/평가로 이식.
- OpenIE 프롬프트·상수·파서·norm_ent 는 track_01.graph_rag.hipporag2_korean 재사용
- 데이터 로딩은 track_02.common.datasets (passage_id 보존 → both@5 평가)
- 평가: Contain-Acc / GPT-Acc(Gemma judge) / both@5 (LinearRAG·Vanilla 와 동일 지표)

그래프: entity(phrase)+passage 노드 / relation·mention·synonym 엣지 (cos≥0.8 top5).
검색: dense fact top-k → LLM recognition filter → 엔티티 PPR reset → PPR(damping0.5) → top-5.
캐시: cache/track_02/hipporag2_{dataset}/{openie,qa}/ 항목별 → resume.
산출: results/track_02/graphrag-org_{dataset}/predictions.json + eval.json

실행: python -m src.track_02.graphrag.hipporag2 --dataset admin_hard2k --stage all --workers 8
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.track_01.common.embedding import Embedder              # noqa: E402
from src.track_01.common.llm_client import LLMClient            # noqa: E402
from src.track_01.common.evaluation import contain_acc, gpt_acc  # noqa: E402
from src.track_02.common.datasets import load_dataset           # noqa: E402
from src.track_01.graph_rag.hipporag2_korean import (           # noqa: E402
    NER_SYSTEM, ONE_SHOT_PARA, ONE_SHOT_NER_OUT,
    RE_SYSTEM, RE_FRAME, ONE_SHOT_RE_OUT, FILTER_PROMPT, QA_SYS,
    parse_json, norm_ent,
    SYNONYM_THRESHOLD, SYNONYM_TOPK, PASSAGE_NODE_WEIGHT,
    DAMPING, FACT_TOPK, RETRIEVAL_TOPK)

CACHE_ROOT = ROOT / "cache" / "track_02"
RESULTS_ROOT = ROOT / "results" / "track_02"


def paths(dataset, variant=""):
    """variant="": 31B OpenIE(기본). variant="4b": 캐시된 4b OpenIE 재사용.
    variant="4bv2": 컨텍스트 한도 교란 제거 후 재추출한 E4B RE(openie_4b_rerun.py) 사용.
    반환 (graph_cache_dir, results_dir, openie_dir)."""
    if variant == "4bv2":
        return (CACHE_ROOT / f"hipporag2_4bv2_{dataset}",
                RESULTS_ROOT / f"graphrag-4bv2_{dataset}",
                CACHE_ROOT / f"openie_4b_v2_{dataset}" / "openie")
    if variant == "4b":
        return (CACHE_ROOT / f"hipporag2_4b_{dataset}",
                RESULTS_ROOT / f"graphrag-4b_{dataset}",
                CACHE_ROOT / f"openie_4b_{dataset}" / "openie")
    return (CACHE_ROOT / f"hipporag2_{dataset}",
            RESULTS_ROOT / f"graphrag-org_{dataset}",
            CACHE_ROOT / f"hipporag2_{dataset}" / "openie")


def load_pp(dataset):
    """track_02 데이터셋 → (texts, passage_ids, questions)."""
    passages, questions = load_dataset(dataset)
    texts = [p["text"] for p in passages]
    pids = [str(p["id"]) for p in passages]
    return texts, pids, questions


# ----------------------------- 1) OpenIE 색인 -----------------------------
def run_openie(dataset, workers):
    texts, _, _ = load_pp(dataset)
    cache, _ = paths(dataset)
    cache = cache / "openie"
    cache.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()

    def work(i):
        cf = cache / f"{i}.json"
        if cf.exists():
            return i
        p = texts[i]
        ner_msgs = [
            {"role": "system", "content": NER_SYSTEM},
            {"role": "user", "content": ONE_SHOT_PARA},
            {"role": "assistant", "content": ONE_SHOT_NER_OUT},
            {"role": "user", "content": p},
        ]
        ents = parse_json(llm.chat(ner_msgs, max_tokens=3000), "named_entities")
        ne_json = json.dumps({"named_entities": ents}, ensure_ascii=False)
        re_msgs = [
            {"role": "system", "content": RE_SYSTEM},
            {"role": "user", "content": RE_FRAME.format(
                passage=ONE_SHOT_PARA, named_entity_json=ONE_SHOT_NER_OUT)},
            {"role": "assistant", "content": ONE_SHOT_RE_OUT},
            {"role": "user", "content": RE_FRAME.format(
                passage=p, named_entity_json=ne_json)},
        ]
        triples_raw = parse_json(llm.chat(re_msgs, max_tokens=6000), "triples")
        triples = [t for t in triples_raw
                   if isinstance(t, list) and len(t) == 3
                   and all(isinstance(x, str) and x.strip() for x in t)]
        json.dump({"named_entities": ents, "triples": triples},
                  open(cf, "w", encoding="utf-8"), ensure_ascii=False)
        return i

    todo = [i for i in range(len(texts)) if not (cache / f"{i}.json").exists()]
    print(f"[openie] total={len(texts)} todo={len(todo)}", flush=True)
    done = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"[openie] ERROR {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"[openie] {done}/{len(todo)} (err={err})", flush=True)


# ------------------------- 2) 그래프 + 임베딩 구축 -------------------------
def build_graph_and_embeddings(dataset, variant=""):
    import igraph as ig
    texts, _, _ = load_pp(dataset)
    cache, _, oe_dir = paths(dataset, variant)
    def _load_oe(i):
        f = oe_dir / f"{i}.json"
        if f.exists():
            return json.load(open(f, encoding="utf-8"))
        return {"named_entities": [], "triples": []}  # 추출 실패분(극소수) 무시
    openie = [_load_oe(i) for i in range(len(texts))]

    entities = {}
    facts = []
    ent_passages = []
    for pi, oe in enumerate(openie):
        seen_in_p = set()
        for t in oe["triples"]:
            s, r, o = (norm_ent(t[0]), str(t[1]).strip(), norm_ent(t[2]))
            if not s or not o:
                continue
            for e in (s, o):
                if e not in entities:
                    entities[e] = len(entities)
                if e not in seen_in_p:
                    seen_in_p.add(e)
                    ent_passages.append((entities[e], pi))
            facts.append((s, r, o, pi))
        for e_raw in oe["named_entities"]:
            e = norm_ent(e_raw)
            if e and e not in seen_in_p:
                if e not in entities:
                    entities[e] = len(entities)
                seen_in_p.add(e)
                ent_passages.append((entities[e], pi))

    ent_list = [None] * len(entities)
    for e, i in entities.items():
        ent_list[i] = e
    print(f"[graph] entities={len(ent_list)} facts={len(facts)} "
          f"mentions={len(ent_passages)}", flush=True)

    nE, nP = len(ent_list), len(texts)
    # 임베딩 캐시(있으면 로드 — 대형 코퍼스 재구축 시 GPU 시간 절약)
    embf = cache / "embeddings.npz"
    if embf.exists():
        z = np.load(embf)
        ent_vecs, fact_vecs, pass_vecs = z["ent"], z["fact"], z["passage"]
        if len(ent_vecs) == nE and len(pass_vecs) == nP:
            print(f"[emb] cached 로드 ent={len(ent_vecs)}", flush=True)
        else:
            embf.unlink(); ent_vecs = None
    else:
        ent_vecs = None
    if ent_vecs is None:
        emb = Embedder()
        print(f"[emb] device={emb.device}", flush=True)
        ent_vecs = np.asarray(emb.encode(ent_list, batch_size=64), dtype="float32")
        fact_texts = [f"{s} {r} {o}" for s, r, o, _ in facts]
        fact_vecs = np.asarray(emb.encode(fact_texts, batch_size=64), dtype="float32")
        pass_vecs = np.asarray(emb.encode(texts, batch_size=16), dtype="float32")
        np.savez(embf, ent=ent_vecs, fact=fact_vecs, passage=pass_vecs)  # sim 전 저장(크래시 대비)
        print("[emb] 저장 완료", flush=True)

    g = ig.Graph(directed=False)
    g.add_vertices(nE + nP)
    edges = {}

    def add_edge(a, b, w):
        k = (a, b) if a < b else (b, a)
        edges[k] = edges.get(k, 0.0) + w

    for s, r, o, pi in facts:
        add_edge(entities[s], entities[o], 1.0)
    for ei, pi in ent_passages:
        add_edge(ei, nE + pi, 1.0)
    # synonym 엣지: 전체 nE×nE 행렬(28GB+)을 피해 블록 단위로 top-k 계산
    BLK = 2048
    for b0 in range(0, nE, BLK):
        b1 = min(b0 + BLK, nE)
        sib = ent_vecs[b0:b1] @ ent_vecs.T          # (blk, nE) — ~0.7GB/블록
        for r, i in enumerate(range(b0, b1)):
            row = sib[r]
            row[i] = 0.0
            idxs = np.argpartition(row, -SYNONYM_TOPK)[-SYNONYM_TOPK:]
            for j in idxs:
                if row[j] >= SYNONYM_THRESHOLD:
                    add_edge(i, int(j), float(row[j]))
        print(f"[graph] synonym {b1}/{nE}", flush=True)

    g.add_edges(list(edges.keys()))
    g.es["weight"] = list(edges.values())
    print(f"[graph] nodes={g.vcount()} edges={g.ecount()}", flush=True)

    json.dump({"entities": ent_list,
               "facts": [[s, r, o, pi] for s, r, o, pi in facts]},
              open(cache / "graph_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    g.write_pickle(str(cache / "graph.pkl"))
    return g, ent_list, entities, facts, ent_vecs, fact_vecs, pass_vecs


# ------------------------------ 3) 검색 + QA ------------------------------
def run_retrieval_qa(dataset, workers, no_judge, variant=""):
    import igraph as ig
    texts, pids, questions = load_pp(dataset)
    cache, results, _ = paths(dataset, variant)
    cache.mkdir(parents=True, exist_ok=True)

    if (cache / "graph.pkl").exists():
        g = ig.Graph.Read_Pickle(str(cache / "graph.pkl"))
        meta = json.load(open(cache / "graph_meta.json", encoding="utf-8"))
        ent_list = meta["entities"]
        entities = {e: i for i, e in enumerate(ent_list)}
        facts = [tuple(f) for f in meta["facts"]]
        z = np.load(cache / "embeddings.npz")
        ent_vecs, fact_vecs, pass_vecs = z["ent"], z["fact"], z["passage"]
        print(f"[load] graph nodes={g.vcount()} edges={g.ecount()}", flush=True)
    else:
        g, ent_list, entities, facts, ent_vecs, fact_vecs, pass_vecs = \
            build_graph_and_embeddings(dataset, variant)

    nE, nP = len(ent_list), len(texts)
    emb = Embedder()
    llm = LLMClient()
    qa_cache = cache / "qa"
    qa_cache.mkdir(parents=True, exist_ok=True)

    def retrieve(question):
        qv = np.asarray(emb.encode(question), dtype="float32")
        fscores = fact_vecs @ qv
        top_f = np.argsort(fscores)[::-1][:FACT_TOPK]
        cand = [(facts[i], float(fscores[i])) for i in top_f]
        facts_json = json.dumps(
            {"fact": [[s, r, o] for (s, r, o, _), _ in cand]},
            ensure_ascii=False)
        kept_raw = parse_json(
            llm.chat([{"role": "user", "content": FILTER_PROMPT.format(
                question=question, facts=facts_json)}], max_tokens=512),
            "fact")
        kept = {tuple(norm_ent(x) for x in f[:3]): True
                for f in kept_raw if isinstance(f, list) and len(f) >= 3}
        reset = np.zeros(nE + nP)
        dpr = pass_vecs @ qv
        dpr_n = (dpr - dpr.min()) / (dpr.max() - dpr.min() + 1e-9)
        reset[nE:] = dpr_n * PASSAGE_NODE_WEIGHT
        linked = 0
        for (s, r, o, pi), sc in cand:
            if (norm_ent(s), norm_ent(r), norm_ent(o)) in kept or \
               (s, norm_ent(r), o) in kept:
                for e in (s, o):
                    if e in entities:
                        reset[entities[e]] += max(sc, 0.0)
                        linked += 1
        if linked == 0:
            order = np.argsort(dpr)[::-1][:RETRIEVAL_TOPK]
            return [int(i) for i in order], "dense_fallback"
        pr = g.personalized_pagerank(damping=DAMPING, reset=reset.tolist(),
                                     weights="weight")
        pscores = np.array(pr[nE:])
        order = np.argsort(pscores)[::-1][:RETRIEVAL_TOPK]
        return [int(i) for i in order], "ppr"

    def work(i):
        cf = qa_cache / f"{i}.json"
        if cf.exists():
            return i
        q = questions[i]
        try:
            idxs, mode = retrieve(q["question"])
            retr = [pids[p] for p in idxs]
            user = "".join(f"{texts[p]}\n" for p in idxs)
            user += f"Question: {q['question']}\n Thought: "
            out = llm.chat([{"role": "system", "content": QA_SYS},
                            {"role": "user", "content": user}],
                           max_tokens=2000)
            pred = out.split("Answer:")[1].strip() if "Answer:" in out else out
        except Exception as e:  # noqa: BLE001
            retr, mode, out, pred = [], "error", "", f"[ERROR] {e}"
        json.dump({"id": q["id"], "question": q["question"],
                   "answer": q["answer"], "prediction": pred.strip(),
                   "retrieved": retr, "supporting": q["supporting"],
                   "mode": mode}, open(cf, "w", encoding="utf-8"),
                  ensure_ascii=False)
        return i

    todo = [i for i in range(len(questions))
            if not (qa_cache / f"{i}.json").exists()]
    print(f"[qa] total={len(questions)} todo={len(todo)}", flush=True)
    done = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"[qa] ERROR {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"[qa] {done}/{len(todo)} (err={err})", flush=True)

    preds = [json.load(open(qa_cache / f"{i}.json", encoding="utf-8"))
             for i in range(len(questions))
             if (qa_cache / f"{i}.json").exists()]

    # 평가: Contain / GPT / both@5
    for p in preds:
        p["contain"] = contain_acc(p["prediction"], p["answer"])
    if not no_judge:
        def judge(p):
            return gpt_acc(p["question"], p["answer"], p["prediction"], llm)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, gv in zip(preds, ex.map(judge, preds)):
                p["gpt"] = gv

    results.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(results / "predictions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n = len(preds)
    both = sum(set(map(str, p["supporting"])) <= set(map(str, p["retrieved"]))
               for p in preds)
    summary = {"dataset": dataset,
               "system": f"graphrag-{variant}" if variant else "graphrag-org",
               "n": n, "topk": RETRIEVAL_TOPK,
               "contain_acc": round(sum(p["contain"] for p in preds) / n, 4),
               "both_at5": round(both / n, 4)}
    if not no_judge:
        summary["gpt_acc"] = round(sum(p["gpt"] for p in preds) / n, 4)
    json.dump(summary, open(results / "eval.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[{dataset}] HippoRAG2  {summary}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", choices=["openie", "graph", "qa", "all"],
                    default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--variant", default="", choices=["", "4b", "4bv2"],
                    help="4b: 캐시된 gemma-4-4b OpenIE로 그래프 구축")
    args = ap.parse_args()
    if args.stage in ("openie", "all") and not args.variant:
        run_openie(args.dataset, args.workers)  # 4b OpenIE는 openie_4b로 별도 추출됨
    if args.stage == "graph":
        build_graph_and_embeddings(args.dataset, args.variant)
    if args.stage in ("qa", "all"):
        run_retrieval_qa(args.dataset, args.workers, args.no_judge, args.variant)


if __name__ == "__main__":
    main()
