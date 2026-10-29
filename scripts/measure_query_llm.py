"""질의 1건당 생성형 LLM 사용량 측정 (가설 재정리 요청 3, 2026-10-18). 원고 <표 34>(2026-10-29 14단계 번호).

admin_full에서 표본 50문항(시드 42)을 뽑아 다음을 잰다. 동시 요청 1건(순차), 31B 서버 한 대.
 ① HippoRAG2 검색 단계의 필터 호출 — 31B 그래프·4B 그래프 각각. 검색을 그대로 재현해
    필터 호출의 입력·출력 토큰(vLLM usage)과 처리 시간을 기록하고, 상위 5개가 저장 결과와 같은지 확인.
 ② 응답 생성 — 같은 표본 × ◆ 4개 시스템. 저장된 상위 5개로 ◆ 응답 조건(QA_SYS, 출력 한도 2,000,
    temperature 0)을 그대로 재현해 토큰·시간을 기록하고, 답이 저장된 답과 같은지 확인.
 ③ LinearRAG 검색 단계 — comp + dense0.5 + PRF(문장)로 같은 표본을 검색하며 생성형 LLM 호출 수를 센다.

임베딩(bge-m3)은 CPU로 돌린다. 결과: results/track_02/query_llm_usage/{calls.jsonl, summary.json}
실행: CUDA_VISIBLE_DEVICES= DEVICE=cpu python scripts/measure_query_llm.py
"""
import json
import os
import random
import re
import statistics as st
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEVICE", "cpu")

import httpx  # noqa: E402
import numpy as np  # noqa: E402


def guard():
    while True:
        avail = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]) // 1024
        if avail < 4096:
            print(f'[중단] 남은 RAM {avail}MB < 4GB', flush=True)
            os._exit(3)
        time.sleep(2)


threading.Thread(target=guard, daemon=True).start()

from src.track_01.common.llm_client import LLMClient  # noqa: E402
from src.track_01.common.embedding import Embedder  # noqa: E402
from src.track_02.common.datasets import load_dataset  # noqa: E402
from src.track_02.graphrag import hipporag2 as H  # noqa: E402
from src.track_02.linearrag import engine as E  # noqa: E402

DS = "admin_full"
N_SAMPLE = 50
OUT = ROOT / "results" / "track_02" / "query_llm_usage"
OUT.mkdir(parents=True, exist_ok=True)
CALLS = open(OUT / "calls.jsonl", "w", encoding="utf-8")
QA_SYSTEMS = ["vanillarag", "graphrag-org", "linearrag-new-comp", "linearrag-new-gazcomp-llm4b"]

client = LLMClient()
BASE = client.url.split("/v1/")[0]


def server_load():
    try:
        met = httpx.get(BASE + "/metrics", headers=client.headers, timeout=10).text
        get = lambda k: float(re.search(rf'vllm:{k}\{{[^}}]*\}} ([0-9.]+)', met).group(1))
        return {"running": get("num_requests_running"), "waiting": get("num_requests_waiting")}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def call(messages, max_tokens):
    """→ (content, usage, 처리 시간 초). 동시 요청 1건."""
    payload = {"model": client.model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    t0 = time.perf_counter()
    r = client.client.post(client.url, headers=client.headers, json=payload)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], j["usage"], dt


def log(**kw):
    CALLS.write(json.dumps(kw, ensure_ascii=False) + "\n")
    CALLS.flush()


def stats(xs):
    return {"n": len(xs), "mean": round(st.mean(xs), 2), "median": round(st.median(xs), 2),
            "min": round(min(xs), 2), "max": round(max(xs), 2)}


passages, questions = load_dataset(DS)
texts = {str(p["id"]): p["text"] for p in passages}
sample = sorted(random.Random(42).sample(range(len(questions)), N_SAMPLE))
Q = [questions[i] for i in sample]
summary = {"dataset": DS, "n_sample": N_SAMPLE, "seed": 42, "server": BASE, "model": client.model,
           "concurrency": 1, "load_before": server_load(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
print(f"[시작] 표본 {N_SAMPLE}문항, 서버 부하 {summary['load_before']}", flush=True)

# ── ① HippoRAG2 필터 호출 ──
emb = Embedder(device="cpu")
qvecs = {q["id"]: np.asarray(emb.encode(q["question"]), dtype="float32") for q in Q}
summary["filter"] = {}
for variant, folder in (("", "graphrag-org"), ("4b", "graphrag-4b")):
    import igraph as ig
    cache, _, _ = H.paths(DS, variant)
    g = ig.Graph.Read_Pickle(str(cache / "graph.pkl"))
    meta = json.load(open(cache / "graph_meta.json", encoding="utf-8"))
    ent_list = meta["entities"]
    entities = {e: i for i, e in enumerate(ent_list)}
    facts = [tuple(f) for f in meta["facts"]]
    z = np.load(cache / "embeddings.npz")
    fact_vecs, pass_vecs = z["fact"], z["passage"]
    _, pids, _ = H.load_pp(DS)
    nE, nP = len(ent_list), len(pids)
    stored = {r["id"]: [str(x) for x in r["retrieved"][:5]]
              for r in json.load(open(ROOT / "results" / "track_02" / f"{folder}_{DS}" / "predictions.json",
                                      encoding="utf-8"))}
    rows = []
    for q in Q:
        qv = qvecs[q["id"]]
        fscores = fact_vecs @ qv
        top_f = np.argsort(fscores)[::-1][:H.FACT_TOPK]
        cand = [(facts[i], float(fscores[i])) for i in top_f]
        facts_json = json.dumps({"fact": [[s, r, o] for (s, r, o, _), _ in cand]}, ensure_ascii=False)
        out, usage, dt = call([{"role": "user", "content": H.FILTER_PROMPT.format(
            question=q["question"], facts=facts_json)}], 512)
        kept_raw = H.parse_json(out, "fact")
        kept = {tuple(H.norm_ent(x) for x in f[:3]): True for f in kept_raw if isinstance(f, list) and len(f) >= 3}
        reset = np.zeros(nE + nP)
        dpr = pass_vecs @ qv
        reset[nE:] = (dpr - dpr.min()) / (dpr.max() - dpr.min() + 1e-9) * H.PASSAGE_NODE_WEIGHT
        linked = 0
        for (s, r, o, pi), sc in cand:
            if (H.norm_ent(s), H.norm_ent(r), H.norm_ent(o)) in kept or (s, H.norm_ent(r), o) in kept:
                for e in (s, o):
                    if e in entities:
                        reset[entities[e]] += max(sc, 0.0)
                        linked += 1
        if linked == 0:
            top, mode = [pids[i] for i in np.argsort(dpr)[::-1][:H.RETRIEVAL_TOPK]], "dense_fallback"
        else:
            pr = np.array(g.personalized_pagerank(damping=H.DAMPING, reset=reset.tolist(), weights="weight")[nE:])
            top, mode = [pids[i] for i in np.argsort(pr)[::-1][:H.RETRIEVAL_TOPK]], "ppr"
        same = top == stored[q["id"]]
        rows.append((usage["prompt_tokens"], usage["completion_tokens"], dt, same, mode))
        log(stage="filter", graph=folder, id=q["id"], prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"], seconds=round(dt, 3), mode=mode, top5_same_as_stored=same)
    summary["filter"][folder] = {
        "calls_per_query": 1, "prompt_tokens": stats([r[0] for r in rows]),
        "completion_tokens": stats([r[1] for r in rows]), "seconds": stats([r[2] for r in rows]),
        "top5_same_as_stored": sum(r[3] for r in rows), "ppr_mode": sum(r[4] == "ppr" for r in rows)}
    print(f"[필터] {folder}: {summary['filter'][folder]}", flush=True)
    del g, meta, facts, z, fact_vecs, pass_vecs
del emb

# ── ② 응답 생성 ──
summary["qa"] = {}
for sysname in QA_SYSTEMS:
    preds = {r["id"]: r for r in json.load(open(ROOT / "results" / "track_02" / f"{sysname}_{DS}" / "predictions.json",
                                                 encoding="utf-8"))}
    rows = []
    for q in Q:
        x = preds[q["id"]]
        ctx = "\n".join(texts[str(r)] for r in x["retrieved"][:5])
        out, usage, dt = call([{"role": "system", "content": E.QA_SYS},
                               {"role": "user", "content": f"{ctx}\nQuestion: {q['question']}\n Thought: "}], 2000)
        pred = out.split("Answer:")[-1].strip() if "Answer:" in out else out.strip()
        same = pred == (x.get("prediction") or "").strip()
        rows.append((usage["prompt_tokens"], usage["completion_tokens"], dt, same))
        log(stage="qa", system=sysname, id=q["id"], prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"], seconds=round(dt, 3), answer_same_as_stored=same)
    summary["qa"][sysname] = {
        "calls_per_query": 1, "prompt_tokens": stats([r[0] for r in rows]),
        "completion_tokens": stats([r[1] for r in rows]), "seconds": stats([r[2] for r in rows]),
        "answer_same_as_stored": sum(r[3] for r in rows)}
    print(f"[응답] {sysname}: {summary['qa'][sysname]}", flush=True)

# ── ③ LinearRAG 검색 단계의 생성형 LLM 호출 수 ──
count = {"n": 0}
_orig_full, _orig_post = LLMClient.chat_full, httpx.Client.post


def spy_full(self, *a, **k):
    count["n"] += 1
    return _orig_full(self, *a, **k)


def spy_post(self, url, *a, **k):
    if "/chat/completions" in str(url) or "/completions" in str(url):
        count["n"] += 1
    return _orig_post(self, url, *a, **k)


LLMClient.chat_full, httpx.Client.post = spy_full, spy_post
rag = E.LinearRAG(DS, "comp", pnw=0.5, prf_beta=0.6, prf_sent=True)
rag.index(passages)
stored = {r["id"]: [str(x) for x in r["retrieved"][:5]]
          for r in json.load(open(ROOT / "results" / "track_02" / f"linearrag-new-comp-pnw0.5-prf0.6sent_{DS}"
                                  / "predictions.json", encoding="utf-8"))}
count["n"] = 0                                   # 색인 로드 이후부터 센다
same, secs = 0, []
for q in Q:
    t0 = time.perf_counter()
    top = [str(x) for x in rag.retrieve_one(q["question"])]
    secs.append(time.perf_counter() - t0)
    same += top == stored[q["id"]]
LLMClient.chat_full, httpx.Client.post = _orig_full, _orig_post
summary["linearrag_retrieval"] = {"system": "comp +dense0.5 +PRF(문장)", "generative_llm_calls": count["n"],
                                  "queries": N_SAMPLE, "top5_same_as_stored": same,
                                  "seconds_cpu": stats(secs)}
print(f"[LinearRAG] {summary['linearrag_retrieval']}", flush=True)

summary["load_after"] = server_load()
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("[완료]", OUT / "summary.json", flush=True)
