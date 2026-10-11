"""Phase 5 - HippoRAG2 한국어 재구현 (faithful reimplementation).

공식 HippoRAG v2 (OSU-NLP-Group/HippoRAG) 의 방법을 동일 환경(Gemma + bge-m3 +
igraph)에서 재구현한다. 공식 패키지를 쓰지 않은 이유: 쿠키 인증 Gemma / bge-m3 /
transformers 4.x 핀과의 호환. 논문 서술 시 "재구현(reimplementation)"으로 명시.

공식 구현과의 대응:
- OpenIE 프롬프트: 공식 templates/ner.py, triple_extraction.py 를 한국어 번역
  (one-shot 예시 포함, JSON 형식 동일)
- 그래프: entity(phrase) 노드 + passage 노드, relation/mention/synonym 엣지
  (synonym: 임베딩 cosine >= 0.8, 공식 기본값)
- 검색: query -> dense fact(triple) top-k -> LLM recognition filter (공식
  filter_default_prompt 의 역할) -> 필터된 fact 의 entity 를 PPR reset 가중치로,
  passage 노드에는 dense 유사도 * passage_node_weight(0.05) -> PPR(damping 0.5)
- QA: LinearRAG 와 동일한 한국어 Thought/Answer 프롬프트 (시스템 간 공정 비교)

체크포인트: cache/track_01/hipporag2/{openie,qa}/ 항목별 캐시 -> resume 가능.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.track_01.common.embedding import Embedder      # noqa: E402
from src.track_01.common.llm_client import LLMClient    # noqa: E402

DATA = ROOT / "data" / "2wiki_ko"
CACHE = ROOT / "cache" / "track_01" / "hipporag2"
RESULTS = ROOT / "results" / "track_01" / "hipporag2"

# ---------------- 공식 프롬프트의 한국어 번역 (one-shot 유지) ----------------
NER_SYSTEM = """주어진 단락에서 개체명(named entity)을 추출하는 것이 당신의 임무입니다.
JSON 객체 {"named_entities": [...]} 형식으로만 응답하세요."""

ONE_SHOT_PARA = """라디오 시티
라디오 시티는 인도 최초의 민영 FM 라디오 방송국으로 2001년 7월 3일에 개국했다.
힌디어, 영어 및 지역 노래를 방송한다.
라디오 시티는 2008년 5월 음악 포털 PlanetRadiocity.com 을 출시하며 뉴미디어에 진출했으며, 이 포털은 음악 관련 뉴스, 비디오, 노래 등의 기능을 제공한다."""

ONE_SHOT_NER_OUT = """{"named_entities":
    ["라디오 시티", "인도", "2001년 7월 3일", "힌디어", "영어", "2008년 5월", "PlanetRadiocity.com"]
}"""

RE_SYSTEM = """주어진 단락과 개체명 리스트로부터 RDF(Resource Description Framework) 그래프를 구성하는 것이 당신의 임무입니다.
RDF 그래프의 관계를 나타내는 트리플들의 JSON 리스트로 응답하세요.

다음 요구사항에 유의하세요:
- 각 트리플은 단락의 개체명 리스트 중 최소 1개, 가급적 2개를 포함해야 합니다.
- 대명사는 명확성을 위해 구체적인 이름으로 해소하세요.
- {"triples": [["주어", "관계", "목적어"], ...]} 형식으로만 응답하세요."""

RE_FRAME = """다음 단락을 개체명 리스트와 트리플 리스트를 갖는 JSON 으로 변환하세요.
단락:
```
{passage}
```

{named_entity_json}"""

ONE_SHOT_RE_OUT = """{"triples": [
            ["라디오 시티", "위치한 국가", "인도"],
            ["라디오 시티", "이다", "민영 FM 라디오 방송국"],
            ["라디오 시티", "개국일", "2001년 7월 3일"],
            ["라디오 시티", "방송 언어", "힌디어"],
            ["라디오 시티", "방송 언어", "영어"],
            ["라디오 시티", "출시함", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "출시 시기", "2008년 5월"],
            ["PlanetRadiocity.com", "이다", "음악 포털"],
            ["PlanetRadiocity.com", "제공", "뉴스"],
            ["PlanetRadiocity.com", "제공", "비디오"],
            ["PlanetRadiocity.com", "제공", "노래"]
    ]
}"""

FILTER_PROMPT = """질문에 답하는 데 유용한 사실(fact)만 남기고 무관한 것은 제거하세요.
입력과 동일한 JSON 형식 {{"fact": [["주어","관계","목적어"], ...]}} 으로만 응답하세요.
유용한 사실이 없으면 {{"fact": []}} 로 응답하세요.

질문: {question}
사실 후보: {facts}"""

QA_SYS = ('당신은 고급 독해 어시스턴트입니다. 주어진 지문들과 질문을 면밀히 분석하세요. '
          '응답은 반드시 "Thought: " 다음에 한국어로 단계별 추론 과정을 서술하고, '
          '마지막에 "Answer: " 다음에 간결하고 명확한 최종 답만 제시하세요(불필요한 부연 없이).')

_JSON_RE = re.compile(r'\{.*\}', re.S)

# 공식 기본값
SYNONYM_THRESHOLD = 0.8
SYNONYM_TOPK = 5
PASSAGE_NODE_WEIGHT = 0.05
DAMPING = 0.5
FACT_TOPK = 5
RETRIEVAL_TOPK = 5


_TRIPLE_RE = re.compile(
    r'\[\s*"((?:[^"\\]|\\.)+?)"\s*,\s*"((?:[^"\\]|\\.)+?)"\s*,\s*"((?:[^"\\]|\\.)+?)"\s*\]')
_STR_RE = re.compile(r'"((?:[^"\\]|\\.)+?)"')


def parse_json(text, key):
    """완전한 JSON 우선, 실패 시(max_tokens 잘림 등) 항목 단위 salvage."""
    text = text or ""
    m = _JSON_RE.search(text)
    if m:
        try:
            v = json.loads(m.group(0)).get(key, [])
            if isinstance(v, list) and v:
                return v
        except json.JSONDecodeError:
            pass
    # salvage: 잘린 출력에서도 개별 항목 복구
    if key == "triples" or key == "fact":
        return [list(t) for t in _TRIPLE_RE.findall(text)]
    if key == "named_entities":
        idx = text.find("named_entities")
        start = text.find('[', idx) if idx >= 0 else text.find('[')
        seg = text[start:] if start >= 0 else text
        return _STR_RE.findall(seg)
    return []


def norm_ent(e):
    return re.sub(r"\s+", " ", str(e).strip().lower())


def load_passages():
    chunks = json.load(open(DATA / "chunks.json", encoding="utf-8"))
    out = []
    for c in chunks:
        m = re.match(r'^\d+:(.*)$', c, flags=re.S)
        out.append((m.group(1).strip() if m else c.strip()))
    return out


# ----------------------------- 1) OpenIE 색인 -----------------------------
def run_openie(workers):
    passages = load_passages()
    cache = CACHE / "openie"
    cache.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()
    stats = {"calls": 0}

    def work(i):
        cf = cache / f"{i}.json"
        if cf.exists():
            return i
        p = passages[i]
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
        stats["calls"] += 2
        json.dump({"named_entities": ents, "triples": triples},
                  open(cf, "w", encoding="utf-8"), ensure_ascii=False)
        return i

    todo = [i for i in range(len(passages))
            if not (cache / f"{i}.json").exists()]
    print(f"[openie] total={len(passages)} todo={len(todo)}", flush=True)
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
            if done % 25 == 0 or done == len(todo):
                print(f"[openie] {done}/{len(todo)} (err={err})", flush=True)


# ------------------------- 2) 그래프 + 임베딩 구축 -------------------------
def build_graph_and_embeddings():
    import igraph as ig
    passages = load_passages()
    cache = CACHE / "openie"
    openie = [json.load(open(cache / f"{i}.json", encoding="utf-8"))
              for i in range(len(passages))]

    entities = {}            # norm -> idx
    facts = []               # (s, r, o, passage_idx)
    ent_passages = []        # (ent_idx, passage_idx)
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

    emb = Embedder()
    print(f"[emb] device={emb.device}", flush=True)
    ent_vecs = np.asarray(emb.encode(ent_list, batch_size=64), dtype="float32")
    fact_texts = [f"{s} {r} {o}" for s, r, o, _ in facts]
    fact_vecs = np.asarray(emb.encode(fact_texts, batch_size=64), dtype="float32")
    pass_vecs = np.asarray(emb.encode(passages, batch_size=16), dtype="float32")

    nE, nP = len(ent_list), len(passages)
    g = ig.Graph(directed=False)
    g.add_vertices(nE + nP)   # 0..nE-1 entity, nE..nE+nP-1 passage
    edges, weights = {}, None

    def add_edge(a, b, w):
        k = (a, b) if a < b else (b, a)
        edges[k] = edges.get(k, 0.0) + w

    for s, r, o, pi in facts:
        add_edge(entities[s], entities[o], 1.0)
    for ei, pi in ent_passages:
        add_edge(ei, nE + pi, 1.0)
    # synonym 엣지 (cosine >= 0.8, top-5)
    sim = ent_vecs @ ent_vecs.T
    np.fill_diagonal(sim, 0)
    for i in range(nE):
        idxs = np.argsort(sim[i])[::-1][:SYNONYM_TOPK]
        for j in idxs:
            if sim[i, j] >= SYNONYM_THRESHOLD:
                add_edge(i, int(j), float(sim[i, j]))

    g.add_edges(list(edges.keys()))
    g.es["weight"] = list(edges.values())
    print(f"[graph] nodes={g.vcount()} edges={g.ecount()}", flush=True)

    np.savez(CACHE / "embeddings.npz", ent=ent_vecs, fact=fact_vecs,
             passage=pass_vecs)
    json.dump({"entities": ent_list,
               "facts": [[s, r, o, pi] for s, r, o, pi in facts]},
              open(CACHE / "graph_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    g.write_pickle(str(CACHE / "graph.pkl"))
    return g, ent_list, entities, facts, ent_vecs, fact_vecs, pass_vecs


# ------------------------------ 3) 검색 + QA ------------------------------
def run_retrieval_qa(workers):
    import igraph as ig
    passages = load_passages()
    questions = json.load(open(DATA / "questions.json", encoding="utf-8"))

    if (CACHE / "graph.pkl").exists():
        g = ig.Graph.Read_Pickle(str(CACHE / "graph.pkl"))
        meta = json.load(open(CACHE / "graph_meta.json", encoding="utf-8"))
        ent_list = meta["entities"]
        entities = {e: i for i, e in enumerate(ent_list)}
        facts = [tuple(f) for f in meta["facts"]]
        z = np.load(CACHE / "embeddings.npz")
        ent_vecs, fact_vecs, pass_vecs = z["ent"], z["fact"], z["passage"]
        print(f"[load] graph nodes={g.vcount()} edges={g.ecount()}", flush=True)
    else:
        g, ent_list, entities, facts, ent_vecs, fact_vecs, pass_vecs = \
            build_graph_and_embeddings()

    nE, nP = len(ent_list), len(passages)
    emb = Embedder()
    llm = LLMClient()
    qa_cache = CACHE / "qa"
    qa_cache.mkdir(parents=True, exist_ok=True)

    def retrieve(question):
        qv = np.asarray(emb.encode(question), dtype="float32")
        # fact recall (dense)
        fscores = fact_vecs @ qv
        top_f = np.argsort(fscores)[::-1][:FACT_TOPK]
        cand = [(facts[i], float(fscores[i])) for i in top_f]
        # recognition filter (LLM)
        facts_json = json.dumps(
            {"fact": [[s, r, o] for (s, r, o, _), _ in cand]},
            ensure_ascii=False)
        kept_raw = parse_json(
            llm.chat([{"role": "user", "content": FILTER_PROMPT.format(
                question=question, facts=facts_json)}], max_tokens=512),
            "fact")
        kept = {tuple(norm_ent(x) for x in f[:3]): True
                for f in kept_raw if isinstance(f, list) and len(f) >= 3}
        # PPR reset 가중치
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
        if linked == 0:   # 필터 전멸 -> dense fallback
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
            pids, mode = retrieve(q["question"])
            user = "".join(f"{passages[p]}\n" for p in pids)
            user += f"Question: {q['question']}\n Thought: "
            out = llm.chat([{"role": "system", "content": QA_SYS},
                            {"role": "user", "content": user}],
                           max_tokens=2000)
            pred = out.split("Answer:")[1].strip() if "Answer:" in out else out
        except Exception as e:  # noqa: BLE001
            pids, mode, out, pred = [], "error", "", f"[ERROR] {e}"
        json.dump({"id": q.get("id"), "question": q["question"],
                   "gold_answer": q.get("answer"), "pred_answer": pred,
                   "retrieved_ids": pids, "mode": mode, "raw": out},
                  open(cf, "w", encoding="utf-8"), ensure_ascii=False)
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

    results = [json.load(open(qa_cache / f"{i}.json", encoding="utf-8"))
               for i in range(len(questions))
               if (qa_cache / f"{i}.json").exists()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "predictions.json"
    json.dump(results, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"wrote {out} ({len(results)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["openie", "graph", "qa", "all"],
                    default="all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.stage in ("openie", "all"):
        run_openie(args.workers)
    if args.stage == "graph":
        build_graph_and_embeddings()
    if args.stage in ("qa", "all"):
        run_retrieval_qa(args.workers)


if __name__ == "__main__":
    main()
