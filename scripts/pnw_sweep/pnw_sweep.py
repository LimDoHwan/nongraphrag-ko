"""dense 가중치(passage_node_weight, pnw) 스윕 — admin R@5, 검색만 (2026-10-18).

engine.py 는 수정하지 않고 내부 함수(_seed_entities, _activate, 그래프·임베딩)를 재사용한다.
pnw 는 PPR 시작 벡터에서 문서 쪽 점수에 곱하는 배수일 뿐이므로(engine._passage_weights),
질문당 시드 선택·활성화 전파·문서별 엔티티 보너스는 한 번만 계산하고 PPR 만 pnw 마다 다시 돌린다.
문서 점수·PRF(문장)·PPR 계산은 엔진과 같은 식·같은 자료형으로 옮겼다(저장 결과 대조로 확인).

실행: python scripts/pnw_sweep/pnw_sweep.py <추출> <집합> [--prf]
  추출: noun | comp | gazcomp | gazcomp_llm4b | gazcomp_llm | compsp
  집합: admin_single | admin_full | admin_hard
출력: cache/track_02/pnw_sweep/{추출}[_prf]_{집합}.json
"""
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402


def guard():
    while True:
        avail = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]) // 1024
        if avail < 4096:
            print(f'[중단] 남은 RAM {avail}MB < 4GB', flush=True)
            os._exit(3)
        time.sleep(2)


threading.Thread(target=guard, daemon=True).start()

import src.track_02.linearrag.engine as E  # noqa: E402
from src.track_02.common.datasets import load_dataset  # noqa: E402

PNWS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
PRF_BETA = 0.6
OUT = E.CACHE / "pnw_sweep"
GLOSS = {  # 논문 gazcomp 와 같은 사전 (systems.csv)
    "gazcomp": {"admin_single": "data/admin_ko/glossary.json", "admin_full": "data/admin_ko/glossary.json",
                "admin_hard": "data/admin_ko/glossary.json"},
    "gazcomp_llm4b": {"admin_single": "data/admin_ko/glossary_llm4b_single.json",
                      "admin_full": "data/admin_ko/glossary_llm4b_full.json",
                      "admin_hard": "data/admin_ko/glossary_llm4b_full.json"},
    "gazcomp_llm": {"admin_single": "data/admin_ko/glossary_llm_single.json",
                    "admin_full": "data/admin_ko/glossary_llm_full.json",
                    "admin_hard": "data/admin_ko/glossary_llm_full.json"},
}


class CompSpNER(E.KiwiNER):
    """compsp: comp 엔티티(연속 명사 결합) + 공백으로 나뉜 어절 조각 (scripts/explore_compsp 와 같음)."""
    def __init__(self):
        super().__init__('comp')

    def _toks_to_ents(self, tokens):
        ents, run = [], []

        def flush():
            if not run:
                return
            ents.append(''.join(t.form for t in run))
            segs = [[run[0]]]
            for a, b in zip(run, run[1:]):
                if b.start > a.start + a.len:
                    segs.append([b])
                else:
                    segs[-1].append(b)
            if len(segs) > 1:
                ents.extend(''.join(t.form for t in s) for s in segs)
        for t in tokens:
            if t.tag in self.NOUN:
                run.append(t)
            else:
                flush()
                run.clear()
        flush()
        return E._dedup(ents)


def build(extract, ds):
    passages, questions = load_dataset(ds)
    if extract in GLOSS:
        tag = extract.split("_", 1)[1] if "_" in extract else ""
        rag = E.LinearRAG(ds, "gazcomp", glossary_path=str(ROOT / GLOSS[extract][ds]), tag=tag)
    elif extract == "compsp":
        rag = E.LinearRAG(ds, "comp")
        rag.ner, rag.extract = CompSpNER(), "compsp"
        rag.cdir = E.CACHE / f"linrag_compsp_{ds}"
    else:
        rag = E.LinearRAG(ds, extract)
    rag.index(passages)
    return rag, questions


def log_bonus(rag, actived):
    """문서별 log(1 + 엔티티 보너스) — 엔진 _passage_weights 와 같은 순서·식 (질의 벡터와 무관)."""
    out = []
    for pi in range(len(rag.passages)):
        text = rag.ptext[pi]
        bonus = 0.0
        for ei, (sc, tier) in actived.items():
            occ = text.count(rag.entities[ei])
            if occ > 0:
                bonus += sc * math.log(1 + occ) / (tier if tier >= 1 else 1)
        out.append(math.log(1 + bonus))
    return np.array(out, dtype=np.float32)


def doc_score(rag, qv, logb):
    """엔진의 score = PASSAGE_RATIO * norm + log(1 + bonus) (float32, pnw 곱하기 전)."""
    sims = rag.pemb @ qv
    smin = sims.min()
    rng = sims.max() - smin or 1.0
    norm = (sims - smin) / rng
    return E.PASSAGE_RATIO * norm + logb


def ppr_order(rag, ent_w, score, pnw):
    pw = np.zeros(rag.g.vcount())
    pw[rag.p_vidx] = score * pnw
    nw = ent_w + pw
    reset = np.where((np.isnan(nw)) | (nw < 0), 0, nw)
    empty = reset.sum() == 0
    pr = rag.g.personalized_pagerank(damping=E.DAMPING, directed=False, weights="weight",
                                     reset=None if empty else reset, implementation="prpack")
    dsc = np.array([pr[v] for v in rag.p_vidx])
    return list(np.argsort(-dsc)), empty


def prf_query(rag, qv, lin_order, actived, sent_cache):
    """엔진 retrieve_one 의 문장-수준 PRF 와 같은 계산. 상위 3개 문서가 같으면 문장 임베딩을 재사용."""
    key = tuple(int(i) for i in lin_order[:3])
    if key not in sent_cache:
        sents = []
        for pi in lin_order[:3]:
            for s in rag.ner.kiwi.split_into_sents(rag.ptext[pi]):
                st = s.text.strip()
                if len(st) >= 10:
                    sents.append(st)
        se = None
        if sents:
            se = np.asarray(rag.emb.encode(sents, batch_size=64), dtype=np.float32)
            se = se / (np.linalg.norm(se, axis=1, keepdims=True) + 1e-9)
        sent_cache[key] = (sents, se)
    sents, se = sent_cache[key]
    ctx = None
    if sents:
        sim = se @ qv
        act_surface = [rag.entities[ei] for ei in actived]
        idxs = [i for i, st in enumerate(sents) if any(e in st for e in act_surface)]
        if not idxs:
            idxs = list(range(len(sents)))
        idxs = sorted(idxs, key=lambda i: -sim[i])[:6]
        ctx = se[idxs].mean(axis=0)
    if ctx is None:
        ctx = rag.pemb[lin_order[:3]]
        ctx = (ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)).mean(axis=0)
    qv2 = qv + PRF_BETA * ctx
    return (qv2 / (np.linalg.norm(qv2) + 1e-9)).astype(np.float32)


def main():
    extract, ds = sys.argv[1], sys.argv[2]
    prf = "--prf" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / f"{extract}{'_prf' if prf else ''}_{ds}.json"
    t0 = time.time()
    rag, questions = build(extract, ds)
    t_load = time.time() - t0
    top = {str(p): {} for p in PNWS}
    noseed, empty = 0, {str(p): 0 for p in PNWS}
    t1 = time.time()
    for k, q in enumerate(questions):
        qv = np.asarray(rag.emb.encode(q["question"], batch_size=1), dtype=np.float32)
        seed_idx, seed_scores = rag._seed_entities(q["question"])
        if not seed_idx:                                     # 질문 엔티티 없음 → 밀집 검색 상위 5개
            noseed += 1
            dense = [rag.pid[i] for i in np.argsort(-(rag.pemb @ qv))[:E.TOPK]]
            for p in PNWS:
                top[str(p)][q["id"]] = [str(x) for x in dense]
            continue
        ent_w, actived = rag._activate(qv, seed_idx, seed_scores)
        logb = log_bonus(rag, actived)
        score = doc_score(rag, qv, logb)
        sent_cache = {}
        for p in PNWS:
            order, emp = ppr_order(rag, ent_w, score, p)
            if prf:
                qv2 = prf_query(rag, qv, order, actived, sent_cache)
                order, emp = ppr_order(rag, ent_w, doc_score(rag, qv2, logb), p)
            empty[str(p)] += int(emp)
            top[str(p)][q["id"]] = [str(rag.pid[i]) for i in order[:E.TOPK]]
        if (k + 1) % 50 == 0:
            print(f"  {extract}{' PRF' if prf else ''} {ds} {k+1}/{len(questions)} ({time.time()-t1:.0f}s)", flush=True)
    meta = {"extract": extract, "dataset": ds, "prf_sent_beta": PRF_BETA if prf else 0.0, "pnws": PNWS,
            "n": len(questions), "no_seed_questions": noseed, "empty_reset_global_pagerank": empty,
            "load_seconds": round(t_load, 1), "query_seconds": round(time.time() - t1, 1)}
    json.dump({"meta": meta, "top5": top}, open(out_f, "w"), ensure_ascii=False)
    print(f"[완료] {out_f.name} 시드 없음 {noseed}/{len(questions)}, 로드 {t_load:.0f}s, "
          f"검색 {time.time()-t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
