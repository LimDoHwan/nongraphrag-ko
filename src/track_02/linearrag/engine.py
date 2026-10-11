"""Track 2 - 메모리 방식 LinearRAG 재구현 (§3.1 원본 baseline / §6·§7 엔진).

원본(DEEP-PolyU/LinearRAG, 한국어 재현)을 충실히 따른다. 색인 = Tri-Graph
([문서]-[문장]-[엔티티]), 검색 = seed 엔티티 → 문장유사도 전파(활성화) →
passage 점수(DPR+엔티티 보너스) → PPR(igraph personalized pagerank).

엔티티 추출은 파라미터화(--extract): 'nnp'(원본 LinearRAG, kiwipiepy 고유명사) /
추후 A1~A5(명사·복합명사·사전·LLM)를 같은 엔진에 얹어 제안법까지 커버.

색인 비용: nnp 추출은 LLM 0회(형태소). QA 답변만 LLM(§9.3).
산출: results/track_02/linear_{extract}_{dataset}/predictions.json + eval.json
캐시: cache/track_02/linrag_{extract}_{dataset}/ (NER·임베딩)

실행: python -m src.track_02.linearrag.engine --dataset admin_ko --extract nnp --workers 8
"""
import argparse
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import igraph as ig
import numpy as np
from kiwipiepy import Kiwi

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache" / "track_02"
RESULTS = ROOT / "results" / "track_02"

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset            # noqa: E402
from src.track_01.common.embedding import Embedder        # noqa: E402
from src.track_01.common.llm_client import LLMClient      # noqa: E402
from src.track_01.common.evaluation import contain_acc, gpt_acc  # noqa: E402

# 원본 config 기본값
TOPK = 5
MAX_ITER = 3
TOPK_SENT = 1
PASSAGE_RATIO = 1.5
PASSAGE_NODE_WEIGHT = 0.05
DAMPING = 0.5
ITER_THRESH = 0.5

QA_SYS = ('당신은 고급 독해 어시스턴트입니다. 주어진 지문들과 질문을 분석해, '
          '"Thought: " 다음에 단계별 추론을, 마지막에 "Answer: " 다음에 간결한 최종 답만 적으세요.')


def _dedup(seq):
    seen, out = set(), []
    for e in seq:
        if e and e not in seen:
            seen.add(e); out.append(e)
    return out


class KiwiNER:
    """문장 분리 + 추출 (A0~A3g).
      nnp(A0)        : 고유명사(NNP)            — 원본 LinearRAG
      noun(A1)       : 명사(NNG+NNP)            — 파편화 베이스라인
      comp(A2)   : 연속 명사 무조건 묶기
      cpmi(A2p): 연속 명사 NPMI 게이팅 묶기 (사전 불요)
      dict(A3)       : 사전 등록 kiwi + 명사추출 (사전 복합명사 단일토큰 보존)
      gazet(A3g) : 사전 표제어 직접 longest-match + 잔여 명사
    """
    NOUN = ("NNG", "NNP")

    def __init__(self, extract="nnp", glossary=None, npmi=None, npmi_thresh=0.3,
                 idf=None, topk_kw=10):
        self.mode = extract
        self.tags = ("NNP",) if extract == "nnp" else self.NOUN
        self.npmi = npmi or {}
        self.npmi_thresh = npmi_thresh
        self.idf = idf or {}          # kw: 전역 IDF(token→idf)
        self.topk_kw = topk_kw        # kw: 패시지당 키프레이즈 수
        self.kiwi = Kiwi()
        if extract in ("dict", "compdict") and glossary:
            for w in glossary:
                self.kiwi.add_user_word(w, "NNP")
        self.auto = None
        if extract in ("gazet", "gazo", "gazcomp") and glossary:
            import ahocorasick
            self.auto = ahocorasick.Automaton()
            for w in glossary:
                if len(w) >= 2:
                    self.auto.add_word(w, w)
            self.auto.make_automaton()

    def _runs_to_ents(self, run):
        """명사 run(forms) -> 엔티티들. mode 에 따라 결합 규칙 적용."""
        if len(run) == 0:
            return []
        if self.mode in ("comp", "compdict"):   # 무조건 결합(+사전 토큰화)
            return ["".join(run)]
        if self.mode == "cpmi":        # NPMI 게이팅: 약한 경계서 분할
            segs, cur = [], [run[0]]
            for a, b in zip(run, run[1:]):
                if self.npmi.get(f"{a}\t{b}", -1.0) >= self.npmi_thresh:
                    cur.append(b)
                else:
                    segs.append("".join(cur)); cur = [b]
            segs.append("".join(cur))
            return segs
        return list(run)                       # noun / dict: 토큰 그대로

    def _toks_to_ents(self, tokens):
        if self.mode in ("nnp", "noun", "dict"):
            return _dedup(t.form for t in tokens if t.tag in self.tags)
        # comp / cpmi: 연속 명사 run 단위
        ents, run = [], []
        for t in tokens:
            if t.tag in self.NOUN:
                run.append(t.form)
            else:
                if run:
                    ents += self._runs_to_ents(run); run = []
        if run:
            ents += self._runs_to_ents(run)
        return _dedup(ents)

    def _gazet_ents(self, sent_text):
        # 사전 longest-match (non-overlap) + 매칭 span 밖의 명사
        spans = [(end - len(w) + 1, end + 1, w) for end, w in self.auto.iter(sent_text)]
        chosen, cov, last = [], [], -1
        for s, e, w in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
            if s > last:
                chosen.append(w); cov.append((s, e)); last = e - 1
        ents = list(chosen)
        if self.mode == "gazo":          # 사전 매칭만 → 없으면 dense fallback
            return _dedup(ents)
        # 사전(최우선)에 안 잡힌 명사: gazet=개별 / gazcomp=comp 묶기 (사전 경계 우선 존중)
        run = []
        for t in self.kiwi.tokenize(sent_text):
            ts, te = t.start, t.start + t.len
            covered = any(cs < te and ts < ce for cs, ce in cov)
            if t.tag in self.NOUN and not covered:
                run.append(t.form)
            else:
                if run:
                    ents += ["".join(run)] if self.mode == "gazcomp" else run
                    run = []
        if run:
            ents += ["".join(run)] if self.mode == "gazcomp" else run
        return _dedup(ents)

    @staticmethod
    def _pagerank(nodes, edges, damping=0.85, iters=30):
        """패시지 내 명사 동시출현 그래프 PageRank (TextRank)."""
        if not nodes:
            return {}
        adj = {n: {} for n in nodes}
        for (a, b), w in edges.items():
            adj[a][b] = adj[a].get(b, 0.0) + w
            adj[b][a] = adj[b].get(a, 0.0) + w
        N = len(nodes)
        pr = {n: 1.0 / N for n in nodes}
        deg = {n: sum(adj[n].values()) or 1.0 for n in nodes}
        for _ in range(iters):
            nxt = {n: (1 - damping) / N for n in nodes}
            for n in nodes:
                share = damping * pr[n] / deg[n]
                for m, w in adj[n].items():
                    nxt[m] += share * w
            pr = nxt
        return pr

    def _kw_runs(self, tokens):
        """연속 명사 run → (phrase, [tokens]) 후보."""
        out, run = [], []
        for t in tokens:
            if t.tag in self.NOUN:
                run.append(t.form)
            else:
                if run:
                    out.append(("".join(run), list(run))); run = []
        if run:
            out.append(("".join(run), list(run)))
        return out

    def _kw_passage(self, text):
        """TF-IDF × TextRank 하이브리드 키프레이즈 추출 (패시지 단위 top-k)."""
        sents = list(self.kiwi.split_into_sents(text, return_tokens=True))
        # 명사 토큰 시퀀스(문장별) + 후보 phrase(문장별)
        sent_nouns, sent_cands, tf = [], [], {}
        for s in sents:
            nouns = [t.form for t in s.tokens if t.tag in self.NOUN]
            sent_nouns.append(nouns)
            sent_cands.append(self._kw_runs(s.tokens))
            for w in nouns:
                tf[w] = tf.get(w, 0) + 1
        toks = list(tf.keys())
        if not toks:
            return [(s.text, []) for s in sents]
        # TextRank: 문장 내 인접(window=2) 명사 동시출현 그래프
        edges = {}
        for nouns in sent_nouns:
            for i, a in enumerate(nouns):
                for b in nouns[i + 1:i + 3]:
                    if a == b:
                        continue
                    k = (a, b) if a < b else (b, a)
                    edges[k] = edges.get(k, 0.0) + 1.0
        tr = self._pagerank(toks, edges)
        # TF-IDF
        med_idf = sorted(self.idf.values())[len(self.idf) // 2] if self.idf else 1.0
        tfidf = {w: tf[w] * self.idf.get(w, med_idf) for w in toks}

        def _norm(d):
            vs = list(d.values()); lo, hi = min(vs), max(vs)
            return {k: (v - lo) / (hi - lo) if hi > lo else 1.0 for k, v in d.items()}
        ntf, ntr = _norm(tfidf), _norm(tr)
        score = {w: ntf[w] * ntr[w] for w in toks}
        # phrase 점수 = 구성 토큰 평균, 패시지 top-k
        cand_score = {}
        for cands in sent_cands:
            for ph, ws in cands:
                cand_score[ph] = sum(score.get(w, 0.0) for w in ws) / len(ws)
        topk = {p for p, _ in sorted(cand_score.items(), key=lambda x: -x[1])[:self.topk_kw]}
        out = []
        for s, cands in zip(sents, sent_cands):
            ents = _dedup(ph for ph, _ in cands if ph in topk)
            out.append((s.text, ents))
        return out

    def passage(self, text):
        if self.mode == "kw":
            return self._kw_passage(text)
        out = []
        for s in self.kiwi.split_into_sents(text, return_tokens=True):
            ents = (self._gazet_ents(s.text) if self.mode in ("gazet","gazo","gazcomp")
                    else self._toks_to_ents(s.tokens))
            out.append((s.text, ents))
        return out

    def question(self, q):
        ents = []
        for s in self.kiwi.split_into_sents(q, return_tokens=True):
            if self.mode == "kw":      # 질의는 후보 명사구 전부(inclusive)
                ents += [ph for ph, _ in self._kw_runs(s.tokens)]
            else:
                ents += (self._gazet_ents(s.text) if self.mode in ("gazet","gazo","gazcomp")
                         else self._toks_to_ents(s.tokens))
        return _dedup(ents)


class LinearRAG:
    def __init__(self, dataset, extract="nnp", npmi_thresh=0.3, glossary_path=None,
                 tag="", pnw=PASSAGE_NODE_WEIGHT, topk_kw=10,
                 weight="b0", alpha=0.5, rrf=False, prf_beta=0.0, prf_sent=False):
        self.dataset, self.extract, self.tag, self.pnw = dataset, extract, tag, pnw
        self.weight, self.alpha, self.rrf = weight, alpha, rrf
        self.prf_beta, self.prf_sent = prf_beta, prf_sent
        self.gloss_df = {}
        self.emb = Embedder()
        glossary, npmi, idf = None, None, None
        ddir = ROOT / "data" / dataset
        if extract in ("dict", "gazet", "gazo", "gazcomp", "compdict"):
            gf = Path(glossary_path) if glossary_path else ddir / "glossary.json"
            if not gf.exists():
                raise FileNotFoundError(
                    f"{gf} 없음 — 먼저 build_glossary 로 {dataset} 사전 구축 필요")
            gl = json.load(open(gf, encoding="utf-8"))
            glossary = [g["word"] for g in gl]
            self.gloss_df = {g["word"]: g.get("df", 1) for g in gl}  # b2/b3 LLM 점수원
            print(f"  사전 {len(glossary)}개 로드 ({extract})")
        if extract == "cpmi":
            nf = ddir / "npmi_bigrams.json"
            if not nf.exists():
                raise FileNotFoundError(
                    f"{nf} 없음 — 먼저 build_glossary 로 {dataset} NPMI 테이블 구축 필요")
            npmi = json.load(open(nf, encoding="utf-8"))
            print(f"  NPMI bigram {len(npmi)}개 로드 (cpmi)")
        if extract == "kw":
            idf = self._load_or_build_idf(dataset)
            print(f"  IDF {len(idf)}토큰 로드 (kw, top-k={topk_kw})")
        self.ner = KiwiNER(extract, glossary=glossary, npmi=npmi,
                           npmi_thresh=npmi_thresh, idf=idf, topk_kw=topk_kw)
        suffix = f"_{tag}" if tag else ""
        self.cdir = CACHE / f"linrag_{extract}{suffix}_{dataset}"
        self.cdir.mkdir(parents=True, exist_ok=True)

    def _load_or_build_idf(self, dataset):
        """kw 모드: 코퍼스 전역 IDF(token→idf) 1회 구축·캐시 (LLM 무관)."""
        import math
        f = CACHE / f"kw_idf_{dataset}.json"
        if f.exists():
            return json.load(open(f, encoding="utf-8"))
        passages, _ = load_dataset(dataset)
        kiwi = Kiwi()
        df = defaultdict(int)
        t0 = time.time()
        for i, p in enumerate(passages):
            toks = {t.form for t in kiwi.tokenize(p["text"]) if t.tag in KiwiNER.NOUN}
            for w in toks:
                df[w] += 1
            if (i + 1) % 5000 == 0:
                print(f"  IDF {i+1}/{len(passages)} ({time.time()-t0:.0f}s)", flush=True)
        N = len(passages)
        idf = {w: math.log(N / (d + 1)) for w, d in df.items()}
        CACHE.mkdir(parents=True, exist_ok=True)
        json.dump(idf, open(f, "w", encoding="utf-8"), ensure_ascii=False)
        return idf

    # ---------------- 색인 ----------------
    def index(self, passages):
        self.passages = passages
        self.pid = [p["id"] for p in passages]
        self.ptext = [p["text"] for p in passages]

        nf = self.cdir / "ner.json"
        if nf.exists():
            d = json.load(open(nf, encoding="utf-8"))
            p2e, s2e = d["p2e"], {k: v for k, v in d["s2e"]}
        else:
            p2e, s2e = {}, {}
            t0 = time.time()
            for i, p in enumerate(passages):
                ents = []
                for sent, se in self.ner.passage(p["text"]):
                    if not se:
                        continue
                    cur = s2e.setdefault(sent, [])
                    for e in se:
                        if e not in cur:
                            cur.append(e)
                        if e not in ents:
                            ents.append(e)
                p2e[p["id"]] = ents
                if (i + 1) % 5000 == 0:
                    print(f"  NER {i+1}/{len(passages)} ({time.time()-t0:.0f}s)", flush=True)
            json.dump({"p2e": p2e, "s2e": list(s2e.items())},
                      open(nf, "w", encoding="utf-8"), ensure_ascii=False)
        self.p2e = p2e
        self.s2e = s2e

        # 엔티티/문장 인덱스
        self.entities = sorted({e for es in p2e.values() for e in es})
        self.eidx = {e: i for i, e in enumerate(self.entities)}
        self.sentences = list(s2e.keys())
        self.sidx = {s: i for i, s in enumerate(self.sentences)}
        # entity -> sentence indices, sentence -> entity indices
        self.e2s = defaultdict(list)
        self.s2e_idx = {}
        for s, es in s2e.items():
            si = self.sidx[s]
            eis = [self.eidx[e] for e in es if e in self.eidx]
            self.s2e_idx[si] = eis
            for ei in eis:
                self.e2s[ei].append(si)

        # 임베딩 (passage 는 vanilla 캐시 재사용 시도)
        self.pemb = self._passage_emb()
        self.semb = self._cached_emb("sent_emb.npy", self.sentences)
        self.eemb = self._cached_emb("ent_emb.npy", self.entities)

        self._build_graph()

    def _passage_emb(self):
        f = CACHE / f"emb_{self.dataset}.npy"
        fi = CACHE / f"emb_{self.dataset}_ids.json"
        if f.exists() and fi.exists() and json.load(open(fi)) == self.pid:
            return np.load(f).astype(np.float32)
        v = np.asarray(self.emb.encode(self.ptext, batch_size=64), dtype=np.float32)
        np.save(f, v.astype(np.float16))
        json.dump(self.pid, open(fi, "w"))
        return v

    def _cached_emb(self, name, texts):
        f = self.cdir / name
        if f.exists():
            return np.load(f).astype(np.float32)
        print(f"  임베딩 {name} ({len(texts)})...", flush=True)
        v = np.asarray(self.emb.encode(texts, batch_size=64), dtype=np.float32) \
            if texts else np.zeros((0, 1024), np.float32)
        np.save(f, v.astype(np.float16))
        return v

    def _entity_weights(self):
        """B축: 엔티티별 가중치 w_j ∈ [0,1] (그래프 엣지에 (1+w_j) 부스팅).
        b0=없음 / b1=TF-IDF / b4=TextRank / b5=TF-IDF×TextRank / b2=LLM(df) / b3=α결합."""
        import math
        if self.weight == "b0":
            return {}
        ents = self.entities
        N = len(self.passages)
        # 엔티티별 df / 총빈도 (코퍼스)
        df = {e: 0 for e in ents}
        tf = {e: 0 for e in ents}
        for p in self.passages:
            t = p["text"]
            for e in self.p2e[p["id"]]:
                c = t.count(e)
                if c > 0:
                    df[e] += 1; tf[e] += c

        def _mm(d):
            vs = list(d.values()); lo, hi = min(vs), max(vs)
            return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in d.items()}
        tfidf = _mm({e: tf[e] * math.log(N / (df[e] + 1)) for e in ents})
        if self.weight == "b1":
            return tfidf
        if self.weight == "b2":   # LLM 점수 = 사전 df(LLM 추출빈도)
            return _mm({e: self.gloss_df.get(e, 1) for e in ents})
        if self.weight in ("b4", "b5"):
            # TextRank: 엔티티 동시출현 그래프(같은 패시지) PageRank
            import igraph as ig2
            ei = {e: i for i, e in enumerate(ents)}
            ew = {}
            for p in self.passages:
                es = [e for e in self.p2e[p["id"]] if e in ei]
                for a in range(len(es)):
                    for b in range(a + 1, len(es)):
                        k = (ei[es[a]], ei[es[b]]) if ei[es[a]] < ei[es[b]] else (ei[es[b]], ei[es[a]])
                        ew[k] = ew.get(k, 0.0) + 1.0
            gg = ig2.Graph(n=len(ents), edges=list(ew.keys()))
            gg.es["weight"] = list(ew.values())
            pr = gg.pagerank(weights="weight") if gg.ecount() else [0.0] * len(ents)
            tr = _mm({e: pr[ei[e]] for e in ents})
            if self.weight == "b4":
                return tr
            return {e: tfidf[e] * tr[e] for e in ents}   # b5
        if self.weight == "b3":   # α·TF-IDF + (1-α)·LLM
            llm = _mm({e: self.gloss_df.get(e, 1) for e in ents})
            return {e: self.alpha * tfidf[e] + (1 - self.alpha) * llm[e] for e in ents}
        return {}

    def _build_graph(self):
        self.ew = self._entity_weights()   # B축 엔티티 가중치
        # 노드: passage(p::id) + entity(e::text)
        pnodes = [f"p::{p}" for p in self.pid]
        enodes = [f"e::{e}" for e in self.entities]
        names = pnodes + enodes
        self.g = ig.Graph(directed=False)
        self.g.add_vertices(names)
        self.name2v = {n: i for i, n in enumerate(names)}
        self.p_vidx = [self.name2v[n] for n in pnodes]

        edges, weights = [], []
        # passage-entity: count/total
        for p in self.passages:
            text = p["text"]
            cnts = {e: text.count(e) for e in self.p2e[p["id"]]}
            tot = sum(cnts.values()) or 1
            for e, c in cnts.items():
                if c > 0:
                    edges.append((self.name2v[f"p::{p['id']}"], self.name2v[f"e::{e}"]))
                    weights.append(c / tot * (1.0 + self.ew.get(e, 0.0)))  # B축 (1+w_j)
        # 인접 passage ("idx:" 접두 있는 경우; wiki_ko)
        idxed = []
        for p in self.passages:
            t = p["text"].lstrip()
            if t[:20].split(":", 1)[0].isdigit():
                idxed.append((int(t.split(":", 1)[0]), p["id"]))
        idxed.sort()
        for a, b in zip(idxed, idxed[1:]):
            edges.append((self.name2v[f"p::{a[1]}"], self.name2v[f"p::{b[1]}"]))
            weights.append(1.0)
        self.g.add_edges(edges)
        self.g.es["weight"] = weights

    # ---------------- 검색 ----------------
    def _seed_entities(self, q):
        qents = self.ner.question(q)
        if not qents or len(self.entities) == 0:
            return [], []
        qv = np.asarray(self.emb.encode(qents, batch_size=32), dtype=np.float32)
        sims = self.eemb @ qv.T  # (E, Q)
        idxs, scores = [], []
        for j in range(len(qents)):
            i = int(np.argmax(sims[:, j]))
            idxs.append(i)
            scores.append(float(sims[i, j]))
        return idxs, scores

    def _activate(self, qv, seed_idx, seed_scores):
        ent_w = np.zeros(self.g.vcount())
        actived = {}
        for ei, sc in zip(seed_idx, seed_scores):
            ent_w[self.name2v[f"e::{self.entities[ei]}"]] = sc
            actived[ei] = (sc, 1)
        sent_sim = self.semb @ qv  # (S,)
        used = set()
        current = {ei: (sc, 1) for ei, sc in zip(seed_idx, seed_scores)}
        it = 1
        while current and it < MAX_ITER:
            nxt = {}
            for ei, (sc, tier) in current.items():
                if sc < ITER_THRESH:
                    continue
                sids = [s for s in self.e2s.get(ei, []) if s not in used]
                if not sids:
                    continue
                sims = sent_sim[sids]
                order = np.argsort(-sims)[:TOPK_SENT]
                for o in order:
                    si = sids[o]
                    used.add(si)
                    for nei in self.s2e_idx.get(si, []):
                        ns = sc * float(sent_sim[si])
                        if ns < ITER_THRESH:
                            continue
                        ent_w[self.name2v[f"e::{self.entities[nei]}"]] += ns
                        nxt[nei] = (ns, it + 1)
            actived.update(nxt)
            current = nxt
            it += 1
        return ent_w, actived

    def _passage_weights(self, qv, actived):
        pw = np.zeros(self.g.vcount())
        sims = self.pemb @ qv
        order = np.argsort(-sims)
        s = sims[order]
        rng = s.max() - s.min() or 1.0
        norm = {int(order[i]): (s[i] - s.min()) / rng for i in range(len(order))}
        for pi in range(len(self.passages)):
            text = self.ptext[pi]
            bonus = 0.0
            for ei, (sc, tier) in actived.items():
                occ = text.count(self.entities[ei])
                if occ > 0:
                    bonus += sc * math.log(1 + occ) / (tier if tier >= 1 else 1)
            score = PASSAGE_RATIO * norm[pi] + math.log(1 + bonus)
            pw[self.p_vidx[pi]] = score * self.pnw
        return pw

    def _rrf(self, lin_order, dense_order, k=60):
        """Reciprocal Rank Fusion: LinearRAG 순위 ⊕ vanilla(dense) 순위."""
        score = defaultdict(float)
        for r, i in enumerate(lin_order):
            score[i] += 1.0 / (k + r)
        for r, i in enumerate(dense_order):
            score[i] += 1.0 / (k + r)
        top = sorted(score, key=lambda i: -score[i])[:TOPK]
        return [self.pid[i] for i in top]

    def _ppr_scores(self, qv, ent_w, actived):
        pw = self._passage_weights(qv, actived)
        nw = ent_w + pw
        reset = np.where((np.isnan(nw)) | (nw < 0), 0, nw)
        if reset.sum() == 0:
            reset = None
        pr = self.g.personalized_pagerank(damping=DAMPING, directed=False,
                                          weights="weight", reset=reset,
                                          implementation="prpack")
        return np.array([pr[v] for v in self.p_vidx])

    def retrieve_one(self, q):
        qv = np.asarray(self.emb.encode(q, batch_size=1), dtype=np.float32)
        dense_order = list(np.argsort(-(self.pemb @ qv)))   # vanilla 순위(RRF용)
        seed_idx, seed_scores = self._seed_entities(q)
        if not seed_idx:
            return [self.pid[i] for i in dense_order[:TOPK]]
        ent_w, actived = self._activate(qv, seed_idx, seed_scores)
        dsc = self._ppr_scores(qv, ent_w, actived)
        lin_order = list(np.argsort(-dsc))
        if self.prf_beta > 0:
            # PRF 2-hop: hop1 top-3 문맥을 질의에 혼합 → dense reset만 재계산 → PPR 재실행
            # (엔티티 seed·활성 불변 → dense-graph 균형 유지. docA→docB semantic bridge 겨냥)
            ctx = None
            if self.prf_sent:
                # 문장-수준 PRF: top-3 문서에서 활성 엔티티 포함 문장 top-6(질의유사순)만 혼합
                sents = []
                for pi in lin_order[:3]:
                    for s in self.ner.kiwi.split_into_sents(self.ptext[pi]):
                        st = s.text.strip()
                        if len(st) >= 10:
                            sents.append(st)
                if sents:
                    se = np.asarray(self.emb.encode(sents, batch_size=64), dtype=np.float32)
                    se = se / (np.linalg.norm(se, axis=1, keepdims=True) + 1e-9)
                    sim = se @ qv
                    act_surface = [self.entities[ei] for ei in actived]
                    idxs = [i for i, st in enumerate(sents)
                            if any(e in st for e in act_surface)]
                    if not idxs:
                        idxs = list(range(len(sents)))
                    idxs = sorted(idxs, key=lambda i: -sim[i])[:6]
                    ctx = se[idxs].mean(axis=0)
            if ctx is None:
                ctx = self.pemb[lin_order[:3]]
                ctx = (ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)).mean(axis=0)
            qv2 = qv + self.prf_beta * ctx
            qv2 = (qv2 / (np.linalg.norm(qv2) + 1e-9)).astype(np.float32)
            dsc = self._ppr_scores(qv2, ent_w, actived)
            lin_order = list(np.argsort(-dsc))
        if self.rrf:
            return self._rrf(lin_order, dense_order)
        return [self.pid[i] for i in lin_order[:TOPK]]


def main():
    global MAX_ITER, TOPK_SENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["wiki_ko", "admin_ko", "admin_hard", "admin_full", "admin_single", "admin_hard2k"])
    ap.add_argument("--extract", default="nnp",
                    choices=["nnp", "noun", "comp", "cpmi", "dict", "gazet", "gazo", "compdict", "gazcomp", "kw"])
    ap.add_argument("--topk-kw", type=int, default=10, help="kw 모드 패시지당 키프레이즈 수")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--glossary", default=None, help="dict/gazet 사전 경로 override")
    ap.add_argument("--tag", default="", help="ID 접미사 (예: oracle)")
    ap.add_argument("--pnw", type=float, default=PASSAGE_NODE_WEIGHT,
                    help="passage_node_weight (dense 비중)")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="QA 생략, 검색 지표(both@5·recall)만 — pnw 스윕용")
    ap.add_argument("--weight", default="b0",
                    choices=["b0", "b1", "b2", "b3", "b4", "b5"],
                    help="B축 엔티티 가중치: b1=TF-IDF b4=TextRank b5=둘곱 b2=LLM b3=α결합")
    ap.add_argument("--alpha", type=float, default=0.5, help="b3 가중합 비율")
    ap.add_argument("--rrf", action="store_true", help="LinearRAG⊕vanilla RRF 융합")
    ap.add_argument("--prf-beta", type=float, default=0.0,
                    help="PRF 2-hop 질의확장 강도 (0=off, 권장 0.6; hop1 top-3 임베딩 혼합 후 PPR 재실행)")
    ap.add_argument("--prf-sent", action="store_true",
                    help="문장-수준 PRF: 문서평균 대신 활성엔티티 포함 문장 top-6 혼합 (--prf-beta 필요)")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER, help="전파 반복 hop 수(traversal 깊이)")
    ap.add_argument("--topk-sent", type=int, default=TOPK_SENT, help="hop당 활성 문장 수(traversal 폭)")
    args = ap.parse_args()
    MAX_ITER, TOPK_SENT = args.max_iter, args.topk_sent

    passages, questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[:args.limit]
    print(f"[{args.dataset}] passages={len(passages)} questions={len(questions)} extract={args.extract} pnw={args.pnw}")

    rag = LinearRAG(args.dataset, args.extract, glossary_path=args.glossary,
                    tag=args.tag, pnw=args.pnw, topk_kw=args.topk_kw,
                    weight=args.weight, alpha=args.alpha, rrf=args.rrf,
                    prf_beta=args.prf_beta, prf_sent=args.prf_sent)
    t0 = time.time()
    rag.index(passages)
    print(f"색인 완료 ({time.time()-t0:.0f}s): 엔티티 {len(rag.entities)} 문장 {len(rag.sentences)} "
          f"그래프 V={rag.g.vcount()} E={rag.g.ecount()}")

    if args.retrieval_only:
        both = any1 = 0
        for q in questions:
            retr = set(map(str, rag.retrieve_one(q["question"])))
            sup = set(map(str, q["supporting"]))
            both += sup <= retr
            any1 += len(sup & retr) > 0
        n = len(questions)
        print(f"[{args.dataset}] {args.extract} pnw={args.pnw}  "
              f"both@5={both/n:.3f}  any@5={any1/n:.3f}  ({time.time()-t0:.0f}s)")
        return

    ptext = {p["id"]: p["text"] for p in passages}
    llm = LLMClient()

    def qa(q):
        retr = rag.retrieve_one(q["question"])
        ctx = "\n".join(ptext[r] for r in retr)
        out = llm.chat([{"role": "system", "content": QA_SYS},
                        {"role": "user", "content": f"{ctx}\nQuestion: {q['question']}\n Thought: "}],
                       temperature=0.0, max_tokens=256)
        pred = out.split("Answer:")[-1].strip() if "Answer:" in out else out.strip()
        return retr, pred

    preds = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (retr, pred) in enumerate(ex.map(qa, questions)):
            q = questions[i]
            preds.append({"id": q["id"], "question": q["question"], "answer": q["answer"],
                          "prediction": pred, "retrieved": retr, "supporting": q["supporting"]})
            if (i + 1) % 50 == 0:
                print(f"  QA {i+1}/{len(questions)} ({time.time()-t0:.0f}s)", flush=True)

    for p in preds:
        p["contain"] = contain_acc(p["prediction"], p["answer"])
    if not args.no_judge:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for p, g in zip(preds, ex.map(
                    lambda p: gpt_acc(p["question"], p["answer"], p["prediction"], llm), preds)):
                p["gpt"] = g

    # ID 체계: linearrag-{org|new}-{extract}  (nnp=원본 논문방법=org)
    variant = "org" if args.extract == "nnp" else "new"
    wsuf = "" if args.weight == "b0" else f"-{args.weight}" + (f"a{args.alpha}" if args.weight == "b3" else "")
    if args.rrf:
        wsuf += "-rrf"
    if abs(args.pnw - PASSAGE_NODE_WEIGHT) > 1e-9:
        wsuf += f"-pnw{args.pnw:g}"
    if args.prf_beta > 0:
        wsuf += f"-prf{args.prf_beta:g}" + ("sent" if args.prf_sent else "")
    sys_id = (f"linearrag-{variant}-{args.extract}"
              + (f"-{args.tag}" if args.tag else "") + wsuf)
    outdir = RESULTS / f"{sys_id}_{args.dataset}"
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(outdir / "predictions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n = len(preds)
    summ = {"dataset": args.dataset, "system": sys_id, "n": n,
            "contain_acc": round(sum(p["contain"] for p in preds) / n, 4)}
    if not args.no_judge:
        summ["gpt_acc"] = round(sum(p["gpt"] for p in preds) / n, 4)
    json.dump(summ, open(outdir / "eval.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[{args.dataset}] {sys_id}  {summ}")


if __name__ == "__main__":
    main()
