"""compsp(comp + 공백 단위 조각) R@5 실험. engine.py 는 수정하지 않고 추출기만 바꿔 끼운다."""
import json, os, shutil, sys, threading, time
ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import torch
torch.set_num_threads(4)
from scipy.stats import binomtest

def guard():
    while True:
        avail = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]) // 1024
        if avail < 4096:
            print(f'[중단] 남은 RAM {avail}MB < 4GB', flush=True); os._exit(3)
        time.sleep(2)
threading.Thread(target=guard, daemon=True).start()

import src.track_02.linearrag.engine as E
_real, _inst = E.Embedder, {}
def _single(*a, **k):
    if 'e' not in _inst: _inst['e'] = _real(*a, **k)
    return _inst['e']
E.Embedder = _single
from src.track_02.common.datasets import load_dataset

class CompSpNER(E.KiwiNER):
    """comp 엔티티(연속 명사 결합) + 공백으로 나뉜 어절 조각."""
    def __init__(self):
        super().__init__('comp')
    def _toks_to_ents(self, tokens):
        ents, run = [], []
        def flush():
            if not run: return
            ents.append(''.join(t.form for t in run))
            segs = [[run[0]]]
            for a, b in zip(run, run[1:]):
                if b.start > a.start + a.len: segs.append([b])
                else: segs[-1].append(b)
            if len(segs) > 1:
                ents.extend(''.join(t.form for t in s) for s in segs)
        for t in tokens:
            if t.tag in self.NOUN: run.append(t)
            else: flush(); run.clear()
        flush()
        return E._dedup(ents)


class GazCompSpNER(E.KiwiNER):
    """gazcomp(사전 최장 일치 우선 + 나머지 명사 결합) + 나머지 명사 run 의 공백 단위 조각."""
    def _gazet_ents(self, sent_text):
        spans = [(end - len(w) + 1, end + 1, w) for end, w in self.auto.iter(sent_text)]
        chosen, cov, last = [], [], -1
        for s, e, w in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
            if s > last:
                chosen.append(w); cov.append((s, e)); last = e - 1
        ents, run = list(chosen), []
        def flush():
            if not run: return
            ents.append("".join(t.form for t in run))
            segs = [[run[0]]]
            for a, b in zip(run, run[1:]):
                if b.start > a.start + a.len: segs.append([b])
                else: segs[-1].append(b)
            if len(segs) > 1:
                ents.extend("".join(t.form for t in g) for g in segs)
        for t in self.kiwi.tokenize(sent_text):
            ts, te = t.start, t.start + t.len
            covered = any(cs < te and ts < ce for cs, ce in cov)
            if t.tag in self.NOUN and not covered:
                run.append(t)
            else:
                flush(); run.clear()
        flush()
        return E._dedup(ents)

GLOSS = {  # 논문 gazcomp 와 같은 사전 (systems.csv)
    '':      {'wiki_ko_v2': 'data/wiki_ko/glossary.json', 'admin_single': 'data/admin_ko/glossary.json',
              'admin_full': 'data/admin_ko/glossary.json', 'admin_hard': 'data/admin_ko/glossary.json'},
    'llm4b': {'wiki_ko_v2': 'data/wiki_ko/glossary_llm4b_wiki_v2.json', 'admin_single': 'data/admin_ko/glossary_llm4b_single.json',
              'admin_full': 'data/admin_ko/glossary_llm4b_full.json', 'admin_hard': 'data/admin_ko/glossary_llm4b_full.json'},
    'llm':   {'wiki_ko_v2': 'data/wiki_ko/glossary_llm.json', 'admin_single': 'data/admin_ko/glossary_llm_single.json',
              'admin_full': 'data/admin_ko/glossary_llm_full.json', 'admin_hard': 'data/admin_ko/glossary_llm_full.json'},
}
def gz_cache(tag, ds):
    t = 'llm4bv2' if (tag == 'llm4b' and ds == 'wiki_ko_v2') else tag
    return E.CACHE / (f'linrag_gazcomp_{t}_{ds}' if t else f'linrag_gazcomp_{ds}')

def build(ds, tag, sp=True):
    passages, questions = load_dataset(ds)
    gpath = f'{ROOT}/' + GLOSS[tag][ds]
    rag = E.LinearRAG(ds, 'gazcomp', glossary_path=gpath, tag=('llm4bv2' if tag == 'llm4b' and ds == 'wiki_ko_v2' else tag))
    if sp:
        words = [g['word'] for g in json.load(open(gpath, encoding='utf-8'))]
        rag.ner = GazCompSpNER('gazcomp', glossary=words)
        name = f'gazcompsp_{tag}' if tag else 'gazcompsp'
        rag.extract = name
        rag.cdir = E.CACHE / f'linrag_{name}_{ds}'
        rag.cdir.mkdir(parents=True, exist_ok=True)
        if not (rag.cdir / 'sent_emb.npy').exists():
            shutil.copy(gz_cache(tag, ds) / 'sent_emb.npy', rag.cdir / 'sent_emb.npy')
    t0 = time.time()
    rag.index(passages)
    if sp:
        ref = [s for s, _ in json.load(open(gz_cache(tag, ds) / 'ner.json'))['s2e']]
        assert rag.sentences == ref, f'{ds} {tag}: 문장 목록이 gazcomp와 다름'
    print(f'  [{ds} {rag.extract}] 색인 {time.time()-t0:.0f}s, 엔티티 {len(rag.entities):,}', flush=True)
    return rag, questions

def stored(folder, ds):
    return {r['id']: r['retrieved'] for r in json.load(open(f'{ROOT}/results/track_02/{folder}_{ds}/predictions.json'))}

rag, Q = build('admin_hard', '', sp=False)
st = stored('linearrag-new-gazcomp', 'admin_hard')
same = sum(1 for q in Q if [str(x) for x in rag.retrieve_one(q['question'])] == [str(x) for x in st[q['id']][:5]])
print(f'[검증] gazcomp(PMI) admin_hard 재실행 상위 5개 일치 {same}/{len(Q)}', flush=True)
del rag
for tag in ('', 'llm4b', 'llm'):
    for ds in ('wiki_ko_v2', 'admin_hard', 'admin_full', 'admin_single'):
        rag, Q = build(ds, tag)
        out = rag.cdir / 'retrieved_top5.json'
        if not out.exists():
            t0 = time.time()
            top = {q['id']: [str(x) for x in rag.retrieve_one(q['question'])] for q in Q}
            json.dump(top, open(out, 'w'))
            print(f'  [{ds} {rag.extract}] 검색 {len(Q)}문항 {time.time()-t0:.0f}s', flush=True)
        del rag
    print(f'[완료] 사전 {tag or "PMI"} 4개 집합 검색', flush=True)
