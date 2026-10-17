"""compsp + dense0.5 + PRF(문장) R@5 실험. engine.py 는 수정하지 않고 추출기만 바꿔 끼운다."""
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

def build(ds, mode):
    passages, questions = load_dataset(ds)
    rag = E.LinearRAG(ds, 'comp', pnw=0.5, prf_beta=0.6, prf_sent=True)
    if mode == 'compsp':
        rag.ner, rag.extract = CompSpNER(), 'compsp'
        rag.cdir = E.CACHE / f'linrag_compsp_{ds}'
        rag.cdir.mkdir(parents=True, exist_ok=True)
        src = E.CACHE / f'linrag_comp_{ds}' / 'sent_emb.npy'   # 문장 목록이 comp와 같으면 재사용
        if not (rag.cdir / 'sent_emb.npy').exists():
            shutil.copy(src, rag.cdir / 'sent_emb.npy')
    t0 = time.time()
    rag.index(passages)
    if mode == 'compsp':
        comp_sents = [s for s, _ in json.load(open(E.CACHE / f'linrag_comp_{ds}' / 'ner.json'))['s2e']]
        assert rag.sentences == comp_sents, '문장 목록이 comp와 다름 — sent_emb 재사용 불가'
    print(f'  [{ds} {mode}] 색인 {time.time()-t0:.0f}s, 엔티티 {len(rag.entities):,}', flush=True)
    return rag, questions

def r5(sup, top): return set(map(str, sup)) <= set(map(str, top[:5]))
def stored(folder, ds):
    return {r['id']: r['retrieved'] for r in json.load(open(f'{ROOT}/results/track_02/{folder}_{ds}/predictions.json'))}


STACK = 'linearrag-new-comp-pnw0.5-prf0.6sent'
rag, Q = build('admin_hard', 'comp')
st = stored(STACK, 'admin_hard')
same = sum(1 for q in Q if [str(x) for x in rag.retrieve_one(q['question'])] == [str(x) for x in st[q['id']][:5]])
print(f'[검증] comp+dense0.5+PRF(문장) admin_hard 재실행 상위 5개 일치 {same}/{len(Q)}', flush=True)
del rag
for ds in ('admin_hard', 'admin_full', 'admin_single'):
    rag, Q = build(ds, 'compsp')
    t0 = time.time()
    top = {q['id']: [str(x) for x in rag.retrieve_one(q['question'])] for q in Q}
    print(f'  [{ds}] 검색 {len(Q)}문항 {time.time()-t0:.0f}s', flush=True)
    json.dump(top, open(rag.cdir / 'retrieved_top5_pnw0.5_prf0.6sent.json', 'w'))
    base = json.load(open(rag.cdir / 'retrieved_top5.json'))
    del rag
    sets = {'compsp+조정': top, 'comp+조정': stored(STACK, ds), 'compsp': base, 'comp': stored('linearrag-new-comp', ds)}
    R = {k: {q['id']: r5(q['supporting'], v[q['id']]) for q in Q} for k, v in sets.items()}
    line = f'== {ds} (n={len(Q)})  R@5  ' + ' | '.join(f'{k} {sum(R[k].values())/len(Q):.3f}' for k in sets)
    for other in ('comp+조정', 'compsp', 'comp'):
        b = sum(1 for q in Q if R['compsp+조정'][q['id']] and not R[other][q['id']])
        c = sum(1 for q in Q if R[other][q['id']] and not R['compsp+조정'][q['id']])
        p = binomtest(b, b + c).pvalue if b + c else 1.0
        line += f'\n   compsp+조정 대 {other}: {100*(b-c)/len(Q):+.1f}%p  b/c {b}/{c}  p={p:.2g}'
    print(line, flush=True)
