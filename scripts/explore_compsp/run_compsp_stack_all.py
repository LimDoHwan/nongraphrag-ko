"""compsp + 검색 조정 6개 구성(comp 와 같은 세트) 검색. engine.py 는 수정하지 않고 추출기만 바꿔 끼운다."""
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

STACKS = [  # (태그, pnw, prf_beta, prf_sent) — 결과 폴더 linearrag-new-comp-{태그} 와 같은 설정
    ('pnw0.5', 0.5, 0.0, False), ('pnw1', 1.0, 0.0, False),
    ('pnw0.5-prf0.6', 0.5, 0.6, False), ('pnw1-prf0.6', 1.0, 0.6, False),
    ('pnw0.5-prf0.6sent', 0.5, 0.6, True), ('pnw1-prf0.6sent', 1.0, 0.6, True),
]
DATASETS = ['wiki_ko_v2', 'admin_hard', 'admin_full', 'admin_single']

def build(ds, pnw, beta, sent, sp=True):
    passages, questions = load_dataset(ds)
    rag = E.LinearRAG(ds, 'comp', pnw=pnw, prf_beta=beta, prf_sent=sent)
    if sp:
        rag.ner, rag.extract = CompSpNER(), 'compsp'
        rag.cdir = E.CACHE / f'linrag_compsp_{ds}'          # compsp 기본 실험의 색인 캐시 재사용
    rag.index(passages)
    return rag, questions

def stored(folder, ds):
    return {r['id']: [str(x) for x in r['retrieved'][:5]]
            for r in json.load(open(f'{ROOT}/results/track_02/{folder}_{ds}/predictions.json'))}

# 검증: comp + dense1.0 + PRF(문서) 를 같은 경로로 재실행 → 저장 결과와 비교
rag, Q = build('admin_hard', 1.0, 0.6, False, sp=False)
st = stored('linearrag-new-comp-pnw1-prf0.6', 'admin_hard')
same = sum(1 for q in Q if [str(x) for x in rag.retrieve_one(q['question'])] == st[q['id']])
print(f'[검증] comp+dense1.0+PRF(문서) admin_hard 재실행 상위 5개 일치 {same}/{len(Q)}', flush=True)
del rag
for tag, pnw, beta, sent in STACKS:
    for ds in DATASETS:
        out = E.CACHE / f'linrag_compsp_{ds}' / f'retrieved_top5_{tag}.json'
        old = E.CACHE / f'linrag_compsp_{ds}' / 'retrieved_top5_pnw0.5_prf0.6sent.json'   # 10/17 실행분
        if not out.exists() and tag == 'pnw0.5-prf0.6sent' and old.exists():
            shutil.copy(old, out)
        if out.exists():
            print(f'  [{ds} {tag}] 기존 결과 사용', flush=True); continue
        rag, Q = build(ds, pnw, beta, sent)
        t0 = time.time()
        top = {q['id']: [str(x) for x in rag.retrieve_one(q['question'])] for q in Q}
        json.dump(top, open(out, 'w'))
        print(f'  [{ds} {tag}] 검색 {len(Q)}문항 {time.time()-t0:.0f}s', flush=True)
        del rag
    print(f'[완료] {tag} 4개 집합 검색', flush=True)
