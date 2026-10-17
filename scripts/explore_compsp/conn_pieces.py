"""comp 대 comp+공백 조각 연결성 비교 (CPU, Kiwi 1개, 문서 단위 스트리밍)."""
import os, sys, threading, time
from collections import Counter
ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
from kiwipiepy import Kiwi
from src.track_02.common.datasets import load_dataset

def guard():
    while True:
        avail = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]) // 1024
        if avail < 4096:
            print(f'[중단] 남은 RAM {avail}MB < 4GB', flush=True); os._exit(3)
        time.sleep(2)
threading.Thread(target=guard, daemon=True).start()

kiwi = Kiwi(num_workers=0)
NOUN = ("NNG", "NNP")

def sent_ents(text, toks):
    """engine comp 규칙(연속 명사 run 결합) + run 내부 공백 조각."""
    comp, pieces = [], []
    run = []
    def flush():
        if not run: return
        comp.append("".join(t.form for t in run))
        segs, cur = [], [run[0]]
        for a, b in zip(run, run[1:]):
            gap = text[a.start + a.len:b.start]
            if gap and gap.isspace():
                segs.append(cur); cur = [b]
            else:
                cur.append(b)
        segs.append(cur)
        if len(segs) > 1:
            pieces.extend("".join(t.form for t in s) for s in segs)
    for t in toks:
        if t.tag in NOUN:
            run.append(t)
        else:
            flush(); run = []
    flush()
    return comp, pieces

def analyze(ds):
    P = load_dataset(ds)[0]
    n = len(P)
    stats = {}
    for mode in ('comp', 'new'):
        stats[mode] = dict(df=Counter(), sdf=Counter(), mentions=0, orphan=0, doc_ents=[], nent=0)
    piece_only = set()
    t0 = time.time()
    for i, p in enumerate(P):
        text = p['text']
        ents = {'comp': set(), 'new': set()}
        sent_seen = {'comp': set(), 'new': set()}
        for s in kiwi.split_into_sents(text, return_tokens=True):
            c, pc = sent_ents(text, s.tokens)
            ents['comp'].update(c); ents['new'].update(c); ents['new'].update(pc)
            sent_seen['comp'].update(c); sent_seen['new'].update(c + pc)
        for mode in ('comp', 'new'):
            st = stats[mode]
            linked = []
            for e in ents[mode]:
                st['mentions'] += 1
                if text.count(e) > 0:
                    st['df'][e] += 1; linked.append(e)
                else:
                    st['orphan'] += 1
            for e in sent_seen[mode]:
                st['sdf'][e] += 1     # 문장 층: 원문 확인 없이 그 문서 문장에 연결
            st['doc_ents'].append(linked); st['nent'] += len(ents[mode])
        piece_only.update(ents['new'] - ents['comp'])
        if (i + 1) % 2000 == 0:
            print(f'  {ds} {i+1}/{n} ({time.time()-t0:.0f}s)', flush=True)
    out = {}
    for mode, st in stats.items():
        df = st['df']
        uniq = set(df) | {e for e in st['sdf']}
        bridge = {e for e, c in df.items() if c >= 2}
        sbridge = {e for e, c in st['sdf'].items() if c >= 2}
        edges = sum(df.values())
        hub = {e for e, c in df.items() if c >= 0.05 * n}
        hub_edges = sum(df[e] for e in hub)
        iso = sum(1 for es in st['doc_ents'] if not (set(es) & bridge))
        out[mode] = (len(uniq), len(bridge), len(sbridge), hub_edges / edges, st['orphan'] / st['mentions'], iso / n, st['nent'] / n)
        if mode == 'new':
            hp = sorted(((df[e], e) for e in hub if e in piece_only), reverse=True)
            out['hub_pieces'] = (len(hp), hp[:15])
    print(f'\n== {ds} ({n}문서, {time.time()-t0:.0f}s)')
    print(f'{"":6s} {"고유":>8s} {"다리(문서)":>10s} {"다리(문장층)":>12s} {"허브끈":>7s} {"끈없는언급":>9s} {"고립문서":>8s} {"문서당엔티티":>10s}')
    for mode in ('comp', 'new'):
        u, b, sb, h, o, iso, pe = out[mode]
        print(f'{mode:6s} {u:8,d} {b:10,d} {sb:12,d} {h:7.1%} {o:9.1%} {iso:8.1%} {pe:10.1f}')
    k, top = out['hub_pieces']
    print(f'새로 생긴 조각 중 허브(문서 5% 이상) {k}개, 상위:', ', '.join(f'{e}({c})' for c, e in top))
    del stats

for ds in sys.argv[1:]:
    analyze(ds)
