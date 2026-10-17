import json, sys
from scipy.stats import binomtest
ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
from src.track_02.common.datasets import load_dataset
C=f'{ROOT}/cache/track_02'; R=f'{ROOT}/results/track_02'
name, ref = sys.argv[1], sys.argv[2]          # 예: gazcompsp linearrag-new-gazcomp
def st(f, ds): return {r['id']:[str(x) for x in r['retrieved'][:5]] for r in json.load(open(f'{R}/{f}_{ds}/predictions.json'))}
for ds in ('admin_single','admin_full','admin_hard'):
    Q=load_dataset(ds)[1]; n=len(Q)
    A=json.load(open(f'{C}/linrag_{name}_{ds}/retrieved_top5.json'))
    sets={name:A, '같은사전 gazcomp':st(ref,ds), 'compsp':json.load(open(f'{C}/linrag_compsp_{ds}/retrieved_top5.json')), 'comp':st('linearrag-new-comp',ds)}
    Rk={k:{q['id']:set(map(str,q['supporting']))<=set(v[q['id']]) for q in Q} for k,v in sets.items()}
    print(f'== {ds} (n={n}) R@5 ' + ' | '.join(f'{k} {sum(Rk[k].values())/n:.3f}' for k in sets))
    for o in list(sets)[1:]:
        b=sum(1 for i in Rk[name] if Rk[name][i] and not Rk[o][i]); c=sum(1 for i in Rk[name] if Rk[o][i] and not Rk[name][i])
        p=binomtest(b,b+c).pvalue if b+c else 1
        print(f'   대 {o:14s} {100*(b-c)/n:+5.1f}%p  b/c {b}/{c}  p={p:.2g}')
