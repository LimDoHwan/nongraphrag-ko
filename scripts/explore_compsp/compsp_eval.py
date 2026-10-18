"""compsp 계열 GPT-Acc·Contain: 응답 생성(gemma-4-31B, QA_SYS) → gpt-4o 판정 → 비교.
단계: qa <ds...> | judge <ds...> | report <ds...>
환경변수: SYSN(compsp|gazcompsp|gazcompsp_llm4b|gazcompsp_llm), STACK(검색 조정 태그, 예: pnw0.5-prf0.6sent),
          MAXTOK(출력 한도: comp 짝과 같게 — 기본·dense0.5+PRF(문장) 2000, 나머지 조정 256), LLM_URLS"""
import glob, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
from src.track_02.common.datasets import load_dataset
C = f'{ROOT}/cache/track_02'
OUT = f'{C}/compsp_eval'
os.makedirs(f'{OUT}/qa', exist_ok=True); os.makedirs(f'{OUT}/judge', exist_ok=True)
SYSN = os.environ.get('SYSN', 'compsp')
STACK = os.environ.get('STACK', '')
MAXTOK = int(os.environ.get('MAXTOK', '2000'))
NAME = f'linearrag-new-{SYSN}' + (f'-{STACK}' if STACK else '')
URLS = [u for u in os.environ.get('LLM_URLS', '').split(',') if u]  # 예: http://<host>:8000/v1,http://<host>:8001/v1
LIN_MAIN = ['linearrag-org-nnp', 'linearrag-new-noun', 'linearrag-new-comp', 'linearrag-new-gazcomp',
            'linearrag-new-gazcomp-llm', 'linearrag-new-gazcomp-llm4b', 'linearrag-new-gazcomp-llm4bv2',
            'linearrag-new-comp-pnw0.5-prf0.6sent']

def h(o): return hashlib.sha1(json.dumps(o, ensure_ascii=False).encode('utf-8')).hexdigest()
def top5(ds): return json.load(open(f'{C}/linrag_{SYSN}_{ds}/' + (f'retrieved_top5_{STACK}.json' if STACK else 'retrieved_top5.json')))
def texts(ds):
    src = 'wiki_ko' if ds == 'wiki_ko_v2' else ds          # 지문은 번호 없는 본문(◆ 조건과 같음)
    return {str(p['id']): p['text'] for p in load_dataset(src)[0]}

def gen_map():
    g = {}
    for d in glob.glob(f'{ROOT}/results/track_02/*/predictions.json'):
        sysname = d.split('/')[-2].rsplit('_', 2)[0] if '_admin_' in d else d.split('/')[-2].split('_wiki_ko')[0]
        if not sysname.startswith('linearrag-'): continue
        try:
            mt = int(json.load(open(d.replace('predictions.json', 'eval.json'))).get('qa_max_tokens') or 0)
        except Exception:
            mt = 0
        mt = mt or (2000 if sysname in LIN_MAIN else 256)       # 기존 결과: ◆ 2000, 나머지 LinearRAG 256
        for x in json.load(open(d)):
            if x.get('prediction') is not None and 'retrieved' in x and 'question' in x:
                g.setdefault(h([x['question'], [str(r) for r in x['retrieved'][:5]], mt]), x['prediction'])
    for f in glob.glob(f'{C}/rerun_wiki_v2/qa/*.json'):
        j = json.load(open(f))
        g.setdefault(os.path.basename(f)[:-5], j['prediction'])     # 키에 출력 한도 포함
    return g

def stage_qa(dss, workers=16):
    from src.track_01.common.llm_client import LLMClient
    from src.track_02.linearrag.engine import QA_SYS
    import hashlib as _h
    g = gen_map()
    jobs = {}
    reused = 0
    for ds in dss:
        T, R = texts(ds), top5(ds)
        for q in load_dataset(ds)[1]:
            ret = R[q['id']]
            k = h([q['question'], ret, MAXTOK])
            if k in g or os.path.exists(f'{OUT}/qa/{k}.json'): reused += k in g; continue
            uk = None if MAXTOK != 2000 else _h.sha1(json.dumps([q['question'], ret], ensure_ascii=False).encode()).hexdigest()
            uf = f'{C}/unify_qa/{uk}.json'
            if uk and os.path.exists(uf):
                json.dump({'prediction': json.load(open(uf))['prediction'], 'source': 'unify_qa'}, open(f'{OUT}/qa/{k}.json', 'w'), ensure_ascii=False)
                continue
            jobs[k] = (q['question'], '\n'.join(T[x] for x in ret))
    print(f'[qa] 재사용 {reused}건, 새로 생성 {len(jobs)}건', flush=True)
    clients = []
    for u in URLS:
        c = LLMClient(timeout=600.0); c.url = u + '/chat/completions'; clients.append(c)
    order = {k: i for i, k in enumerate(jobs)}
    def run(k):
        qq, ctx = jobs[k]
        try:
            out, fin = clients[order[k] % len(clients)].chat_full([{'role': 'system', 'content': QA_SYS},
                         {'role': 'user', 'content': f'{ctx}\nQuestion: {qq}\n Thought: '}], temperature=0.0, max_tokens=MAXTOK)
        except Exception as e:
            print(f'  [실패] {k[:10]} {e}', flush=True); return False
        pred = out.split('Answer:')[-1].strip() if 'Answer:' in out else out.strip()
        json.dump({'prediction': pred, 'finish_reason': fin, 'max_tokens': MAXTOK, 'raw': out}, open(f'{OUT}/qa/{k}.json', 'w'), ensure_ascii=False)
        return True
    t0, done, fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(run, list(jobs)):
            done += 1; fail += not ok
            if done % 100 == 0: print(f'  {done}/{len(jobs)} ({time.time()-t0:.0f}s, 실패 {fail})', flush=True)
    print(f'[qa] 완료 {done-fail}건, 실패 {fail}건 ({time.time()-t0:.0f}s)', flush=True)

def rows(ds):
    g = gen_map(); R = top5(ds); out = []
    for q in load_dataset(ds)[1]:
        k = h([q['question'], R[q['id']], MAXTOK])
        pred = g[k] if k in g else json.load(open(f'{OUT}/qa/{k}.json'))['prediction']
        out.append({'id': q['id'], 'question': q['question'], 'answer': q['answer'], 'prediction': pred,
                    'retrieved': R[q['id']], 'supporting': q['supporting'], 'qa_source': 'reused' if k in g else 'new'})
    return out

def judge_map():
    j = {}
    for d in glob.glob(f'{ROOT}/results/track_02/*/predictions.json'):
        for x in json.load(open(d)):
            if 'gpt4o' in x and x.get('prediction') is not None:
                j[h([x['question'], x['answer'], x['prediction']])] = bool(x['gpt4o'])
    for f in glob.glob(f'{C}/rerun_wiki_v2/judge/*.json') + glob.glob(f'{OUT}/judge/*.json'):
        j.setdefault(os.path.basename(f)[:-5], json.load(open(f))['gpt4o'])
    return j

def stage_judge(dss, workers=8):
    from src.track_01.common.llm_client import OpenAIClient
    from src.track_01.common.evaluation import gpt_acc
    J = judge_map(); jobs = {}
    for ds in dss:
        for x in rows(ds):
            k = h([x['question'], x['answer'], x['prediction']])
            if k not in J: jobs[k] = (x['question'], x['answer'], x['prediction'])
    print(f'[judge] 새로 판정 {len(jobs)}건', flush=True)
    cl = OpenAIClient(model='gpt-4o')
    def run(k):
        qq, a, p = jobs[k]
        json.dump({'gpt4o': gpt_acc(qq, a, p, cl)}, open(f'{OUT}/judge/{k}.json', 'w'))
    with ThreadPoolExecutor(max_workers=workers) as ex: list(ex.map(run, list(jobs)))
    print('[judge] 완료', flush=True)

def stage_report(dss):
    from src.track_01.common.evaluation import contain_acc
    from scipy.stats import binomtest
    J = judge_map()
    for ds in dss:
        rs = rows(ds)
        for x in rs:
            x['gpt4o'] = J[h([x['question'], x['answer'], x['prediction']])]
            x['contain'] = contain_acc(x['prediction'], x['answer'])
        d = f'{ROOT}/results/track_02/{NAME}_{ds}'; os.makedirs(d, exist_ok=True)
        json.dump(rs, open(f'{d}/predictions.json', 'w'), ensure_ascii=False, indent=1)
        n = len(rs); A = {x['id']: x for x in rs}
        ev = {'dataset': ds, 'system': NAME, 'n': n, 'gpt4o_acc': round(sum(x['gpt4o'] for x in rs)/n, 4),
              'contain_acc': round(sum(x['contain'] for x in rs)/n, 4), 'gen_model': 'gemma-4-31B-it',
              'qa_prompt': 'linearrag.QA_SYS', 'qa_max_tokens': MAXTOK, 'qa_new': sum(x['qa_source'] == 'new' for x in rs),
              'note': f'탐색적 추가 실험(2026-10-17~18): {NAME}'}
        json.dump(ev, open(f'{d}/eval.json', 'w'), ensure_ascii=False, indent=1)
        print(f'\n== {ds} (n={n})  {NAME} GPT-Acc {ev["gpt4o_acc"]:.3f}  Contain {ev["contain_acc"]:.3f}  (새 답변 {ev["qa_new"]})')
        sfx = 'wiki_ko_v2' if ds == 'wiki_ko_v2' else ds
        GZ = {'gazcompsp': 'linearrag-new-gazcomp', 'gazcompsp_llm': 'linearrag-new-gazcomp-llm',
              'gazcompsp_llm4b': 'linearrag-new-gazcomp-llm4bv2' if ds == 'wiki_ko_v2' else 'linearrag-new-gazcomp-llm4b'}
        extra = ((('같은사전 gazcomp', GZ[SYSN]), ('compsp', 'linearrag-new-compsp')) if SYSN in GZ else ())
        if STACK:
            extra = (('같은조정 comp', f'linearrag-new-comp-{STACK}'), ('compsp 기본', 'linearrag-new-compsp'))
        for name, folder in extra + (('comp', 'linearrag-new-comp'), ('noun', 'linearrag-new-noun'),
                             ('HippoRAG2', 'graphrag-org'), ('comp+조정', 'linearrag-new-comp-pnw0.5-prf0.6sent')):
            sf = 'wiki_ko' if (ds == 'wiki_ko_v2' and folder == 'graphrag-org') else sfx
            B = {x['id']: x for x in json.load(open(f'{ROOT}/results/track_02/{folder}_{sf}/predictions.json'))}
            for m in ('gpt4o', 'contain'):
                b = sum(1 for i in A if A[i][m] and not B[i][m]); c = sum(1 for i in A if B[i][m] and not A[i][m])
                p = binomtest(b, b+c).pvalue if b+c else 1.0
                rb = sum(bool(B[i][m]) for i in A)/n
                print(f'   대 {name:10s} {m:7s} {rb:.3f} → 차이 {100*(b-c)/n:+5.1f}%p  b/c {b}/{c}  p={p:.2g}')

if __name__ == '__main__':
    st, dss = sys.argv[1], sys.argv[2:]
    {'qa': stage_qa, 'judge': stage_judge, 'report': stage_report}[st](dss)
