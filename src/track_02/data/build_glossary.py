"""Track 2 - 사용자 사전 구축 (PMI 자동추출, §12).

dict-pool 코퍼스(전체 풀 − 평가 gold 문서)에서 통계적으로 응집된 복합명사를
NPMI + 빈도로 자동 발굴 → kiwipiepy 사용자 사전 후보.

leakage 차단: 평가셋(questions_admin.json)이 쓴 supporting 문서(gold)를 제외.
정답을 보지 않고 코퍼스 통계로만 구축 → 재현 가능(부록 공개).

후보 = 연속 명사(NNG/NNP) run(길이≥2). 점수 = run 내부 인접 bigram NPMI 의 최소값.
채택 = df ≥ --min-df 이고 min-NPMI ≥ --npmi.

산출: data/admin_ko/glossary.json  (복합명사 리스트 + df/npmi)
실행: python -m src.track_02.data.build_glossary --min-df 5 --npmi 0.3
"""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from kiwipiepy import Kiwi

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"
_NOUN = {"NNG", "NNP"}


def noun_runs(tokens):
    """연속 명사 run 들을 [(forms_tuple)] 로. (run 길이>=2 만)"""
    runs, cur = [], []
    for t in tokens:
        if t.tag in _NOUN:
            cur.append(t.form)
        else:
            if len(cur) >= 2:
                runs.append(tuple(cur))
            cur = []
    if len(cur) >= 2:
        runs.append(tuple(cur))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--npmi", type=float, default=0.3)
    args = ap.parse_args()

    docs = (json.load(open(DATA / "docs_train.json", encoding="utf-8"))
            + json.load(open(DATA / "docs_val.json", encoding="utf-8")))
    # leakage 차단: 평가 gold 문서 제외
    qs = json.load(open(DATA / "questions_admin.json", encoding="utf-8"))
    gold = {str(p) for q in qs for p in q["supporting_facts"]}
    pool = [d for d in docs if d["passage_id"] not in gold]
    print(f"dict-pool {len(pool)} (전체 {len(docs)} − gold {len(gold)})")

    kiwi = Kiwi()
    uni = Counter()           # 명사 unigram
    big = Counter()           # 인접 명사 bigram
    comp_df = Counter()       # 복합명사 run(joined) -> df
    comp_parts = {}           # joined -> tuple(parts)
    n_uni = n_big = 0
    t0 = time.time()
    for i, d in enumerate(pool):
        toks = kiwi.tokenize(d["text"])
        nouns = [t.form for t in toks if t.tag in _NOUN]
        for w in nouns:
            uni[w] += 1
        n_uni += len(nouns)
        runs = noun_runs(toks)
        doc_comps = set()
        for r in runs:
            for a, b in zip(r, r[1:]):
                big[(a, b)] += 1
                n_big += 1
            j = "".join(r)
            doc_comps.add(j)
            comp_parts[j] = r
        for j in doc_comps:
            comp_df[j] += 1
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(pool)} ({time.time()-t0:.0f}s)", flush=True)

    def npmi(a, b):
        c = big[(a, b)]
        if c == 0:
            return -1.0
        p_ab = c / n_big
        p_a, p_b = uni[a] / n_uni, uni[b] / n_uni
        return math.log(p_ab / (p_a * p_b)) / (-math.log(p_ab))

    glossary = []
    for j, df in comp_df.items():
        if df < args.min_df:
            continue
        parts = comp_parts[j]
        mn = min(npmi(a, b) for a, b in zip(parts, parts[1:]))
        if mn >= args.npmi:
            glossary.append({"word": j, "df": df, "npmi": round(mn, 3),
                             "n_parts": len(parts)})
    glossary.sort(key=lambda x: (-x["df"]))

    json.dump(glossary, open(DATA / "glossary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # A2p(인라인 NPMI 게이팅)용 bigram NPMI 테이블 저장 (빈도 충분한 것만)
    npmi_tbl = {f"{a}\t{b}": round(npmi(a, b), 3)
                for (a, b), c in big.items() if c >= args.min_df}
    json.dump(npmi_tbl, open(DATA / "npmi_bigrams.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\n사전 {len(glossary)}개 / NPMI bigram {len(npmi_tbl)}개 "
          f"(min_df={args.min_df}, npmi≥{args.npmi})  {time.time()-t0:.0f}s")
    print("상위 15:", ", ".join(f"{g['word']}({g['df']})" for g in glossary[:15]))
    print("saved data/admin_ko/glossary.json")


if __name__ == "__main__":
    main()
