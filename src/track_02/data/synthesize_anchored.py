"""Track 2 - entity-anchored multi-hop 재합성 (option 2).

기존 QA-앵커 합성은 bridge를 숨기고 서술 → dense로 풀려서 엔티티 그래프 무용.
재설계: **질문에 anchor(문서A의 고유명사)를 명시하고, bridge를 타야 문서B의 답에
도달**하게 만든다 (2WikiMQA 구조). dense는 anchor로 docA만 찾고 docB는 못 찾아야 함.

씨앗: bridge 인덱스(bridges_val) — X가 docA·docB에 공유.
Gemma: docA에서 anchor, docB에서 답(C) 선택 + 2-hop 질문 작성.
       질문엔 anchor만, X·C·docB지칭어 금지.
검증: anchor∈질문, X∉질문, C∉질문, C∈docB, anchor∈docA.

산출: data/admin_ko/questions_anchored.json
실행: python -m src.track_02.data.synthesize_anchored --n 40 --workers 8
"""
import argparse
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"
CACHE = ROOT / "cache" / "track_02" / "synth_anchored"

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.track_01.common.llm_client import LLMClient  # noqa: E402

SYSTEM = ("당신은 한국어 multi-hop QA 작성자입니다. 두 문서가 공유하는 연결어를 매개로, "
          "한 문서만으론 못 풀고 두 문서를 모두 거쳐야 풀리는 2-hop 질문을 만듭니다. JSON만 답합니다.")

PROMPT = """[문서 A] 제목: {ta}
{ca}

[문서 B] 제목: {tb}
{cb}

[연결어]: {x}   (문서 A와 B에 공통 등장)

작업:
1. **문서 A** 에서, 문서 A를 특정하는 **구체적 고유명사 anchor** 하나 선택 (연결어 "{x}" 아닌, 기관/사업/제도/인물/장소/법령 등).
2. **문서 B** 에서, 답이 될 **구체적 사실/값 C** 하나 선택 (연결어·anchor 아닌).
3. 2-hop 질문 작성 — 다음 모두 충족:
   - 질문에 **anchor를 명시**(그 단어를 그대로 포함).
   - **질문 본문은 문서 A의 표현만 사용**한다. 연결어 "{x}" 를 매개로 문서 B로 가야 답이 나오게 하되,
     **"{x}" 와 답 C 는 쓰지 말 것.**
   - ★**문서 B의 고유한 단어·명칭·수치·표현을 질문에 절대 넣지 말 것.** 문서 B 에서는 "무엇을 묻는지"
     (일반 속성: 날짜·금액·인물·장소·수량·명칭 등)만 가져오고, 문서 B를 식별할 단서는 일절 쓰지 않는다.
     → dense 임베딩 검색만으로는 문서 B를 절대 못 찾고, anchor→연결어 경로를 거쳐야만 찾게.
   - 답은 정확히 C.
4. **두 문서가 연결어 "{x}" 에 대해 실제로 같은 대상/맥락을 공유할 때만** valid=true.
   단지 비슷한 성격이라고 "동일한 성격의/유사한 목적의" 식으로 무관한 문서를 억지로 잇지 말 것.
   억지 연결이면 valid=false.

JSON만 출력:
{{"valid": true/false, "anchor": "...", "question": "...", "answer": "C", "reason": "..."}}"""


def parse_json(t):
    t = re.sub(r"^```(json)?|```$", "", t.strip(), flags=re.MULTILINE).strip()
    i, j = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="씨앗 수(상한)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-df", type=int, default=6)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    # train+val 풀링 (규모 확장 — topical bridge 상한 ↑, 카테고리 다양성 ↑)
    CT = ROOT / "cache/track_02"
    bt = json.load(open(CT / "bridges_train.json", encoding="utf-8"))
    bv = json.load(open(CT / "bridges_val.json", encoding="utf-8"))
    bridges = {}
    for x in set(bt) | set(bv):
        bridges[x] = bt.get(x, []) + bv.get(x, [])
    docs = {d["passage_id"]: d
            for split in ("docs_train", "docs_val")
            for d in json.load(open(DATA / f"{split}.json", encoding="utf-8"))}
    print(f"풀링: bridges {len(bridges)}, docs {len(docs)}")

    rng = random.Random(42)
    # 카테고리 균형: bridge 의 topical 문서들의 doc_class 로 라운드로빈 선택
    from collections import defaultdict
    by_cat = defaultdict(list)
    for x, pids in bridges.items():
        if not (args.min_df <= len(pids) <= args.max_df and len(x) >= 4):
            continue
        topical = [p for p in pids if p in docs and x in docs[p]["doc_title"]]
        if len(topical) < 2:
            continue
        cat = docs[topical[0]]["doc_class"]
        by_cat[cat].append((x, topical))
    for c in by_cat:
        rng.shuffle(by_cat[c])
    print("카테고리별 topical 씨앗 가용:", {c: len(v) for c, v in sorted(by_cat.items())})

    # 라운드로빈으로 카테고리 균형 추출
    cats = [c for c in by_cat if c != "기타"]
    seeds = []
    ptr = {c: 0 for c in cats}
    while len(seeds) < args.n:
        progressed = False
        for c in cats:
            if ptr[c] >= len(by_cat[c]):
                continue
            x, topical = by_cat[c][ptr[c]]; ptr[c] += 1; progressed = True
            a, b = rng.sample(topical, 2)
            if docs[a]["doc_title"] != docs[b]["doc_title"]:
                seeds.append({"id": f"anc_{len(seeds)}", "bridge": x, "docs": [a, b]})
            if len(seeds) >= args.n:
                break
        if not progressed:
            break

    llm = LLMClient()

    def gen(s):
        cf = CACHE / f"{s['id']}.json"
        if cf.exists():
            return json.load(open(cf, encoding="utf-8"))
        da, db = docs[s["docs"][0]], docs[s["docs"][1]]
        prompt = PROMPT.format(ta=da["doc_title"], ca=da["text"],
                               tb=db["doc_title"], cb=db["text"], x=s["bridge"])
        try:
            out = llm.chat([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
                           temperature=0.0, max_tokens=512)
            r = parse_json(out) or {"valid": False, "reason": "PARSE_FAIL"}
        except Exception as e:  # noqa: BLE001
            r = {"valid": False, "reason": f"ERR:{e}"}
        json.dump(r, open(cf, "w", encoding="utf-8"), ensure_ascii=False)
        return r

    results = []
    n_valid = n_pass = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for s, r in zip(seeds, ex.map(gen, seeds)):
            n_valid += bool(r.get("valid"))
            q = (r.get("question") or "").strip()
            anc = (r.get("anchor") or "").strip()
            ans = (r.get("answer") or "").strip()
            db = docs[s["docs"][1]]; da = docs[s["docs"][0]]
            ok = (r.get("valid") and q and anc and ans
                  and anc in q                       # anchor 명시됨
                  and s["bridge"] not in q           # bridge 숨김
                  and ans not in q                   # 답 숨김
                  and ans in db["text"]              # 답이 docB에
                  and anc in da["text"])             # anchor가 docA에
            if ok:
                n_pass += 1
                results.append({"id": s["id"], "hops": 2, "bridge": s["bridge"],
                                "anchor": anc, "docs": s["docs"],
                                "supporting_facts": s["docs"], "question": q,
                                "answer": ans, "question_type": "multi-hop",
                                "source": "admin_anchored"})

    json.dump(results, open(DATA / "questions_anchored.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"씨앗 {len(seeds)} → valid {n_valid} → 검증통과 {n_pass}")
    for r in results[:6]:
        print(f"\n— anchor=({r['anchor']}) bridge=({r['bridge']}) 답=({r['answer']})")
        print(f"   Q: {r['question']}")
    print(f"\nsaved data/admin_ko/questions_anchored.json")


if __name__ == "__main__":
    main()
