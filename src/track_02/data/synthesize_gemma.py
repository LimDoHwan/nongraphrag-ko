"""Track 2 - Gemma 판정+생성 (3단계 / §11.3 하이브리드 c + [4]).

QA-앵커 씨앗(synth_seeds.json)을 Gemma 가:
  ① bridge X 가 두 문서에서 '같은 특정 실체'인지 판정 (일반 범주는 기각)
  ② 같다면, 문서A 로 X 를 우회 지칭하고 문서B 에서 정답을 얻는 2홉 질문 작성
정답(A_B)은 고정. 이후 자동검증: same_entity & valid & 질문에 X 미포함 & 정답이 문서B에 존재.

동시성(--workers), per-id 캐싱(resume). 카테고리별 통과 수 집계.
산출: data/admin_ko/synth_gemma.json (검증 통과 전체, category 포함)

실행: python -m src.track_02.data.synthesize_gemma --workers 8
"""
import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"
CACHE = ROOT / "cache" / "track_02" / "synth_gemma"

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.track_01.common.llm_client import LLMClient  # noqa: E402

SYSTEM = (
    "당신은 한국어 multi-hop QA 데이터셋의 엄격한 검수자 겸 작성자입니다. "
    "두 행정문서를 모두 거쳐야 풀리는 2-hop 질문을 만들거나, 불가능하면 기각합니다. "
    "반드시 JSON 으로만 답합니다."
)

PROMPT = """[문서 A] 제목: {ta}
{ca}

[문서 B] 제목: {tb}
{cb}

[연결어]: {x}
[문서 B에서 도출된 정답(고정, 변경 금지)]: {ans}
[문서 B의 단서 질문(참고)]: {qb}

판정 기준 — 다음을 모두 만족할 때만 same_entity=true:
- 연결어 "{x}" 가 두 문서에서 **동일한 특정 대상**(같은 그 사업/기관/제도/행사/사건)을 가리킨다.
- 단지 같은 단어를 쓸 뿐 서로 다른 대상이거나, "{x}" 가 일반 범주어
  (예: 공동주택·어린이집·심의위원회·지역사회 처럼 종류를 뜻하는 보통명사)이면 false.

valid=true 이면 2-hop 질문을 한국어로 작성(아래 모두 충족):
- 문서 A 의 정보로 "{x}" 를 **우회 지칭**(질문에 "{x}" 표현을 절대 직접 쓰지 말 것).
- 그 대상에 대해 문서 B 에서 답할 수 있는 것을 물어, 정답이 정확히 "{ans}" 가 되게 한다.
- 문서 A·B 를 **모두** 봐야 풀려야 한다(한 문서만으론 불가).
불가능하면 valid=false, question 은 빈 문자열.

JSON 만 출력:
{{"same_entity": true/false, "valid": true/false, "question": "질문 또는 \"\"", "reason": "근거"}}"""


def parse_json(text: str):
    t = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0=전체")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    seeds = json.load(open(DATA / "synth_seeds.json", encoding="utf-8"))
    if args.limit:
        seeds = seeds[:args.limit]
    docs = (json.load(open(DATA / "docs_train.json", encoding="utf-8"))
            + json.load(open(DATA / "docs_val.json", encoding="utf-8")))
    by_pid = {d["passage_id"]: d for d in docs}
    llm = LLMClient()

    def judge(c):
        cf = CACHE / f"{c['id']}.json"
        if cf.exists():
            return json.load(open(cf, encoding="utf-8"))
        da, db = by_pid[c["docs"][0]], by_pid[c["docs"][1]]
        prompt = PROMPT.format(ta=da["doc_title"], ca=da["text"],
                               tb=db["doc_title"], cb=db["text"],
                               x=c["bridge"], ans=c["answer"], qb=c["_q_b"])
        try:
            out = llm.chat([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
                           temperature=0.0, max_tokens=512)
            r = parse_json(out) or {"same_entity": False, "valid": False,
                                    "question": "", "reason": "PARSE_FAIL"}
        except Exception as e:  # noqa: BLE001
            r = {"same_entity": False, "valid": False, "question": "",
                 "reason": f"ERR:{e}"}
        json.dump(r, open(cf, "w", encoding="utf-8"), ensure_ascii=False)
        return r

    done = 0
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for c, r in zip(seeds, ex.map(judge, seeds)):
            done += 1
            q = (r.get("question") or "").strip()
            db = by_pid[c["docs"][1]]
            ok = (r.get("same_entity") and r.get("valid") and q
                  and c["bridge"] not in q and c["answer"] in db["text"])
            if ok:
                results.append({
                    "id": c["id"], "category": c["category"], "hops": 2,
                    "bridge": c["bridge"], "docs": c["docs"],
                    "supporting_facts": c["docs"], "question": q,
                    "answer": c["answer"], "question_type": "multi-hop",
                    "source": "admin_synth_gemma",
                })
            if done % 100 == 0:
                print(f"  {done}/{len(seeds)}  통과 {len(results)}", flush=True)

    json.dump(results, open(DATA / "synth_gemma.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    bycat = Counter(r["category"] for r in results)
    print(f"\n씨앗 {len(seeds)} → 검증통과 {len(results)} ({100*len(results)/len(seeds):.0f}%)")
    print(f"{'카테고리':<8}{'통과':>6}{'≥50':>6}")
    for c in ["과학기술", "환경기상", "공공행정", "국토관리", "식품건강",
              "사회복지", "법률", "농축수산", "산업고용", "문화관광"]:
        n = bycat.get(c, 0)
        print(f"{c:<8}{n:>6}{'O' if n >= 50 else 'X':>6}")
    print("saved data/admin_ko/synth_gemma.json")


if __name__ == "__main__":
    main()
