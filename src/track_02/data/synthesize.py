"""Track 2 - QA-앵커 씨앗 생성 (3단계 전반 / §11.3).

순수 co-occurrence 는 관계가 없어 실패 → 사람이 라벨링한 정답 span 을 앵커로
2홉 씨앗을 만든다. 카테고리 균형(목표 50/카테고리)을 위해 train+val 을 풀링하고
카테고리별로 캡(--cap)을 둔다. (leakage: 최종 선택된 eval 문서는 추후 사전 구축
코퍼스에서 제외한다 — §12.)

2홉 씨앗 (compositional):
  문서A: X 가 '정답'이면서 '제목'에 등장(= 문서A 는 X 에 관함). docA 후보.
  문서B: 어떤 질문 q_b 안에 X 가 등장 → 그 질문의 정답 A_B.
  → 합성 시 X 를 문서A 정보로 우회 지칭하면, 답(A_B)은 문서B, X 식별은 문서A 필요 = 2홉.
카테고리는 문서B(정답문서) 기준. (doc A 는 전 카테고리 차용 허용 → 소형 카테고리 수율↑)

산출: data/admin_ko/synth_seeds.json
실행: python -m src.track_02.data.synthesize --cap 150
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"

CATS = ["과학기술", "환경기상", "공공행정", "국토관리", "식품건강",
        "사회복지", "법률", "농축수산", "산업고용", "문화관광"]


def is_entity(a: str) -> bool:
    a = a.strip()
    if not (3 <= len(a) <= 25):
        return False
    if not re.search(r"[가-힣]", a):
        return False
    if sum(c.isdigit() for c in a) / len(a) > 0.3:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=150, help="카테고리별 씨앗 상한")
    args = ap.parse_args()

    docs = (json.load(open(DATA / "docs_train.json", encoding="utf-8"))
            + json.load(open(DATA / "docs_val.json", encoding="utf-8")))
    by_pid = {d["passage_id"]: d for d in docs}

    # 1) topical answer entity: 정답(len>=4, entity) 이면서 자기 제목에 등장 → docA 후보
    topical = defaultdict(list)  # X -> [pid]
    for d in docs:
        t = d["doc_title"]
        for q in d["orig_qas"]:
            if q["is_impossible"] or not q["answer"]:
                continue
            a = q["answer"].strip()
            if len(a) >= 4 and is_entity(a) and a in t:
                topical[a].append(d["passage_id"])

    # 2) Aho-Corasick 으로 각 질문에서 topical X 탐색
    A = ahocorasick.Automaton()
    for x in topical:
        A.add_word(x, x)
    A.make_automaton()

    seeds = []
    seen = set()
    cat_cnt = Counter()
    for d in docs:
        b, tb, cb = d["passage_id"], d["doc_title"], d["doc_class"]
        if cb not in CATS or cat_cnt[cb] >= args.cap:
            continue
        for q in d["orig_qas"]:
            if q["is_impossible"] or not q["answer"]:
                continue
            qb, ab = q["question"], q["answer"].strip()
            for x in set(v for _, v in A.iter(qb)):
                a_srcs = [p for p in topical[x]
                          if p != b and by_pid[p]["doc_title"] != tb]
                if not a_srcs:
                    continue
                key = (a_srcs[0], b, x)
                if key in seen:
                    continue
                seen.add(key)
                seeds.append({
                    "id": f"adm_{len(seeds)}",
                    "category": cb,
                    "hops": 2,
                    "bridge": x,
                    "docs": [a_srcs[0], b],
                    "answer": ab,
                    "_q_b": qb,
                })
                cat_cnt[cb] += 1
                break  # 문서B QA 당 1개
            if cat_cnt[cb] >= args.cap:
                break

    json.dump(seeds, open(DATA / "synth_seeds.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"씨앗 {len(seeds)}개 (cap={args.cap}/카테고리)")
    print(f"{'카테고리':<8}{'씨앗':>7}")
    for c in CATS:
        print(f"{c:<8}{cat_cnt[c]:>7}")
    print(f"saved data/admin_ko/synth_seeds.json")


if __name__ == "__main__":
    main()
