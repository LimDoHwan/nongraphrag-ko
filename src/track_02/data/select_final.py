"""Track 2 - 최종 평가셋 선택 (카테고리 균형).

Gemma 검증 통과분(synth_gemma.json)에서 지정 카테고리별 N건을 뽑아 균형 평가셋을
만든다. 기본: 환경기상·산업고용 제외(통과 30 미달), 나머지 8종 × 30 = 240쌍.

산출: data/admin_ko/questions_admin.json (wiki_ko 병렬 스키마)
실행: python -m src.track_02.data.select_final --per 30 --exclude 환경기상 산업고용
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"
CATS = ["과학기술", "환경기상", "공공행정", "국토관리", "식품건강",
        "사회복지", "법률", "농축수산", "산업고용", "문화관광"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=30, help="카테고리별 선택 수")
    ap.add_argument("--exclude", nargs="*", default=["환경기상", "산업고용"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    passed = json.load(open(DATA / "synth_gemma.json", encoding="utf-8"))
    bycat = defaultdict(list)
    for r in passed:
        bycat[r["category"]].append(r)

    cats = [c for c in CATS if c not in args.exclude]
    final = []
    print(f"{'카테고리':<8}{'보유':>6}{'선택':>6}")
    for c in cats:
        pool = bycat.get(c, [])
        rng.shuffle(pool)
        take = pool[:args.per]
        if len(take) < args.per:
            print(f"{c:<8}{len(pool):>6}{len(take):>6}  ⚠️ 부족")
        else:
            print(f"{c:<8}{len(pool):>6}{len(take):>6}")
        final.extend(take)

    # 정리: 디버그 필드 제거, 최종 스키마
    out = [{"id": r["id"], "category": r["category"], "question": r["question"],
            "answer": r["answer"], "supporting_facts": r["supporting_facts"],
            "hops": r["hops"], "question_type": r["question_type"],
            "source": r["source"]} for r in final]
    json.dump(out, open(DATA / "questions_admin.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n최종 {len(out)}쌍  (제외: {', '.join(args.exclude)})")
    print(f"saved data/admin_ko/questions_admin.json")


if __name__ == "__main__":
    main()
