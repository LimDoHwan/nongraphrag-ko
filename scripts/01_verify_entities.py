"""Phase 2 - 데이터 검증 (sanity check).

번역본 품질 확인: 질문의 정답(answer)이 실제로 청크 코퍼스에 등장하는지
매칭률을 측정한다. entity 표기 불일치가 있으면 LinearRAG 의 entity activation
이 무력화되므로, 모든 실험 전에 반드시 통과해야 한다.

- yes/no(예/아니오)형 정답은 코퍼스에 없는 게 정상 -> 매칭 대상에서 제외
- 한국어 조사 제거 + 공백/구두점 제거 후 substring 매칭
- 결과: data/2wiki_ko/entity_consistency.json (통계 + 불일치 목록)
- 판정: 매칭률 >= 95% 통과
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "2wiki_ko"

_JOSA = re.compile(r"(은|는|을|를|이|가|와|과|에서|에게|으로|로|에|의|도|만|"
                   r"까지|부터|이다|입니다|였다|이었다)$")
_BOOL = {"예", "아니오", "아니요", "yes", "no", "네"}


def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = _JOSA.sub("", s)
    # 공백·구두점 제거 (표기 띄어쓰기 차이 흡수)
    return re.sub(r"[\s\.,'\"()\-·]", "", s)


def main():
    questions = json.load(open(DATA / "questions.json", encoding="utf-8"))
    chunks = json.load(open(DATA / "chunks.json", encoding="utf-8"))
    corpus = norm(" ".join(chunks))

    total = len(questions)
    boolean = []
    matched = []
    mismatched = []

    for q in questions:
        ans = str(q.get("answer", "")).strip()
        if not ans:
            mismatched.append({"id": q.get("id"), "answer": ans,
                               "question": q.get("question"),
                               "reason": "empty_answer"})
            continue
        if ans.lower() in _BOOL:
            boolean.append(q.get("id"))
            continue
        na = norm(ans)
        toks = [norm(t) for t in re.split(r"[\s,·]+", ans) if len(norm(t)) >= 2]
        core = max(toks, key=len) if toks else na
        # 전체 일치 OR 핵심 토큰(최장) 코퍼스 존재 OR 토큰 과반 존재
        hit = (na in corpus) or (core and core in corpus) or \
              (toks and sum(1 for t in toks if t in corpus) >= max(1, len(toks)//2))
        if hit:
            matched.append(q.get("id"))
        else:
            mismatched.append({"id": q.get("id"), "answer": ans,
                               "question": q.get("question"),
                               "type": q.get("question_type"),
                               "reason": "answer_not_in_corpus"})

    applicable = len(matched) + len(mismatched)
    rate = len(matched) / applicable * 100 if applicable else 0.0

    report = {
        "total_questions": total,
        "boolean_excluded": len(boolean),
        "applicable": applicable,
        "matched": len(matched),
        "mismatched": len(mismatched),
        "match_rate_pct": round(rate, 2),
        "verdict": ("PASS (>=95%)" if rate >= 95 else
                    "REVIEW (90-95%)" if rate >= 90 else "FAIL (<90%)"),
        "mismatch_samples": mismatched[:50],
    }
    out = DATA / "entity_consistency.json"
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"전체 질문        : {total}")
    print(f"yes/no 제외      : {len(boolean)}")
    print(f"검증 대상        : {applicable}")
    print(f"정답 코퍼스 매칭 : {len(matched)}")
    print(f"불일치           : {len(mismatched)}")
    print(f"매칭률           : {rate:.2f}%  ->  {report['verdict']}")
    print(f"리포트 저장      : {out}")
    if mismatched:
        print("\n불일치 표본 (최대 8):")
        for m in mismatched[:8]:
            print(f"  [{m.get('type','?')}] A='{m['answer']}'  Q={m.get('question','')[:40]}")


if __name__ == "__main__":
    main()
