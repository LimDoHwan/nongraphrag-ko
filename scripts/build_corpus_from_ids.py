"""AI-Hub 원천에서 admin 검색 코퍼스와 admin single 질의를 복원한다.

AI-Hub 이용정책상 원문이 든 코퍼스와 원본 질의응답은 재배포할 수 없으므로, 이 저장소에는
문서 ID 목록만 들어 있다. 이 스크립트는 그 목록으로 원천 문서에서 코퍼스를 다시 만들고,
data/admin_ko/CHECKSUMS.json 의 정규화 해시와 비교해 실험에 쓴 것과 같은지 확인한다.

선행: AI-Hub 「행정 문서 대상 기계독해 데이터」(데이터셋 번호 569)의 span_extraction 라벨링 데이터를
      data/admin_ko/raw/ 에 두고 `python -m src.track_02.data.load_raw` 를 실행해
      data/admin_ko/docs_train.json, docs_val.json 을 만든다 (README 참조).

산출: data/admin_ko/chunks_{single,full,hard,hard2k}.json, data/admin_ko/questions_admin_single.json
실행: python scripts/build_corpus_from_ids.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "admin_ko"
CORPORA = ("single", "full", "hard", "hard2k")


def canon_md5(obj):
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def main():
    docs = {}
    for split in ("docs_train.json", "docs_val.json"):
        f = D / split
        if not f.exists():
            sys.exit(f"{f} 없음 — 먼저 AI-Hub 원천을 받아 load_raw 를 실행하세요 (README 참조)")
        for d in json.load(open(f, encoding="utf-8")):
            docs[str(d["passage_id"])] = d
    expected = json.load(open(D / "CHECKSUMS.json", encoding="utf-8"))
    ok = True

    for name in CORPORA:
        ids = json.load(open(D / "corpus_ids" / f"{name}.json", encoding="utf-8"))
        chunks = [{"chunk_id": cid, "passage_id": pid, "text": docs[pid]["text"], "role": role}
                  for cid, pid, role in ids]
        json.dump(chunks, open(D / f"chunks_{name}.json", "w", encoding="utf-8"), ensure_ascii=False)
        h = canon_md5(chunks)
        good = h == expected[f"chunks_{name}.json"]
        ok &= good
        print(f"chunks_{name}.json  {len(chunks):,}건  {'일치' if good else '불일치'}")

    qids = json.load(open(D / "questions_admin_single_ids.json", encoding="utf-8"))
    qs = []
    for qid, pid, k in qids:
        qa = docs[pid]["orig_qas"][k]
        qs.append({"id": qid, "question": qa["question"], "answer": qa["answer"].strip(),
                   "supporting_facts": [pid]})
    json.dump(qs, open(D / "questions_admin_single.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    h = canon_md5(qs)
    good = h == expected["questions_admin_single.json"]
    ok &= good
    print(f"questions_admin_single.json  {len(qs)}문항  {'일치' if good else '불일치'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
