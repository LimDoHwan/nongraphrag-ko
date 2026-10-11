"""Track 2 - admin_ko raw 로더 (1단계).

AI-Hub "행정문서 대상 기계독해"(dataSetSn=569) span_extraction 라벨 zip 을
정규화된 passage 레코드로 변환한다. 계획서 §11.1.1 데이터 사용 정의 기준:
  - 코퍼스: span_extraction 만
  - passage = 각 문서의 paragraph.context (+ doc 메타)
  - split: Training / Validation 분리 유지 (사전=Train, 평가합성=Val)
  - 원본 single-hop QA 는 미사용이나 참고용으로 orig_qas 에 보존

산출:
  data/admin_ko/docs_train.json   (Training passage 레코드)
  data/admin_ko/docs_val.json     (Validation passage 레코드)

passage 레코드 스키마:
  {passage_id, doc_id, doc_title, doc_class, doc_source, doc_published,
   text, orig_qas:[{question, answer, answer_start, qa_type, is_impossible}]}

실행: python -m src.track_02.data.load_raw
"""
import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "admin_ko" / "raw"
OUT = ROOT / "data" / "admin_ko"

# split -> 라벨 zip 경로 (span_extraction 만)
ZIPS = {
    "train": RAW / "1.Training" / "라벨링데이터" / "TL_span_extraction.zip",
    "val":   RAW / "2.Validation" / "라벨링데이터" / "VL_span_extraction.zip",
}


def read_label_zip(zp: Path):
    """zip 안의 단일 JSON 을 메모리에서 읽어 data 리스트 반환."""
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith(".json"))
        with z.open(name) as f:
            return json.load(f)["data"]


def to_passages(docs):
    """원본 문서 리스트 -> passage 레코드 리스트 (paragraph 단위)."""
    out = []
    for d in docs:
        meta = {
            "doc_id": d.get("doc_id"),
            "doc_title": d.get("doc_title"),
            "doc_class": (d.get("doc_class") or {}).get("code"),
            "doc_source": d.get("doc_source"),
            "doc_published": d.get("doc_published"),
        }
        for p in d.get("paragraphs", []):
            orig_qas = []
            for qa in p.get("qas", []):
                ans = qa.get("answers") or {}
                orig_qas.append({
                    "question": qa.get("question"),
                    "answer": ans.get("text"),
                    "answer_start": ans.get("answer_start"),
                    "qa_type": qa.get("qa_type"),
                    "is_impossible": qa.get("is_impossible"),
                })
            out.append({
                "passage_id": p.get("context_id"),
                **meta,
                "text": p.get("context", ""),
                "orig_qas": orig_qas,
            })
    return out


def stats(passages, label):
    n = len(passages)
    avg_len = sum(len(p["text"]) for p in passages) // max(n, 1)
    cls = Counter(p["doc_class"] for p in passages)
    src = Counter(p["doc_source"] for p in passages)
    print(f"[{label}] passages={n}  avg_len={avg_len}자  "
          f"분류={len(cls)}종  부서={len(src)}종")
    top = ", ".join(f"{k}:{v}" for k, v in cls.most_common(5))
    print(f"         분류 상위: {top}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for sp in args.splits:
        zp = ZIPS[sp]
        if not zp.exists():
            raise FileNotFoundError(f"라벨 zip 없음: {zp}")
        docs = read_label_zip(zp)
        passages = to_passages(docs)
        out_path = OUT / f"docs_{sp}.json"
        json.dump(passages, open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False)
        stats(passages, sp)
        print(f"         saved {out_path.relative_to(ROOT)} "
              f"({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
