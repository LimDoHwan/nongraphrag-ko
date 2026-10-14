"""Track 2 - 데이터셋 추상화 (baseline 공통 로더 / §3.1).

wiki_ko(Track 1) 와 admin_ko(Track 2) 를 동일 인터페이스로 반환한다.
  load_dataset(name) -> (passages, questions)
    passages  = [{"id": str, "text": str}, ...]   # 검색 코퍼스
    questions = [{"id": str, "question": str, "answer": str,
                  "supporting": [pid...]}, ...]
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"


def _load_wiki_ko(keep_index=False):
    """keep_index=False: 본문만(기존 wiki_ko). True: 원 LinearRAG처럼 "번호:본문"을 유지(wiki_ko_v2).
    원 구현은 이 번호로 인접 문단 간선을 만들고 번호째 임베딩한다(engine._build_graph 의 "idx:" 처리)."""
    chunks = json.load(open(DATA / "2wiki_ko" / "chunks.json", encoding="utf-8"))
    passages = []
    for s in chunks:                       # "idx:본문"
        i, _, text = s.partition(":")
        i, text = i.strip(), text.strip()
        passages.append({"id": i, "text": f"{i}:{text}" if keep_index else text})
    qs = json.load(open(DATA / "2wiki_ko" / "questions.json", encoding="utf-8"))
    questions = [{"id": str(q["id"]), "question": q["question"],
                  "answer": q["answer"], "supporting": q.get("evidence", [])}
                 for q in qs]
    return passages, questions


def _load_admin_ko():
    chunks = json.load(open(DATA / "admin_ko" / "chunks.json", encoding="utf-8"))
    passages = [{"id": str(c["passage_id"]), "text": c["text"]} for c in chunks]
    qs = json.load(open(DATA / "admin_ko" / "questions_admin.json", encoding="utf-8"))
    questions = [{"id": q["id"], "question": q["question"], "answer": q["answer"],
                  "supporting": [str(p) for p in q.get("supporting_facts", [])]}
                 for q in qs]
    return passages, questions


def _load_admin_hard():
    """적대적 필터로 선별한 traversal-필수 hard 셋 + 전용 코퍼스."""
    chunks = json.load(open(DATA / "admin_ko" / "chunks_hard.json", encoding="utf-8"))
    passages = [{"id": str(c["passage_id"]), "text": c["text"]} for c in chunks]
    qs = json.load(open(DATA / "admin_ko" / "questions_admin_hard.json", encoding="utf-8"))
    questions = [{"id": q["id"], "question": q["question"], "answer": q["answer"],
                  "supporting": [str(p) for p in q.get("supporting_facts", [])]}
                 for q in qs]
    return passages, questions


def _load_admin_full():
    chunks = json.load(open(DATA / "admin_ko" / "chunks_full.json", encoding="utf-8"))
    passages = [{"id": str(c["passage_id"]), "text": c["text"]} for c in chunks]
    qs = json.load(open(DATA / "admin_ko" / "questions_anchored.json", encoding="utf-8"))
    questions = [{"id": q["id"], "question": q["question"], "answer": q["answer"],
                  "supporting": [str(p) for p in q.get("supporting_facts", [])]}
                 for q in qs]
    return passages, questions


def _load_admin_single():
    chunks = json.load(open(DATA / "admin_ko" / "chunks_single.json", encoding="utf-8"))
    passages = [{"id": str(c["passage_id"]), "text": c["text"]} for c in chunks]
    qs = json.load(open(DATA / "admin_ko" / "questions_admin_single.json", encoding="utf-8"))
    questions = [{"id": q["id"], "question": q["question"], "answer": q["answer"],
                  "supporting": [str(p) for p in q.get("supporting_facts", [])]} for q in qs]
    return passages, questions


def _load_admin_hard2k():
    chunks = json.load(open(DATA / "admin_ko" / "chunks_hard2k.json", encoding="utf-8"))
    passages = [{"id": str(c["passage_id"]), "text": c["text"]} for c in chunks]
    qs = json.load(open(DATA / "admin_ko" / "questions_admin_hard.json", encoding="utf-8"))
    questions = [{"id": q["id"], "question": q["question"], "answer": q["answer"],
                  "supporting": [str(p) for p in q.get("supporting_facts", [])]} for q in qs]
    return passages, questions


def load_dataset(name: str):
    if name == "wiki_ko":
        return _load_wiki_ko()
    if name == "wiki_ko_v2":
        return _load_wiki_ko(keep_index=True)
    if name == "admin_ko":
        return _load_admin_ko()
    if name == "admin_hard":
        return _load_admin_hard()
    if name == "admin_full":
        return _load_admin_full()
    if name == "admin_single":
        return _load_admin_single()
    if name == "admin_hard2k":
        return _load_admin_hard2k()
    raise ValueError(f"unknown dataset: {name}")
