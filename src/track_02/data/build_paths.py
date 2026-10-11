"""Track 2 - 추론 경로 구성 (3단계 / §11.3 [2]).

bridge 인덱스를 이용해 2-hop / 3-hop 경로 후보를 deterministic 하게 만든다.
질문/정답 문장은 아직 만들지 않는다 — 먼저 경로 후보(어떤 문서가 어떤 bridge로
이어지고, 시작/정답 entity 후보가 무엇인지)를 점검하기 위함.

경로 구조 (계획 §11.3 [2]):
  2-hop: [시작] → 문서A → (b1) → 문서B → [정답]
  3-hop: [시작] → 문서A → (b1) → 문서B → (b2) → 문서C → [정답]
   - 시작  = 첫 문서에만 있는 distinctive 복합명사
   - 정답  = 끝 문서에만 있는 distinctive 복합명사
   - bridge = 인접 두 문서의 공통 복합명사

bridge 선택: 등장 문서수(df) 가 [min_df, max_df] 구간인 것만 (너무 흔한 양식어/
조직명·너무 드문 노이즈 완화). stopword 필터는 두지 않음(계획대로).

실행: python -m src.track_02.data.build_paths --split val --hops 2 3 --n 40
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"
CACHE = ROOT / "cache" / "track_02"


def load(split):
    bridges = json.load(open(CACHE / f"bridges_{split}.json", encoding="utf-8"))
    ents = json.load(open(CACHE / f"entities_compound_{split}.json", encoding="utf-8"))
    docs = {d["passage_id"]: d
            for d in json.load(open(DATA / f"docs_{split}.json", encoding="utf-8"))}
    return bridges, {k: set(v) for k, v in ents.items()}, docs


def distinctive(pid, entset, exclude, ent_sets):
    """pid 문서에만 있고 exclude(다른 경로 문서들의 entity합집합)에 없는 복합명사 중
    가장 긴(=구체적) 것 하나. 없으면 None."""
    cand = [e for e in ent_sets[pid] if e not in exclude]
    cand.sort(key=len, reverse=True)
    return cand[0] if cand else None


def build_2hop(bridges, ent_sets, docs, sel_bridges, n, rng):
    out = []
    rng.shuffle(sel_bridges)
    for b in sel_bridges:
        pids = bridges[b]
        if len(pids) < 2:
            continue
        a, c = rng.sample(pids, 2)
        sa, sc = ent_sets[a], ent_sets[c]
        start = distinctive(a, sa, sc | {b}, ent_sets)
        ans = distinctive(c, sc, sa | {b}, ent_sets)
        if not start or not ans:
            continue
        out.append({"hops": 2, "bridges": [b], "docs": [a, c],
                    "start_entity": start, "answer_entity": ans})
        if len(out) >= n:
            break
    return out


def build_3hop(bridges, ent_sets, docs, sel_set, n, rng):
    """가운데 문서 B 가 서로 다른 bridge b1,b2 를 갖고, b1→A, b2→C 로 잇는다."""
    out = []
    mids = list(ent_sets.keys())
    rng.shuffle(mids)
    for bmid in mids:
        # B 안에 있는 선택가능 bridge 들
        bs = [e for e in ent_sets[bmid] if e in sel_set and len(bridges[e]) >= 2]
        if len(bs) < 2:
            continue
        rng.shuffle(bs)
        b1, b2 = bs[0], bs[1]
        a_cand = [p for p in bridges[b1] if p != bmid]
        c_cand = [p for p in bridges[b2] if p != bmid]
        if not a_cand or not c_cand:
            continue
        a = rng.choice(a_cand)
        c = rng.choice(c_cand)
        if c == a:
            continue
        sa, sb, sc = ent_sets[a], ent_sets[bmid], ent_sets[c]
        start = distinctive(a, sa, sb | sc | {b1, b2}, ent_sets)
        ans = distinctive(c, sc, sa | sb | {b1, b2}, ent_sets)
        if not start or not ans:
            continue
        out.append({"hops": 3, "bridges": [b1, b2], "docs": [a, bmid, c],
                    "start_entity": start, "answer_entity": ans})
        if len(out) >= n:
            break
    return out


def snippet(text, term, width=45):
    i = text.find(term)
    if i < 0:
        return text[:width * 2].replace("\n", " ")
    s = max(0, i - width)
    return ("…" if s > 0 else "") + text[s:i + len(term) + width].replace("\n", " ") + "…"


def show(path, docs):
    ds = [docs[p] for p in path["docs"]]
    print(f"\n[{path['hops']}-hop]  시작=({path['start_entity']})  정답=({path['answer_entity']})  bridge={path['bridges']}")
    for k, (p, d) in enumerate(zip(path["docs"], ds)):
        print(f"   문서{k+1} [{d['doc_class']}/{d['doc_source']}] {d['doc_title'][:40]}")
        for b in path["bridges"]:
            if b in d["text"]:
                print(f"       …{b}: {snippet(d['text'], b)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--hops", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-df", type=int, default=8)
    ap.add_argument("--n", type=int, default=40, help="hop 종류별 생성 수")
    ap.add_argument("--show", type=int, default=6, help="hop 종류별 출력 예시 수")
    args = ap.parse_args()
    rng = random.Random(42)

    bridges, ent_sets, docs = load(args.split)
    sel = [b for b, p in bridges.items() if args.min_df <= len(p) <= args.max_df]
    sel_set = set(sel)
    print(f"[{args.split}] bridge {len(bridges)}종 중 df {args.min_df}-{args.max_df} = {len(sel)}종 선택")

    all_paths = []
    if 2 in args.hops:
        p2 = build_2hop(bridges, ent_sets, docs, list(sel), args.n, rng)
        print(f"2-hop 경로 후보: {len(p2)}개")
        for p in p2[:args.show]:
            show(p, docs)
        all_paths += p2
    if 3 in args.hops:
        p3 = build_3hop(bridges, ent_sets, docs, sel_set, args.n, rng)
        print(f"\n3-hop 경로 후보: {len(p3)}개")
        for p in p3[:args.show]:
            show(p, docs)
        all_paths += p3

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"paths_{args.split}.json"
    json.dump(all_paths, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {out.relative_to(ROOT)} ({len(all_paths)}개)")


if __name__ == "__main__":
    main()
