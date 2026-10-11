"""Track 2 - bridge 인덱서 (2단계, 계획서 §11.3 [1]).

각 passage 에서 복합명사를 추출하고, "복합명사 -> 등장 passage" 역인덱스를 만든다.
2개 이상 문서(passage)에 등장하는 복합명사 = **bridge 후보**.
(span_extraction 은 문서당 paragraph 1개이므로 passage ≈ 문서.)

stopword 필터는 두지 않는다(계획대로 복합명사 필터만). 무의미 양식어 경로는
이후 합성 품질 검증(§11.3 [4])에서 거른다.

산출 (split=train|val 별):
  cache/track_02/entities_compound_{split}.json   # {passage_id: [복합명사...]} (passage내 dedup)
  cache/track_02/bridges_{split}.json             # {복합명사: [passage_id...]}  (2+ passage 만)

leakage 정의(§11.1.1): 사전 후보=train, 합성 bridge=val 에서 각각 사용.

실행: python -m src.track_02.data.build_bridge_index            # train, val 둘 다
      python -m src.track_02.data.build_bridge_index --splits val
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from kiwipiepy import Kiwi

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "admin_ko"
CACHE = ROOT / "cache" / "track_02"

from src.track_02.common.extract import extract_compounds  # noqa: E402


def build(split: str, kiwi: Kiwi):
    passages = json.load(open(DATA / f"docs_{split}.json", encoding="utf-8"))
    per_passage = {}            # passage_id -> [복합명사...] (dedup)
    ent2pid = defaultdict(set)  # 복합명사 -> {passage_id}

    t0 = time.time()
    for i, p in enumerate(passages):
        pid = p["passage_id"]
        ents = sorted(set(extract_compounds(p["text"], kiwi)))
        per_passage[pid] = ents
        for e in ents:
            ent2pid[e].add(pid)
        if (i + 1) % 10000 == 0:
            print(f"  [{split}] {i+1}/{len(passages)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    bridges = {e: sorted(pids) for e, pids in ent2pid.items() if len(pids) >= 2}

    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(per_passage, open(CACHE / f"entities_compound_{split}.json", "w",
              encoding="utf-8"), ensure_ascii=False)
    json.dump(bridges, open(CACHE / f"bridges_{split}.json", "w",
              encoding="utf-8"), ensure_ascii=False)

    n_ent = len(ent2pid)
    n_br = len(bridges)
    print(f"[{split}] passages={len(passages)}  복합명사={n_ent}종  "
          f"bridge후보(2+ passage)={n_br}종 ({100*n_br/max(n_ent,1):.1f}%)  "
          f"{time.time()-t0:.0f}s")
    top = sorted(bridges.items(), key=lambda x: -len(x[1]))[:10]
    print("  bridge 상위:", ", ".join(f"{e}({len(p)})" for e, p in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val"])
    args = ap.parse_args()
    kiwi = Kiwi()
    for sp in args.splits:
        build(sp, kiwi)


if __name__ == "__main__":
    main()
