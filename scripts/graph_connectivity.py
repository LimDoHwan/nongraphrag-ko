"""추출 방식별 엔티티 그래프 연결성 지표 (탐색적 분석, 2026-10-10).

LinearRAG engine._build_graph 와 같은 규칙(text.count(e) > 0 일 때만 문서–엔티티 끈)으로
캐시된 추출 결과(cache/track_02/linrag_{mode}_{dataset}/ner.json 의 p2e)에서 다음을 센다.

  - 고유 엔티티     : 추출된 서로 다른 엔티티 문자열 수
  - 다리 엔티티     : 2개 이상의 문서와 끈으로 이어진 엔티티 수
  - 허브 끈 비율    : 전체 문서–엔티티 끈 중, 문서의 5% 이상과 이어진 엔티티에 걸린 끈의 비율
  - 끈 없는 언급    : (문서, 엔티티) 언급 중 원문에 그 문자열이 그대로 없어 끈이 생기지 않은 비율
  - 다리 없는 문서  : 다리 엔티티와 하나도 이어지지 않은 문서의 비율 (그래프에서 고립)

실행: python scripts/graph_connectivity.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset  # noqa: E402

HUB_FRAC = 0.05


def main():
    for ds in ("wiki_ko", "admin_full"):
        passages = load_dataset(ds)[0]
        texts = {str(p["id"]): p["text"] for p in passages}
        n_doc = len(passages)
        for mode in ("nnp", "noun", "comp"):
            p2e = json.load(open(ROOT / f"cache/track_02/linrag_{mode}_{ds}/ner.json"))["p2e"]
            if not set(texts) & set(p2e):  # wiki_ko 캐시는 순번 키
                texts = {str(i): p["text"] for i, p in enumerate(passages)}
            df, mentions, orphan = Counter(), 0, 0
            for pid, ents in p2e.items():
                for e in set(ents):
                    mentions += 1
                    if texts[pid].count(e) > 0:
                        df[e] += 1
                    else:
                        orphan += 1
            uniq = {e for es in p2e.values() for e in es}
            bridge = {e for e, c in df.items() if c >= 2}
            edges = sum(df.values())
            hub_edges = sum(c for c in df.values() if c >= HUB_FRAC * n_doc)
            isolated = sum(1 for pid, es in p2e.items() if not (set(es) & bridge))
            print(f"{ds:10s} {mode:5s} 고유 {len(uniq):7,d}  다리 {len(bridge):6,d}  "
                  f"허브끈 {hub_edges / edges:.1%}  끈없는언급 {orphan / mentions:.1%}  "
                  f"다리없는문서 {isolated / len(p2e):.1%}")


if __name__ == "__main__":
    main()
