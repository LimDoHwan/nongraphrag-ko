"""Track 2 - OpenIE 캐시 → LLM 증류 사전(glossary) 구축.

2026-06 증류 사전을 만든 원 스크립트가 리포에 남아 있지 않아, 산출물과 캐시를 대조해
규칙을 역추적·복원했다(2026-09). wiki_ko E4B 기준 기존 사전을 (word, df) 집합까지 완전 재현함.
색인 엔진(gazcomp)은 사전을 Aho-Corasick 집합으로만 쓰므로 동률 df의 순서 차이는 결과에 무관.

규칙
- 후보: ner   = 문서별 named_entities
        nerre = named_entities + 트리플 주어·목적어(t[0], t[2])
- df  : 후보 문자열의 원시 등장 횟수(문서 내 중복 포함, 정규화 없음)
- 필터: df >= 2, 공백 없음, 2자 이상, 숫자 미포함, 한글 1자 이상 포함
- 정렬: df 내림차순
- merge: PMI 사전 + (NER-only 증류 사전 − PMI 단어)   ※ wiki_ko merge4b 기준

실행:
  python -m src.track_02.data.distill_glossary --openie cache/track_02/openie_4b_wiki_ko/openie \
      --mode nerre --out data/wiki_ko/glossary_llm4b_wiki.json
  python -m src.track_02.data.distill_glossary --verify-wiki-e4b   # 기존 사전 재현 검증
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def valid(w):
    return (bool(re.fullmatch(r"\S{2,}", w)) and not re.search(r"\d", w)
            and bool(re.search(r"[가-힣]", w)))


def load_openie(openie_dir):
    files = sorted(Path(openie_dir).glob("*.json"), key=lambda f: int(f.stem))
    return [json.load(open(f, encoding="utf-8")) for f in files]


def distill(docs, mode, min_df=2):
    cnt = Counter()
    for d in docs:
        cnt.update(d["named_entities"])
        if mode == "nerre":
            cnt.update(x for t in d["triples"] for x in (t[0], t[2]))
    items = [{"word": w, "df": c} for w, c in cnt.items() if c >= min_df and valid(w)]
    return sorted(items, key=lambda x: -x["df"])


def merge(pmi, ner_glossary):
    pw = {x["word"] for x in pmi}
    return pmi + [x for x in ner_glossary if x["word"] not in pw]


def _same(a, b):
    return {(x["word"], x["df"]) for x in a} == {(x["word"], x["df"]) for x in b}


def verify_wiki_e4b():
    docs = load_openie(ROOT / "cache/track_02/openie_4b_wiki_ko/openie")
    D = ROOT / "data/wiki_ko"
    ld = lambda n: json.load(open(D / n, encoding="utf-8"))  # noqa: E731
    ner, nerre = distill(docs, "ner"), distill(docs, "nerre")
    ok = {
        "glossary_llm4bner_wiki.json": _same(ner, ld("glossary_llm4bner_wiki.json")),
        "glossary_llm4b_wiki.json": _same(nerre, ld("glossary_llm4b_wiki.json")),
        "glossary_merge4b.json": merge(ld("glossary.json"), ld("glossary_llm4bner_wiki.json"))
                                 == ld("glossary_merge4b.json"),
    }
    for k, v in ok.items():
        print(f"  {k:32s} 재현 {'일치' if v else '불일치'}")
    return all(ok.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openie", help="OpenIE 캐시 디렉토리({i}.json)")
    ap.add_argument("--mode", choices=["ner", "nerre"], default="nerre")
    ap.add_argument("--out")
    ap.add_argument("--verify-wiki-e4b", action="store_true")
    args = ap.parse_args()
    if args.verify_wiki_e4b:
        raise SystemExit(0 if verify_wiki_e4b() else 1)
    g = distill(load_openie(args.openie), args.mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(g, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"사전 {len(g)}개 → {args.out}")


if __name__ == "__main__":
    main()
