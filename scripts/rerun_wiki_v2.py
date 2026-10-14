"""wiki_ko 재실행 (2026-10-14): 원 LinearRAG와 같은 입력 처리("번호:본문" 유지)로 LinearRAG 계열 38개 재실행.

배경: 기존 wiki_ko 로더가 문단 앞의 "번호:"를 떼어, 원 구현이 만드는 인접 문단 간선이 생기지 않았다
(엔진의 간선 코드는 있었으나 작동하지 않음). wiki_ko_v2 데이터셋은 번호를 유지한다.
응답 생성·판정 조건은 기존 실험과 같다.
  - 지문: 번호 없는 본문(기존과 같음). 검색만 바뀐다.
  - 생성: gemma-4-31B-it, QA_SYS, temperature 0, 비교 10개 시스템(◆)은 max_tokens 2000, 나머지 256.
  - 판정: gpt-4o, evaluation.gpt_acc(YES/NO).
  - 재사용: 같은 (질문, 상위 5개, 출력 한도)의 기존 답변과 같은 (질문, 정답, 답변)의 기존 판정은 그대로 쓴다
    (temperature 0이므로 같은 입력이면 같은 출력).

단계(각 단계는 캐시에 저장하고 이어서 실행 가능):
  validate : 원 저장소 코드 결과(Track 1)와 같은 설정으로 대조
  retrieve : 38개 시스템 검색 → cache/track_02/rerun_wiki_v2/retrieved/{system}.json
  qa       : 바뀐 문항만 31B로 생성 → cache/track_02/rerun_wiki_v2/qa/{key}.json
  judge    : 새 (질문, 정답, 답변)만 gpt-4o 판정 → cache/track_02/rerun_wiki_v2/judge/{key}.json
  assemble : results/track_02/{system}_wiki_ko_v2/{predictions,eval}.json

공개 저장소에는 원 저장소 코드 결과(Track 1)와 기존 wiki_ko LinearRAG 결과가 없으므로,
validate 는 건너뛰고 qa·judge 는 모든 답을 새로 생성·판정한다.

실행: python scripts/rerun_wiki_v2.py <단계> [--workers 8] [--urls URL1 URL2]
"""
import argparse
import csv
import gc
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "cache" / "track_02" / "rerun_wiki_v2"
RES = ROOT / "results" / "track_02"
DS_NEW, DS_OLD = "wiki_ko_v2", "wiki_ko"
SYSTEMS_CSV = next(p for p in (ROOT / "analysis" / "systems.csv", ROOT / "nongraphRAG-ko" / "analysis" / "systems.csv")
                   if p.exists())


def h(obj):
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False).encode("utf-8")).hexdigest()


def systems():
    rows = [r for r in csv.DictReader(open(SYSTEMS_CSV, encoding="utf-8")) if r["runner"] == "linearrag"]
    for r in rows:
        tag = r["tag"]
        if r["extract"] == "kw":
            tag = f"k{r['topk_kw']}"
        elif tag == "llm4b":
            tag = "llm4bv2"          # wiki_ko 4B 증류는 재추출판(기존 실험과 같음)
        r["cache_tag"] = tag
        r["system"] = r["wiki_folder"]
    return rows


def engine():
    """임베딩 모델을 한 번만 올리도록 Embedder를 싱글턴으로 바꾼 엔진 모듈."""
    import src.track_02.linearrag.engine as E
    real = E.Embedder
    inst = {}

    def single(*a, **k):
        if "e" not in inst:
            inst["e"] = real(*a, **k)
        return inst["e"]
    E.Embedder = single
    return E


def build(E, extract, tag, gloss, topk_kw):
    from src.track_02.common.datasets import load_dataset
    passages, questions = load_dataset(DS_NEW)
    rag = E.LinearRAG(DS_NEW, extract, glossary_path=gloss or None, tag=tag,
                      topk_kw=int(topk_kw) if topk_kw else 10)
    rag.index(passages)
    pset = set(rag.p_vidx)
    adj = sum(1 for e in rag.g.es if e.source in pset and e.target in pset)
    assert adj == len(passages) - 1, f"인접 문단 간선 {adj}개 (기대 {len(passages) - 1})"
    return rag, questions


def stage_validate():
    vpath = ROOT / "results/track_01/linear_rag/predictions.json"
    if not vpath.exists():
        sys.exit(f"[validate] 원 저장소 코드 결과가 없어 건너뜁니다: {vpath} (retrieve 단계부터 실행)")
    E = engine()
    E.ITER_THRESH, E.PASSAGE_RATIO = 0.4, 0.05       # Track 1 원 코드 실행값
    rag, questions = build(E, "nnp", "", None, None)
    vend = {str(r["id"]): [int(re.match(r"^(\d+):", p).group(1)) for p in r["sorted_passage"]]
            for r in json.load(open(vpath, encoding="utf-8"))}
    same = sset = 0
    ov = 0.0
    for q in questions:
        mine = [int(x) for x in rag.retrieve_one(q["question"])]
        v = vend[str(q["id"])]
        same += mine == v
        sset += set(mine) == set(v)
        ov += len(set(mine) & set(v)) / 5
    n = len(questions)
    print(f"[validate] 원 코드 대비 순서까지 동일 {same}/{n}, 집합 동일 {sset}/{n}, 평균 겹침 {ov / n:.3f}", flush=True)


def stage_retrieve():
    import torch
    E = engine()
    rows = systems()
    (OUT / "retrieved").mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in rows:
        groups.setdefault((r["extract"], r["cache_tag"], r["glossary_wiki_ko"], r["topk_kw"]), []).append(r)
    for gi, ((extract, tag, gloss, kw), members) in enumerate(groups.items(), 1):
        todo = [r for r in members if not (OUT / "retrieved" / f"{r['system']}.json").exists()]
        if not todo:
            continue
        t0 = time.time()
        rag, questions = build(E, extract, tag, gloss, kw)
        print(f"[retrieve] 캐시 {gi}/{len(groups)} {extract}/{tag or '-'} 색인 {time.time() - t0:.0f}s", flush=True)
        for r in todo:
            rag.pnw = float(r["pnw"]) if r["pnw"] else E.PASSAGE_NODE_WEIGHT
            rag.prf_beta = float(r["prf_beta"]) if r["prf_beta"] else 0.0
            rag.prf_sent = r["prf_sent"] == "1"
            t1 = time.time()
            out = {q["id"]: [str(x) for x in rag.retrieve_one(q["question"])] for q in questions}
            json.dump(out, open(OUT / "retrieved" / f"{r['system']}.json", "w"), ensure_ascii=False)
            print(f"  {r['system']}: {time.time() - t1:.0f}s", flush=True)
        del rag
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def old_maps():
    """기존 wiki_ko 결과에서 재사용할 답변·판정. 답변은 (질문, 상위5, 출력한도), 판정은 (질문, 정답, 답변)."""
    gen, judge = {}, {}
    rows = systems()
    mt = {r["system"]: (2000 if r["main10"] == "1" else 256) for r in rows}
    for d in RES.glob(f"*_{DS_OLD}"):
        sysname = d.name[: -len(DS_OLD) - 1]
        for x in json.load(open(d / "predictions.json", encoding="utf-8")):
            if "gpt4o" in x:
                judge[h([x["question"], x["answer"], x["prediction"]])] = bool(x["gpt4o"])
            if sysname in mt:
                gen[h([x["question"], [str(r) for r in x["retrieved"]], mt[sysname]])] = x["prediction"]
    # unify_qa 캐시(출력 한도 2000)
    uq = ROOT / "cache" / "track_02" / "unify_qa"
    return gen, judge, mt, uq


def stage_qa(workers, main_only=False, urls=None):
    from src.track_01.common.llm_client import LLMClient
    from src.track_02.common.datasets import load_dataset
    from src.track_02.linearrag.engine import QA_SYS
    passages, questions = load_dataset(DS_OLD)              # 지문은 번호 없는 본문(기존과 같음)
    text = {p["id"]: p["text"] for p in passages}
    gen, _, mt, uq = old_maps()
    (OUT / "qa").mkdir(parents=True, exist_ok=True)
    jobs = {}
    reused = 0
    for r in systems():
        if main_only and r["main10"] != "1":
            continue
        retr = json.load(open(OUT / "retrieved" / f"{r['system']}.json"))
        for q in questions:
            k = h([q["question"], retr[q["id"]], mt[r["system"]]])
            if k in gen or (OUT / "qa" / f"{k}.json").exists():
                reused += k in gen
                continue
            if mt[r["system"]] == 2000:
                uk = hashlib.sha1(json.dumps([q["question"], retr[q["id"]]], ensure_ascii=False).encode()).hexdigest()
                if (uq / f"{uk}.json").exists():
                    c = json.load(open(uq / f"{uk}.json", encoding="utf-8"))
                    json.dump({"prediction": c["prediction"], "source": "unify_qa"}, open(OUT / "qa" / f"{k}.json", "w"),
                              ensure_ascii=False)
                    continue
            jobs[k] = (q["question"], retr[q["id"]], mt[r["system"]])
    print(f"[qa] 기존 답변 재사용 {reused}건, 새로 생성할 고유 프롬프트 {len(jobs)}건", flush=True)
    if not jobs:
        return
    # 31B 서버가 여러 대면 요청을 번갈아 보낸다(같은 모델·설정)
    clients = []
    for u in (urls or [None]):
        c = LLMClient(timeout=600.0)
        if u:
            c.url = u.rstrip("/") + ("" if u.rstrip("/").endswith("/chat/completions") else "/chat/completions")
        clients.append(c)
    order = {k: i for i, k in enumerate(jobs)}
    print(f"[qa] 서버 {len(clients)}대: {[c.url for c in clients]}", flush=True)

    def run(k):
        llm = clients[order[k] % len(clients)]
        qq, ret, m = jobs[k]
        ctx = "\n".join(text[x] for x in ret)
        try:
            out, finish = llm.chat_full([{"role": "system", "content": QA_SYS},
                                     {"role": "user", "content": f"{ctx}\nQuestion: {qq}\n Thought: "}],
                                    temperature=0.0, max_tokens=m)
        except Exception as e:  # noqa: BLE001  한 건 실패로 전체가 멈추지 않게: 기록만 하고 다음 실행에서 다시 시도
            print(f"  [실패] {k[:10]} {e}", flush=True)
            return None
        pred = out.split("Answer:")[-1].strip() if "Answer:" in out else out.strip()
        json.dump({"prediction": pred, "finish_reason": finish, "max_tokens": m, "raw": out},
                  open(OUT / "qa" / f"{k}.json", "w", encoding="utf-8"), ensure_ascii=False)
        return k

    done = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(run, list(jobs)):
            done += 1
            fail += res is None
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} ({time.time() - t0:.0f}s, 실패 {fail})", flush=True)
    print(f"[qa] 완료 {done - fail}건, 실패 {fail}건 ({time.time() - t0:.0f}s)", flush=True)
    if fail:
        raise SystemExit(f"[qa] 실패 {fail}건 — 같은 명령을 다시 실행하면 실패 건만 재시도")


def predictions(main_only=False):
    """시스템별 새 결과(답변까지). 판정 전 단계."""
    from src.track_02.common.datasets import load_dataset
    _, questions = load_dataset(DS_OLD)
    gen, judge, mt, _ = old_maps()
    out = {}
    for r in systems():
        if main_only and r["main10"] != "1":
            continue
        retr = json.load(open(OUT / "retrieved" / f"{r['system']}.json"))
        rows = []
        for q in questions:
            k = h([q["question"], retr[q["id"]], mt[r["system"]]])
            if k in gen:
                pred, src = gen[k], "reused"
            else:
                pred, src = json.load(open(OUT / "qa" / f"{k}.json", encoding="utf-8"))["prediction"], "new"
            rows.append({"id": q["id"], "question": q["question"], "answer": q["answer"],
                         "prediction": pred, "retrieved": retr[q["id"]], "supporting": q["supporting"],
                         "qa_source": src})
        out[r["system"]] = (r, rows)
    return out, judge


def stage_judge(workers, main_only=False):
    from src.track_01.common.llm_client import OpenAIClient
    from src.track_01.common.evaluation import gpt_acc
    preds, judge = predictions(main_only)
    (OUT / "judge").mkdir(parents=True, exist_ok=True)
    jobs = {}
    for _, rows in preds.values():
        for x in rows:
            k = h([x["question"], x["answer"], x["prediction"]])
            if k not in judge and not (OUT / "judge" / f"{k}.json").exists():
                jobs[k] = (x["question"], x["answer"], x["prediction"])
    total = sum(len(rows) for _, rows in preds.values())
    print(f"[judge] 전체 {total}건 중 새로 판정할 고유 (질문, 정답, 답변) {len(jobs)}건", flush=True)
    if not jobs:
        return
    client = OpenAIClient(model="gpt-4o")

    def run(k):
        qq, a, p = jobs[k]
        json.dump({"gpt4o": gpt_acc(qq, a, p, client)}, open(OUT / "judge" / f"{k}.json", "w"))
        return k

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(run, list(jobs)):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"[judge] 완료 {done}건", flush=True)


def stage_assemble(main_only=False):
    from src.track_01.common.evaluation import contain_acc
    preds, judge = predictions(main_only)
    for name, (r, rows) in preds.items():
        for x in rows:
            k = h([x["question"], x["answer"], x["prediction"]])
            x["gpt4o"] = judge[k] if k in judge else json.load(open(OUT / "judge" / f"{k}.json"))["gpt4o"]
            x["contain"] = contain_acc(x["prediction"], x["answer"])
        d = RES / f"{name}_{DS_NEW}"
        d.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(d / "predictions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n = len(rows)
        ev = {"dataset": DS_NEW, "system": name, "n": n,
              "gpt4o_acc": round(sum(x["gpt4o"] for x in rows) / n, 4),
              "contain_acc": round(sum(x["contain"] for x in rows) / n, 4),
              "gen_model": "gemma-4-31B-it", "qa_prompt": "linearrag.QA_SYS",
              "qa_max_tokens": 2000 if r["main10"] == "1" else 256,
              "qa_new": sum(x["qa_source"] == "new" for x in rows),
              "note": "원 LinearRAG 입력 처리(번호:본문, 인접 문단 간선) 반영 재실행 2026-10-14"}
        json.dump(ev, open(d / "eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{name}: GPT-Acc {ev['gpt4o_acc']:.3f}  Contain {ev['contain_acc']:.3f}  (새 답변 {ev['qa_new']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["validate", "retrieve", "qa", "judge", "assemble"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--main-only", action="store_true", help="비교 10개 시스템(◆)의 LinearRAG 7개만 먼저")
    ap.add_argument("--urls", nargs="*", default=None, help="31B 서버 주소들(예: http://host:8000/v1 http://host:8001/v1)")
    a = ap.parse_args()
    {"validate": stage_validate, "retrieve": stage_retrieve, "assemble": lambda: stage_assemble(a.main_only),
     "qa": lambda: stage_qa(a.workers, a.main_only, a.urls), "judge": lambda: stage_judge(a.workers, a.main_only)}[a.stage]()


if __name__ == "__main__":
    main()
