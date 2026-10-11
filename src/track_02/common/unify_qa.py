"""Track 2 - 답변 생성 조건 통일: 공통 프롬프트(LinearRAG QA_SYS) + 출력 한도 2,000 (2026-09).

시스템별로 답변 생성 조건이 달랐다(Vanilla: 별도 프롬프트·128 / LinearRAG: QA_SYS·256 /
HippoRAG2: 다른 QA_SYS·2,000). 검색 결과(retrieved)는 그대로 두고 답변만 공통 조건으로 맞춘다.

- LinearRAG 계열: 이미 같은 프롬프트이므로 **출력 한도(256)에 걸려 잘린 답변만** 재생성
  (온도 0 생성이라 256 안에 끝난 답변은 한도를 늘려도 동일).
- Vanilla·HippoRAG2: 프롬프트가 달랐으므로 **전량** 재생성.
- 재생성한 항목은 gpt4o/gpt 채점을 지워 rejudge_gpt4o.py가 다시 채점하게 한다.
- 같은 (질문, top-5) 프롬프트는 한 번만 생성(캐시: cache/track_02/unify_qa/, 재시작 가능).

실행: python -m src.track_02.common.unify_qa --workers 8
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.track_01.common.evaluation import contain_acc               # noqa: E402
from src.track_01.common.llm_client import LLMClient                # noqa: E402
from src.track_02.common.datasets import load_dataset               # noqa: E402
from src.track_02.linearrag.engine import QA_SYS                    # noqa: E402

RESULTS = ROOT / "results" / "track_02"
CACHE = ROOT / "cache" / "track_02" / "unify_qa"
MAX_TOKENS = 2000
DATASETS = ["wiki_ko", "admin_single", "admin_full", "admin_hard"]
LINEAR = ["linearrag-org-nnp", "linearrag-new-noun", "linearrag-new-comp", "linearrag-new-gazcomp",
          "linearrag-new-comp-pnw0.5-prf0.6sent", "linearrag-new-gazcomp-llm4b", "linearrag-new-gazcomp-llm"]
FULL = ["vanillarag", "graphrag-org"]


def run_dir(system, ds):
    if ds == "wiki_ko" and system == "linearrag-new-gazcomp-llm4b":
        system = "linearrag-new-gazcomp-llm4bv2"
    if ds == "wiki_ko" and system == "graphrag-4b":
        system = "graphrag-4bv2"
    return RESULTS / f"{system}_{ds}"


def truncated(x):
    p = (x.get("prediction") or "").strip()
    return (not p) or p.startswith("Thought")


def key(q, retrieved):
    return hashlib.sha1(json.dumps([q, list(map(str, retrieved))], ensure_ascii=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--linear", nargs="*", default=None, help="잘린 답변만 재생성할 시스템(기본: LINEAR 전체)")
    ap.add_argument("--full", nargs="*", default=None, help="전량 재생성할 시스템(기본: FULL 전체)")
    args = ap.parse_args()
    linear = LINEAR if args.linear is None else args.linear
    full = FULL if args.full is None else args.full
    CACHE.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()
    texts = {ds: {str(p["id"]): p["text"] for p in load_dataset(ds)[0]} for ds in DATASETS}

    plan = []  # (dir, ds, preds, todo_idx)
    jobs = {}  # key -> (ds, question, retrieved)
    for system in linear + full:
        for ds in DATASETS:
            d = run_dir(system, ds)
            preds = json.load(open(d / "predictions.json", encoding="utf-8"))
            todo = [i for i, x in enumerate(preds) if system in full or truncated(x)]
            for i in todo:
                x = preds[i]
                jobs.setdefault(key(x["question"], x["retrieved"]), (ds, x["question"], [str(r) for r in x["retrieved"]]))
            plan.append((d, ds, preds, todo))
    pending = {k: v for k, v in jobs.items() if not (CACHE / f"{k}.json").exists()}
    print(f"[unify] 대상 런 {len(plan)}개, 재생성 항목 {sum(len(t) for *_, t in plan)}건, "
          f"고유 프롬프트 {len(jobs)}건(남은 {len(pending)}건), 생성기 {llm.model}", flush=True)

    def gen(k):
        ds, q, ret = pending[k]
        ctx = "\n".join(texts[ds][r] for r in ret)
        out, finish = llm.chat_full([{"role": "system", "content": QA_SYS},
                                     {"role": "user", "content": f"{ctx}\nQuestion: {q}\n Thought: "}],
                                    temperature=0.0, max_tokens=MAX_TOKENS)
        pred = out.split("Answer:")[-1].strip() if "Answer:" in out else out.strip()
        json.dump({"prediction": pred, "finish_reason": finish, "raw": out},
                  open(CACHE / f"{k}.json", "w", encoding="utf-8"), ensure_ascii=False)
        return k

    done = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(gen, k) for k in pending]):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"[unify] ERROR {e}", flush=True)
            done += 1
            if done % 100 == 0 or done == len(pending):
                print(f"[unify] 생성 {done}/{len(pending)} (err={err})", flush=True)
    if err:
        print(f"[unify] 생성 실패 {err}건 — 결과 파일 갱신 안 함(재실행하면 이어서 진행)", flush=True)
        return

    for d, ds, preds, todo in plan:
        for i in todo:
            x = preds[i]
            c = json.load(open(CACHE / f"{key(x['question'], x['retrieved'])}.json", encoding="utf-8"))
            x.update(prediction=c["prediction"], contain=contain_acc(c["prediction"], x["answer"]),
                     gen_model=llm.model, qa_max_tokens=MAX_TOKENS, qa_prompt="linearrag.QA_SYS",
                     qa_finish=c["finish_reason"])
            x.pop("gpt4o", None)
            x.pop("gpt", None)
        json.dump(preds, open(d / "predictions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        ef = d / "eval.json"
        e = json.load(open(ef, encoding="utf-8")) if ef.exists() else {}
        n = len(preds)
        e.update(contain_acc=round(sum(bool(x["contain"]) for x in preds) / n, 4),
                 qa_prompt="linearrag.QA_SYS", qa_max_tokens=MAX_TOKENS, qa_regenerated=len(todo))
        for k_ in ("gpt4o_acc", "gpt4o_n", "gpt_acc"):
            e.pop(k_, None)
        json.dump(e, open(ef, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[unify] {d.name}: 재생성 {len(todo)} / {n}, contain={e['contain_acc']}", flush=True)
    print("[unify] 완료", flush=True)


if __name__ == "__main__":
    main()
