"""Track 2 - 재실험 QA: 저장된 검색 결과로 답변 생성, 검색이 기존과 같으면 기존 답변 재사용 (2026-09).

rerun_retrieve.py가 저장한 top-5로 QA를 수행해 engine.py와 같은 형식의 결과를 만든다.
- 기존 런(태그의 'v2' 제거 → 예: llm4bv2 → llm4b)의 같은 문항과 top-5가 순서까지 같으면
  기존 prediction·contain·gpt4o를 그대로 재사용(같은 생성기·같은 프롬프트이므로 동일 조건).
- 다르면 31B(.env GEMMA_URL)로 engine.py와 동일 프롬프트(QA_SYS, temperature 0, max_tokens 256) 생성.
- gpt-4o 채점은 rejudge_gpt4o.py(--glob)로 수행: 'gpt4o'가 없는 새 생성분만 채점된다.

산출: results/track_02/{system_id}_{dataset}/predictions.json, eval.json

실행: python -m src.track_02.common.rerun_qa --dataset wiki_ko --tag-new llm4bv2 --workers 32
"""
import argparse
import glob
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

RETR = ROOT / "cache" / "track_02" / "rerun_v2" / "retrieved"
RESULTS = ROOT / "results" / "track_02"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag-new", required=True, help="예: llm4bv2")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    tag_old = args.tag_new.removesuffix("v2")

    texts = {str(p["id"]): p["text"] for p in load_dataset(args.dataset)[0]}
    llm = LLMClient()
    files = sorted(glob.glob(str(RETR / f"*-{args.tag_new}*_{args.dataset}.json")))
    print(f"대상 {len(files)}개, 생성기 {llm.model}", flush=True)
    for f in files:
        R = json.load(open(f, encoding="utf-8"))
        sid_new = R["system"]
        sid_old = sid_new.replace(f"-{args.tag_new}", f"-{tag_old}", 1)
        old = {x["id"]: x for x in json.load(open(
            RESULTS / f"{sid_old}_{args.dataset}" / "predictions.json", encoding="utf-8"))}
        preds, todo = [], []
        for row in R["rows"]:
            o = old.get(row["id"])
            p = dict(row)
            if o and [str(r) for r in o["retrieved"]] == row["retrieved"]:
                p.update(prediction=o["prediction"], contain=o["contain"], reused=True)
                if "gpt4o" in o:
                    p["gpt4o"] = o["gpt4o"]
                if o.get("gen_model"):
                    p["gen_model"] = o["gen_model"]
            else:
                p["reused"] = False
                todo.append(len(preds))
            preds.append(p)

        def gen(i):
            x = preds[i]
            ctx = "\n".join(texts[r] for r in x["retrieved"])
            out = llm.chat([{"role": "system", "content": QA_SYS},
                            {"role": "user", "content": f"{ctx}\nQuestion: {x['question']}\n Thought: "}],
                           temperature=0.0, max_tokens=256)
            return i, out.split("Answer:")[-1].strip() if "Answer:" in out else out.strip()

        err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(gen, i) for i in todo]):
                try:
                    i, pred = fut.result()
                except Exception as e:  # noqa: BLE001
                    err += 1
                    print(f"  ERROR: {e}", flush=True)
                    continue
                preds[i].update(prediction=pred, contain=contain_acc(pred, preds[i]["answer"]),
                                gen_model=llm.model)
        missing = [i for i in todo if "prediction" not in preds[i]]
        if missing:
            print(f"[{sid_new}] 생성 실패 {len(missing)}건 — 저장하지 않음(재실행 필요)", flush=True)
            continue
        outdir = RESULTS / f"{sid_new}_{args.dataset}"
        outdir.mkdir(parents=True, exist_ok=True)
        json.dump(preds, open(outdir / "predictions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n = len(preds)
        summ = {"dataset": args.dataset, "system": sid_new, "n": n,
                "contain_acc": round(sum(p["contain"] for p in preds) / n, 4),
                "reused": n - len(todo), "regenerated": len(todo), "gen_model": llm.model,
                "retrieval_device": R.get("device"), "glossary": R.get("glossary"),
                "baseline_system": sid_old}
        json.dump(summ, open(outdir / "eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{sid_new}] n={n} 재사용 {n-len(todo)} / 생성 {len(todo)} contain={summ['contain_acc']}", flush=True)


if __name__ == "__main__":
    main()
