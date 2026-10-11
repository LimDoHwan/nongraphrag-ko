"""Track 2 - 저장된 predictions를 gpt-4o로 재채점 (judge 교체, 검색·QA 재실행 0).

각 predictions.json 의 prediction을 gpt-4o judge로 평가 → 'gpt4o' 필드 추가.
gemma judge('gpt') / contain('contain')은 보존. eval.json에 gpt4o_acc 추가.
resumable: 이미 'gpt4o' 있는 항목은 건너뜀.

실행: python -m src.track_02.common.rejudge_gpt4o --workers 24
"""
import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.track_01.common.llm_client import OpenAIClient
from src.track_01.common.evaluation import gpt_acc

RESULTS = "results/track_02"


def rejudge_file(pf, client, workers):
    preds = json.load(open(pf, encoding="utf-8"))
    todo = [i for i, x in enumerate(preds) if "gpt4o" not in x]
    if todo:
        def judge(i):
            x = preds[i]
            try:
                return i, gpt_acc(x["question"], x["answer"], x["prediction"], client)
            except Exception as e:  # noqa: BLE001
                return i, None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(judge, i) for i in todo]):
                i, v = fut.result()
                if v is not None:
                    preds[i]["gpt4o"] = v
        json.dump(preds, open(pf, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    # eval.json 갱신
    n = len(preds)
    ef = os.path.join(os.path.dirname(pf), "eval.json")
    e = json.load(open(ef, encoding="utf-8")) if os.path.exists(ef) else {}
    judged = [x for x in preds if "gpt4o" in x]
    if judged:
        e["gpt4o_acc"] = round(sum(x["gpt4o"] for x in judged) / len(judged), 4)
        e["gpt4o_n"] = len(judged)
    json.dump(e, open(ef, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return n, len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--glob", default="*", help="결과 디렉토리 필터")
    args = ap.parse_args()
    client = OpenAIClient(model="gpt-4o")
    files = sorted(glob.glob(f"{RESULTS}/{args.glob}/predictions.json"))
    print(f"재채점 대상 {len(files)}개 디렉토리", flush=True)
    for k, pf in enumerate(files, 1):
        name = os.path.basename(os.path.dirname(pf))
        n, done = rejudge_file(pf, client, args.workers)
        ef = json.load(open(os.path.join(os.path.dirname(pf), "eval.json")))
        print(f"[{k}/{len(files)}] {name}: n={n} judged+{done} "
              f"gpt4o_acc={ef.get('gpt4o_acc')}", flush=True)
    print("=== gpt-4o 재채점 완료 ===", flush=True)


if __name__ == "__main__":
    main()
