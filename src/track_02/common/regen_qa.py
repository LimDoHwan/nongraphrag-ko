"""Track 2 - F0: ‡ 행 QA 재생성 (생성기 일관성 정합화).

양자화 gemma-4-31B-it-qat-w4a16 로 생성됐던 predictions 의 QA 답변을
원본 gemma-4-31B-it (.env GEMMA_MODEL) 로 재생성한다.
검색(retrieved)은 기존 것을 그대로 재사용 — QA 생성만 재실행 (계획서 F0 §방법 2).

- 프롬프트·디코딩은 engine.py 의 QA 단계와 동일 (QA_SYS, temperature=0, max_tokens=256)
- 각 항목에 gen_model 필드 기록; 이미 gen_model == 현재 모델인 항목은 skip (resumable)
- 낡은 judge 필드(gpt4o, gpt) 삭제 → 이후 rejudge_gpt4o.py 로 재채점
- eval.json: contain_acc 갱신, gpt4o_acc/gpt_acc 등 낡은 지표 제거, gen_model 기록

실행: python -m src.track_02.common.regen_qa [--glob 'linearrag-new-comp-*'] [--workers 8]
"""
import argparse
import fnmatch
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.track_01.common.evaluation import contain_acc
from src.track_01.common.llm_client import LLMClient
from src.track_02.common.datasets import load_dataset
from src.track_02.linearrag.engine import QA_SYS

RESULTS = Path("results/track_02")
DATASETS = ("admin_full", "admin_hard", "admin_single", "wiki_ko")

# 마스터표 ‡ 행 14설정 (2026-07 양자화 서버 시기 QA 생성분)
QAT_CONFIGS = [
    "linearrag-new-comp-pnw0.5",
    "linearrag-new-comp-pnw1",
    "linearrag-new-comp-pnw0.5-prf0.6",
    "linearrag-new-comp-pnw1-prf0.6",
    "linearrag-new-comp-pnw0.5-prf0.6sent",
    "linearrag-new-comp-pnw1-prf0.6sent",
    "linearrag-new-gazcomp-pnw0.5-prf0.6",
    "linearrag-new-gazcomp-pnw1-prf0.6",
    "linearrag-new-gazcomp-pnw0.5-prf0.6sent",
    "linearrag-new-gazcomp-pnw1-prf0.6sent",
    "linearrag-new-gazcomp-llm-pnw0.5-prf0.6",
    "linearrag-new-gazcomp-llm-pnw0.5-prf0.6sent",
    "linearrag-new-gazcomp-llm4b-pnw0.5-prf0.6",
    "linearrag-new-gazcomp-llm4b-pnw0.5-prf0.6sent",
]

_ptext_cache: dict = {}


def ptext(dataset: str) -> dict:
    if dataset not in _ptext_cache:
        passages, _ = load_dataset(dataset)
        _ptext_cache[dataset] = {str(p["id"]): p["text"] for p in passages}
    return _ptext_cache[dataset]


def regen_dir(outdir: Path, dataset: str, llm: LLMClient, workers: int,
              limit: int = 0) -> tuple[int, int]:
    pf = outdir / "predictions.json"
    preds = json.load(open(pf, encoding="utf-8"))
    texts = ptext(dataset)
    todo = [i for i, x in enumerate(preds) if x.get("gen_model") != llm.model]
    if limit:
        todo = todo[:limit]
    if not todo:
        return len(preds), 0

    def gen(i):
        x = preds[i]
        ctx = "\n".join(texts[str(r)] for r in x["retrieved"])
        out = llm.chat([{"role": "system", "content": QA_SYS},
                        {"role": "user",
                         "content": f"{ctx}\nQuestion: {x['question']}\n Thought: "}],
                       temperature=0.0, max_tokens=256)
        return i, out.split("Answer:")[-1].strip() if "Answer:" in out else out.strip()

    def save():
        json.dump(preds, open(pf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(gen, i) for i in todo]):
            i, pred = fut.result()
            x = preds[i]
            x["prediction"] = pred
            x["contain"] = contain_acc(pred, x["answer"])
            x["gen_model"] = llm.model
            x.pop("gpt4o", None)   # 낡은 judge 결과 무효화 → rejudge 대상
            x.pop("gpt", None)
            done += 1
            if done % 100 == 0:
                save()
    save()

    n = len(preds)
    ef = outdir / "eval.json"
    e = json.load(open(ef, encoding="utf-8")) if ef.exists() else {}
    e = {"dataset": e.get("dataset", dataset), "system": e.get("system", ""),
         "n": n, "gen_model": llm.model,
         "contain_acc": round(sum(p["contain"] for p in preds) / n, 4),
         "valid_qa": sum(1 for p in preds if p["prediction"].strip() != "")}
    json.dump(e, open(ef, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return n, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*", help="설정 이름 필터 (QAT_CONFIGS 내)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="디렉토리당 재생성 문항 수 제한(파일럿)")
    args = ap.parse_args()

    llm = LLMClient()
    print(f"생성기: {llm.model} @ {llm.url}", flush=True)
    targets = [(c, d) for c in QAT_CONFIGS if fnmatch.fnmatch(c, args.glob)
               for d in args.datasets.split(",")
               if (RESULTS / f"{c}_{d}").is_dir()]
    print(f"대상 {len(targets)}개 디렉토리", flush=True)
    t0 = time.time()
    for k, (c, d) in enumerate(targets, 1):
        n, done = regen_dir(RESULTS / f"{c}_{d}", d, llm, args.workers, args.limit)
        ef = json.load(open(RESULTS / f"{c}_{d}" / "eval.json", encoding="utf-8"))
        print(f"[{k}/{len(targets)}] {c}_{d}: n={n} regen+{done} "
              f"contain={ef.get('contain_acc')} ({time.time()-t0:.0f}s)", flush=True)
    print("=== QA 재생성 완료 ===", flush=True)


if __name__ == "__main__":
    main()
