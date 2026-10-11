"""Track 2 - E4B 관계추출(RE) 재실행 — 컨텍스트 한도 교란 제거용 (2026-09).

배경: 6월 E4B OpenIE(openie_4b.py)는 서버 max_model_len=4096 추정 환경에서 돌아,
wiki_ko 문서의 RE 요청이 거절→출력 한도 축소(2048→1024/512)되어 출력이 잘렸다.
admin 문서는 한도에 걸린 요청이 0건이었다.

재실행 원칙:
- 요청 구성은 openie_4b.py와 동일(system→user 병합, one-shot 우선·실패 시 zero-shot,
  컨텍스트 초과 시 max_tokens 절반 재요청). 달라지는 것은 서버 한도뿐.
- NER은 6월 캐시(openie_4b_{dataset})의 named_entities를 그대로 재사용 → RE만 호출.
- 6월 캐시는 건드리지 않고 openie_4b_v2_{dataset}/ 에 저장.
- 이번에는 원 응답·finish_reason·usage·재시도 이력(raw/)과 서버 메타(run_meta.json)를 기록.

대조군: --sample N 으로 admin 문서 표본을 재추출해 6월 캐시와 비교
(admin은 원래 한도 비구속이므로 차이 = 서버·vLLM 버전 차이).

실행:
  python -m src.track_02.graphrag.openie_4b_rerun --dataset wiki_ko --workers 16
  python -m src.track_02.graphrag.openie_4b_rerun --dataset admin_full --sample 200 --workers 16
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import os

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset            # noqa: E402
from src.track_01.graph_rag.hipporag2_korean import (            # noqa: E402
    ONE_SHOT_PARA, ONE_SHOT_NER_OUT, RE_SYSTEM, RE_FRAME, ONE_SHOT_RE_OUT, parse_json)

BASE = os.environ["OPENIE_4B_BASE"]  # 예: http://<vllm-host>:8000
CACHE_ROOT = ROOT / "cache" / "track_02"
RE_MAX_TOKENS = 6000          # 31B(hipporag2_korean.py RE max_tokens=6000)와 동일
LAUNCH_FLAGS = "--max-model-len 32768 --gpu-memory-utilization 0.92 --port 8000"

_client = httpx.Client(trust_env=False, timeout=120.0)


def server_meta():
    ver = _client.get(f"{BASE}/version").json().get("version")
    m = _client.get(f"{BASE}/v1/models").json()["data"][0]
    return {"url": BASE, "vllm_version": ver, "model_id": m["id"], "model_root": m.get("root"),
            "max_model_len": m.get("max_model_len"), "launch_flags_reported": LAUNCH_FLAGS}


def chat(model, max_model_len, messages, max_tokens):
    """openie_4b.chat과 동일한 로직 + 시도 이력·usage·finish_reason 반환."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":          # Gemma system 병합(6월과 동일)
        sys_txt = msgs.pop(0)["content"]
        for m in msgs:
            if m["role"] == "user":
                m["content"] = sys_txt + "\n\n" + m["content"]
                break
    attempts = []
    mt = min(max_tokens, max_model_len - 256)
    while mt >= 128:
        r = _client.post(f"{BASE}/v1/chat/completions",
                         json={"model": model, "messages": msgs,
                               "max_tokens": mt, "temperature": 0.0})
        attempts.append({"max_tokens": mt, "status": r.status_code})
        if r.status_code == 400 and any(k in r.text for k in
                                        ("max_tokens", "context", "maximum", "length")):
            mt = mt // 2
            continue
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        return ch["message"]["content"], {"attempts": attempts,
                                          "finish_reason": ch.get("finish_reason"),
                                          "usage": j.get("usage")}
    raise RuntimeError(f"컨텍스트 초과로 요청 불가: {attempts}")


def run(dataset, workers, sample):
    meta = server_meta()
    passages, _ = load_dataset(dataset)
    texts = [p["text"] for p in passages]
    src = CACHE_ROOT / f"openie_4b_{dataset}" / "openie"
    tag = f"openie_4b_v2_{dataset}" + (f"_s{sample}" if sample else "")
    out = CACHE_ROOT / tag
    (out / "openie").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    idx = [i for i in range(len(texts)) if (src / f"{i}.json").exists()]
    if sample:
        idx = sorted(random.Random(42).sample(idx, sample))

    def work(i):
        cf = out / "openie" / f"{i}.json"
        if cf.exists():
            return i
        ents = json.load(open(src / f"{i}.json", encoding="utf-8"))["named_entities"]
        ne_json = json.dumps({"named_entities": ents}, ensure_ascii=False)
        p = texts[i]
        re_oneshot = [
            {"role": "system", "content": RE_SYSTEM},
            {"role": "user", "content": RE_FRAME.format(
                passage=ONE_SHOT_PARA, named_entity_json=ONE_SHOT_NER_OUT)},
            {"role": "assistant", "content": ONE_SHOT_RE_OUT},
            {"role": "user", "content": RE_FRAME.format(passage=p, named_entity_json=ne_json)}]
        re_zeroshot = [
            {"role": "system", "content": RE_SYSTEM},
            {"role": "user", "content": RE_FRAME.format(passage=p, named_entity_json=ne_json)}]
        triples_raw, rec = [], {"path": None, "errors": []}
        for name, msgs in (("oneshot", re_oneshot), ("zeroshot", re_zeroshot)):
            try:
                content, info = chat(meta["model_id"], meta["max_model_len"], msgs, RE_MAX_TOKENS)
                triples_raw = parse_json(content, "triples")
                rec.update(path=name, content=content, **info)
                break
            except Exception as e:  # noqa: BLE001 — 6월과 동일하게 zero-shot으로 넘어감
                rec["errors"].append(f"{name}: {type(e).__name__}: {e}")
                continue
        triples = [t for t in triples_raw
                   if isinstance(t, list) and len(t) == 3
                   and all(isinstance(x, str) and x.strip() for x in t)]
        json.dump(rec, open(out / "raw" / f"{i}.json", "w", encoding="utf-8"), ensure_ascii=False)
        json.dump({"named_entities": ents, "triples": triples},
                  open(cf, "w", encoding="utf-8"), ensure_ascii=False)
        return i

    todo = [i for i in idx if not (out / "openie" / f"{i}.json").exists()]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"[rerun] {tag} total={len(idx)} todo={len(todo)} server={meta}", flush=True)
    done = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                if err <= 5:
                    print(f"[rerun] ERROR {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"[rerun] {done}/{len(todo)} (err={err})", flush=True)
    meta.update(dataset=dataset, sample=sample, docs=len(idx), workers=workers,
                re_max_tokens=RE_MAX_TOKENS, ner_source=str(src.relative_to(ROOT)),
                started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_sec=round(time.time() - t0, 1), errors=err)
    json.dump(meta, open(out / "run_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[rerun] {tag} 완료 (err={err}, {meta['elapsed_sec']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0, help="대조군: 문서 N개 무작위 표본(seed 42)")
    args = ap.parse_args()
    run(args.dataset, args.workers, args.sample)


if __name__ == "__main__":
    main()
