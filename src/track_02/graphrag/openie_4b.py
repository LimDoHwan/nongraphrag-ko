"""Track 2 - 소형 모델(gemma-4-4b) 엔티티 추출 (OpenIE NER+RE) — 메모리-safe 1단계.

로컬 vllm(gemma4-4b-it)으로 OpenIE만 수행 → openie 캐시 저장.
임베딩(bge-m3)·그래프(igraph) 미사용 → vllm과 GPU/메모리 충돌 없음.
이후 2단계(vllm 내린 뒤)에서 이 캐시로 사전 증류 + gazcomp 실행.

프롬프트·파서는 track_01 HippoRAG2 재구현과 동일(31B와 공정 비교).
캐시: cache/track_02/openie_4b_{dataset}/openie/{i}.json  (resume 가능)

실행: python -m src.track_02.graphrag.openie_4b --dataset admin_full --workers 16
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.track_02.common.datasets import load_dataset            # noqa: E402
from src.track_01.graph_rag.hipporag2_korean import (            # noqa: E402
    NER_SYSTEM, ONE_SHOT_PARA, ONE_SHOT_NER_OUT,
    RE_SYSTEM, RE_FRAME, ONE_SHOT_RE_OUT, parse_json)

URL = os.environ["OPENIE_4B_URL"]  # 예: http://<vllm-host>:8000/v1/chat/completions
MODEL = "gemma4-4b-it"
CACHE_ROOT = ROOT / "cache" / "track_02"

# 프록시 우회(trust_env=False) — 로컬 vllm은 사내 프록시 거치면 안 됨
_client = httpx.Client(trust_env=False, timeout=120.0)


MAX_MODEL_LEN = 4096  # 4b vllm 컨텍스트(프롬프트+출력) 상한


def chat(messages, max_tokens):
    # Gemma는 system 역할 미지원 → 첫 user 메시지에 병합
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        sys_txt = msgs.pop(0)["content"]
        for m in msgs:
            if m["role"] == "user":
                m["content"] = sys_txt + "\n\n" + m["content"]
                break
    mt = min(max_tokens, MAX_MODEL_LEN - 256)
    while mt >= 128:
        r = _client.post(URL, json={"model": MODEL, "messages": msgs,
                                    "max_tokens": mt, "temperature": 0.0})
        if r.status_code == 400 and any(k in r.text for k in
                                        ("max_tokens", "context", "maximum", "length")):
            mt = mt // 2   # 컨텍스트 초과 → 출력 토큰 축소 재시도
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("프롬프트가 컨텍스트(4096)에 비해 너무 김")


def run_openie(dataset, workers):
    passages, _ = load_dataset(dataset)
    texts = [p["text"] for p in passages]
    cache = CACHE_ROOT / f"openie_4b_{dataset}" / "openie"
    cache.mkdir(parents=True, exist_ok=True)

    def work(i):
        cf = cache / f"{i}.json"
        if cf.exists():
            return i
        p = texts[i]
        ents = parse_json(chat([
            {"role": "system", "content": NER_SYSTEM},
            {"role": "user", "content": ONE_SHOT_PARA},
            {"role": "assistant", "content": ONE_SHOT_NER_OUT},
            {"role": "user", "content": p}], 1024), "named_entities")
        ne_json = json.dumps({"named_entities": ents}, ensure_ascii=False)
        # one-shot RE 우선, 컨텍스트 초과 시 zero-shot RE로 fallback(긴 wiki 문서 대응)
        re_oneshot = [
            {"role": "system", "content": RE_SYSTEM},
            {"role": "user", "content": RE_FRAME.format(
                passage=ONE_SHOT_PARA, named_entity_json=ONE_SHOT_NER_OUT)},
            {"role": "assistant", "content": ONE_SHOT_RE_OUT},
            {"role": "user", "content": RE_FRAME.format(
                passage=p, named_entity_json=ne_json)}]
        re_zeroshot = [
            {"role": "system", "content": RE_SYSTEM},
            {"role": "user", "content": RE_FRAME.format(
                passage=p, named_entity_json=ne_json)}]
        triples_raw = []
        for msgs in (re_oneshot, re_zeroshot):
            try:
                triples_raw = parse_json(chat(msgs, 2048), "triples")
                break
            except Exception:  # noqa: BLE001
                continue
        triples = [t for t in triples_raw
                   if isinstance(t, list) and len(t) == 3
                   and all(isinstance(x, str) and x.strip() for x in t)]
        json.dump({"named_entities": ents, "triples": triples},
                  open(cf, "w", encoding="utf-8"), ensure_ascii=False)
        return i

    todo = [i for i in range(len(texts)) if not (cache / f"{i}.json").exists()]
    print(f"[openie-4b] {dataset} total={len(texts)} todo={len(todo)}", flush=True)
    done = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                if err <= 5:
                    print(f"[openie-4b] ERROR {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"[openie-4b] {done}/{len(todo)} (err={err})", flush=True)
    print(f"[openie-4b] {dataset} 완료 (err={err})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="테스트용 N개만")
    args = ap.parse_args()
    if args.limit:
        # 테스트: 앞 N개만 (load_dataset 후 자르기 위해 monkey 처리)
        import src.track_02.common.datasets as D
        orig = D.load_dataset
        def limited(name):
            p, q = orig(name); return p[:args.limit], q
        D.load_dataset = limited
        globals()["load_dataset"] = limited
    run_openie(args.dataset, args.workers)


if __name__ == "__main__":
    main()
