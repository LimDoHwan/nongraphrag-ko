"""2WikiMultiHopQA (LinearRAG 공식 Zly0523) EN -> KO 재번역 파이프라인.

분실한 한국어 데이터 복원용. 내려받은 영어 원본(질문 1000 / 청크 658)을
Gemma 31B(외부 vLLM)로 번역한다.

특징:
- 체크포인트 기반 resume (항목별 파일 캐시) -> 중단되어도 이어서 진행
- ThreadPoolExecutor 동시 호출
- 청크 잘림(finish_reason != stop) 감지 시 문장 단위 분할 후 재번역
- 출력 구조는 원본과 동일 (chunks: "N:..." 문자열 리스트, questions: dict 리스트)

사용:
  python scripts/translate_dataset.py --what all
  python scripts/translate_dataset.py --what chunks --workers 8
  python scripts/translate_dataset.py --what questions
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# src 패키지 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.track_01.common.llm_client import LLMClient, make_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "2wiki_ko"
SRC_DIR = DATA_DIR / "_source_en"
CACHE_DIR = DATA_DIR / "_cache"

HF_BASE = ("https://huggingface.co/datasets/Zly0523/linear-rag/"
           "resolve/f7971bb84fe205d0c080df5f48b1393d9f580c68/2wikimultihop")  # 2026-06-08 취득 시점 커밋으로 고정

CHUNK_PROMPT = """다음 영어 텍스트를 한국어로 번역하세요.
규칙:
- 모든 사실과 정보를 빠짐없이 번역합니다. 절대 요약하거나 생략하지 마세요.
- 고유명사(인명, 지명, 작품명, 단체명 등)는 표준 한국어 표기로 음역합니다.
- 연도/날짜는 한국어 방식으로 자연스럽게 옮깁니다.
- 번역문만 출력하고 그 외 설명·머리말은 절대 붙이지 마세요.
{glossary}
[영어 텍스트]
{text}"""

QA_PROMPT = """다음 multi-hop QA 데이터의 질문과 정답을 한국어로 번역하세요.
규칙:
- 고유명사는 표준 한국어 표기로 음역합니다.
- 정답이 yes/no 이면 각각 "예"/"아니오" 로 번역합니다.
- 날짜/연도는 한국어 방식으로 옮깁니다.
- 반드시 아래 JSON 형식 한 줄로만 출력하세요.
{glossary}
{{"question": "<번역된 질문>", "answer": "<번역된 정답>"}}

[질문] {question}
[정답] {answer}"""

# EN(lower) -> KO 용어집. main() 에서 로드.
GLOSSARY: dict[str, str] = {}
# 긴 키 우선 매칭을 위해 정렬된 (key, ko) 리스트
_GLOSSARY_ITEMS: list[tuple[str, str]] = []


def load_glossary():
    global GLOSSARY, _GLOSSARY_ITEMS
    gp = DATA_DIR / "glossary.json"
    if gp.exists():
        GLOSSARY = json.load(open(gp, encoding="utf-8"))
        _GLOSSARY_ITEMS = sorted(GLOSSARY.items(), key=lambda kv: -len(kv[0]))
        print(f"[glossary] loaded {len(GLOSSARY)} terms")
    else:
        print("[glossary] WARNING: glossary.json 없음 -> 용어집 강제 없이 번역")


def glossary_block(text: str, cap: int = 80) -> str:
    """영어 text 에 등장하는 용어집 항목으로 '고정 표기' 지시문 생성."""
    if not _GLOSSARY_ITEMS:
        return ""
    low = text.lower()
    hits = []
    for key, ko in _GLOSSARY_ITEMS:
        if key in low and ko and ko.lower() != key:
            # 단어 경계 확인 (영문 알파벳에 둘러싸인 부분일치 배제)
            if re.search(r'(?<![a-z])' + re.escape(key) + r'(?![a-z])', low):
                hits.append((key, ko))
                if len(hits) >= cap:
                    break
    if not hits:
        return ""
    lines = "\n".join(f"- {k} → {v}" for k, v in hits)
    return ("- 다음 고유명사는 반드시 지정된 한국어 표기를 그대로 사용하세요:\n"
            + lines + "\n")


# --------------------------------------------------------------------------- #
def download_source():
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for fname in ("questions.json", "chunks.json"):
        dst = SRC_DIR / fname
        if dst.exists():
            continue
        url = f"{HF_BASE}/{fname}"
        print(f"[download] {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)


def split_sentences(text: str) -> list[str]:
    # 단순 마침표 기반 분할 (잘림 발생 청크 재처리용)
    parts = re.split(r'(?<=\.)\s+', text)
    return [p for p in parts if p.strip()]


def translate_text(client: LLMClient, text: str, depth: int = 0) -> str:
    """청크 텍스트 번역. 잘리면 절반으로 나눠 재귀 번역. 용어집 강제."""
    gloss = glossary_block(text)
    content, finish = client.chat_full(
        [{"role": "user",
          "content": CHUNK_PROMPT.format(glossary=gloss, text=text)}],
        temperature=0.0, max_tokens=4096,
    )
    if finish == "stop" or depth >= 3:
        return content.strip()
    # 잘림 -> 문장 단위로 절반 분할 후 각각 번역
    sents = split_sentences(text)
    if len(sents) < 2:
        return content.strip()
    mid = len(sents) // 2
    left = translate_text(client, " ".join(sents[:mid]), depth + 1)
    right = translate_text(client, " ".join(sents[mid:]), depth + 1)
    return (left + " " + right).strip()


def translate_chunk(client: LLMClient, idx: int, raw: str) -> str:
    # "N:" 접두사 보존, 본문만 번역
    m = re.match(r'^(\d+):(.*)$', raw, flags=re.S)
    prefix, body = (m.group(1), m.group(2)) if m else (str(idx), raw)
    ko = translate_text(client, body.strip())
    return f"{prefix}:{ko}"


_JSON_RE = re.compile(r'\{.*\}', re.S)


def translate_qa(client: LLMClient, item: dict) -> dict:
    out = dict(item)
    gloss = glossary_block(f'{item["question"]} {item["answer"]}')
    prompt = QA_PROMPT.format(glossary=gloss, question=item["question"],
                             answer=item["answer"])
    content = client.chat([{"role": "user", "content": prompt}],
                          temperature=0.0, max_tokens=512)
    parsed = None
    mt = _JSON_RE.search(content)
    if mt:
        try:
            parsed = json.loads(mt.group(0))
        except json.JSONDecodeError:
            parsed = None
    if parsed and "question" in parsed and "answer" in parsed:
        out["question"] = str(parsed["question"]).strip()
        out["answer"] = str(parsed["answer"]).strip()
    else:
        # JSON 파싱 실패 -> 개별 번역 폴백
        out["question"] = translate_text(client, item["question"])
        out["answer"] = translate_text(client, item["answer"])
    return out


# --------------------------------------------------------------------------- #
def run_chunks(client, workers: int):
    src = json.load(open(SRC_DIR / "chunks.json", encoding="utf-8"))
    cache = CACHE_DIR / "chunks"
    cache.mkdir(parents=True, exist_ok=True)

    todo = [i for i in range(len(src)) if not (cache / f"{i}.txt").exists()]
    print(f"[chunks] total={len(src)} todo={len(todo)}")

    def work(i):
        ko = translate_chunk(client, i, src[i])
        (cache / f"{i}.txt").write_text(ko, encoding="utf-8")
        return i

    _run_pool(work, todo, workers, "chunks")

    result = [(cache / f"{i}.txt").read_text(encoding="utf-8")
              for i in range(len(src))]
    out = DATA_DIR / "chunks.json"
    json.dump(result, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[chunks] wrote {out} ({len(result)} items)")


def run_questions(client, workers: int):
    src = json.load(open(SRC_DIR / "questions.json", encoding="utf-8"))
    cache = CACHE_DIR / "questions"
    cache.mkdir(parents=True, exist_ok=True)

    todo = [i for i in range(len(src)) if not (cache / f"{i}.json").exists()]
    print(f"[questions] total={len(src)} todo={len(todo)}")

    def work(i):
        ko = translate_qa(client, src[i])
        json.dump(ko, open(cache / f"{i}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        return i

    _run_pool(work, todo, workers, "questions")

    result = [json.load(open(cache / f"{i}.json", encoding="utf-8"))
              for i in range(len(src))]
    out = DATA_DIR / "questions.json"
    json.dump(result, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[questions] wrote {out} ({len(result)} items)")


def _run_pool(work, todo, workers, tag):
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[{tag}] ERROR idx={i}: {e}", flush=True)
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"[{tag}] {done}/{len(todo)} (errors={errors})",
                      flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["all", "chunks", "questions"],
                    default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backend", choices=["gemma", "openai"], default="gemma")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    download_source()
    load_glossary()
    client = make_client(args.backend, args.model)
    print(f"[backend] {args.backend} model={getattr(client, 'model', '?')} "
          f"workers={args.workers}")
    if args.what in ("all", "chunks"):
        run_chunks(client, args.workers)
    if args.what in ("all", "questions"):
        run_questions(client, args.workers)


if __name__ == "__main__":
    main()
