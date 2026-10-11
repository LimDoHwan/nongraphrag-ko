"""논문의 시스템 하나를 한 데이터셋에서 실행한다 (analysis/systems.csv 의 설정 사용).

예:
  python scripts/run_system.py --system "comp +dense0.5 +PRF(문장)" --dataset admin_full
  python scripts/run_system.py --system "gazcomp(PMI)" --dataset wiki_ko --dry-run
  python scripts/run_system.py --list

--dry-run 은 실행할 명령만 출력한다. HippoRAG2 는 OpenIE(색인 LLM 호출) → 그래프 → QA 순으로 돈다.
"""
import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["wiki_ko", "admin_single", "admin_full", "admin_hard", "admin_hard2k"]


def command(s, dataset, workers):
    py = [sys.executable, "-m"]
    if s["runner"] == "vanillarag":
        return py + ["src.track_02.vanillarag.run", "--dataset", dataset, "--workers", str(workers)]
    if s["runner"] == "hipporag2":
        cmd = py + ["src.track_02.graphrag.hipporag2", "--dataset", dataset, "--stage", "all",
                    "--workers", str(workers)]
        if s["tag"]:
            cmd += ["--variant", "4bv2" if dataset == "wiki_ko" else "4b"]
        return cmd
    cmd = py + ["src.track_02.linearrag.engine", "--dataset", dataset, "--extract", s["extract"],
                "--workers", str(workers)]
    tag = s["tag"]
    if tag and s["extract"] != "kw":
        if dataset == "wiki_ko" and tag == "llm4b":
            tag = "llm4bv2"
        cmd += ["--tag", tag]
    if s["topk_kw"]:
        cmd += ["--topk-kw", s["topk_kw"], "--tag", f"k{s['topk_kw']}"]
    gkey = "glossary_admin_hard" if dataset == "admin_hard2k" else f"glossary_{dataset}"
    gloss = s.get(gkey, "")
    if dataset == "admin_hard2k":
        gloss = {"llm": "data/admin_ko/glossary_llm.json",
                 "llmner": "data/admin_ko/glossary_llm_ner.json"}.get(s["tag"], gloss)
    if gloss:
        cmd += ["--glossary", gloss]
    if s["pnw"]:
        cmd += ["--pnw", s["pnw"]]
    if s["prf_beta"]:
        cmd += ["--prf-beta", s["prf_beta"]]
    if s["prf_sent"] == "1":
        cmd += ["--prf-sent"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "analysis" / "systems.csv", encoding="utf-8")))
    if a.list:
        for r in rows:
            print(f"{r['no']:>2}  {r['name']}{'  ◆' if r['main10'] == '1' else ''}")
        return
    if not (a.system and a.dataset):
        ap.error("--system 과 --dataset 이 필요합니다 (--list 로 목록 확인)")
    s = next((r for r in rows if r["name"] == a.system), None)
    if s is None:
        sys.exit(f"시스템 없음: {a.system} (--list 로 확인)")
    cmd = command(s, a.dataset, a.workers)
    print(" ".join(shlex.quote(c) for c in cmd))
    if not a.dry_run:
        sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
