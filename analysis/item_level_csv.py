"""제출용 문항별 결과 CSV (2026-11-02 원고 요청, 공유 문서 26절).

학과 제출 자료의 "데이터 파일(CSV)"용으로, 원고 검정에 쓴 문항별 결과를 긴 형식(한 행 = 집합 × 시스템 × 문항)으로 낸다.
열 설명(데이터 딕셔너리)은 analysis/item_level_data_dictionary.md이며, 출력 폴더에 데이터딕셔너리.md로 함께 복사한다.

- 문항별결과_시스템.csv : 부록 B <표 54>의 51행(41개 시스템 + ☆ 탐색 조건 10개) × 4개 집합. <표 21>의 행은 모두 여기에 들어 있다
- 문항별결과_스윕.csv   : 문단 가중 계수 스윕, 8개 계열 × 계수 11단계 × admin 3개 집합의 R@5 (<표 39>, 부록 F <표 81>~<표 83>)
- 문항별결과_스윕확인용.csv : 스윕 중 GPT-Acc를 확인용으로 낸 4개 점(comp 0.5·1.0, 4B 증류 사전 0.2·1.0) × admin 3개 집합

질의·정답 문장은 넣지 않고 문항 ID로만 식별한다(AI 허브 원본 질의응답 재배포 금지).
UTF-8(BOM 포함)로 써서 Excel에서 한글이 깨지지 않게 한다.
쓴 뒤 CSV를 다시 읽어 집계한 값을 analysis/reference/appendix_b.csv(부록 B 수록값)와 대조한다.

실행: python analysis/item_level_csv.py [--out 출력 폴더]   (기본 출력: item_level/)
"""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from common import DATASETS, RESULTS, ROOT, load, systems

EXPLORE = list(csv.DictReader(open(ROOT / "analysis" / "systems_explore.csv", encoding="utf-8")))
# 탐색 행 중 ◆와 같은 응답 조건(출력 한도 2,000)으로 생성한 행 (integrated_table.py와 같음)
EXPLORE_LONG = {"compsp", "gazcompsp(PMI)", "gazcompsp(4B 증류 NER+RE)", "gazcompsp(31B 증류 NER+RE)",
                "compsp +dense0.5 +PRF(문장)"}
SETS = ["admin_single", "admin_full", "admin_hard"]
# 스윕 파일 접두어 → (원고 부록 F의 계열 이름, PRF(문장) 사용, <표 54>의 대응 시스템)
SERIES = {
    "comp": ("comp", 0, "comp"),
    "noun": ("noun", 0, "noun"),
    "gazcomp": ("gazcomp(PMI)", 0, "gazcomp(PMI)"),
    "gazcomp_llm4b": ("4B 증류 사전", 0, "gazcomp(4B 증류 NER+RE)"),
    "gazcomp_llm": ("31B 증류 사전", 0, "gazcomp(31B 증류 NER+RE)"),
    "compsp": ("compsp (탐색 조건)", 0, "compsp"),
    "comp_prf": ("comp + PRF(문장)", 1, "comp"),
    "gazcomp_llm4b_prf": ("4B 증류 사전 + PRF(문장)", 1, "gazcomp(4B 증류 NER+RE)"),
}
CONFIRM = [("comp", "comp", "0.5"), ("comp", "comp", "1.0"),
           ("gazcomp_llm4b", "4B 증류 사전", "0.2"), ("gazcomp_llm4b", "4B 증류 사전", "1.0")]


def b(v):
    return "" if v is None else int(v)


def write(path, head, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    print(f"{path.name}: {len(rows)}행")


def system_rows():
    out = []
    for s in systems():
        group = "비교 시스템(◆)" if s["main10"] == "1" else "기타 시스템"
        tokens = 2000 if s["main10"] == "1" else 256
        out.append((s["no"], s["name"], group, tokens, s))
    for i, r in enumerate(EXPLORE, 1):
        out.append((f"E{i}", r["name"], "탐색 조건(☆)", 2000 if r["name"] in EXPLORE_LONG else 256, r))
    return out


def supporting(ds):
    rows = json.load(open(RESULTS / f"vanillarag_{ds}" / "predictions.json", encoding="utf-8"))
    return {r["id"]: set(map(str, r["supporting"])) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "item_level"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 시스템 × 집합 × 문항
    rows = []
    for no, name, group, tokens, row in system_rows():
        for d in DATASETS:
            for qid, v in sorted(load(row, d).items()):
                rows.append([d, no, name, group, tokens, qid, int(v["acc"]), int(v["contain"]), b(v["r5"])])
    write(out / "문항별결과_시스템.csv",
          ["dataset", "system_no", "system", "system_group", "answer_max_tokens", "question_id",
           "gpt_acc", "contain", "r5"], rows)

    # 2. 스윕 (검색만, R@5)
    sup = {d: supporting(d) for d in SETS}
    rows = []
    for key, (series, prf, base) in SERIES.items():
        for d in SETS:
            j = json.load(open(RESULTS / "pnw_sweep" / f"{key}_{d}.json", encoding="utf-8"))
            for pnw, top in j["top5"].items():
                for qid, t in sorted(top.items()):
                    rows.append([d, series, base, prf, pnw, qid, int(sup[d][qid] <= set(map(str, t[:5])))])
    write(out / "문항별결과_스윕.csv",
          ["dataset", "series", "base_system", "prf_sentence", "passage_node_weight", "question_id", "r5"], rows)

    # 3. 스윕 확인용 GPT-Acc (◆와 같은 응답 조건)
    rows = []
    for key, series, pnw in CONFIRM:
        row = {"wiki_folder": "", "admin_folder": f"linearrag-new-{key}-sweep-pnw{pnw}"}
        for d in SETS:
            for qid, v in sorted(load(row, d).items()):
                rows.append([d, series, pnw, qid, int(v["acc"]), int(v["contain"]), b(v["r5"])])
    write(out / "문항별결과_스윕확인용.csv",
          ["dataset", "series", "passage_node_weight", "question_id", "gpt_acc", "contain", "r5"], rows)

    shutil.copyfile(ROOT / "analysis" / "item_level_data_dictionary.md", out / "데이터딕셔너리.md")
    verify(out)


def verify(out):
    """시스템 CSV를 다시 읽어 집계하고 부록 B 수록값(analysis/reference/appendix_b.csv, 41행)과 대조한다."""
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])   # n, acc, contain, r5 합, r5 수
    with open(out / "문항별결과_시스템.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            a = agg[(r["system"], r["dataset"])]
            a[0] += 1
            a[1] += int(r["gpt_acc"])
            a[2] += int(r["contain"])
            if r["r5"] != "":
                a[3] += int(r["r5"])
                a[4] += 1
    ref = list(csv.DictReader(open(ROOT / "analysis" / "reference" / "appendix_b.csv", encoding="utf-8")))
    bad = cells = 0
    for r in ref:
        for col, v in r.items():
            if col == "name" or not v:
                continue
            d, m = col.rsplit("_", 1)
            n, acc, con, r5, nr5 = agg[(r["name"], d)]
            got = {"acc": acc / n, "contain": con / n, "r5": r5 / nr5 if nr5 else None}[m]
            cells += 1
            if got is None or f"{got:.3f}" != v:
                bad += 1
                print(f"  불일치: {r['name']} {col} 부록 B {v} / CSV {got}")
    print(f"부록 B 수록값 대조: {cells}칸 중 불일치 {bad}칸")


if __name__ == "__main__":
    main()
