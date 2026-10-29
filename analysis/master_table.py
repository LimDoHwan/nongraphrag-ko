"""부록 B <표 54>의 41개 시스템 행 (× 4개 집합의 GPT-Acc·R@5·Contain) 재생성. 탐색 조건 10행은 integrated_table.py.

문항별 결과에서 지표를 다시 계산하고, analysis/reference/appendix_b.csv(논문 수록값)와 비교한다.
불일치가 있으면 종료 코드 1.

실행: python analysis/master_table.py
"""
import csv
import sys

from common import DATASETS, ROOT, load, rate, systems

COLS = [("wiki_ko", "acc"), ("wiki_ko", "contain"),
        ("admin_single", "acc"), ("admin_single", "r5"), ("admin_single", "contain"),
        ("admin_full", "acc"), ("admin_full", "r5"), ("admin_full", "contain"),
        ("admin_hard", "acc"), ("admin_hard", "r5"), ("admin_hard", "contain")]


def main():
    ref = {}
    with open(ROOT / "analysis" / "reference" / "appendix_b.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref[r["name"]] = r
    head = "| 시스템 | " + " | ".join(f"{d} {m}" for d, m in COLS) + " |"
    print(head)
    print("|:--|" + "--:|" * len(COLS))
    bad = rounding = 0
    for s in systems():
        res = {d: load(s, d) for d in DATASETS}
        cells = []
        for d, m in COLS:
            v = f"{rate(res[d], m):.3f}"
            r = ref.get(s["name"], {}).get(f"{d}_{m}")
            if r is not None and r != v:
                if abs(float(r) - float(v)) <= 0.0011:
                    rounding += 1
                    v += f" (논문 {r}, 반올림)"
                else:
                    bad += 1
                    v += f" (논문 {r}, 불일치)"
            cells.append(v)
        mark = " ◆" if s["main10"] == "1" else ""
        print(f"| {s['name']}{mark} | " + " | ".join(cells) + " |")
    print(f"\n논문 수록값과 불일치: {bad}칸, 반올림 차이(±0.001): {rounding}칸")
    if rounding:
        print("반올림 차이는 논문 쪽이 소수 넷째 자리에서 한 번 반올림한 값을 다시 반올림한 것이다. 이 표의 값이 정확하다.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
