"""분석 공용 함수: 시스템 목록과 문항별 결과를 읽는다.

결과 파일: results/track_02/{폴더}_{데이터셋}/predictions.json
  - gpt4o   : gpt-4o 판정 정오 (bool)
  - contain : 정답 문자열 포함 여부 (bool)
  - retrieved / supporting : 검색 상위 문서 ID / 정답 근거 문서 ID
R@5 는 정답 근거 문서가 모두 검색 상위 5개 안에 있으면 1 이다 (wiki_ko 는 산출하지 않음).
"""
import csv
import json
import math
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "track_02"
DATASETS = ["wiki_ko", "admin_single", "admin_full", "admin_hard"]


def systems():
    """analysis/systems.csv → [{no, name, main10, wiki_folder, admin_folder, ...}]"""
    with open(ROOT / "analysis" / "systems.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def system(name):
    for s in systems():
        if s["name"] == name:
            return s
    raise KeyError(name)


def folder(sysrow, dataset):
    return sysrow["wiki_folder"] if dataset == "wiki_ko" else sysrow["admin_folder"]


def load(name_or_row, dataset):
    """문항 ID → {"acc", "contain", "r5"} (r5 는 wiki_ko 에서 None)"""
    row = system(name_or_row) if isinstance(name_or_row, str) else name_or_row
    path = RESULTS / f"{folder(row, dataset)}_{dataset}" / "predictions.json"
    out = {}
    for r in json.load(open(path, encoding="utf-8")):
        r5 = None
        if dataset != "wiki_ko":
            r5 = set(map(str, r["supporting"])) <= set(map(str, r["retrieved"][:5]))
        out[r["id"]] = {"acc": bool(r["gpt4o"]), "contain": bool(r["contain"]), "r5": r5}
    return out


def rate(res, key):
    vals = [v[key] for v in res.values()]
    return sum(vals) / len(vals)


def mcnemar(a, b, key, ids=None):
    """짝지은 정확 McNemar(양측). 반환: A율, B율, b(A만), c(B만), 차이(%p), 95% CI(%p), p"""
    ids = sorted(a) if ids is None else ids
    assert set(ids) <= set(b), "두 조건의 문항 ID가 다르다"
    n = len(ids)
    bb = sum(1 for i in ids if a[i][key] and not b[i][key])
    cc = sum(1 for i in ids if b[i][key] and not a[i][key])
    p = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
    ra = sum(a[i][key] for i in ids) / n
    rb = sum(b[i][key] for i in ids) / n
    d = (bb - cc) / n
    se = math.sqrt(max(bb + cc - (bb - cc) ** 2 / n, 0)) / n
    return ra, rb, bb, cc, 100 * d, (100 * (d - 1.96 * se), 100 * (d + 1.96 * se)), p


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"
