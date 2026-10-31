"""H5 검정: 문단 가중 계수(pnw)별 단일 도약 대 Vanilla, H5 판정표, 정점 위치 부트스트랩 (2026-10-28).

H5. 질의–문단 유사도의 가중 계수(pnw)를 원논문 기본값(0.05)보다 높이면 단일 도약 검색은 그래프 없는
    밀집 검색 수준까지 개선되지만, 다중 도약 검색은 일정 수준까지만 함께 개선되고 그 이상에서는 저하된다.

입력: results/track_02/pnw_sweep/{추출}[_prf]_{집합}.json (pnw_sweep.py와 같음), Vanilla RAG 문항별 결과,
      확인용 ◆ 응답(pnw_sweep.CONFIRM). 새 검색·응답 생성 없음.

  R5  단일 도약: 8개 계열 × pnw 11개의 admin single R@5 대 Vanilla RAG. 판정용 칸은 계열별 full 최고점
      (full R@5 최고, 같은 값이면 가장 작은 pnw). 확인용 GPT-Acc는 ◆ 응답이 있는 6개 점만 Vanilla ◆와 비교
  R6  H5 판정표: ① 최고점 single 대 0.05, ② 최고점 single 대 Vanilla,
      ③-1 full 2.0·5.0 대 full 최고점, ③-2 hard 정점(같은 규칙)과 hard 2.0·5.0 대 hard 정점
  R7  정점 위치의 불확실성: admin_full 문항 단위 부트스트랩(2,000회, 시드 고정)으로 full 최고 pnw의 분포.
      판정에는 쓰지 않는다

정확 McNemar(양측), 95% CI는 짝지은 비율 차이의 정규 근사(common.mcnemar). 차이는 A − B (%p). 다중 비교 보정 없음.
최고점은 full R@5를 보고 고른 점이므로 ①·② 검정은 탐색적이다.

원고 대응(2026-10-31 15단계 번호): 제4장 제5절 1항 <표 39>(R6 판정표, ③의 2.0·5.0 비교 포함), <표 40>(R7 부트스트랩).

실행: python analysis/h5_tests.py
"""
import numpy as np

from common import fmt_p, load, mcnemar
from pnw_sweep import CONFIRM, SETS, load_sweep, rate

SERIES = [  # (추출, PRF(문장), 표시 이름)
    ("comp", False, "comp"), ("noun", False, "noun"), ("gazcomp", False, "gazcomp(PMI)"),
    ("gazcomp_llm4b", False, "4B 증류"), ("gazcomp_llm", False, "31B 증류"), ("compsp", False, "compsp (탐색)"),
    ("comp", True, "comp + PRF(문장)"), ("gazcomp_llm4b", True, "4B 증류 + PRF(문장)"),
]
BASE_P = "0.05"
HIGH = ["2.0", "5.0"]
N_BOOT, SEED = 2000, 20261028

HEAD = "| A | n | A | B | b/c | 차이(%p) | 95% CI | p |\n|:--|--:|--:|--:|--:|--:|:--|--:|"


def peak(curve):
    """R@5가 가장 높은 pnw (같은 값이면 가장 작은 pnw)와 동률 목록"""
    top = max(curve.values())
    ties = sorted((p for p, v in curve.items() if abs(v - top) < 1e-12), key=float)
    return ties[0], ties


def test(a, b, key="r5"):
    ra, rb, bb, cc, diff, (lo, hi), p = mcnemar(a, b, key)
    return dict(n=len(a), ra=ra, rb=rb, bb=bb, cc=cc, diff=diff, lo=lo, hi=hi, p=p)


def bold(t, s):
    return f"**{s}**" if t["p"] < 0.05 else s


def line(label, t):
    diff = bold(t, f"{t['diff']:+.1f}")
    return (f"| {label} | {t['n']} | {t['ra']:.3f} | {t['rb']:.3f} | {t['bb']}/{t['cc']} | {diff} | "
            f"[{t['lo']:+.1f}, {t['hi']:+.1f}] | {bold(t, fmt_p(t['p']))} |")


def short(t, ci=False):
    s = bold(t, f"{t['diff']:+.1f}")
    if ci:
        s += f" [{t['lo']:+.1f}, {t['hi']:+.1f}]"
    return f"{s} ({t['bb']}/{t['cc']}, p={fmt_p(t['p'])})"


def main():
    van = {d: load("Vanilla RAG", d) for d in SETS}
    print("# H5 검정 (정확 McNemar 양측, 굵은 글씨 = p < 0.05, 다중 비교 보정 없음)")
    print(f"\nVanilla RAG admin single: R@5 {rate(van['admin_single']):.3f}, "
          f"GPT-Acc(◆) {sum(v['acc'] for v in van['admin_single'].values()) / len(van['admin_single']):.3f}")

    rows, boot = [], []
    print("\n## R5. 단일 도약 R@5: pnw별 대 Vanilla RAG (A = 해당 pnw, B = Vanilla)\n")
    for x, prf, name in SERIES:
        S, _ = load_sweep(x, prf)
        pnws = list(S["admin_full"].keys())
        assert pnws == sorted(pnws, key=float)
        best, ties = peak({p: rate(S["admin_full"][p]) for p in pnws})
        hpeak, hties = peak({p: rate(S["admin_hard"][p]) for p in pnws})
        print(f"\n### {name} — full 최고 pnw {best}" + (f" (동률 {', '.join(ties)})" if len(ties) > 1 else "") + "\n")
        print(HEAD.replace("| A |", "| pnw |", 1))
        for p in pnws:
            mark = " ★" if p == best else (" (기본값)" if p == BASE_P else "")
            print(line(f"{p}{mark}", test(S["admin_single"][p], van["admin_single"])))
        rows.append(dict(
            name=name, best=best, ties=ties, hpeak=hpeak, hties=hties,
            s_best=rate(S["admin_single"][best]), s_base=rate(S["admin_single"][BASE_P]),
            r1=test(S["admin_single"][best], S["admin_single"][BASE_P]),
            r2=test(S["admin_single"][best], van["admin_single"]),
            r31={p: test(S["admin_full"][p], S["admin_full"][best]) for p in HIGH},
            r32={p: test(S["admin_hard"][p], S["admin_hard"][hpeak]) for p in HIGH if p != hpeak}))
        boot.append((name, best, pnws, S["admin_full"]))

    print("\n## R5 확인용 GPT-Acc: ◆ 응답이 있는 점 대 Vanilla RAG ◆ (admin single)\n")
    print(HEAD.replace("| A |", "| 추출 · pnw |", 1))
    for name, pts in CONFIRM.items():
        for p, f in pts.items():
            res = load({"admin_folder": f, "wiki_folder": f}, "admin_single")
            print(line(f"{name} · {p}", test(res, van["admin_single"], "acc")))

    print("\n## R6. H5 판정표 (원고 <표 39>, admin R@5, 차이는 A − B %p, 괄호는 b/c와 p)\n")
    print("| 계열 | full 최고 pnw | single (0.05 → 최고점) | ① 최고점 대 0.05 | ② 최고점 대 Vanilla [95% CI] "
          "| ③-1 full 2.0 대 최고점 | ③-1 full 5.0 대 최고점 | ③-2 hard 정점 | ③-2 hard 2.0 대 정점 | ③-2 hard 5.0 대 정점 |")
    print("|:--|--:|--:|:--|:--|:--|:--|--:|:--|:--|")
    for r in rows:
        tie = lambda b, t: b + (f" (동률 {','.join(t)})" if len(t) > 1 else "")
        h = [short(r["r32"][p]) if p in r["r32"] else "— (정점)" for p in HIGH]
        print(f"| {r['name']} | {tie(r['best'], r['ties'])} | {r['s_base']:.3f} → {r['s_best']:.3f} | "
              f"{short(r['r1'])} | {short(r['r2'], ci=True)} | {short(r['r31']['2.0'])} | {short(r['r31']['5.0'])} | "
              f"{tie(r['hpeak'], r['hties'])} | {h[0]} | {h[1]} |")

    print(f"\n## R7. full 최고 pnw의 부트스트랩 분포 (원고 <표 40>, admin_full 문항 단위, {N_BOOT:,}회, 시드 {SEED}, "
          "같은 값이면 가장 작은 pnw)\n")
    pnws = boot[0][2]
    print("| 계열 | 표본 최고점 | 최빈값 | " + " | ".join(pnws) + " |")
    print("|:--|--:|--:|" + "--:|" * len(pnws))
    for name, best, ps, full in boot:
        ids = sorted(full[ps[0]])
        M = np.array([[full[p][i]["r5"] for p in ps] for i in ids], dtype=np.int32)  # 문항 × pnw
        idx = np.random.default_rng(SEED).integers(0, len(ids), size=(N_BOOT, len(ids)))
        sums = M[idx].sum(axis=1)  # 반복 × pnw
        winner = sums.argmax(axis=1)  # 같은 값이면 첫 열 = 가장 작은 pnw (ps는 오름차순)
        share = np.bincount(winner, minlength=len(ps)) / N_BOOT
        print(f"| {name} | {best} | {ps[int(share.argmax())]} | "
              + " | ".join(f"**{s:.1%}**" if p == best else (f"{s:.1%}" if s else "—") for p, s in zip(ps, share)) + " |")


if __name__ == "__main__":
    main()
