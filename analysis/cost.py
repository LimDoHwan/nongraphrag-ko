"""제4장 제3절 비용 표 재생성(2026-10-19 원고 번호): <표 32> 토큰, <표 33> GPU-시간·화폐 환산, <표 38> ICER, <표 36> 비용 구조.

입력: analysis/cost_inputs.csv
  - 호출 수 = 문서 수 × 2 (OpenIE의 NER 1회 + RE 1회)
  - 입력 토큰 = 실제 요청을 재구성해 Gemma 4 토크나이저로 센 정확값
  - 출력 토큰 = 저장된 추출 결과를 재직렬화한 하한값 (4B wiki_ko 의 RE 는 서버 실측)
  - minutes = OpenIE 캐시 파일 생성 시각으로 복원한 처리 시간(동시 16요청, 공유 부하 포함, 잠정치)
    wiki_ko 는 처리 시간이 기록되지 않았다.
잠재가격: 2026-09-14 조회 클라우드 온디맨드 GPU 단가(USD/GPU·h), 제3장 <표 16>.

실행: python analysis/cost.py
"""
import csv

from common import ROOT, load, rate

PRICE = {  # 기준, 하한, 상한
    "RTX PRO 6000": (2.21, 1.69, 3.36),
    "RTX 4090": (0.44, 0.27, 0.74),
}


def rows():
    with open(ROOT / "analysis" / "cost_inputs.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    R = rows()
    print("## <표 32> 색인 단계 토큰 수 (OpenIE, 문서당 2회 호출)\n")
    print("| 모델 | 코퍼스 | 문서 | 호출 | 입력(정확) | 출력(하한) | 문서당 입력 / 출력 |")
    print("|:--|:--|--:|--:|--:|--:|:--|")
    for r in R:
        d = int(r["docs"])
        print(f"| {r['model']} | {r['corpus']} | {d:,} | {int(r['calls']):,} | {int(r['input_tokens']):,} | "
              f"≥ {int(r['output_tokens_lb']):,} | {int(r['input_tokens']) / d:,.0f} / ≥ {int(r['output_tokens_lb']) / d:,.0f} |")

    print("\n## <표 33> GPU-시간과 화폐 환산\n")
    print("| 모델 (GPU) | 코퍼스 | 문서 | GPU-시간 | 기준 단가 | 민감도 범위 |")
    print("|:--|:--|--:|--:|--:|:--|")
    gpu_h = {}
    for r in R:
        if not r["minutes"]:
            continue
        h = float(r["minutes"]) / 60
        gpu_h[(r["model"], r["corpus"])] = h
        base, lo, hi = PRICE[r["gpu"]]
        print(f"| {r['model']} ({r['gpu']}) | {r['corpus']} | {int(r['docs']):,} | {h:.2f} | "
              f"{h * base:.2f}달러 | {h * lo:.2f} ~ {h * hi:.2f}달러 |")
    for corpus in ("admin_full", "admin_single"):
        c31 = gpu_h[("31B", corpus)] * PRICE["RTX PRO 6000"][0]
        c4 = gpu_h[("4B", corpus)] * PRICE["RTX 4090"][0]
        print(f"\n{corpus}: 31B 비용 / 4B 비용 = {c31 / c4:.1f}배")

    print("\n## <표 38> wiki_ko 증분 비용-효과비 (호출·입력 토큰 기준)\n")
    tok = {r["model"]: int(r["input_tokens"]) for r in R if r["corpus"] == "wiki_ko"}
    calls = 1316
    pairs = [("HippoRAG2 (31B 그래프)", "comp", "31B"),
             ("HippoRAG2 (31B 그래프)", "gazcomp(PMI)", "31B"),
             ("gazcomp(4B 증류 NER+RE)", "gazcomp(PMI)", "4B"),
             ("gazcomp(31B 증류 NER+RE)", "gazcomp(PMI)", "31B")]
    print("| 비교 (B 대 A) | 효과 차이(%p) | 추가 호출 | 추가 입력 토큰 | 1%p당 호출 | 1%p당 입력 토큰 |")
    print("|:--|--:|--:|--:|--:|--:|")
    for b, a, m in pairs:
        de = 100 * (rate(load(b, "wiki_ko"), "acc") - rate(load(a, "wiki_ko"), "acc"))
        print(f"| {b} 대 {a} | {de:.1f} | {calls:,} | {tok[m]:,} | {calls / de:.1f}회 | 약 {tok[m] / de / 1e4:.1f}만 개 |")

    print("\n## <표 36> 관련: 문서 규모별 GPU 비용 (예시)\n")
    print("1,000문서당 GPU-시간 범위는 admin_full·admin_single 의 관측값에서 구한다.\n")
    for m, gpu in (("4B", "RTX 4090"), ("31B", "RTX PRO 6000")):
        per = sorted(gpu_h[(m, c)] / int(next(r["docs"] for r in R if r["model"] == m and r["corpus"] == c)) * 1000
                     for c in ("admin_full", "admin_single"))
        base = PRICE[gpu][0]
        print(f"- {m}: 1,000문서당 {per[0]:.3f}~{per[1]:.3f} GPU-시간 ({per[0] * base:.3f}~{per[1] * base:.3f}달러)")
        for n in (100_000, 1_000_000):
            lo, hi = per[0] * n / 1000, per[1] * n / 1000
            print(f"  - HippoRAG2({m}) 신규 {n:,}문서: {lo:,.1f}~{hi:,.1f} GPU-시간 "
                  f"({lo * base:,.1f}~{hi * base:,.1f}달러), GPU 1장 연속 {lo / 24:.1f}~{hi / 24:.1f}일")
    print("- 형태소 규칙·통계 사전: GPU 0. 증류 사전: 구축 시 1회(위 1,000문서당 비용 × 구축 코퍼스), 재사용 시 증분 GPU 0")


if __name__ == "__main__":
    main()
