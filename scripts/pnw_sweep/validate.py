"""스윕 결과를 저장된 결과(같은 설정의 기존 실행)와 문항 단위로 대조한다.
실행: python scripts/pnw_sweep/validate.py [파일...]  (인자 없으면 cache/track_02/pnw_sweep/*.json 전부)"""
import glob, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "track_02"
BASE = {"noun": "linearrag-new-noun", "comp": "linearrag-new-comp", "gazcomp": "linearrag-new-gazcomp",
        "gazcomp_llm4b": "linearrag-new-gazcomp-llm4b", "gazcomp_llm": "linearrag-new-gazcomp-llm",
        "compsp": "linearrag-new-compsp"}
def stored_folders(extract, prf):
    b = BASE[extract]
    if prf:
        return {"0.5": f"{b}-pnw0.5-prf0.6sent", "1.0": f"{b}-pnw1-prf0.6sent"}
    return {"0.05": b, "0.5": f"{b}-pnw0.5", "1.0": f"{b}-pnw1"}
files = sys.argv[1:] or sorted(glob.glob(str(ROOT / "cache/track_02/pnw_sweep/*.json")))
for f in files:
    d = json.load(open(f)); m = d["meta"]; prf = m["prf_sent_beta"] > 0
    for p, folder in stored_folders(m["extract"], prf).items():
        pf = R / f"{folder}_{m['dataset']}" / "predictions.json"
        if not pf.exists():
            continue
        st = {r["id"]: [str(x) for x in r["retrieved"][:5]] for r in json.load(open(pf))}
        sw = d["top5"][p]
        same = sum(sw[i] == st[i] for i in st)
        print(f"{os.path.basename(f)}  pnw {p}  대 {folder}: 상위 5개 일치 {same}/{len(st)}")
