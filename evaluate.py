"""검증 스크립트.

사용:  python3 evaluate.py [URL] [validation.jsonl] [N] [태그]

평가 서버는 실수 점수를 반올림해 정수로 만든 뒤 채점한다(2026-08-06 규칙).
따라서 여기서도 반올림 후 값으로 RMSE/Spearman을 낸다.

DEBUG_RAW=1 로 서버를 띄우면 캘리브레이션 이전의 raw 예측값을
raw_<태그>.jsonl 에 저장한다. 격자 탐색은 이 파일만 있으면 GPU 없이 돌아간다.
"""
import json, sys, os, httpx, numpy as np
from scipy.stats import spearmanr

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
PATH = sys.argv[2] if len(sys.argv) > 2 else "validation.jsonl"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
TAG = sys.argv[4] if len(sys.argv) > 4 else "run"
DIMS = ["content", "organization", "expression"]

rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()][:N]
P, G, RAW = [], [], []
raw_f = open(f"raw_{TAG}.jsonl", "w", encoding="utf-8")

with httpx.Client(timeout=600.0, trust_env=False) as c:
    for i, r in enumerate(rows):
        msg = f"[prompt_text]\n{r['prompt']}\n\n[essay_text]\n{r['essay']}"
        d = c.post(f"{URL}/v1/chat/completions", json={
            "model": "kwriting-scorer",
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": 2048, "temperature": 0.0, "seed": 42})
        o = json.loads(d.json()["choices"][0]["message"]["content"])
        P.append([o[k]["score"] for k in DIMS])
        G.append([r["score"][k] for k in DIMS])

        rec = {"id": r.get("id", i), "gold": {k: r["score"][k] for k in DIMS}}
        try:
            dbg = c.get(f"{URL}/debug/last", timeout=10.0).json()
            rec["raw"] = dbg.get("raw", {})
            if i == 0:
                rec["analysis"] = dbg.get("analysis", "")
                print("--- 1번 글의 구조 분석 ---")
                print(dbg.get("analysis", "(비어 있음 — 분석 실패 또는 비활성)"))
                print("--------------------------")
        except Exception:
            rec["raw"] = {}
        RAW.append(rec)
        raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if i == 0:
            print(json.dumps(o, ensure_ascii=False, indent=2)[:700])
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(rows)}", flush=True)
raw_f.close()

P, G = np.array(P, dtype=float), np.array(G, dtype=float)
Pr = np.round(P)          # 평가 서버가 적용하는 반올림

print(f"\n=== 결과 (n={len(rows)}, 반올림 적용) ===")
rs, ss = [], []
for i, d in enumerate(DIMS):
    rmse = float(np.sqrt(((Pr[:, i] - G[:, i]) ** 2).mean()))
    sp = float(spearmanr(Pr[:, i], G[:, i]).statistic)
    rs.append(rmse); ss.append(sp)
    uniq = np.unique(Pr[:, i])
    print(f"{d:13s} RMSE={rmse:.4f} Spearman={sp:.4f} "
          f"| 예측평균={Pr[:,i].mean():.3f} 정답평균={G[:,i].mean():.3f} "
          f"| 출력값={[int(u) for u in uniq]}")
print(f"{'평균':13s} RMSE={np.mean(rs):.4f} Spearman={np.mean(ss):.4f}")

print("\n[기존 1회차 성적 — 비교용]")
print("content       RMSE 0.6842  Spearman 0.2785")
print("organization  RMSE 0.9077  Spearman 0.3438")
print("expression    RMSE 0.6726  Spearman 0.5675")
print("평균          RMSE 0.7548  Spearman 0.3716")
print("\n[새 규칙에서의 이론적 한계 — 정답을 알아도 정수만 낼 수 있음]")
print("content RMSE 0.291 / organization 0.303 / expression 0.303")

nraw = sum(1 for r in RAW if r.get("raw"))
print(f"\nraw 저장: raw_{TAG}.jsonl ({nraw}/{len(RAW)}건)")
if nraw == 0:
    print("  ! raw가 비었다. 서버를 DEBUG_RAW=1 로 띄웠는지 확인할 것.")
