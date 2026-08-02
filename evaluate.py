import json, sys, httpx, numpy as np
from scipy.stats import spearmanr

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
PATH = sys.argv[2] if len(sys.argv) > 2 else "validation.jsonl"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
DIMS = ["content", "organization", "expression"]

rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()][:N]
P, G, raws = [], [], []
with httpx.Client(timeout=300.0) as c:
    for i, r in enumerate(rows):
        msg = f"[prompt_text]\n{r['prompt']}\n\n[essay_text]\n{r['essay']}"
        d = c.post(f"{URL}/v1/chat/completions", json={
            "model": "kwriting-scorer",
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": 2048, "temperature": 0.0, "seed": 42})
        o = json.loads(d.json()["choices"][0]["message"]["content"])
        P.append([o[k]["score"] for k in DIMS])
        G.append([r["score"][k] for k in DIMS])
        if i == 0:
            print(json.dumps(o, ensure_ascii=False, indent=2)[:700])
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(rows)}")

P, G = np.array(P), np.array(G)
print("\n=== 결과 ===")
rs, ss = [], []
for i, d in enumerate(DIMS):
    rmse = float(np.sqrt(((P[:, i] - G[:, i]) ** 2).mean()))
    sp = float(spearmanr(P[:, i], G[:, i]).statistic)
    rs.append(rmse); ss.append(sp)
    # 최적 캘리브레이션 계수
    a, b = np.polyfit(P[:, i], G[:, i], 1)
    print(f"{d:13s} RMSE={rmse:.4f} Spearman={sp:.4f} "
          f"| 예측평균={P[:,i].mean():.3f} 정답평균={G[:,i].mean():.3f} "
          f"| 권장 a={a:.4f} b={b:.4f}")
print(f"{'평균':13s} RMSE={np.mean(rs):.4f} Spearman={np.mean(ss):.4f}")
print("\n리더보드 1위: RMSE 0.4176 / Spearman 0.7366")
