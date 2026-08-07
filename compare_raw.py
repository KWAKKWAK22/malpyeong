"""Pod에서 잰 raw 값과 제출 이미지의 raw 값이 일치하는지 대조.

사용: python3 compare_raw.py raw_v2full.jsonl http://127.0.0.1:8000 [건수]

계수 a가 1 근처이므로 raw가 0.05 어긋나면 반올림 경계를 넘어 점수가 1점 바뀐다.
CUDA 11.8 Pod과 12.8 이미지는 다른 빌드이므로 반드시 확인할 것.
"""
import json, sys, httpx, numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw_v2full.jsonl"
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
VAL = "validation.jsonl"
DIMS = ["content", "organization", "expression"]

ref = {json.loads(l)["id"]: json.loads(l)["raw"]
       for l in open(RAW, encoding="utf-8") if l.strip()}
rows = [json.loads(l) for l in open(VAL, encoding="utf-8") if l.strip()][:N]

diffs = {d: [] for d in DIMS}
with httpx.Client(timeout=600.0, trust_env=False) as c:
    for r in rows:
        msg = f"[prompt_text]\n{r['prompt']}\n\n[essay_text]\n{r['essay']}"
        c.post(f"{URL}/v1/chat/completions", json={
            "model": "kwriting-scorer", "messages": [{"role": "user", "content": msg}],
            "max_tokens": 2048, "temperature": 0.0, "top_p": 1.0, "seed": 42})
        now = c.get(f"{URL}/debug/last").json().get("raw", {})
        old = ref.get(r.get("id"), {})
        for d in DIMS:
            if d in now and d in old:
                diffs[d].append(abs(now[d] - old[d]))

print(f"=== raw 재현성 ({len(rows)}건) ===")
bad = False
for d in DIMS:
    a = np.array(diffs[d]) if diffs[d] else np.array([np.nan])
    over = int((a > 0.05).sum())
    if over: bad = True
    print(f"  {d:13s} 최대차 {np.nanmax(a):.4f}  평균차 {np.nanmean(a):.4f}  "
          f"| 0.05 초과 {over}/{len(a)}건")
print()
if bad:
    print(">>> 어긋남. 이미지에서 raw를 다시 뽑아 계수를 재추정해야 한다.")
    sys.exit(1)
print(">>> 일치. Pod에서 정한 계수를 그대로 써도 된다.")
