"""격자 탐색 캘리브레이션 + 프롬프트 효과 진단.

사용:
    python3 calibrate.py raw_v2.jsonl                    # 캘리브레이션
    python3 calibrate.py raw_v2.jsonl raw_v1base.jsonl   # + 프롬프트 A/B 비교

GPU 불필요. raw 파일만 있으면 몇 초면 끝난다.

두 가지를 분리해서 본다.
  1) raw Spearman  — 캘리브레이션과 무관. 프롬프트가 신호를 늘렸는지 여기서만 알 수 있다.
  2) 반올림 후 성능 — a, b를 어떻게 잡느냐의 문제. raw가 같아도 크게 달라진다.
"""
import json, sys, itertools
import numpy as np
from scipy.stats import spearmanr

DIMS = ["content", "organization", "expression"]
# 대회 배점: RMSE 45 / Spearman 45 / Judge 10 → 둘을 같은 비중으로 본다.
# RMSE는 낮을수록, Spearman은 높을수록 좋으므로 부호를 맞춘다.
W_RMSE, W_SP = 0.5, 0.5
KFOLD = 5
SEED = 42


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = {}
    for d in DIMS:
        raw = np.array([r["raw"].get(d, np.nan) for r in rows], dtype=float)
        gold = np.array([r["gold"][d] for r in rows], dtype=float)
        m = ~np.isnan(raw)
        out[d] = (raw[m], gold[m])
    return out, len(rows)


def apply(a, b, raw):
    return np.round(np.clip(a * raw + b, 1, 5))


def metrics(p, g):
    rmse = float(np.sqrt(((p - g) ** 2).mean()))
    if len(np.unique(p)) < 2:
        return rmse, 0.0          # 전부 같은 값이면 순위 정보 없음
    return rmse, float(spearmanr(p, g).statistic)


def score_of(rmse, sp):
    # RMSE 하한 0.30, 상한 1.0 근처를 0~1로 정규화해 Spearman과 합산
    return W_SP * sp + W_RMSE * (1 - min(rmse, 1.2) / 1.2)


def grid(raw, gold, a_range, b_range):
    best = None
    for a in a_range:
        ar = a * raw
        for b in b_range:
            p = np.round(np.clip(ar + b, 1, 5))
            rmse, sp = metrics(p, gold)
            s = score_of(rmse, sp)
            if best is None or s > best[0]:
                best = (s, a, b, rmse, sp)
    return best


def cv_grid(raw, gold, a_range, b_range, k=KFOLD):
    """fold별로 train에서 a,b를 고르고 held-out에서 평가 → 정직한 추정치."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(raw))
    folds = np.array_split(idx, k)
    rs, ss, picks = [], [], []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        _, a, b, _, _ = grid(raw[tr], gold[tr], a_range, b_range)
        rmse, sp = metrics(apply(a, b, raw[te]), gold[te])
        rs.append(rmse); ss.append(sp); picks.append((a, b))
    return float(np.mean(rs)), float(np.mean(ss)), picks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "raw_v2.jsonl"
    data, n = load(path)
    print(f"=== {path} (n={n}) ===\n")

    A = np.arange(0.20, 6.01, 0.05)
    B = np.arange(-14.0, 8.01, 0.05)

    print("[1] raw 자체의 신호 — 캘리브레이션 이전, 프롬프트 품질의 척도")
    for d in DIMS:
        raw, gold = data[d]
        print(f"  {d:13s} Spearman={spearmanr(raw, gold).statistic:+.4f} "
              f"| raw 평균={raw.mean():.3f} 표준편차={raw.std():.3f} "
              f"범위={raw.min():.2f}~{raw.max():.2f}")

    print("\n[2] 현재 계수 (구 규칙 최소제곱값) 적용 시")
    CUR = {"content": (0.5115, 1.1582), "organization": (0.7812, 0.1981),
           "expression": (0.8437, 0.2960)}
    cur_r, cur_s = [], []
    for d in DIMS:
        raw, gold = data[d]
        a, b = CUR[d]
        p = apply(a, b, raw)
        rmse, sp = metrics(p, gold)
        cur_r.append(rmse); cur_s.append(sp)
        print(f"  {d:13s} RMSE={rmse:.4f} Spearman={sp:.4f} "
              f"| 출력값={sorted(int(u) for u in np.unique(p))} "
              f"| 예측평균={p.mean():.2f} 정답평균={gold.mean():.2f}")
    print(f"  {'평균':13s} RMSE={np.mean(cur_r):.4f} Spearman={np.mean(cur_s):.4f}")

    print("\n[3] 격자 탐색 — 전체 데이터에 최적화 (낙관적, 과적합 포함)")
    best = {}
    fit_r, fit_s = [], []
    for d in DIMS:
        raw, gold = data[d]
        _, a, b, rmse, sp = grid(raw, gold, A, B)
        best[d] = (a, b)
        fit_r.append(rmse); fit_s.append(sp)
        p = apply(a, b, raw)
        print(f"  {d:13s} a={a:5.2f} b={b:6.2f} -> RMSE={rmse:.4f} Spearman={sp:.4f} "
              f"| 출력값={sorted(int(u) for u in np.unique(p))}")
    print(f"  {'평균':13s} RMSE={np.mean(fit_r):.4f} Spearman={np.mean(fit_s):.4f}")

    print(f"\n[4] {KFOLD}-fold 교차검증 — 처음 보는 데이터에서 기대되는 성능")
    cv_r, cv_s = [], []
    for d in DIMS:
        raw, gold = data[d]
        rmse, sp, picks = cv_grid(raw, gold, A, B)
        cv_r.append(rmse); cv_s.append(sp)
        aa = [float(p[0]) for p in picks]
        spread = max(aa) - min(aa)
        flag = "  <- fold마다 a가 크게 흔들림. 신뢰도 낮음" if spread > 1.0 else ""
        print(f"  {d:13s} RMSE={rmse:.4f} Spearman={sp:.4f} "
              f"| fold별 a={[round(x,2) for x in aa]}{flag}")
    print(f"  {'평균':13s} RMSE={np.mean(cv_r):.4f} Spearman={np.mean(cv_s):.4f}")

    gap_s = np.mean(fit_s) - np.mean(cv_s)
    print(f"\n  과적합 폭(Spearman): {gap_s:+.4f}"
          f"{'  <- 크다. [3]의 값을 믿지 말 것' if gap_s > 0.05 else '  <- 허용 범위'}")

    print("\n[5] server.py에 넣을 CALIB")
    print(json.dumps({d: {"a": round(best[d][0], 4), "b": round(best[d][1], 4)}
                      for d in DIMS}, indent=4, ensure_ascii=False))

    # ---- 프롬프트 A/B ----
    if len(sys.argv) > 2:
        other = sys.argv[2]
        d2, n2 = load(other)
        print(f"\n\n=== 프롬프트 A/B: {path} vs {other} ===")
        print("raw Spearman으로 비교한다. 반올림 후 수치는 a,b에 좌우되므로 프롬프트 판정에 쓸 수 없다.\n")
        print(f"  {'영역':13s} {'v2(분석)':>10s} {'v1base':>10s} {'차이':>9s}")
        for d in DIMS:
            r1, g1 = data[d]; r2, g2 = d2[d]
            s1 = spearmanr(r1, g1).statistic
            s2 = spearmanr(r2, g2).statistic
            mark = "개선" if s1 - s2 > 0.02 else ("악화" if s2 - s1 > 0.02 else "차이없음")
            print(f"  {d:13s} {s1:+10.4f} {s2:+10.4f} {s1-s2:+9.4f}  {mark}")
        n_eff = min(len(data['organization'][0]), len(d2['organization'][0]))
        se = 1.0 / np.sqrt(max(n_eff - 3, 1))
        print(f"\n  n={n_eff} 기준 Spearman 표준오차 ≈ {se:.3f}. "
              f"차이가 {2*se:.2f} 미만이면 우연과 구별되지 않는다.")


if __name__ == "__main__":
    main()
