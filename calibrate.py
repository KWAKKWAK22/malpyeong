"""격자 탐색 캘리브레이션 + 프롬프트 효과 진단 (고속판).

사용:
    python3 calibrate.py raw_v2full.jsonl
    python3 calibrate.py raw_v2.jsonl raw_v1base.jsonl   # raw 기준 A/B


핵심: p = round(clip(a*raw+b,1,5)) 는 raw 위의 4개 경계로 완전히 결정된다.
raw를 한 번 정렬해 두면 각 (a,b)는 searchsorted 4번 + 누적합 조회로 O(1)에 끝난다.
"""
import json, sys
import numpy as np
from scipy.stats import rankdata, spearmanr

DIMS = ["content", "organization", "expression"]
MIN_SPREAD = 0.6


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = {}
    for d in DIMS:
        raw = np.array([r["raw"].get(d, np.nan) for r in rows], float)
        gold = np.array([r["gold"][d] for r in rows], float)
        m = ~np.isnan(raw)
        out[d] = (raw[m], gold[m])
    return out, len(rows)


class Fast:
    """정렬 + 누적합 사전계산. 이후 (a,b) 평가는 상수 시간."""
    def __init__(self, raw, gold):
        o = np.argsort(raw, kind="mergesort")
        self.rs = raw[o]; self.gs = gold[o]
        self.rk = rankdata(gold)[o]              # 정답의 타이 보정 순위
        self.n = len(raw)
        z = lambda v: np.concatenate([[0.0], np.cumsum(v)])
        self.c_rk = z(self.rk); self.c_g = z(self.gs); self.c_g2 = z(self.gs ** 2)
        self.rk_mean = self.rk.mean()
        self.rk_sd = self.rk.std()
        self.g_sd = gold.std()

    def eval(self, a, b):
        if a <= 0:
            return None
        # a*raw+b 가 k+0.5 를 넘는 지점 = 예측이 k+1로 바뀌는 raw 경계
        t = [(k + 0.5 - b) / a for k in range(1, 5)]
        idx = np.searchsorted(self.rs, t, side="left")
        idx = np.clip(idx, 0, self.n)
        bounds = [0] + list(idx) + [self.n]
        n_k, sum_rk, sum_g, sum_g2, lv = [], [], [], [], []
        for k in range(5):
            i, j = bounds[k], bounds[k + 1]
            if j <= i:
                continue
            n_k.append(j - i); lv.append(k + 1.0)
            sum_rk.append(self.c_rk[j] - self.c_rk[i])
            sum_g.append(self.c_g[j] - self.c_g[i])
            sum_g2.append(self.c_g2[j] - self.c_g2[i])
        if len(n_k) < 3:
            return None
        n_k = np.array(n_k, float); lv = np.array(lv)
        sum_rk = np.array(sum_rk); sum_g = np.array(sum_g); sum_g2 = np.array(sum_g2)

        # RMSE
        sse = (lv ** 2 * n_k - 2 * lv * sum_g + sum_g2).sum()
        rmse = float(np.sqrt(sse / self.n))

        # 예측 분포의 표준편차 (변별력 붕괴 방지 제약)
        mu = (lv * n_k).sum() / self.n
        sd = float(np.sqrt((lv ** 2 * n_k).sum() / self.n - mu ** 2))
        if sd < MIN_SPREAD * self.g_sd:
            return None

        # Spearman = 예측의 타이 보정 순위와 정답 순위의 Pearson
        cum = np.concatenate([[0.0], np.cumsum(n_k)])
        ar = cum[:-1] + (n_k + 1) / 2.0
        p_mean = (ar * n_k).sum() / self.n
        p_sd = np.sqrt((ar ** 2 * n_k).sum() / self.n - p_mean ** 2)
        if p_sd <= 0:
            return None
        cov = (ar * sum_rk).sum() / self.n - p_mean * self.rk_mean
        sp = float(cov / (p_sd * self.rk_sd))
        return rmse, sp, sd, [int(x) for x in lv]


def obj(rmse, sp):
    return 0.5 * sp + 0.5 * (1 - min(rmse, 1.2) / 1.2)


A = np.arange(0.30, 3.51, 0.05)
B = np.arange(-6.0, 3.01, 0.05)


def grid(raw, gold):
    f = Fast(raw, gold); best = None
    for a in A:
        for b in B:
            r = f.eval(a, b)
            if r is None:
                continue
            s = obj(r[0], r[1])
            if best is None or s > best[0]:
                best = (s, float(a), float(b), r[0], r[1], r[3])
    return best


def cv(raw, gold, k=5, seed=42):
    idx = np.random.default_rng(seed).permutation(len(raw))
    F = np.array_split(idx, k); R, S, P = [], [], []
    for i in range(k):
        te = F[i]; tr = np.concatenate([F[j] for j in range(k) if j != i])
        bb = grid(raw[tr], gold[tr])
        if bb is None:
            continue
        _, a, b, *_ = bb
        p = np.round(np.clip(a * raw[te] + b, 1, 5))
        R.append(np.sqrt(((p - gold[te]) ** 2).mean()))
        S.append(spearmanr(p, gold[te]).statistic if len(np.unique(p)) > 1 else 0.0)
        P.append((round(a, 2), round(b, 2)))
    return float(np.mean(R)), float(np.mean(S)), P


def main():
    path = sys.argv[1]
    data, n = load(path)
    print(f"=== {path} (n={n}) ===\n")

    print("[1] raw 신호 (캘리브레이션 무관)")
    for d in DIMS:
        raw, gold = data[d]
        print(f"  {d:13s} Spearman={spearmanr(raw, gold).statistic:+.4f} "
              f"| raw 평균={raw.mean():.3f} sd={raw.std():.3f} "
              f"| 정답 평균={gold.mean():.3f} sd={gold.std():.3f}")

    CUR = {"content": (0.95, -0.40), "organization": (1.15, -0.95),
           "expression": (1.10, -0.90)}
    OLD = {"content": (0.5115, 1.1582), "organization": (0.7812, 0.1981),
           "expression": (0.8437, 0.2960)}
    for tag, C in (("[2] 1회차 계수", OLD), ("[3] 현재 계수 (100건에서 뽑은 값)", CUR)):
        print(f"\n{tag}")
        rs, ss = [], []
        for d in DIMS:
            raw, gold = data[d]; a, b = C[d]
            p = np.round(np.clip(a * raw + b, 1, 5))
            rm = np.sqrt(((p - gold) ** 2).mean())
            sp = spearmanr(p, gold).statistic if len(np.unique(p)) > 1 else 0.0
            rs.append(rm); ss.append(sp)
            print(f"  {d:13s} RMSE={rm:.4f} Spearman={sp:.4f} "
                  f"| 예측평균={p.mean():.3f} 출력값={sorted(int(u) for u in np.unique(p))}")
        print(f"  {'평균':13s} RMSE={np.mean(rs):.4f} Spearman={np.mean(ss):.4f}")

    print("\n[4] 400건 격자 탐색")
    best = {}
    rs, ss = [], []
    for d in DIMS:
        raw, gold = data[d]
        _, a, b, rm, sp, lv = grid(raw, gold)
        best[d] = (round(a, 4), round(b, 4)); rs.append(rm); ss.append(sp)
        p = np.round(np.clip(a * raw + b, 1, 5))
        print(f"  {d:13s} a={a:5.2f} b={b:6.2f} -> RMSE={rm:.4f} Spearman={sp:.4f} "
              f"| 예측평균={p.mean():.3f} 출력값={lv}")
    print(f"  {'평균':13s} RMSE={np.mean(rs):.4f} Spearman={np.mean(ss):.4f}")

    print("\n[5] 5-fold 교차검증 (처음 보는 데이터 기준)")
    cr, cs = [], []
    for d in DIMS:
        raw, gold = data[d]
        rm, sp, picks = cv(raw, gold)
        cr.append(rm); cs.append(sp)
        aa = [p[0] for p in picks]
        flag = "  <- fold간 a 편차 큼" if max(aa) - min(aa) > 0.6 else ""
        print(f"  {d:13s} RMSE={rm:.4f} Spearman={sp:.4f} | fold별 a={aa}{flag}")
    print(f"  {'평균':13s} RMSE={np.mean(cr):.4f} Spearman={np.mean(cs):.4f}")
    print(f"\n  과적합 폭(Spearman): {np.mean(ss)-np.mean(cs):+.4f}")

    print("\n[6] server.py CALIB")
    print(json.dumps({d: {"a": best[d][0], "b": best[d][1]} for d in DIMS},
                     indent=4, ensure_ascii=False))



def ab_compare(p1, p2):
    d1, _ = load(p1); d2, _ = load(p2)
    print(f"\n\n=== 프롬프트 A/B: {p1} vs {p2} ===")
    print("raw Spearman으로 비교한다. 반올림 후 수치는 a,b에 좌우되어 프롬프트 판정에 쓸 수 없다.\n")
    print(f"  {'영역':13s} {'A':>10s} {'B':>10s} {'차이':>9s}")
    for d in DIMS:
        r1, g1 = d1[d]; r2, g2 = d2[d]
        s1 = spearmanr(r1, g1).statistic; s2 = spearmanr(r2, g2).statistic
        mark = "개선" if s1 - s2 > 0.02 else ("악화" if s2 - s1 > 0.02 else "차이없음")
        print(f"  {d:13s} {s1:+10.4f} {s2:+10.4f} {s1-s2:+9.4f}  {mark}")
    n = min(len(d1['content'][0]), len(d2['content'][0]))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    print(f"\n  n={n} 기준 표준오차 ≈ {se:.3f}. 차이가 {2*se:.2f} 미만이면 우연과 구별되지 않는다.")



main()
if len(sys.argv) > 2:
    ab_compare(sys.argv[1], sys.argv[2])
