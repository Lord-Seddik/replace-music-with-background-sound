"""T14 throwaway: list suspect background moments from background_scan.py features.

Usage: python experiments/background_find.py FEATURES.npz
"""
import numpy as np, sys
f = np.load(sys.argv[1])
g = lambda k: f[k]
L = {n: g(n + "_level") for n in ("mix","speech","music","effects","out","residual")}
B = {n: g(n + "_bands") for n in ("mix","speech","music","effects","out","residual")}
FL = {n: g(n + "_flat") for n in ("mix","speech","music","effects")}
T = len(L["mix"])
def sec(i): s = i / 10; return f"{int(s//60)}:{s%60:04.1f}"
def runs(mask, min_len):
    m = np.concatenate([[0], mask.astype(int), [0]]); d = np.diff(m)
    return [(a, b) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= min_len]
def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")
def merge(rs, gap=10):
    out = []
    for a, b in rs:
        if out and a - out[-1][1] <= gap: out[-1] = (out[-1][0], b)
        else: out.append((a, b))
    return out
def show(title, rs, score):
    print(f"\n== {title}: {len(rs)} runs, {sum(b-a for a,b in rs)/10:.0f} s")
    for a, b in sorted(rs, key=lambda r: -score(*r))[:12]:
        print(f"  {sec(a)}-{sec(b)} ({(b-a)/10:4.1f}s) score {score(a,b):6.1f}  mix {L['mix'][a:b].mean():6.1f} sp {np.median(L['speech'][a:b]):6.1f}"
              f" mu {L['music'][a:b].mean():6.1f} fx {L['effects'][a:b].mean():6.1f} out {L['out'][a:b].mean():6.1f} res {L['residual'][a:b].mean():6.1f}"
              f" muflat {np.median(FL['music'][a:b]):.3f}")

# B: noise-like sound in the music stem (wind, rain, crowd, hits) -> removed with the music
noisy = (smooth(FL["music"], 10) > 0.12) & (L["music"] > -45) & (L["music"] > L["effects"] + 6)
show("B noise-like sound in music stem", merge(runs(noisy, 5)), lambda a, b: (b - a) / 10 * (L["music"][a:b].mean() + 60))

# D: stems do not add up to the original (sound lost entirely)
lost = smooth(L["residual"] - L["mix"], 10) > -22
show("D residual close to original level", merge(runs(lost & (L["mix"] > -50), 5)), lambda a, b: (L["residual"][a:b] - L["mix"][a:b]).mean() + 40)

# A: steady background split between stems: music+effects steady, effects alone jumping
def db_sum(a, b): return 10 * np.log10(10 ** (a / 10) + 10 ** (b / 10))
W = 20
flut = np.zeros(T, bool); fscore = np.zeros(T)
for k in range(1, 6):
    comb = db_sum(B["music"][:, k], B["effects"][:, k]); e = B["effects"][:, k]
    from numpy.lib.stride_tricks import sliding_window_view as sw
    cs = sw(comb, W).std(axis=1); es = sw(e, W).std(axis=1); em = sw(e, W).mean(axis=1); cm = sw(comb, W).mean(axis=1)
    m = (cs < 2.5) & (es > 6) & (em > cm - 20) & (em > -80)
    idx = np.where(m)[0] + W // 2
    flut[idx] = True; fscore[idx] = np.maximum(fscore[idx], (es - cs)[m])
show("A effects flutter under steady music+effects", merge(runs(flut, 5)), lambda a, b: fscore[a:b].mean() * (b - a) / 10)

# C: effects stem tracking the music (music leaking into effects -> ghost music in output)
from numpy.lib.stride_tricks import sliding_window_view as sw
W = 50
mu = B["music"][:, 1:6].mean(axis=1); fx = B["effects"][:, 1:6].mean(axis=1)
mw, fw = sw(mu, W), sw(fx, W)
r = np.array([np.corrcoef(a, b)[0, 1] if a.std() > 1 and b.std() > 1 else 0 for a, b in zip(mw, fw)])
rr = np.zeros(T); rr[W // 2: W // 2 + len(r)] = r
leak = (rr > 0.85) & (L["music"] > -45) & (L["effects"] > L["music"] - 30) & (FL["effects"] < 0.02)
show("C effects follows music (leak)", merge(runs(leak, 10)), lambda a, b: (b - a) / 10 * (L["effects"][a:b].mean() - L["music"][a:b].mean() + 40))
