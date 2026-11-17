"""Beat tracking."""
from __future__ import annotations

import numpy as np

SR = 100  # envelope samples per second


def _envelope(onsets: np.ndarray) -> np.ndarray:
    n = int(onsets.max() * SR) + SR * 2
    env = np.zeros(n)
    np.add.at(env, np.round(onsets * SR).astype(int), 1.0)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    return np.convolve(env, k, mode="same")


def _period(env: np.ndarray, bpm_hint: float | None) -> float:
    lo, hi = int(SR * 60 / 240), int(SR * 60 / 40)
    e = env - env.mean()
    ac = np.array([np.dot(e[:-l], e[l:]) for l in range(lo, hi)])
    bpm = 60 * SR / np.arange(lo, hi)
    prior = np.exp(-0.5 * (np.log2(bpm / (bpm_hint or 120)) / (0.25 if bpm_hint else 1.0)) ** 2)
    return float(np.arange(lo, hi)[np.argmax(ac * prior)])


def track_beats(onsets: list[float], bpm_hint: float | None = None,
                tightness: float = 100.0) -> np.ndarray:
    """Beat times (s) following tempo drift, covering all onsets."""
    on = np.asarray(sorted(onsets))
    env = _envelope(on)
    p = _period(env, bpm_hint)
    n = len(env)
    score = env.copy()
    back = np.full(n, -1)
    offs = np.arange(-int(2 * p), -int(p / 2) + 1)
    pen = -tightness * np.log(-offs / p) ** 2
    for t in range(int(p / 2), n):
        prev = t + offs
        ok = prev >= 0
        if not ok.any():
            continue
        cand = score[prev[ok]] + pen[ok]
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[t] += cand[j]
            back[t] = prev[ok][j]
    # backtrack from the best beat near the end
    tail = np.arange(max(0, n - int(2 * p)), n)
    t = int(tail[np.argmax(score[tail])])
    beats = [t]
    while back[t] >= 0:
        t = int(back[t])
        beats.append(t)
    b = np.array(beats[::-1], float) / SR
    # extend to cover everything
    while b[0] > on[0] - 0.05:
        b = np.r_[b[0] - (b[1] - b[0]), b]
    while b[-1] < on[-1] + 0.05:
        b = np.r_[b, b[-1] + (b[-1] - b[-2])]
    return b


def to_beats(t: np.ndarray | float, beats: np.ndarray) -> np.ndarray:
    """Seconds -> beat position (linear between beats, extrapolated outside)."""
    idx = np.arange(len(beats), dtype=float)
    t = np.asarray(t, float)
    out = np.interp(t, beats, idx)
    lo, hi = t < beats[0], t > beats[-1]
    out = np.where(lo, (t - beats[0]) / (beats[1] - beats[0]), out)
    out = np.where(hi, idx[-1] + (t - beats[-1]) / (beats[-1] - beats[-2]), out)
    return out
