"""Note detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .keyboard import Keyboard


@dataclass
class Note:
    pitch: int
    start: float
    end: float
    color: tuple[float, float, float]
    track: int = 0


class Sampler:
    def __init__(self, kb: Keyboard):
        self.kb = kb
        self.white = [k for k in kb.keys if not k.black]
        self.black = [k for k in kb.keys if k.black]
        order = self.white + self.black
        self.pitches = np.array([k.pitch for k in order])
        self.black_mask = np.array([k.black for k in order])
        self._wy = (self.white[0].y0, self.white[0].y1) if self.white else (0, 1)
        self._by = (self.black[0].y0, self.black[0].y1) if self.black else (0, 1)
        self._wx = np.array([(k.x0, k.x1) for k in self.white]).reshape(-1, 2)
        self._bx = np.array([(k.x0, k.x1) for k in self.black]).reshape(-1, 2)

    @staticmethod
    def _band(frame, ys, xs):
        if len(xs) == 0:
            return np.zeros((0, 3))
        row = frame[ys[0]:ys[1]].astype(np.float32).mean(axis=0)
        cs = np.vstack([np.zeros((1, 3), np.float32), np.cumsum(row, axis=0)])
        return (cs[xs[:, 1]] - cs[xs[:, 0]]) / (xs[:, 1] - xs[:, 0])[:, None]

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return np.vstack([self._band(frame, self._wy, self._wx),
                          self._band(frame, self._by, self._bx)])


def sample_all(sampler: Sampler, frames: Iterable[tuple[float, np.ndarray]],
               progress=None) -> tuple[np.ndarray, np.ndarray]:
    ts, cols = [], []
    for t, f in frames:
        ts.append(t)
        cols.append(sampler(f))
        if progress:
            progress(t)
    return np.array(ts), np.array(cols, np.float32)


def find_notes(ts: np.ndarray, cols: np.ndarray, base: np.ndarray, pitches: np.ndarray,
               black: np.ndarray | None = None, on: float = 60.0, off: float = 35.0, min_len: float = 0.0,
               release: float = 0.6, rearm: float = 20.0, dip: float = 0.12) -> list[Note]:
    dist = np.linalg.norm(cols - base[None], axis=2)
    dist[occluded_frames(dist, on, black)] = 0.0
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 60
    notes: list[Note] = []
    T, K = dist.shape

    def emit(k: int, s: int, e: int, peak: float) -> None:
        d = dist[:, k]
        while e - 1 > s and d[e - 1] < 0.9 * peak:
            e -= 1
        if (e - s) * dt >= min_len:
            c = np.median(cols[s:e, k], axis=0)
            notes.append(Note(int(pitches[k]), float(ts[s]), float(ts[e - 1] + dt),
                              tuple(map(float, c))))

    for k in range(K):
        d = dist[:, k]
        start, peak, floor, dip_at = None, 0.0, None, None
        for i in range(T):
            x = d[i]
            if start is None:
                if floor is not None:
                    floor = min(floor, x)
                    if x < off or x > floor + rearm:
                        floor = None
                if floor is None and x > on:
                    start, peak, dip_at = i, x, None
            elif x < max(off, release * peak):
                emit(k, start, i, peak)
                start, floor = None, x  # wait for a rise before re-triggering
            elif dip and x < (1 - dip) * peak:  # possible re-press
                if dip_at is None:
                    dip_at = i
            elif dip_at is not None and x > (1 - dip / 3) * peak:
                lo = dip_at + int(np.argmin(d[dip_at:i]))
                emit(k, start, lo, peak)
                start, peak, dip_at = lo + 1, x, None
            elif dip_at is None:
                peak = max(peak, x)
        if start is not None:
            emit(k, start, T, peak)
    notes = drop_grey_notes(notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def occluded_frames(dist: np.ndarray, on: float, black: np.ndarray | None = None) -> np.ndarray:
    """Frames covered by an overlay."""
    lit = dist > on
    bad = (np.median(dist, axis=1) > on / 4) | (lit.mean(axis=1) > 0.4)
    if black is not None and black.any() and (~black).any():
        bad |= (lit[:, black].mean(axis=1) > 0.4) | (lit[:, ~black].mean(axis=1) > 0.4)
    return bad


def drop_grey_notes(notes: list[Note]) -> list[Note]:
    """Grey notes in a coloured video are overlays."""
    if len(notes) < 10:
        return notes
    _, sat = _hue(np.array([n.color for n in notes]))
    if np.median(sat) < 0.25:
        return notes
    return [n for n, s in zip(notes, sat) if s >= 0.08]


def _hue(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.clip(rgb / 255.0, 0, 1)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    a = r - 0.5 * (g + b)
    bb = (np.sqrt(3) / 2) * (g - b)
    mag = np.hypot(a, bb)
    vec = np.stack([a, bb], 1) / np.maximum(mag, 1e-6)[:, None]
    return vec, mag


def split_hands(notes: list[Note], min_sep_deg: float = 35.0) -> int:
    """Cluster by hue into right/left hand. Returns track count."""
    if len(notes) < 4:
        return 1
    vec, w = _hue(np.array([n.color for n in notes]))
    if np.median(w) < 0.08:
        return 1
    i0 = int(np.argmax(w))
    i1 = int(np.argmax(np.where(w > 0.08, -(vec @ vec[i0]), -np.inf)))
    cent = vec[[i0, i1]].copy()
    for _ in range(20):
        lab = np.argmax(vec @ cent.T, axis=1)
        for j in range(2):
            m = (vec[lab == j] * w[lab == j, None]).sum(0)
            if np.linalg.norm(m) > 0:
                cent[j] = m / np.linalg.norm(m)
    sep = np.degrees(np.arccos(np.clip(cent[0] @ cent[1], -1, 1)))
    if sep < min_sep_deg or min(np.sum(lab == 0), np.sum(lab == 1)) < 2:
        return 1
    mean_p = [np.mean([n.pitch for n, l in zip(notes, lab) if l == j]) for j in range(2)]
    right = int(np.argmax(mean_p))
    for n, l in zip(notes, lab):
        n.track = 0 if l == right else 1
    return 2
