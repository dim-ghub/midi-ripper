"""Key detection."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WHITE_SEMIS = [0, 2, 4, 5, 7, 9, 11]
HAS_BLACK_RIGHT = [1, 1, 0, 1, 1, 1, 0]


def note_name(p: int) -> str:
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


def parse_note(s: str) -> int:
    s = s.strip()
    if s.lstrip("-").isdigit():
        return int(s)
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", s)
    if not m:
        raise ValueError(f"bad note name: {s!r}")
    semi = WHITE_SEMIS["CDEFGAB".index(m[1].upper())] + {"#": 1, "b": -1, "": 0}[m[2]]
    return semi + 12 * (int(m[3]) + 1)


@dataclass
class Key:
    pitch: int
    black: bool
    x0: int  # sample patch
    x1: int
    y0: int
    y1: int
    left: int = 0  # key extent
    right: int = 0


@dataclass
class Keyboard:
    keys: list[Key]
    black_rows: tuple[int, int]
    white_rows: tuple[int, int]

    def describe(self) -> str:
        lo = min(k.pitch for k in self.keys)
        hi = max(k.pitch for k in self.keys)
        nw = sum(not k.black for k in self.keys)
        return (f"{len(self.keys)} keys ({nw} white, {len(self.keys) - nw} black), "
                f"{note_name(lo)}..{note_name(hi)}")


def _otsu(g: np.ndarray) -> float:
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return float(np.nanargmax(var))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _find_rows(g: np.ndarray, thr: float) -> tuple[tuple[int, int], tuple[int, int]]:
    dark = g < thr
    frac = dark.mean(axis=1)
    bright = g.mean(axis=1)
    # black-key rows alternate dark/bright
    nruns = np.array([len(_runs(r)) for r in dark])
    W = g.shape[1]
    min_runs = max(5, W // 200)
    black_mask = (frac > 0.15) & (frac < 0.7) & (nruns >= min_runs)
    runs = _runs(black_mask)
    if not runs:
        raise RuntimeError("could not find the black keys; select the keyboard more tightly")
    b0, b1 = max(runs, key=lambda r: r[1] - r[0])
    white_mask = (frac < 0.15) & (bright > thr)
    white_mask[:b1] = False
    wruns = _runs(white_mask)
    if not wruns:
        raise RuntimeError("could not find white-key area below the black keys")
    w0, w1 = max(wruns, key=lambda r: r[1] - r[0])
    return (int(b0), int(b1)), (int(w0), int(w1))


def _white_boundaries(profile: np.ndarray, black_centers: np.ndarray,
                      black_w: float) -> np.ndarray:
    W = len(profile)
    # adjacent black keys in a group are ~1 white key apart
    if len(black_centers) >= 3:
        gaps = np.diff(black_centers)
        ww = float(np.median(gaps[gaps < np.median(gaps) * 1.2]))
    else:
        ww = black_w / 0.6
    base = np.array([np.max(profile[max(0, i - int(ww / 2)):i + int(ww / 2) + 1])
                     for i in range(W)])
    depth = base - profile
    cand = [i for i in range(1, W - 1)
            if depth[i] >= depth[i - 1] and depth[i] > depth[i + 1] and depth[i] > 4]
    cand.sort(key=lambda i: -depth[i])
    chosen: list[int] = []
    for i in cand:
        if all(abs(i - j) > ww * 0.5 for j in chosen):
            chosen.append(i)
    chosen.sort()
    bounds = [0] + chosen + [W]
    # fill in missed separators
    out = [float(bounds[0])]
    for b in bounds[1:]:
        a = out[-1]
        n = max(1, round((b - a) / ww))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    out_a = np.array(out, float)
    # drop partial keys at the edges
    keep = [out_a[0]]
    for b in out_a[1:]:
        if b - keep[-1] < ww * 0.45 and b != out_a[-1]:
            continue
        keep.append(b)
    if len(keep) > 2 and keep[-1] - keep[-2] < ww * 0.45:
        keep.pop(-2)
    return np.array(keep)


def detect(ref: np.ndarray, lowest: int | None = None, white_keys: int | None = None) -> Keyboard:
    g = ref.astype(np.float32).mean(axis=2)
    H, W = g.shape
    thr = _otsu(g)
    (b0, b1), (w0, w1) = _find_rows(g, thr)

    bh = b1 - b0
    band = g[b0 + bh // 4: b1 - bh // 4].mean(axis=0)
    runs = _runs(band < thr)
    widths = np.array([e - s for s, e in runs])
    if len(widths) == 0:
        raise RuntimeError("no black keys found")
    mw = np.median(widths)
    blacks = [(s, e) for s, e in runs if 0.5 * mw <= e - s <= 1.8 * mw and s > 0 and e < W]
    bc = np.array([(s + e) / 2 for s, e in blacks])

    wprof = g[w0:w1].mean(axis=0)
    if white_keys:
        bounds = np.linspace(0, W, white_keys + 1)
    else:
        bounds = _white_boundaries(wprof, bc, mw)
    nwhite = len(bounds) - 1

    has_black = np.zeros(nwhite, int)
    black_owner: list[int] = []
    for c in bc:
        i = int(np.argmin(np.abs(bounds[1:-1] - c))) if nwhite > 1 else 0
        has_black[i] = 1
        black_owner.append(i)

    # match the 2-3 black key pattern to find C
    best, best_off = -1, 0
    for off in range(7):
        pat = np.array([HAS_BLACK_RIGHT[(off + i) % 7] for i in range(nwhite)])
        score = int((pat[:-1] == has_black[:-1]).sum())
        if score > best:
            best, best_off = score, off

    rel = []
    for i in range(nwhite):
        letter = (best_off + i) % 7
        octave = (best_off + i) // 7
        rel.append(12 * octave + WHITE_SEMIS[letter])
    rel = np.array(rel)

    if lowest is not None:
        shift = lowest - rel[0]
    elif nwhite == 52 and best_off == 5:  # 88 keys
        shift = 21 - rel[0]
    else:
        mid = (rel[0] + rel[-1]) / 2
        shift = 12 * round((64.5 - mid) / 12)
    keys: list[Key] = []
    wy0 = w0 + (w1 - w0) // 4
    wy1 = w1 - (w1 - w0) // 4
    for i in range(nwhite):
        l, r = bounds[i], bounds[i + 1]
        c, half = (l + r) / 2, (r - l) * 0.25
        keys.append(Key(int(rel[i] + shift), False, int(c - half), int(c + half) + 1,
                        wy0, max(wy1, wy0 + 1), int(l), int(r)))
    by0 = b0 + bh // 5
    by1 = b1 - bh // 5
    for (s, e), owner in zip(blacks, black_owner):
        c, half = (s + e) / 2, (e - s) * 0.25
        p = int(rel[owner] + shift + 1)
        if any(k.pitch == p for k in keys):
            continue
        keys.append(Key(p, True, int(c - half), int(c + half) + 1, by0, by1, int(s), int(e)))
    keys.sort(key=lambda k: k.pitch)
    if best < (nwhite - 1) * 0.9:
        print(f"warning: black-key pattern only matched {best}/{nwhite - 1} white keys; "
              "pitches may be off (try --white-keys / --lowest-note)")
    return Keyboard(keys, (b0, b1), (w0, w1))


def debug_image(ref: np.ndarray, kb: Keyboard) -> np.ndarray:
    img = ref.astype(np.uint8).copy()

    def rect(x0, y0, x1, y1, col):
        x1, y1 = max(x1 - 1, x0), max(y1 - 1, y0)
        img[y0:y1 + 1, [x0, x1]] = col
        img[[y0, y1], x0:x1 + 1] = col

    for k in kb.keys:
        if k.pitch % 12 == 0:
            img[-6:, k.left:k.right] = (255, 200, 0)
            if k.pitch == 60:
                img[-12:, k.left:k.right] = (255, 0, 255)
        rect(k.x0, k.y0, k.x1, k.y1, (0, 120, 255) if k.black else (255, 0, 0))
    return img
