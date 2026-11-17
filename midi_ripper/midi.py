"""MIDI file writer."""
from __future__ import annotations

import struct

import numpy as np

from .detect import Note

PPQ = 480


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _track(events: list[tuple[int, bytes]]) -> bytes:
    events.sort(key=lambda e: e[0])
    data, now = b"", 0
    for tick, msg in events:
        data += _vlq(tick - now) + msg
        now = tick
    data += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + data


def _name(s: str) -> bytes:
    b = s.encode()
    return b"\xff\x03" + _vlq(len(b)) + b


def write(path: str, notes: list[Note], bpm: float = 120.0, ntracks: int = 1,
          velocity: int = 80, quantize: int = 0, program: int = 0) -> None:
    tps = PPQ * bpm / 60.0
    shift = 0.0
    if quantize and notes:
        # align grid phase to onsets
        g = PPQ / quantize
        on = np.array([n.start for n in notes]) * tps
        phases = np.linspace(0, g, 64, endpoint=False)
        err = [np.abs((on - ph) / g - np.round((on - ph) / g)).sum() for ph in phases]
        shift = -phases[int(np.argmin(err))]

    def tick(t: float) -> int:
        x = t * tps + shift
        if quantize:
            g = PPQ / quantize
            return max(0, int(round(x / g) * g))
        return max(0, int(round(x)))

    tempo = int(round(60_000_000 / bpm))
    meta = [(0, b"\xff\x51\x03" + tempo.to_bytes(3, "big")),
            (0, b"\xff\x58\x04\x04\x02\x18\x08")]
    tracks = [_track(meta)]
    names = ["Right hand", "Left hand"] if ntracks == 2 else ["Piano"]
    for tr in range(ntracks):
        ch = tr
        ev: list[tuple[int, bytes]] = [(0, _name(names[tr])),
                                       (0, bytes([0xC0 | ch, program]))]
        for n in notes:
            if n.track != tr:
                continue
            a = tick(n.start)
            b = max(tick(n.end), a + (int(PPQ / quantize) if quantize else 1))
            ev.append((a, bytes([0x90 | ch, n.pitch, velocity])))
            ev.append((b, bytes([0x80 | ch, n.pitch, 0])))
        ev.sort(key=lambda e: (e[0], 0 if e[1][0] & 0xF0 == 0x80 else 1))
        tracks.append(_track(ev))
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), PPQ))
        for t in tracks:
            f.write(t)
