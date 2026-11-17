"""Sheet music (MusicXML / PDF)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from xml.sax.saxutils import escape

import numpy as np

from . import tempo
from .detect import Note

DIV = 4  # 16th notes
BAR = 4 * DIV
TYPES = {1: ("16th", 0), 2: ("eighth", 0), 3: ("eighth", 1), 4: ("quarter", 0),
         6: ("quarter", 1), 8: ("half", 0), 12: ("half", 1), 16: ("whole", 0)}
SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_fifths(notes: list[Note]) -> int:
    h = np.zeros(12)
    for n in notes:
        h[n.pitch % 12] += n.end - n.start
    best, tonic = -2.0, 0
    for prof, shift in ((MAJOR, 0), (MINOR, 3)):
        for k in range(12):
            r = np.corrcoef(h, np.roll(prof, k))[0, 1]
            if r > best:
                best, tonic = r, (k + shift) % 12
    f = (tonic * 7) % 12
    return f - 12 if f > 6 else f


def is_swing(phase: np.ndarray) -> bool:
    swung = np.sum((phase > 0.6) & (phase < 0.72))
    straight = np.sum((phase > 0.44) & (phase < 0.56))
    return swung > 2 * max(straight, 5)


def _straighten(pos: np.ndarray) -> np.ndarray:
    b, f = np.floor(pos), pos % 1
    return b + np.interp(f, [0, 2 / 3, 1], [0, 0.5, 1])


def quantize(notes: list[Note], beats: np.ndarray, swing: bool | None):
    s = tempo.to_beats(np.array([n.start for n in notes]), beats)
    e = tempo.to_beats(np.array([n.end for n in notes]), beats)
    if swing is None:
        swing = is_swing(s % 1)
    if swing:
        s, e = _straighten(s), _straighten(e)
    q = np.round(s * DIV).astype(int)
    qe = np.maximum(q + 1, np.round(e * DIV).astype(int))
    # downbeat: the beat phase with the most bass onsets
    w = np.array([2.0 if n.track == 1 else 1.0 for n in notes]) * (q % DIV == 0)
    k = int(np.argmax([w[(q // DIV) % 4 == j].sum() for j in range(4)]))
    shift = -k * DIV
    first = (q.min() + shift) // BAR * BAR
    shift -= first
    return q + shift, qe + shift, swing


def _pieces(p: int, d: int):
    """Split [p, p+d) into notatable values that respect beats."""
    while d > 0:
        for v in (16, 12, 8, 6, 4, 3, 2, 1):
            if v > d or (p % BAR) + v > BAR:
                continue
            m = p % BAR
            if v == 16 and m:
                continue
            if v in (8, 12) and m % 8 and not (v == 12 and m == 4):
                continue
            if v in (3, 4, 6) and m % 4:
                continue
            if v == 2 and m % 2:
                continue
            yield p, v
            p += v
            d -= v
            break


def _stream(events):
    """[(q, qe, pitch)] -> [(pos, dur, [pitches])] as a chord sequence."""
    by = {}
    for q, qe, pitch in events:
        by.setdefault(q, []).append((qe, pitch))
    starts = sorted(by)
    out = []
    for i, q in enumerate(starts):
        end = max(qe for qe, _ in by[q])
        if i + 1 < len(starts):
            nxt = starts[i + 1]
            end = max(end, q + 2)
            if nxt - end <= max(2, (end - q) // 2):  # close small gaps
                end = nxt
            end = min(end, nxt)
        out.append((q, end - q, sorted({p for _, p in by[q]})))
    return out


def _pitch_xml(p: int, flats: bool) -> str:
    name = (FLAT if flats else SHARP)[p % 12]
    alter = {"#": 1, "b": -1}.get(name[1:], 0)
    alt = f"<alter>{alter}</alter>" if alter else ""
    return f"<pitch><step>{name[0]}</step>{alt}<octave>{p // 12 - 1}</octave></pitch>"


def musicxml(notes: list[Note], beats: np.ndarray, title: str = "",
             swing: bool | None = None) -> tuple[str, bool]:
    q, qe, swing = quantize(notes, beats, swing)
    split_by_track = len({n.track for n in notes}) > 1
    staves = {1: [], 2: []}
    for n, a, b in zip(notes, q, qe):
        st = (2 if n.track == 1 else 1) if split_by_track else (1 if n.pitch >= 60 else 2)
        staves[st].append((int(a), int(b), n.pitch))
    nbars = int(max(qe) // BAR + 1)
    fifths = key_fifths(notes)
    flats = fifths < 0
    bpm = int(round(60 / np.median(np.diff(beats))))

    measures = {st: [[] for _ in range(nbars)] for st in staves}
    for st, ev in staves.items():
        pos = 0
        for q0, d, pitches in _stream(ev) + [(nbars * BAR, 0, [])]:
            if q0 > pos:  # rest
                for p, v in _pieces(pos, q0 - pos):
                    measures[st][p // BAR].append((v, None, False, False))
            parts = list(_pieces(q0, d))
            for i, (p, v) in enumerate(parts):
                measures[st][p // BAR].append((v, pitches, i > 0, i < len(parts) - 1))
            pos = max(pos, q0 + d)

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">',
           '<score-partwise version="4.0">',
           f"<work><work-title>{escape(title)}</work-title></work>",
           '<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>',
           '<part id="P1">']
    for m in range(nbars):
        out.append(f'<measure number="{m + 1}">')
        if m == 0:
            out.append(f"<attributes><divisions>{DIV}</divisions>"
                       f"<key><fifths>{fifths}</fifths></key>"
                       "<time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>"
                       '<clef number="1"><sign>G</sign><line>2</line></clef>'
                       '<clef number="2"><sign>F</sign><line>4</line></clef></attributes>')
            words = "<direction-type><words>Swing</words></direction-type>" if swing else ""
            out.append(f'<direction placement="above">{words}<direction-type><metronome>'
                       f"<beat-unit>quarter</beat-unit><per-minute>{bpm}</per-minute>"
                       f'</metronome></direction-type><staff>1</staff><sound tempo="{bpm}"/>'
                       "</direction>")
        for st in (1, 2):
            if st == 2:
                out.append(f"<backup><duration>{BAR}</duration></backup>")
            voice = 1 if st == 1 else 5
            evs = measures[st][m]
            if all(p is None for _, p, _, _ in evs):
                out.append(f'<note><rest measure="yes"/><duration>{BAR}</duration>'
                           f"<voice>{voice}</voice><staff>{st}</staff></note>")
                continue
            for v, pitches, tie_stop, tie_start in evs:
                typ, dot = TYPES[v]
                tail = (f"<voice>{voice}</voice><type>{typ}</type>{'<dot/>' * dot}"
                        f"<staff>{st}</staff>")
                if pitches is None:
                    out.append(f"<note><rest/><duration>{v}</duration>{tail}</note>")
                    continue
                for i, p in enumerate(pitches):
                    ties = (('<tie type="stop"/>' if tie_stop else "")
                            + ('<tie type="start"/>' if tie_start else ""))
                    tied = (('<tied type="stop"/>' if tie_stop else "")
                            + ('<tied type="start"/>' if tie_start else ""))
                    out.append("<note>" + ("<chord/>" if i else "") + _pitch_xml(p, flats)
                               + f"<duration>{v}</duration>{ties}{tail}"
                               + (f"<notations>{tied}</notations>" if tied else "") + "</note>")
        out.append("</measure>")
    out += ["</part>", "</score-partwise>"]
    return "\n".join(out), swing


def _musescore() -> str | None:
    for b in ("mscore", "musescore", "mscore4", "MuseScore4", "musescore4", "mscore3"):
        if shutil.which(b):
            return b
    return None


def write(path: str, notes: list[Note], beats: np.ndarray, title: str = "",
          swing: bool | None = None) -> str:
    """Write .musicxml, or .pdf via MuseScore. Returns a status message."""
    xml, swing = musicxml(notes, beats, title, swing)
    stem, ext = os.path.splitext(path)
    if ext.lower() in (".musicxml", ".xml"):
        open(path, "w").write(xml)
        return f"sheet music -> {path}"
    mscore = _musescore()
    if not mscore:
        open(stem + ".musicxml", "w").write(xml)
        return (f"MuseScore not found, wrote {stem}.musicxml instead "
                "(install musescore for PDF output)")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "score.musicxml")
        open(src, "w").write(xml)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        r = subprocess.run([mscore, "-o", os.path.abspath(path), src], env=env,
                           capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(path):
        open(stem + ".musicxml", "w").write(xml)
        return f"MuseScore failed, wrote {stem}.musicxml instead"
    return f"sheet music -> {path}" + (" (swing)" if swing else "")
