"""midi-ripper: turn a 'falling notes' piano video into a MIDI file."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from . import capture, detect, keyboard, midi


def _crop(s: str) -> capture.Crop:
    try:
        x, y, w, h = (int(v) for v in s.replace("x", ",").replace(" ", ",").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    return x, y, w, h


def _safe_name(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "", title).strip(" .")
    return name[:120] or "out"


def _save_png(path: str, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-i", "-", path], input=img.tobytes(), check=True)


def reference_frame(path: str, crop, start: float, end: float | None) -> np.ndarray:
    """Median frame: the keyboard with nothing pressed."""
    W, H, dur = capture.probe(path)
    w, h = (crop[2], crop[3]) if crop else (W, H)
    span = (end or dur or 60) - start
    n = int(np.clip(300e6 / (w * h * 3), 15, 200))
    fps = max(n / max(span, 1), 0.05)
    stack = [f for _, f in capture.frames(path, crop, fps, start, end)][:n]
    if not stack:
        sys.exit("error: could not read any frames")
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def analyze(path: str, a: argparse.Namespace) -> None:
    crop = a.crop
    print("building reference frame...")
    ref = reference_frame(path, crop, a.start, a.end)
    lowest = keyboard.parse_note(a.lowest_note) if a.lowest_note else None
    try:
        kb = keyboard.detect(ref, lowest, a.white_keys)
    except RuntimeError as e:
        if a.debug:
            _save_png(a.debug, ref)
        sys.exit(f"error: {e}")
    print("keyboard:", kb.describe())
    if a.debug:
        _save_png(a.debug, keyboard.debug_image(ref, kb))
        print("calibration image:", a.debug)

    sampler = detect.Sampler(kb)
    _, _, dur = capture.probe(path)
    total = (a.end or dur) - a.start

    def progress(t):
        if int(t * a.fps) % int(a.fps) == 0:
            print(f"\rscanning {t - a.start:6.1f}/{total:.1f}s", end="", flush=True)

    ts, cols = detect.sample_all(sampler, capture.frames(path, crop, a.fps, a.start, a.end),
                                 progress)
    print()
    if len(ts) == 0:
        sys.exit("error: no frames decoded")
    base = np.median(cols, axis=0)
    notes = detect.find_notes(ts, cols, base, sampler.pitches, sampler.black_mask, on=a.on, off=a.off,
                              min_len=a.min_len, release=a.release, rearm=a.rearm, dip=a.dip)
    if not notes:
        sys.exit("no notes detected (try a lower --on threshold, check --debug image)")
    ntracks = 1 if a.no_split else detect.split_hands(notes)
    if a.trim:
        t0 = notes[0].start - a.lead_in
        for n in notes:
            n.start -= t0
            n.end -= t0
    midi.write(a.output, notes, bpm=a.bpm, ntracks=ntracks, velocity=a.velocity,
               quantize=a.quantize)
    lo, hi = min(n.pitch for n in notes), max(n.pitch for n in notes)
    print(f"{len(notes)} notes, {keyboard.note_name(lo)}..{keyboard.note_name(hi)}, "
          f"{ntracks} track(s), {notes[-1].end:.1f}s -> {a.output}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="midi-ripper", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("output")
    g.add_argument("-o", "--output",
                   help="MIDI file to write (default: named after the video, or out.mid)")
    g.add_argument("--bpm", type=float, default=120, help="tempo written to the file (default 120)")
    g.add_argument("--quantize", type=int, default=0, metavar="N",
                   help="snap to N divisions per beat at --bpm (4 = 16ths); default off")
    g.add_argument("--velocity", type=int, default=80)
    g.add_argument("--no-split", action="store_true", help="don't split hands by key colour")
    g.add_argument("--no-trim", dest="trim", action="store_false",
                   help="keep leading silence (default: first note at --lead-in)")
    g.add_argument("--lead-in", type=float, default=0.5, help="seconds before the first note")
    g = common.add_argument_group("keyboard")
    g.add_argument("--lowest-note", help="pitch of the leftmost key, e.g. A0, C2 (default: guess)")
    g.add_argument("--white-keys", type=int, help="force N equally spaced white keys")
    g.add_argument("--debug", metavar="PNG", help="write a calibration image showing detected keys")
    g = common.add_argument_group("detection")
    g.add_argument("--fps", type=float, default=60, help="analysis frame rate (default 60)")
    g.add_argument("--on", type=float, default=60, help="colour distance to count as pressed")
    g.add_argument("--off", type=float, default=35, help="colour distance to count as released")
    g.add_argument("--release", type=float, default=0.6,
                   help="also release when distance < this x the note's peak (default 0.6)")
    g.add_argument("--rearm", type=float, default=20,
                   help="rise needed after a release before a new press counts (default 20)")
    g.add_argument("--dip", type=float, default=0.12,
                   help="split a held note when it dims by this fraction and recovers "
                        "(repeated presses); 0 disables (default 0.12)")
    g.add_argument("--min-len", type=float, default=0.0, help="drop notes shorter than this (s)")
    g.add_argument("--start", type=float, default=0.0, help="start time in the video (s)")
    g.add_argument("--end", type=float, help="end time in the video (s)")

    r = sub.add_parser("record", parents=[common],
                       help="select a screen area with the mouse, record it, convert to MIDI")
    r.add_argument("--duration", type=float, help="stop after N seconds (default: Ctrl+C)")
    r.add_argument("--keep", metavar="FILE",
                   help="keep the recording (e.g. rec.mkv) to re-run 'video' on it later")
    r.add_argument("--region", type=_crop, help="x,y,w,h instead of selecting with the mouse")

    v = sub.add_parser("video", parents=[common], help="convert a video file to MIDI")
    v.add_argument("file")
    v.add_argument("--crop", type=_crop,
                   help="x,y,w,h of the keyboard in the video (default: auto-detect)")

    u = sub.add_parser("url", parents=[common],
                       help="download a YouTube (or any yt-dlp supported) video and convert it")
    u.add_argument("url")
    u.add_argument("--crop", type=_crop,
                   help="x,y,w,h of the keyboard in the video (default: auto-detect)")
    u.add_argument("--keep-video", metavar="FILE", help="also save the downloaded video here")

    a = p.parse_args(argv)
    if a.cmd == "video":
        a.output = a.output or os.path.splitext(os.path.basename(a.file))[0] + ".mid"
        analyze(a.file, a)
        return
    if a.cmd == "url":
        with tempfile.TemporaryDirectory() as d:
            path, title = capture.download(a.url, d)
            a.output = a.output or _safe_name(title) + ".mid"
            if a.keep_video:
                shutil.copyfile(path, a.keep_video)
                print("video saved:", a.keep_video)
            analyze(path, a)
        return
    a.output = a.output or "out.mid"

    a.crop = None
    region = a.region or capture.select_region()
    if a.keep:
        capture.record(region, a.keep, a.duration)
        analyze(a.keep, a)
    else:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rec.mkv")
            capture.record(region, path, a.duration)
            analyze(path, a)


if __name__ == "__main__":
    main()
