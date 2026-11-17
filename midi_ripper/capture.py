"""Screen capture, download and decoding."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterator

import numpy as np

Crop = tuple[int, int, int, int]  # x, y, w, h


def _need(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"error: '{tool}' not found in PATH")


def select_region() -> Crop:
    _need("slurp")
    out = subprocess.run(["slurp", "-f", "%x %y %w %h"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("selection cancelled")
    x, y, w, h = map(int, out.stdout.split())
    return x, y, w, h


def record(region: Crop, path: str, duration: float | None = None, fps: int = 60) -> None:
    _need("wf-recorder")
    x, y, w, h = region
    cmd = [
        "wf-recorder", "-y", "-g", f"{x},{y} {w}x{h}", "-r", str(fps),
        "-c", "libx264", "-p", "preset=ultrafast", "-p", "crf=14", "-f", path,
    ]
    errlog = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errlog,
                            start_new_session=True)
    print(f"recording {w}x{h}+{x}+{y} -> {path}")
    print("play the video now; press Ctrl+C to stop" if duration is None
          else f"recording for {duration:.0f}s (Ctrl+C stops early)")
    try:
        if duration is None:
            proc.wait()
        else:
            proc.wait(timeout=duration)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(0.2)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        errlog.seek(0)
        msg = errlog.read().decode(errors="replace").strip().splitlines()[-3:]
        sys.exit("error: wf-recorder produced no video:\n  " + "\n  ".join(msg))
    print("recording stopped")


def download(url: str, folder: str) -> tuple[str, str]:
    _need("yt-dlp")
    cmd = ["yt-dlp", "--no-playlist", "--no-simulate", "--quiet", "--no-warnings",
           "--progress", "--newline",
           "-f", "bv*[height<=1080]/b[height<=1080]/bv*/b", "-S", "res,fps",
           "-o", os.path.join(folder, "video.%(ext)s"),
           "--print", "after_move:title", "--print", "after_move:filepath", url]
    print("downloading video...")
    out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if out.returncode != 0 or len(lines) < 2 or not os.path.exists(lines[-1]):
        sys.exit("error: download failed (check the link; 'yt-dlp -U' updates yt-dlp)")
    return lines[-1], lines[-2]


def probe(path: str) -> tuple[int, int, float]:
    _need("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    return int(s["width"]), int(s["height"]), dur


def frames(path: str, crop: Crop | None, fps: float, start: float = 0.0,
           end: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    _need("ffmpeg")
    W, H, _ = probe(path)
    if crop:
        x, y, w, h = crop
        w, h = min(w, W - x), min(h, H - y)
        vf = f"crop={w}:{h}:{x}:{y},fps={fps}"
    else:
        w, h = W, H
        vf = f"fps={fps}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if end is not None:
        cmd += ["-t", str(end - start)]
    cmd += ["-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start + i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
