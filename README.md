# midi-ripper

Convert falling-notes piano videos into MIDI.

## Install

Requires `ffmpeg`, plus `yt-dlp` for links, `musescore` for PDF sheet music, and `slurp` + `wf-recorder` for screen recording (wlroots Wayland only).

```sh
uv tool install --editable .
```

## Usage

From a YouTube link:

```sh
midi-ripper url 'https://www.youtube.com/watch?v=...'
```

From a video file:

```sh
midi-ripper video song.mp4
```

Sheet music too (works with any command):

```sh
midi-ripper url 'https://www.youtube.com/watch?v=...' --pdf song.pdf
```

From the screen (select the keyboard, play the video, Ctrl+C to stop):

```sh
midi-ripper record -o song.mid
```

## Options

| Option | |
|---|---|
| `-o FILE` | output file |
| `--pdf FILE` | sheet music (`.pdf`, or `.musicxml` to edit in MuseScore) |
| `--swing on/off` | override swing detection for sheet music |
| `--sheet-bpm N` | approximate tempo, if the sheet music's is off by 2x |
| `--debug calib.png` | save an image of the detected keys |
| `--lowest-note A0` | fix the octave if it's wrong |
| `--white-keys N` | force N white keys if detection fails |
| `--crop x,y,w,h` | keyboard area in the video |
| `--start` / `--end` | only use part of the video (seconds) |
| `--on N` | lower to catch faint notes (default 60) |
| `--dip N` | lower if repeated notes merge (default 0.12) |
| `--bpm N --quantize 4` | snap to 16ths (steady-tempo songs only) |
| `--no-split` | single track instead of left/right hands |
| `--keep-video FILE` | keep the downloaded video (`url`) |
| `--keep FILE` | keep the recording (`record`) |

Run `midi-ripper <command> --help` for all options.
