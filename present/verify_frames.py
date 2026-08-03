"""Pixel verification for presentation media (plan.md "Verification rule").

Decode a frame from an mp4 as raw RGB via ffmpeg's rawvideo decoder and
assert that the palette colours from `animate_budget.py` actually span the
expected regions. Pure standard library — no numpy, no PIL. The layout
constants come from `animate_budget.py` so the check can never drift from
what the renderer draws.

Usage:
    python present/verify_frames.py                      # check budget_reel.mp4
    python present/verify_frames.py --mp4 out.mp4 --spec budget
    python present/verify_frames.py --mp4 out.mp4 --dump frame.png   # eyeball
"""

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from animate_budget import (  # noqa: E402  (palette + layout are the spec)
    BUDGET_LINE_BOX,
    CAPTION_BAND,
    LANE_STRIPS,
    PALETTE,
)


# ---------------------------------------------------------------------------
# ffmpeg plumbing
# ---------------------------------------------------------------------------
def probe_streams(mp4_path):
    """Return (width, height, duration_sec, fps) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "default=noprint_wrappers=1", mp4_path],
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    info = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        info[key] = value
    fps_num, _, fps_den = info["r_frame_rate"].partition("/")
    fps = float(fps_num) / (float(fps_den) if float(fps_den) else 1.0)
    duration = float(info["duration"])
    return int(info["width"]), int(info["height"]), duration, fps


def decode_frame(mp4_path, seconds):
    """Return (width, height, rgb bytes) of the frame at `seconds`."""
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{seconds}", "-i", mp4_path,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )
    w, h, nb, fps = probe_streams(mp4_path)
    expected = w * h * 3
    if len(res.stdout) != expected:
        raise AssertionError(f"decoded {len(res.stdout)} bytes, expected {expected}")
    return w, h, res.stdout


def decode_png(mp4_path, seconds, png_path):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{seconds}", "-i", mp4_path,
         "-frames:v", "1", png_path],
        capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# pixel counting (pure python, region-scoped)
# ---------------------------------------------------------------------------
def count_in_region(data, w, h, x0, y0, x1, y1, hex_colour, tol=12):
    """Count pixels within `tol` (per channel) of `hex_colour` in the box.

    Layout coords (from animate_budget) are matplotlib-style: y grows UP.
    Rawvideo rows are top-down, so flip the y band here.
    """
    rgb = tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    x0, x1 = max(x0, 0), min(x1, w)
    y0, y1 = max(y0, 0), min(y1, h)
    r_top, r_bot = h - y1, h - y0
    count = 0
    for y in range(r_top, r_bot):
        row = y * w * 3 + x0 * 3
        for x in range(x0, x1):
            r = data[row]
            g = data[row + 1]
            b = data[row + 2]
            if abs(r - rgb[0]) <= tol and abs(g - rgb[1]) <= tol and abs(b - rgb[2]) <= tol:
                count += 1
            row += 3
    return count


# ---------------------------------------------------------------------------
# check registry: one function per spec
# ---------------------------------------------------------------------------
def _check_canvas(w, h, expected):
    assert (w, h) == expected, f"canvas {w}x{h} != expected {expected[0]}x{expected[1]}"


def _check_budget(mp4_path):
    w, h, duration, fps = probe_streams(mp4_path)
    _check_canvas(w, h, (1080, 1080))
    assert duration > 0, "empty video"
    end_t = max(duration - 0.7, duration * 0.9)

    start = decode_frame(mp4_path, 0.05)
    end = decode_frame(mp4_path, end_t)

    band_area = (CAPTION_BAND[2] - CAPTION_BAND[0]) * (CAPTION_BAND[3] - CAPTION_BAND[1])
    band = count_in_region(end[2], w, h, *CAPTION_BAND, PALETTE["CAPTION_BG"])
    assert band >= band_area * 0.5, f"caption band too sparse: {band}/{band_area}"

    counts_end, counts_start = {}, {}
    for lane, (x0, y0, x1, y1) in LANE_STRIPS.items():
        colour = {"python-naive": "LANE_PY", "c-blocked": "LANE_C",
                  "asm-blocked": "LANE_ASM"}[lane]
        counts_end[lane] = count_in_region(end[2], w, h, x0, y0, x1, y1, PALETTE[colour])
        counts_start[lane] = count_in_region(start[2], w, h, x0, y0, x1, y1, PALETTE[colour])
        assert counts_end[lane] >= 1000, f"{lane}: lane colour missing at end ({counts_end[lane]})"
        assert counts_end[lane] > counts_start[lane], (
            f"{lane}: no motion (end {counts_end[lane]} <= start {counts_start[lane]})")

    assert counts_end["asm-blocked"] >= 5 * counts_end["python-naive"], (
        "asm bar should dwarf python bar: "
        f"asm {counts_end['asm-blocked']} vs py {counts_end['python-naive']}")

    line = count_in_region(end[2], w, h, *BUDGET_LINE_BOX, PALETTE["ACCENT"])
    assert line >= 50, f"5 s budget line not visible ({line} px)"

    text = count_in_region(end[2], w, h, *CAPTION_BAND, PALETTE["TEXT"])
    assert text >= 500, f"headline text missing from caption band ({text} px)"
    print("budget spec OK: canvas, caption band, 3 lanes (fill + motion + ratio), "
          "budget line, headline text")
    print(f"lane pixel counts at end: "
          f"python-naive={counts_end['python-naive']} c-blocked={counts_end['c-blocked']} "
          f"asm-blocked={counts_end['asm-blocked']}")


SPECS = {"budget": _check_budget}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp4", default=os.path.join("present", "budget_reel.mp4"))
    parser.add_argument("--spec", choices=sorted(SPECS), default="budget")
    parser.add_argument("--dump", type=float, default=None,
                        help="also decode a frame at this video-time (s) to a PNG next to the mp4")
    args = parser.parse_args()

    if args.dump is not None:
        png = os.path.splitext(args.mp4)[0] + f"_t{args.dump}.png"
        decode_png(args.mp4, args.dump, png)
        print(f"dumped frame at {args.dump}s -> {png}")

    SPECS[args.spec](args.mp4)


if __name__ == "__main__":
    main()
