"""MP3 merge, silence-aware splitting and silence detection (ffmpeg + PyDub)."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import warnings

import imageio_ffmpeg

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # pydub warns about ffmpeg not on PATH; we set it below
    from pydub import AudioSegment
    from pydub.silence import detect_silence

SAMPLE_RATE = 16_000
BITRATE = "64k"
DIRECT_UPLOAD_LIMIT = 25_000_000  # OpenAI accepts files up to 25 MB
PART_LIMIT = 20_000_000  # split target from the OpenAI "longer inputs" guide
SEARCH_WINDOW_SEC = 120.0  # look for a pause in the last 2 minutes before the limit
SILENT_MAX_DBFS = -50.0

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = FFMPEG
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AudioError(Exception):
    pass


@dataclass
class Part:
    path: Path
    start: float
    end: float


def run_ffmpeg(args: list[str], *, input: bytes | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", *args] if input is None else [FFMPEG, "-hide_banner", *args],
        input=input,
        capture_output=True,
        creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise AudioError("ffmpeg 失败：" + " | ".join(tail))
    return proc


def encoder_args(output: Path) -> list[str]:
    return ["-c:a", "libmp3lame", "-b:a", BITRATE, "-f", "mp3", "-y", str(output)]


def encode_pcm(pcm: bytes, output: Path) -> None:
    """Encode 16 kHz mono s16le PCM into an MP3 file (used by tests and tools)."""
    run_ffmpeg(["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0", *encoder_args(output)], input=pcm)


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def duration(path: Path) -> float:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-i", str(path)],
        capture_output=True,
        creationflags=_NO_WINDOW,
    )
    match = _DURATION.search(proc.stderr.decode("utf-8", "replace"))
    if not match:
        raise AudioError(f"无法读取音频时长：{path.name}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


_MAX_VOLUME = re.compile(r"max_volume:\s*(-?inf|-?\d+(?:\.\d+)?) dB")


def max_dbfs(path: Path, start: float | None = None, end: float | None = None) -> float:
    """Peak level in dBFS, computed by ffmpeg's volumedetect (streams, low memory)."""
    args = ["-nostdin"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if start is not None and end is not None:
        args += ["-t", f"{end - start:.3f}"]
    args += ["-af", "volumedetect", "-f", "null", "-"]
    proc = subprocess.run([FFMPEG, "-hide_banner", *args], capture_output=True, creationflags=_NO_WINDOW)
    match = _MAX_VOLUME.search(proc.stderr.decode("utf-8", "replace"))
    if not match:
        return float("-inf")
    value = match.group(1)
    return float("-inf") if "inf" in value else float(value)


def is_silent(path: Path, start: float | None = None, end: float | None = None) -> bool:
    return max_dbfs(path, start, end) < SILENT_MAX_DBFS


def merge_chunks(chunks: list[Path], output: Path) -> float:
    """Concatenate MP3 chunks without re-encoding; delete chunks once verified."""
    if not chunks:
        raise AudioError("没有可合并的录音分块。")
    expected = sum(duration(c) for c in chunks)
    tmp = output.with_name(output.stem + ".merging.mp3")
    if len(chunks) == 1:
        shutil.copyfile(chunks[0], tmp)
    else:
        listing = output.with_name(output.stem + ".concat.txt")
        lines = []
        for chunk in chunks:
            escaped = chunk.resolve().as_posix().replace("'", r"'\''")
            lines.append(f"file '{escaped}'")
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-f", "mp3", "-y", str(tmp)])
        finally:
            listing.unlink(missing_ok=True)
    actual = duration(tmp)
    if abs(actual - expected) > max(1.0, 0.05 * len(chunks)):
        tmp.unlink(missing_ok=True)
        raise AudioError(f"合并校验失败：分块合计 {expected:.1f}s，合并后 {actual:.1f}s，分块已保留。")
    tmp.replace(output)
    for chunk in chunks:
        chunk.unlink(missing_ok=True)
    return actual


def load_segment(path: Path, start: float, length: float) -> AudioSegment:
    """Decode only [start, start+length) into a PyDub segment."""
    proc = run_ffmpeg(
        ["-ss", f"{start:.3f}", "-i", str(path), "-t", f"{length:.3f}",
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    )
    return AudioSegment(data=proc.stdout, sample_width=2, frame_rate=SAMPLE_RATE, channels=1)


def find_cut(segment: AudioSegment) -> int:
    """Best cut position (ms) inside the segment: the middle of a pause.

    Prefers the latest pause of at least 1.5 s, otherwise the longest pause,
    otherwise the quietest 50 ms frame, so sentences are not split mid-word.
    """
    length = len(segment)
    if length == 0:
        return 0
    if segment.dBFS == float("-inf"):
        return length
    pauses = detect_silence(segment, min_silence_len=500, silence_thresh=segment.dBFS - 16, seek_step=10)
    if pauses:
        start, end = max(pauses, key=lambda p: (min(p[1] - p[0], 1500), p[0]))
        return (start + end) // 2
    frame = 50
    best_pos, best_rms = length, None
    for pos in range(0, length - frame + 1, frame):
        rms = segment[pos:pos + frame].rms
        if best_rms is None or rms <= best_rms:
            best_pos, best_rms = pos + frame // 2, rms
    return best_pos


def cut_part(source: Path, start: float, end: float | None, output: Path) -> None:
    args = ["-ss", f"{start:.3f}", "-i", str(source)]
    if end is not None:
        args += ["-t", f"{end - start:.3f}"]
    run_ffmpeg([*args, "-c", "copy", "-f", "mp3", "-y", str(output)])


def plan_uploads(
    source: Path,
    parts_dir: Path,
    *,
    direct_limit: int = DIRECT_UPLOAD_LIMIT,
    part_limit: int = PART_LIMIT,
) -> list[Part]:
    """Upload the whole file when it is under 25 MB, otherwise split it into ~20 MB parts at pauses."""
    total = duration(source)
    size = source.stat().st_size
    if size < direct_limit:
        return [Part(source, 0.0, total)]

    bytes_per_sec = size / max(total, 0.001)
    max_part_sec = part_limit / bytes_per_sec * 0.97
    parts_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    for old in parts_dir.glob(f"{stem}-*.mp3"):
        old.unlink()

    parts: list[Part] = []
    start = 0.0
    while start < total - 0.05:
        index = len(parts) + 1
        output = parts_dir / f"{stem}-{index:03d}.mp3"
        if total - start <= max_part_sec:
            cut_part(source, start, None, output)
            parts.append(Part(output, start, total))
            break
        limit = start + max_part_sec
        window_start = max(start + max_part_sec / 2, limit - SEARCH_WINDOW_SEC)
        segment = load_segment(source, window_start, limit - window_start)
        cut = window_start + find_cut(segment) / 1000.0
        if cut <= start + 1.0:
            cut = limit
        cut_part(source, start, cut, output)
        if output.stat().st_size > part_limit:
            raise AudioError(f"分段 {output.name} 超过 {part_limit // 1_000_000}MB。")
        parts.append(Part(output, start, cut))
        start = cut
    return parts
