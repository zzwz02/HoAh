"""Microphone / system-loopback recording into 5-minute MP3 chunks per track."""

from __future__ import annotations

import ctypes
import json
import subprocess
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc  # must be imported on the main thread: it initialises COM at import time

from . import audio

CHUNK_SECONDS = 5 * 60
BLOCK_FRAMES = 1600  # 100 ms at 16 kHz
TRACKS_FOR_SOURCE = {"microphone": ["microphone"], "system": ["system"], "both": ["microphone", "system"]}


class RecordingError(Exception):
    pass


class Mp3ChunkWriter:
    """Streams 16 kHz mono PCM into ffmpeg, rotating to a new MP3 every chunk_seconds."""

    def __init__(self, directory: Path, name: str, chunk_seconds: int = CHUNK_SECONDS):
        self.directory = directory
        self.name = name
        self.chunk_samples = chunk_seconds * audio.SAMPLE_RATE
        self.index = 0
        self.samples_in_chunk = 0
        self.total_samples = 0
        self.paths: list[Path] = []
        self._proc: subprocess.Popen | None = None
        self._part: Path | None = None

    def _open(self) -> None:
        self._part = self.directory / f"{self.name}-{self.index:04d}.mp3.part"
        self._proc = subprocess.Popen(
            [audio.FFMPEG, "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", str(audio.SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
             *audio.encoder_args(self._part)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.samples_in_chunk = 0

    def _close(self) -> None:
        proc, part = self._proc, self._part
        self._proc = self._part = None
        if proc is None or part is None:
            return
        _, err = proc.communicate(timeout=120)
        if self.samples_in_chunk == 0:
            part.unlink(missing_ok=True)
        elif proc.returncode != 0:
            raise RecordingError(f"MP3 编码失败：{err.decode('utf-8', 'replace').strip()[-300:]}")
        else:
            final = part.with_suffix("")  # drop ".part"
            part.replace(final)
            self.paths.append(final)
        self.index += 1

    def write(self, pcm: np.ndarray) -> None:
        """pcm: int16 mono samples."""
        offset = 0
        while offset < len(pcm):
            if self._proc is None:
                self._open()
            take = min(len(pcm) - offset, self.chunk_samples - self.samples_in_chunk)
            self._proc.stdin.write(pcm[offset:offset + take].tobytes())
            self.samples_in_chunk += take
            self.total_samples += take
            offset += take
            if self.samples_in_chunk >= self.chunk_samples:
                self._close()

    def finish(self) -> list[Path]:
        self._close()
        return self.paths


def to_int16_mono(block: np.ndarray) -> np.ndarray:
    mono = block.mean(axis=1) if block.ndim == 2 else block
    return (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)


def _co_initialize() -> None:
    try:
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED; ignore "already initialised"
    except (AttributeError, OSError):
        pass


def _open_device(track: str):
    if track == "microphone":
        device = sc.default_microphone()
    else:
        device = sc.get_microphone(str(sc.default_speaker().name), include_loopback=True)
    return device


class Recorder:
    """One capture thread per track; each feeds its own Mp3ChunkWriter."""

    def __init__(self, session_dir: Path, source: str):
        self.session_dir = session_dir
        self.source = source
        self.tracks = TRACKS_FOR_SOURCE[source]
        self.chunks_dir = session_dir / "chunks"
        self.writers = {t: Mp3ChunkWriter(self.chunks_dir, t) for t in self.tracks}
        self.levels = {t: 0.0 for t in self.tracks}
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._ready: dict[str, threading.Event] = {t: threading.Event() for t in self.tracks}
        self.started_at = 0.0

    @staticmethod
    def new_session(source: str) -> "Recorder":
        from .config import sessions_dir

        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = sessions_dir() / name
        suffix = 1
        while session.exists():
            suffix += 1
            session = sessions_dir() / f"{name}-{suffix}"
        session.mkdir(parents=True)
        return Recorder(session, source)

    def start(self) -> None:
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        write_meta(self.session_dir, {
            "source": self.source,
            "tracks": self.tracks,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "recording",
        })
        for track in self.tracks:
            thread = threading.Thread(target=self._capture, args=(track,), name=f"capture-{track}", daemon=True)
            self._threads.append(thread)
            thread.start()
        for track in self.tracks:
            self._ready[track].wait(timeout=8)
        if self.errors:
            self.stop()
            raise RecordingError("；".join(self.errors))

    def _capture(self, track: str) -> None:
        _co_initialize()
        writer = self.writers[track]
        try:
            device = _open_device(track)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with device.recorder(samplerate=audio.SAMPLE_RATE, blocksize=BLOCK_FRAMES) as rec:
                    self._ready[track].set()
                    while not self._stop.is_set():
                        block = rec.record(numframes=BLOCK_FRAMES)
                        pcm = to_int16_mono(block)
                        peak = int(np.abs(pcm).max()) if len(pcm) else 0
                        self.levels[track] = peak / 32767
                        writer.write(pcm)
        except Exception as exc:  # device unplugged, access denied, encoder died...
            label = "麦克风" if track == "microphone" else "系统声音"
            self.errors.append(f"{label}录音失败：{exc}")
        finally:
            self._ready[track].set()

    @property
    def elapsed(self) -> float:
        return max((w.total_samples for w in self.writers.values()), default=0) / audio.SAMPLE_RATE

    def level(self) -> float:
        return max(self.levels.values(), default=0.0)

    def stop(self) -> dict[str, list[Path]]:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=10)
        result: dict[str, list[Path]] = {}
        for track, writer in self.writers.items():
            try:
                result[track] = writer.finish()
            except Exception as exc:
                self.errors.append(str(exc))
                result[track] = writer.paths
        meta = read_meta(self.session_dir)
        meta.update({"status": "recorded", "duration_sec": round(self.elapsed, 1)})
        if self.errors:
            meta["recording_errors"] = self.errors
        write_meta(self.session_dir, meta)
        return result


# --- session metadata & crash repair ----------------------------------------

def read_meta(session_dir: Path) -> dict:
    try:
        return json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_meta(session_dir: Path, meta: dict) -> None:
    path = session_dir / "session.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def repair_incomplete(sessions_root: Path) -> list[str]:
    """Turn chunks left as .mp3.part by a crash into playable .mp3 files."""
    repaired = []
    for part in sessions_root.glob("*/chunks/*.mp3.part"):
        final = part.with_suffix("")
        try:
            audio.run_ffmpeg(["-i", str(part), "-c", "copy", "-f", "mp3", "-y", str(final)])
            part.unlink()
            repaired.append(str(final))
        except audio.AudioError:
            final.unlink(missing_ok=True)
            if part.stat().st_size == 0:
                part.unlink()
    for session in sessions_root.iterdir():
        meta = read_meta(session) if session.is_dir() else {}
        if meta.get("status") in ("recording", "preparing", "transcribing", "polishing"):
            meta["status"] = "recorded"
            meta["interrupted"] = True
            write_meta(session, meta)
    return repaired
