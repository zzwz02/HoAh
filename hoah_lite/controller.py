"""Recording / transcription state shared by the main window, floating windows and tray."""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from . import config
from .config import Settings
from .recorder import Recorder, RecordingError, read_meta, write_meta
from .transcriber import describe_parts, polish_only, prepare_session, reset_state, run_session


class Controller(QObject):
    recording_changed = Signal(bool)
    tick = Signal(float, float)  # elapsed seconds, level 0..1
    source_changed = Signal(str)
    # {"type": clear|rejected|start|stage|progress|delta|polished_delta|prepared|done|error, ...}
    # stage/progress events carry "phase": "transcribe" | "polish"
    transcribe_event = Signal(object)  # plain Python dict; object avoids QVariantMap conversion
    sessions_changed = Signal(object)  # session Path to select, or None
    notice = Signal(str, str)  # title, message

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.recorder: Recorder | None = None
        self.worker: threading.Thread | None = None
        self.worker_session: Path | None = None
        self.live_text: dict[str, str] = {}  # streamed raw transcription per track
        self.live_polished: dict[str, str] = {}  # polished paragraphs finished so far per track
        self.phase = "transcribe"
        self.progress_text = ""
        self.progress_pct = 0.0
        self.last_error = ""
        self.mode = "transcribe"
        self._pending_prepare: list[Path] = []
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._on_tick)
        self.transcribe_event.connect(self._track_event)

    # --- state -------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self.recorder is not None

    @property
    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def save_settings(self) -> None:
        self.settings.save()

    def set_source(self, source: str) -> None:
        if self.is_recording or source not in config.SOURCES or source == self.settings.source:
            return
        self.settings.source = source
        self.save_settings()
        self.source_changed.emit(source)

    def cycle_source(self) -> None:
        order = list(config.SOURCES)
        self.set_source(order[(order.index(self.settings.source) + 1) % len(order)])

    # --- recording ---------------------------------------------------------------

    def toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.is_recording:
            return
        recorder = Recorder.new_session(self.settings.source)
        try:
            recorder.start()
        except RecordingError as exc:
            self.notice.emit("无法开始录音", str(exc))
            self.sessions_changed.emit(None)
            return
        self.recorder = recorder
        self._timer.start()
        self.recording_changed.emit(True)
        self.sessions_changed.emit(recorder.session_dir)

    def stop_recording(self, *, prepare: bool = True) -> None:
        """Stop and save; then merge + split in the background. Transcription is started by the user."""
        recorder = self.recorder
        if recorder is None:
            return
        self.recorder = None
        self._timer.stop()
        recorder.stop()
        self.recording_changed.emit(False)
        self.tick.emit(0.0, 0.0)
        if recorder.errors:
            self.notice.emit("录音过程中出现问题", "\n".join(recorder.errors))
        self.sessions_changed.emit(recorder.session_dir)
        if prepare:
            self._pending_prepare.append(recorder.session_dir)
            self._run_pending()

    def _run_pending(self) -> None:
        if not self._pending_prepare:
            return
        if self.is_busy:
            QTimer.singleShot(200, self._run_pending)
            return
        session = self._pending_prepare.pop(0)
        self.transcribe_event.emit({"type": "clear"})
        self._start_worker(session, "prepare", None)

    def _on_tick(self) -> None:
        recorder = self.recorder
        if recorder is None:
            return
        self.tick.emit(recorder.elapsed, recorder.level())
        if recorder.errors and all(not t.is_alive() for t in recorder._threads):
            self.stop_recording()

    # --- transcription -------------------------------------------------------------

    def start_transcription(self, session: Path | None, redo: bool = False) -> None:
        # Every submission starts by clearing the previous error everywhere.
        self.transcribe_event.emit({"type": "clear"})
        error = self._check_can_transcribe(session)
        if error:
            self.transcribe_event.emit({"type": "rejected", "text": error})
            return
        if redo:
            reset_state(session)
        self._start_worker(session, "transcribe", config.get_api_key()[0])

    def start_polish(self, session: Path | None, redo: bool = False) -> None:
        self.transcribe_event.emit({"type": "clear"})
        error = self._check_can_transcribe(session)
        if not error and not (session / "transcript.md").exists():
            error = "这个会话还没有转写，请先点击“转写 / 继续”。"
        if error:
            self.transcribe_event.emit({"type": "rejected", "text": error})
            return
        self._start_worker(session, "polish", config.get_api_key()[0], redo=redo)

    def _start_worker(self, session: Path, mode: str, key: str | None, redo: bool = False) -> None:
        """mode "prepare": merge + split only; "transcribe": prepare if needed, then call gpt-transcribe;
        "polish": gpt-6-luna strict polish of an existing transcript."""
        meta = read_meta(session)
        meta.pop("last_error", None)
        meta["status"] = {"prepare": "preparing", "transcribe": "transcribing", "polish": "polishing"}[mode]
        write_meta(session, meta)

        self.mode = mode
        self.live_text = {}
        self.live_polished = {}
        self.worker_session = session
        settings = Settings.load()
        emit = self.transcribe_event.emit  # Qt queues cross-thread emissions to the GUI thread

        def work() -> None:
            try:
                if mode == "prepare":
                    _, state = prepare_session(session, emit)
                    m = read_meta(session)
                    m["status"] = "recorded"
                    write_meta(session, m)
                    emit({"type": "prepared", "session": session, "text": describe_parts(state)})
                elif mode == "polish":
                    markdown = polish_only(session, settings, key, emit, redo=redo)
                    emit({"type": "done", "session": session, "text": markdown, "mode": mode})
                else:
                    markdown = run_session(session, settings, key, emit)
                    emit({"type": "done", "session": session, "text": markdown, "mode": mode})
            except Exception as exc:  # API, ffmpeg, disk: all shown in the UI
                message = str(exc) or exc.__class__.__name__
                if mode == "prepare":
                    message = f"合并/切分失败：{message}"
                m = read_meta(session)
                m["status"] = "error"
                m["last_error"] = message
                write_meta(session, m)
                emit({"type": "error", "session": session, "text": message})

        self.worker = threading.Thread(target=work, name=mode, daemon=True)
        self.transcribe_event.emit({"type": "start", "session": session, "mode": mode})
        self.worker.start()
        self.sessions_changed.emit(session)

    def _check_can_transcribe(self, session: Path | None) -> str:
        if session is None:
            return "请先选择一个录音会话。"
        if self.is_busy:
            return "正在处理另一个任务（合并/切分或转写），请等它完成。"
        if self.recorder and self.recorder.session_dir == session:
            return "这个会话还在录音，请先停止录音。"
        if not config.get_api_key()[0]:
            return "还没有设置 OpenAI API Key，请在主窗口点击“设置…”。"
        return ""

    def _track_event(self, event: dict) -> None:
        kind = event["type"]
        if kind in ("clear", "start"):
            self.last_error = ""
            self.phase = "polish" if kind == "start" and event.get("mode") == "polish" else "transcribe"
            self.progress_text, self.progress_pct = ("准备中…", 0.0) if kind == "start" else ("", 0.0)
        elif kind == "stage":
            self.progress_text = event["text"]
            if event.get("phase") == "polish" and self.phase != "polish":
                self.phase, self.progress_pct = "polish", 0.0
        elif kind == "progress":
            self.phase = event.get("phase", self.phase)
            self.progress_pct = 100.0 * event["done"] / event["total"]
            self.progress_text = event["text"]
        elif kind == "delta":
            self.live_text[event["track"]] = event["text"]
        elif kind == "polished_delta":
            self.live_polished[event["track"]] = event["text"]
        elif kind == "done":
            self.progress_text = "润色完成" if event.get("mode") == "polish" else "转写完成"
            self.progress_pct = 100.0
        elif kind == "prepared":
            self.progress_text, self.progress_pct = "录音已处理，待转写", 0.0
        elif kind in ("error", "rejected"):
            self.last_error = event["text"]
            if kind == "error":
                if self.mode == "prepare":
                    self.progress_text = "合并/切分失败"
                else:
                    self.progress_text = "润色失败" if self.phase == "polish" else "转写失败"
        if kind in ("done", "prepared", "error"):
            self.worker_session = None
            self.sessions_changed.emit(event["session"])
            # the worker thread emits its final event just before exiting; start queued work once it has
            QTimer.singleShot(50, self._run_pending)
