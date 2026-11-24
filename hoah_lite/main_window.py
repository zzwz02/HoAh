"""Main window and settings dialog."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from . import config
from .config import SOURCES, TRACK_LABELS
from .controller import Controller
from .recorder import read_meta
from . import polisher
from .transcriber import MODEL, POLISHED_MD, TranscribeError, check_api_key
from .widgets import MAIN_QSS, LevelMeter, RecordButton, TranscriptView, app_icon, clock

STATUS_LABELS = {
    "recording": "录音中",
    "recorded": "待转写",
    "preparing": "合并/切分中",
    "transcribing": "转写中",
    "polishing": "润色中",
    "transcribed": "已转写",
    "error": "失败，可继续",
}
VIEWS = {"raw": "原始转写", "polished": "润色稿"}
SEGMENT_QSS = """
QPushButton#segLeft, QPushButton#segRight { padding: 6px 14px; background: #ffffff; }
QPushButton#segLeft { border-top-right-radius: 0; border-bottom-right-radius: 0; margin-right: -7px; }
QPushButton#segRight { border-top-left-radius: 0; border-bottom-left-radius: 0; }
QPushButton#segLeft:checked, QPushButton#segRight:checked { background: #e3f5f7; border-color: #1fa9b5; color: #13808a; }
"""


class MainWindow(QMainWindow):
    visibility_changed = Signal(bool)  # the floating window hides while the main window is shown

    def __init__(self, controller: Controller):
        super().__init__()
        self.c = controller
        self.sessions: list[Path] = []
        self.quitting = False
        self.setWindowTitle("HoAh-lite")
        self.setWindowIcon(app_icon())
        self.resize(1040, 700)
        self.setStyleSheet(MAIN_QSS + SEGMENT_QSS)  # scoped to this window and its dialogs, not the floating bar
        self._build()
        self._wire()
        self.refresh_key_status()
        self.refresh_sessions()

    # --- layout ----------------------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        top = QHBoxLayout()
        info = QVBoxLayout()
        info.setSpacing(3)
        self.key_label = QLabel()
        info.addWidget(self.key_label)
        sessions = config.sessions_dir()
        details = QLabel(
            f"转写模型：<b>{MODEL}</b>　·　润色模型：<b>{polisher.MODEL}</b>　·　录音保存在："
            f'<a href="{QUrl.fromLocalFile(str(sessions)).toString()}" style="color:#1a8f99;">{sessions}</a>'
        )
        details.setObjectName("muted")
        details.setTextFormat(Qt.RichText)
        details.setTextInteractionFlags(Qt.TextBrowserInteraction)
        details.setOpenExternalLinks(True)
        details.setToolTip("点击打开录音文件夹")
        info.addWidget(details)
        top.addLayout(info)
        top.addStretch()
        settings_btn = QPushButton("设置…")
        settings_btn.clicked.connect(self.open_settings)
        top.addWidget(settings_btn)
        layout.addLayout(top)

        rec = QGroupBox("录音")
        rl = QHBoxLayout(rec)
        rl.setSpacing(12)
        self.record_btn = RecordButton(46)
        rl.addWidget(self.record_btn)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setStyleSheet("font-family: Consolas; font-size: 18pt;")
        rl.addWidget(self.timer_label)
        self.meter = LevelMeter(height=8)
        self.meter.setMinimumWidth(160)
        rl.addWidget(self.meter, 1)
        rl.addWidget(QLabel("音源"))
        self.source_box = QComboBox()
        for key, label in SOURCES.items():
            self.source_box.addItem(label, key)
        self.source_box.setCurrentIndex(list(SOURCES).index(self.c.settings.source))
        rl.addWidget(self.source_box)
        layout.addWidget(rec)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        title = QLabel("录音会话")
        title.setObjectName("muted")
        ll.addWidget(title)
        self.session_list = QListWidget()
        ll.addWidget(self.session_list)
        splitter.addWidget(left)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        self.transcribe_btn = QPushButton("转写 / 继续")
        self.transcribe_btn.setObjectName("primary")
        self.redo_btn = QPushButton("全部重新转写")
        self.copy_btn = copy_btn = QPushButton("复制")
        copy_btn.setToolTip("复制当前显示的全部文字（原始转写或润色稿）。\n只复制一部分：在文本框里选中后按 Ctrl+C，或右键“复制”。")
        folder_btn = QPushButton("打开文件夹")
        self.transcribe_btn.setToolTip(
            f"用 {MODEL} 转写（不做任何后处理）。\n"
            "之前中断或失败时，跳过已转好的上传文件；未切分的录音会整段重转。")
        self.polish_btn = QPushButton("润色 / 继续")
        self.polish_btn.setObjectName("primary")
        self.polish_btn.setToolTip(
            f"用 {polisher.MODEL} 对原始转写只做润色和分段（修正错字、标点、去口头禅，不总结、不改写）。\n"
            "之前中断或失败时，只补做未完成的块；已有润色稿时会询问是否重新润色。")
        for b in (self.transcribe_btn, self.polish_btn, self.redo_btn, copy_btn, folder_btn):
            actions.addWidget(b)
        actions.addStretch()
        # segmented toggle: 原始转写 | 润色稿
        self.view_buttons = {}
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        for i, (key, label) in enumerate(VIEWS.items()):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setObjectName("segLeft" if i == 0 else "segRight")
            self.view_group.addButton(b)
            self.view_buttons[key] = b
            actions.addWidget(b)
        actions.setSpacing(6)
        self.view_buttons["raw"].setChecked(True)
        copy_btn.clicked.connect(self.copy_text)
        folder_btn.clicked.connect(self.open_folder)
        rlay.addLayout(actions)

        self.error_banner = QLabel()
        self.error_banner.setObjectName("errorBanner")
        self.error_banner.setWordWrap(True)
        self.error_banner.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.error_banner.hide()
        rlay.addWidget(self.error_banner)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("muted")
        rlay.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        rlay.addWidget(self.progress)

        self.text = TranscriptView()
        self.text.setStyleSheet("font-size: 11pt; line-height: 150%;")
        self.text.copied.connect(self.show_copied)
        self._copy_reset = QTimer(self)
        self._copy_reset.setSingleShot(True)
        self._copy_reset.timeout.connect(lambda: self.copy_btn.setText("复制"))
        self._live_timer = QTimer(self)  # coalesce streamed updates: redraw at most ~4 times a second
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(250)
        self._live_timer.timeout.connect(self.render_live)
        rlay.addWidget(self.text, 1)
        splitter.addWidget(right)
        splitter.setSizes([300, 720])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _wire(self) -> None:
        c = self.c
        self.record_btn.clicked.connect(c.toggle_recording)
        self.source_box.currentIndexChanged.connect(lambda _i: c.set_source(self.source_box.currentData()))
        self.session_list.currentRowChanged.connect(lambda _r: self.show_selected())
        self.view_group.buttonClicked.connect(lambda _b: self.show_selected())
        self.transcribe_btn.clicked.connect(lambda: c.start_transcription(self.selected_session()))
        self.polish_btn.clicked.connect(self.polish)
        self.redo_btn.clicked.connect(self.redo)
        c.recording_changed.connect(self.on_recording_changed)
        c.tick.connect(self.on_tick)
        c.source_changed.connect(lambda s: self.source_box.setCurrentIndex(list(SOURCES).index(s)))
        c.transcribe_event.connect(self.on_event)
        c.sessions_changed.connect(self.refresh_sessions)
        c.notice.connect(lambda title, msg: QMessageBox.warning(self, title, msg))

    # --- error banner ------------------------------------------------------------------

    def show_error(self, message: str) -> None:
        self.error_banner.setText(message)
        self.error_banner.show()

    def clear_error(self) -> None:
        self.error_banner.clear()
        self.error_banner.hide()

    # --- controller events ----------------------------------------------------------------

    def on_recording_changed(self, recording: bool) -> None:
        self.record_btn.set_recording(recording)
        self.source_box.setEnabled(not recording)
        if not recording:
            self.timer_label.setText("00:00:00")
            self.meter.set_level(0, False)

    def on_tick(self, elapsed: float, level: float) -> None:
        self.timer_label.setText(clock(elapsed, hours=True))
        self.meter.set_level(level, True)

    def on_event(self, event: dict) -> None:
        kind = event["type"]
        if kind == "clear":
            self.clear_error()
        elif kind == "rejected":
            self.show_error(event["text"])
        elif kind == "start":
            self._set_busy(True)
            mode = event.get("mode")
            self.progress_label.setText({"prepare": "正在合并录音分块…", "polish": "准备润色…"}.get(mode, "准备中…"))
            self.progress.setRange(0, 0 if mode == "prepare" else 100)
            self.progress.setValue(0)
            if mode in ("transcribe", "polish") and self.selected_session() == event["session"]:
                self.set_view("polished" if mode == "polish" else "raw")  # follow the step that is running
                self.render_live()
        elif kind == "prepared":
            self._set_busy(False)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress_label.setText(f"录音已合并（{event['text']}），点击“转写 / 继续”开始转写。")
        elif kind == "stage":
            self.progress_label.setText(event["text"])
            self.progress.setRange(0, 0)  # indeterminate
        elif kind == "progress":
            pct = 100.0 * event["done"] / event["total"]
            self.progress_label.setText(f"{event['text']}　已完成 {pct:.0f}%")
            if event.get("busy"):
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 100)
                self.progress.setValue(int(pct))
        elif kind in ("delta", "polished_delta"):
            if self.selected_session() == self.c.worker_session and not self._live_timer.isActive():
                self._live_timer.start()
        elif kind == "done":
            self._set_busy(False)
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            if event.get("mode") == "polish":
                self.progress_label.setText(f"润色完成，已保存到 {POLISHED_MD}；可在右侧切换“原始转写 / 润色稿”对照。")
            else:
                self.progress_label.setText("转写完成，原始转写已保存到 transcript.md；需要润色时点击“润色 / 继续”。")
        elif kind == "error":
            self._set_busy(False)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            if self.c.mode == "prepare":
                self.progress_label.setText("合并/切分失败，录音分块已保留；点击“转写 / 继续”会重新尝试。")
            elif self.c.mode == "polish":
                self.progress_label.setText("润色失败；已润色的块都已保存，点击“润色 / 继续”只补做剩下的块。")
            else:
                self.progress_label.setText("转写失败，已完成的部分已保存；修正后点击“转写 / 继续”会从失败处接着转。")
            self.show_error(event["text"])

    def _set_busy(self, busy: bool) -> None:
        for b in (self.transcribe_btn, self.polish_btn, self.redo_btn):
            b.setEnabled(not busy)

    def view(self) -> str:
        return next(k for k, b in self.view_buttons.items() if b.isChecked())

    def set_view(self, key: str) -> None:
        self.view_buttons[key].setChecked(True)

    # --- sessions ----------------------------------------------------------------------------

    def refresh_sessions(self, select: Path | None = None) -> None:
        current = select or self.selected_session()
        root = config.sessions_dir()
        self.sessions = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for session in self.sessions:
            meta = read_meta(session)
            status = meta.get("status", "recorded")
            if session == self.c.worker_session and self.c.is_busy:
                status = {"prepare": "preparing", "polish": "polishing"}.get(self.c.mode, "transcribing")
            secs = meta.get("duration_sec")
            length = f"   {clock(secs, hours=True)}" if secs else ""
            label = STATUS_LABELS.get(status, status)
            if status == "transcribed":
                label = "已转写 · 已润色" if (session / POLISHED_MD).exists() else "已转写 · 未润色"
            item = QListWidgetItem(f"{session.name}{length}\n{label}")
            if status == "error":
                item.setForeground(Qt.red)
            self.session_list.addItem(item)
        if current in self.sessions:
            self.session_list.setCurrentRow(self.sessions.index(current))
        elif self.sessions:
            self.session_list.setCurrentRow(0)
        self.session_list.blockSignals(False)
        self.show_selected()

    def selected_session(self) -> Path | None:
        row = self.session_list.currentRow()
        return self.sessions[row] if 0 <= row < len(self.sessions) else None

    def show_selected(self) -> None:
        session = self.selected_session()
        if session is None:
            self.text.setPlainText("还没有录音。点击上方红/青色圆形按钮开始录音。")
            return
        raw, polished = session / "transcript.md", session / POLISHED_MD
        live = session == self.c.worker_session and self.c.is_busy
        view = self.view()
        if live and (self.c.mode == "transcribe" and view == "raw" or self.c.mode == "polish" and view == "polished"):
            self.render_live()
            return
        if view == "polished":
            if polished.exists():
                self.text.setPlainText(polished.read_text(encoding="utf-8"))
            elif raw.exists():
                self.text.setPlainText("这个会话还没有润色稿。\n\n点击“润色 / 继续”，用 "
                                       f"{polisher.MODEL} 对原始转写只做润色和分段；完成后可在这里和“原始转写”来回切换对照。")
            else:
                self.text.setPlainText("还没有转写，也就没有润色稿。请先点击“转写 / 继续”。")
            return
        if raw.exists():
            self.text.setPlainText(raw.read_text(encoding="utf-8"))
            return
        meta = read_meta(session)
        status = meta.get("status")
        if status == "recording":
            hint = "录音中…"
        elif status == "preparing":
            hint = "正在合并和切分录音…"
        else:
            hint = "录音已保存，还没有转写。点击“转写 / 继续”开始。"
        if meta.get("last_error"):
            hint += f"\n\n上次转写失败：{meta['last_error']}"
        self.text.setPlainText(hint)

    def render_live(self) -> None:
        session = self.selected_session()
        if session is None or session != self.c.worker_session:  # worker_session is cleared when the run ends
            return
        if self.c.mode == "polish":  # polished paragraphs appear as chunks finish, in order
            if self.view() != "polished":
                return
            blocks = [txt for txt in self.c.live_polished.values() if txt]
            text = "\n\n".join(blocks) if blocks else "正在润色，第一块完成后这里会显示润色稿…"
        else:
            if self.view() != "raw":
                return
            blocks = [f"【{TRACK_LABELS[t]}】\n{txt}" for t, txt in self.c.live_text.items() if txt]
            text = "\n\n".join(blocks) if blocks else "转写中…"
        self.text.set_text_preserving(text, follow_end=True)

    # --- actions -----------------------------------------------------------------------------------

    def redo(self) -> None:
        session = self.selected_session()
        if session and QMessageBox.question(
            self, "全部重新转写", "将丢弃这个会话已有的转写结果和润色稿并重新转写，确定吗？"
        ) != QMessageBox.Yes:
            return
        self.c.start_transcription(session, redo=True)

    def polish(self) -> None:
        session = self.selected_session()
        redo = False
        if session and (session / POLISHED_MD).exists():
            answer = QMessageBox.question(
                self, "重新润色", "这个会话已经有润色稿。要丢弃它并重新润色吗？\n（原始转写不受影响）")
            if answer != QMessageBox.Yes:
                return
            redo = True
        self.c.start_polish(session, redo=redo)

    def copy_text(self) -> None:
        self.text.copy_all()

    def show_copied(self, chars: int) -> None:
        self.copy_btn.setText(f"已复制 {chars} 字 ✓")
        self._copy_reset.start(1800)

    def open_folder(self) -> None:
        os.startfile(self.selected_session() or config.sessions_dir())

    def refresh_key_status(self) -> None:
        key, source = config.get_api_key()
        if key:
            self.key_label.setText(f"OpenAI API Key：已设置 {config.mask_key(key)}（{source}）")
            self.key_label.setStyleSheet("")
        else:
            self.key_label.setText("OpenAI API Key：未设置，请点击右侧“设置…”填写")
            self.key_label.setStyleSheet("color: #d0392f;")

    def open_settings(self) -> None:
        SettingsDialog(self).exec()
        self.refresh_key_status()

    def show_and_raise(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:  # minimised counts as hidden
            self.visibility_changed.emit(self.isVisible() and not self.isMinimized())

    def closeEvent(self, event) -> None:
        # Closing the main window keeps the app (floating window + tray) running,
        # except when the app itself is quitting (Qt 6 quit() closes every window first).
        if self.quitting:
            event.accept()
            return
        event.ignore()
        self.hide()


class SettingsDialog(QDialog):
    key_checked = Signal(str, bool)

    def __init__(self, main: MainWindow):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("设置")
        self.resize(680, 640)
        s = main.c.settings
        layout = QVBoxLayout(self)

        key_box = QGroupBox("OpenAI API Key")
        kl = QVBoxLayout(key_box)
        self.key_state = QLabel()
        kl.addWidget(self.key_state)
        row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("粘贴新的 sk-… Key 后点击“保存 / 替换”")
        row.addWidget(self.key_edit, 1)
        show = QCheckBox("显示")
        show.toggled.connect(lambda on: self.key_edit.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        row.addWidget(show)
        kl.addLayout(row)
        buttons = QHBoxLayout()
        for text, slot in (("保存 / 替换 Key", self.save_key), ("删除 Key", self.delete_key), ("测试 Key", self.test_key)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch()
        kl.addLayout(buttons)
        self.key_msg = QLabel()
        self.key_msg.setWordWrap(True)
        kl.addWidget(self.key_msg)
        layout.addWidget(key_box)
        self.key_checked.connect(self._show_key_result)
        self.refresh_key_state()

        form = QFormLayout()
        self.lang_edit = QLineEdit(s.languages)
        self.lang_edit.setPlaceholderText("zh-cn, en")
        form.addRow("语言（逗号分隔）", self.lang_edit)
        self.context_edit = QPlainTextEdit(s.context)
        self.context_edit.setFixedHeight(100)
        form.addRow("场景说明（prompt）", self.context_edit)
        self.vocab_edit = QPlainTextEdit(s.vocabulary)
        self.vocab_edit.setPlaceholderText("每行一个词，例如：\nKubernetes\nPHS Epic\nJIRA")
        form.addRow("自定义词汇\n（每行一个）", self.vocab_edit)
        hint = QLabel("自定义词汇会作为 keywords 发给 gpt-transcribe，最多取前 100 个；< > 会被自动去掉。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        form.addRow("", hint)
        self.autostart = QCheckBox("开机自动启动（只显示悬浮窗，主窗口不弹出）")
        self.autostart.setChecked(config.is_autostart_enabled())
        form.addRow("", self.autostart)
        layout.addLayout(form, 1)

        bottom = QHBoxLayout()
        bottom.addStretch()
        save = QPushButton("保存")
        save.setObjectName("primary")
        save.clicked.connect(self.save)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        bottom.addWidget(save)
        bottom.addWidget(cancel)
        layout.addLayout(bottom)

    def refresh_key_state(self) -> None:
        key, source = config.get_api_key()
        self.key_state.setText(f"当前：{config.mask_key(key)}（来自{source}）" if key else "当前：未设置")
        self.main.refresh_key_status()

    def _msg(self, text: str, ok: bool | None = None) -> None:
        color = "" if ok is None else ("color:#1e8449;" if ok else "color:#d0392f;")
        self.key_msg.setStyleSheet(color)
        self.key_msg.setText(text)

    def save_key(self) -> None:
        try:
            config.set_api_key(self.key_edit.text())
        except Exception as exc:
            self._msg(f"保存失败：{exc}", False)
            return
        self.key_edit.clear()
        self._msg("已保存到 Windows 凭据管理器，下次转写立即生效。", True)
        self.refresh_key_state()

    def delete_key(self) -> None:
        removed = config.delete_api_key()
        self._msg("已从凭据管理器删除。" if removed else "凭据管理器里没有保存的 Key。")
        self.refresh_key_state()

    def test_key(self) -> None:
        key = self.key_edit.text().strip() or config.get_api_key()[0]
        if not key:
            self._msg("没有可测试的 Key。", False)
            return
        self._msg("测试中…")

        def work() -> None:
            try:
                self.key_checked.emit(check_api_key(key), True)
            except TranscribeError as exc:
                self.key_checked.emit(f"测试失败：{exc}", False)

        threading.Thread(target=work, daemon=True).start()

    def _show_key_result(self, text: str, ok: bool) -> None:
        self._msg(text, ok)

    def save(self) -> None:
        s = self.main.c.settings
        s.languages = self.lang_edit.text().strip()
        s.context = self.context_edit.toPlainText().strip() or config.DEFAULT_CONTEXT
        s.vocabulary = self.vocab_edit.toPlainText().strip()
        self.main.c.save_settings()
        try:
            if self.autostart.isChecked() != config.is_autostart_enabled():
                config.set_autostart(self.autostart.isChecked())
        except OSError as exc:
            QMessageBox.critical(self, "开机自启设置失败", str(exc))
            return
        self.accept()
