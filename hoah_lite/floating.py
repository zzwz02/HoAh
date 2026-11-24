"""Always-on-top floating bar: record button, timer and status. Double-click opens the main window."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QActionGroup, QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QWidget

from .config import SOURCES
from .controller import Controller
from .widgets import ACCENT, ERROR, MENU_QSS, MUTED, RECORD, SUCCESS, TEXT, RecordButton, clock

SIZE = (128, 30)

FLOAT_QSS = f"""
QWidget {{ background: transparent; color: {TEXT}; font-family: "Microsoft YaHei UI"; }}
""" + MENU_QSS


class FloatingWindow(QWidget):
    open_main = Signal()
    quit_requested = Signal()
    mode_changed = Signal()  # visibility toggled; the tray refreshes its checkmark

    def __init__(self, controller: Controller):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.c = controller
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("HoAh-lite 悬浮窗")
        self.setStyleSheet(FLOAT_QSS)
        self.setFixedSize(*SIZE)
        self._drag_offset: QPoint | None = None
        self._dragged = False
        self._suppressed = False  # true while the main window is on screen

        lay = QHBoxLayout(self)
        lay.setContentsMargins(5, 0, 8, 0)
        lay.setSpacing(5)
        self.record = RecordButton(20)
        self.record.clicked.connect(self.c.toggle_recording)
        lay.addWidget(self.record)
        self.time = QLabel("00:00")
        self.time.setStyleSheet("font-family: Consolas; font-size: 9.5pt;")
        lay.addWidget(self.time)
        self.status_label = QLabel()
        lay.addWidget(self.status_label, 1, Qt.AlignRight | Qt.AlignVCenter)

        c = controller
        c.recording_changed.connect(lambda _r: self.refresh())
        c.tick.connect(lambda elapsed, _level: self.time.setText(clock(elapsed)))
        c.source_changed.connect(lambda _s: self.refresh())
        c.transcribe_event.connect(lambda _e: self.refresh())
        self.refresh()

    # --- state ----------------------------------------------------------------------------

    def status(self) -> tuple[str, str]:
        c = self.c
        if c.is_recording:
            return "录音中", RECORD.name()
        if c.is_busy:
            if c.mode == "prepare":
                return "处理中…", ACCENT.name()
            verb = "润色" if c.phase == "polish" else "转写"
            return f"{verb} {c.progress_pct:.0f}%", ACCENT.name()
        if c.last_error:
            return c.progress_text if c.progress_text.endswith("失败") else "出错", ERROR
        if c.progress_text in ("转写完成", "润色完成"):
            return c.progress_text, SUCCESS
        if c.progress_text == "录音已处理，待转写":
            return "待转写", SUCCESS
        return "就绪", MUTED

    def refresh(self) -> None:
        c = self.c
        self.record.set_recording(c.is_recording)
        text, color = self.status()
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 7.5pt;")
        detail = c.last_error or c.progress_text
        self.setToolTip(
            f"音源：{SOURCES[c.settings.source]}\n" + (f"{detail}\n" if detail else "") + "双击打开主窗口；右键更多选项"
        )
        if not c.is_recording:
            self.time.setText("00:00")

    # --- visibility & position -------------------------------------------------------------

    def set_shown(self, shown: bool) -> None:
        """User preference: whether the bar appears when the main window is not shown."""
        self.c.settings.floating = shown
        self.c.save_settings()
        self._update_visibility()
        self.mode_changed.emit()

    def set_suppressed(self, suppressed: bool) -> None:
        self._suppressed = suppressed
        self._update_visibility()

    def _update_visibility(self) -> None:
        if self.c.settings.floating and not self._suppressed:
            if not self.isVisible():
                self.place()
                self.show()
            self.raise_()
        else:
            self.hide()

    def place(self) -> None:
        s = self.c.settings
        if s.float_x >= 0 and s.float_y >= 0 and QGuiApplication.screenAt(QPoint(s.float_x + 10, s.float_y + 10)):
            self.move(s.float_x, s.float_y)
        else:
            area = QGuiApplication.primaryScreen().availableGeometry()
            self.move(area.right() - self.width() - 24, area.bottom() - self.height() - 24)
        self._clamp_to_screen()

    def _clamp_to_screen(self) -> None:
        screen = QGuiApplication.screenAt(self.geometry().center()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(max(self.x(), area.left()), area.right() - self.width())
        y = min(max(self.y(), area.top()), area.bottom() - self.height())
        self.move(x, y)

    def _save_position(self) -> None:
        self.c.settings.float_x, self.c.settings.float_y = self.x(), self.y()
        self.c.save_settings()

    # --- painting & mouse ----------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(0, 0, 0, 38), 1))
        p.setBrush(QColor(255, 255, 255, 240))
        p.drawRoundedRect(rect, 9, 9)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self._dragged = False

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._dragged = True

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_offset is not None and self._dragged:
            self._clamp_to_screen()
            self._save_position()
        self._drag_offset = None

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.open_main.emit()

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.addAction("打开主窗口", self.open_main.emit)
        menu.addAction("停止录音" if self.c.is_recording else "开始录音", self.c.toggle_recording)
        sources = menu.addMenu("音源")
        sources.setEnabled(not self.c.is_recording)
        group = QActionGroup(sources)
        for key, label in SOURCES.items():
            action = sources.addAction(label, lambda k=key: self.c.set_source(k))
            action.setCheckable(True)
            action.setChecked(key == self.c.settings.source)
            group.addAction(action)
        menu.addAction("隐藏悬浮窗（可从托盘恢复）", lambda: self.set_shown(False))
        menu.addSeparator()
        menu.addAction("退出 HoAh-lite", self.quit_requested.emit)
        menu.exec(event.globalPos())
