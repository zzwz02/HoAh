"""Small painted widgets and the shared colour theme."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import QAbstractButton, QPlainTextEdit, QWidget

from . import clipboard

ACCENT = QColor("#1fa9b5")
RECORD = QColor("#e5534b")
TEXT = "#1f2629"
MUTED = "#6b7a80"
ERROR = "#d0392f"
SUCCESS = "#1e8449"

MENU_QSS = f"""
QMenu {{ background: #ffffff; color: {TEXT}; border: 1px solid #d5dde0; border-radius: 6px; padding: 4px;
    font-family: "Microsoft YaHei UI"; font-size: 9.5pt; }}
QMenu::item {{ background: transparent; padding: 6px 22px 6px 26px; border-radius: 4px; }}
QMenu::item:selected {{ background: #e3f5f7; color: {TEXT}; }}
QMenu::item:disabled {{ color: #a9b4b8; }}
QMenu::separator {{ height: 1px; background: #e6ebed; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 6px; }}
"""

MAIN_QSS = f"""
QWidget {{ background: #ffffff; color: {TEXT}; font-family: "Microsoft YaHei UI"; font-size: 10pt; }}
QGroupBox {{ border: 1px solid #dfe5e8; border-radius: 8px; margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QPushButton {{ background: #f3f6f7; border: 1px solid #d5dde0; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ background: #e8eef0; }}
QPushButton:disabled {{ color: #a9b4b8; }}
QPushButton#primary {{ background: {ACCENT.name()}; border-color: #1a96a1; color: #ffffff; }}
QPushButton#primary:hover {{ background: #1a96a1; }}
QPushButton#primary:disabled {{ background: #9fd6db; border-color: #9fd6db; color: #ffffff; }}
QLineEdit, QPlainTextEdit, QTextEdit, QListWidget, QComboBox {{
    background: #ffffff; border: 1px solid #d5dde0; border-radius: 6px; padding: 4px;
    selection-background-color: #bfe8ec; selection-color: {TEXT}; }}
QListWidget::item {{ padding: 6px 4px; }}
QListWidget::item:selected {{ background: #e3f5f7; color: {TEXT}; }}
QProgressBar {{ background: #eef2f3; border: 1px solid #dfe5e8; border-radius: 5px; height: 10px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT.name()}; border-radius: 4px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#errorBanner {{ background: #fdecea; color: #a52a22; border: 1px solid #f5c2bd; border-radius: 6px; padding: 8px 10px; }}
QSplitter::handle {{ background: #ffffff; }}
""" + MENU_QSS


class RecordButton(QAbstractButton):
    """Round button: teal dot when idle, red rounded square while recording."""

    def __init__(self, diameter: int = 44, parent: QWidget | None = None):
        super().__init__(parent)
        self.diameter = diameter
        self.recording = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(diameter, diameter)

    def set_recording(self, recording: bool) -> None:
        self.recording = recording
        self.setToolTip("停止录音" if recording else "开始录音")
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.diameter, self.diameter)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = RECORD if self.recording else ACCENT
        if not self.isEnabled():
            color = QColor(color.red(), color.green(), color.blue(), 110)
        ring = QColor(color)
        ring.setAlpha(60 if self.underMouse() else 35)
        d = self.diameter
        p.setPen(Qt.NoPen)
        p.setBrush(ring)
        p.drawEllipse(QRectF(0, 0, d, d))
        p.setBrush(color)
        if self.recording:
            s = d * 0.36
            p.drawRoundedRect(QRectF((d - s) / 2, (d - s) / 2, s, s), d * 0.06, d * 0.06)
        else:
            s = d * 0.46
            p.drawEllipse(QRectF((d - s) / 2, (d - s) / 2, s, s))


class LevelMeter(QWidget):
    def __init__(self, parent: QWidget | None = None, height: int = 6):
        super().__init__(parent)
        self.level = 0.0
        self.recording = False
        self.setFixedHeight(height)

    def set_level(self, level: float, recording: bool) -> None:
        self.level = max(0.0, min(1.0, level))
        self.recording = recording
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        radius = r.height() / 2
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 22))
        p.drawRoundedRect(r, radius, radius)
        if self.level > 0:
            w = max(r.height(), int(r.width() * min(1.0, self.level * 1.4)))
            p.setBrush(RECORD if self.recording else ACCENT)
            p.drawRoundedRect(QRectF(0, 0, w, r.height()), radius, radius)


class TranscriptView(QPlainTextEdit):
    """Read-only text box whose Ctrl+C / context-menu copy writes the clipboard immediately."""

    copied = Signal(int)  # number of characters copied

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)

    def selected_text(self) -> str:
        return self.textCursor().selectedText().replace(" ", "\n")  # Qt marks paragraph breaks with U+2029

    def copy_selection(self) -> None:
        text = self.selected_text()
        if text and clipboard.set_text(text):
            self.copied.emit(len(text))

    def copy_all(self) -> None:
        text = self.toPlainText()
        if text and clipboard.set_text(text):
            self.copied.emit(len(text))

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = self.createStandardContextMenu()
        for action in menu.actions():
            if action.objectName() == "edit-copy" or action.shortcut() == QKeySequence(QKeySequence.Copy):
                try:
                    action.triggered.disconnect()
                except (RuntimeError, TypeError):
                    pass
                action.triggered.connect(self.copy_selection)
        menu.addSeparator()
        menu.addAction("复制全部", self.copy_all)
        menu.exec(event.globalPos())

    def set_text_preserving(self, text: str, follow_end: bool = False) -> None:
        """Replace the text without losing the reader's place.

        While the user has a selection the update is skipped (it would wipe the selection);
        the next update after they click elsewhere catches up. With follow_end the view sticks
        to the bottom only if it was already there.
        """
        if text == self.toPlainText():
            return
        if self.textCursor().hasSelection():
            return
        bar = self.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 4
        position = bar.value()
        self.setPlainText(text)
        bar.setValue(bar.maximum() if follow_end and at_end else min(position, bar.maximum()))


def app_icon(recording: bool = False) -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#14181b"))
    p.drawEllipse(2, 2, 60, 60)
    p.setBrush(RECORD if recording else ACCENT)
    if recording:
        p.drawRoundedRect(20, 20, 24, 24, 4, 4)
    else:
        p.drawEllipse(18, 18, 28, 28)
    p.end()
    return QIcon(pix)


def clock(seconds: float, hours: bool = False) -> str:
    s = int(seconds)
    if hours or s >= 3600:
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"
