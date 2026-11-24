"""Application entry: single instance, tray icon, main window and floating window."""

from __future__ import annotations

import ctypes
import sys

from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from . import config
from .config import Settings
from .controller import Controller
from .floating import FloatingWindow
from .main_window import MainWindow
from .recorder import repair_incomplete
from .widgets import MENU_QSS, app_icon


class Tray(QSystemTrayIcon):
    def __init__(self, app: "HoAhApp"):
        super().__init__(app_icon())
        self.a = app
        self.setToolTip("HoAh-lite")
        menu = QMenu()
        menu.setStyleSheet(MENU_QSS)
        menu.addAction("打开主窗口", app.main.show_and_raise)
        self.record_action = menu.addAction("开始录音", app.controller.toggle_recording)
        menu.addSeparator()
        self.floating_action = QAction("显示悬浮窗", menu, checkable=True)
        self.floating_action.toggled.connect(lambda on: app.floating.set_shown(on))
        menu.addAction(self.floating_action)
        self.autostart_action = QAction("开机自动启动", menu, checkable=True)
        self.autostart_action.toggled.connect(self._toggle_autostart)
        menu.addAction(self.autostart_action)
        menu.addSeparator()
        menu.addAction("退出", app.quit)
        self.setContextMenu(menu)
        self._menu = menu
        self.activated.connect(self._on_activated)
        app.controller.recording_changed.connect(self._on_recording)
        app.floating.mode_changed.connect(self.sync)
        menu.aboutToShow.connect(self.sync)
        self.sync()

    def sync(self) -> None:
        for action, value in (
            (self.floating_action, self.a.controller.settings.floating),
            (self.autostart_action, config.is_autostart_enabled()),
        ):
            action.blockSignals(True)
            action.setChecked(value)
            action.blockSignals(False)

    def _toggle_autostart(self, on: bool) -> None:
        try:
            config.set_autostart(on)
        except OSError as exc:
            QMessageBox.critical(None, "开机自启设置失败", str(exc))
        self.sync()

    def _on_recording(self, recording: bool) -> None:
        self.setIcon(app_icon(recording))
        self.record_action.setText("停止录音" if recording else "开始录音")
        self.setToolTip("HoAh-lite · 录音中" if recording else "HoAh-lite")

    def _on_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.a.main.show_and_raise()


class HoAhApp:
    def __init__(self, qt: QApplication, autostart: bool):
        self.qt = qt
        self.controller = Controller(Settings.load())
        self.main = MainWindow(self.controller)
        self.floating = FloatingWindow(self.controller)
        self.floating.open_main.connect(self.main.show_and_raise)
        self.floating.quit_requested.connect(self.quit)
        # The floating window is hidden whenever the main window is on screen.
        self.main.visibility_changed.connect(self.floating.set_suppressed)
        self.tray = Tray(self)
        self.tray.show()

        if not autostart:
            self.main.show()
        else:
            self.floating.set_suppressed(False)
            if not self.controller.settings.floating:
                self.tray.showMessage("HoAh-lite", "已在后台启动，点击托盘图标打开。", app_icon(), 4000)
        self.tray.sync()

    def quit(self) -> None:
        c = self.controller
        if c.is_recording:
            answer = QMessageBox.question(None, "正在录音", "正在录音，退出会停止录音并保存已录内容。确定退出吗？")
            if answer != QMessageBox.Yes:
                return
            c.stop_recording(prepare=False)
        elif c.is_busy:
            answer = QMessageBox.question(
                None, "正在转写", "转写还在进行，退出后下次可以点“转写 / 继续”接着转。确定退出吗？")
            if answer != QMessageBox.Yes:
                return
        self.main.quitting = True
        self.tray.hide()
        self.qt.exit(0)


def _single_instance() -> bool:
    if sys.platform != "win32":
        return True
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _single_instance.handle = kernel32.CreateMutexW(None, False, "Local\\HoAhLite")
    return kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def main() -> None:
    qt = QApplication(sys.argv)
    qt.setApplicationName("HoAh-lite")
    qt.setQuitOnLastWindowClosed(False)  # main window hides to tray; floating window is a tool window
    qt.setWindowIcon(app_icon())
    # At fractional scaling (e.g. 150%) Qt renders unhinted glyphs, which smears CJK strokes
    # across pixel rows. Full hinting snaps them to the grid like native Windows apps.
    font = QFont("Microsoft YaHei UI", 10)
    font.setHintingPreference(QFont.PreferFullHinting)
    qt.setFont(font)
    if not _single_instance():
        QMessageBox.information(None, "HoAh-lite", "HoAh-lite 已经在运行了，请在托盘区或悬浮窗中使用。")
        return
    repair_incomplete(config.sessions_dir())
    app = HoAhApp(qt, autostart="--autostart" in sys.argv)
    qt.aboutToQuit.connect(lambda: app.controller.stop_recording(prepare=False))
    sys.exit(qt.exec())
