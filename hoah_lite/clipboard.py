"""Copy text to the Windows clipboard immediately.

Qt's Windows clipboard uses delayed rendering: it only promises the data and hands it over when
another program pastes, which needs this app's GUI thread to be free at that moment. While a long
transcript is streaming into the text box the GUI thread is busy, and a paste can come back empty.
Writing CF_UNICODETEXT directly puts the text on the clipboard right away.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


def _win32_set_text(text: str) -> bool:
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]

    data = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-16-le") + b"\x00\x00"
    for _ in range(10):  # another program may hold the clipboard for a moment
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        return False
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)  # ownership passes to the system only on success
            return False
        return True
    finally:
        user32.CloseClipboard()


def set_text(text: str) -> bool:
    """Put text on the system clipboard; returns True on success."""
    if sys.platform == "win32":
        try:
            if _win32_set_text(text):
                return True
        except OSError:
            pass
    from PySide6.QtWidgets import QApplication

    QApplication.clipboard().setText(text)
    return True
