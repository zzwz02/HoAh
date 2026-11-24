"""Paths, settings, API key storage and Windows autostart."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import keyring
from keyring.errors import PasswordDeleteError

APP_NAME = "HoAhLite"
KEYRING_SERVICE = "HoAhLite/OpenAI"
KEYRING_USER = "api_key"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = PROJECT_ROOT / "HoAh-lite.vbs"

DEFAULT_CONTEXT = (
    "日常工作录音，可能包括办公室沟通、线上会议、客户现场和电脑播放的视频。"
    "以普通话为主，夹杂英文技术术语、产品名和缩写。"
    "请逐字转写，保留原话和语气词，英文术语保持英文原样，不翻译、不总结、不润色。"
)

SOURCES = {"microphone": "麦克风", "system": "系统声音", "both": "麦克风 + 系统声音"}
TRACK_LABELS = {"microphone": "麦克风", "system": "系统声音"}


def data_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    path = Path(root) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def sessions_dir() -> Path:
    path = data_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Settings:
    source: str = "both"
    languages: str = "zh-cn, en"
    context: str = DEFAULT_CONTEXT
    vocabulary: str = ""
    floating: bool = True  # show the floating bar while the main window is hidden
    float_x: int = -1
    float_y: int = -1

    @classmethod
    def load(cls) -> "Settings":
        path = data_dir() / "settings.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        defaults = cls()
        values = {}
        for f in fields(cls):
            value = raw.get(f.name)
            if type(value) is type(getattr(defaults, f.name)):
                values[f.name] = value
        settings = cls(**values)
        if settings.source not in SOURCES:
            settings.source = "both"
        if not settings.context.strip():
            settings.context = DEFAULT_CONTEXT
        return settings

    def save(self) -> None:
        path = data_dir() / "settings.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


# --- API key -----------------------------------------------------------------

def get_api_key() -> tuple[str | None, str | None]:
    """Return (key, source); source is "凭据管理器" or "环境变量"."""
    try:
        key = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:
        key = None
    if key and key.strip():
        return key.strip(), "凭据管理器"
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env, "环境变量"
    return None, None


def set_api_key(key: str) -> None:
    key = key.strip()
    if not key:
        raise ValueError("API Key 不能为空。")
    keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key)


def delete_api_key() -> bool:
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
        return True
    except PasswordDeleteError:
        return False


def mask_key(key: str) -> str:
    return f"{key[:3]}…{key[-4:]}" if len(key) > 8 else "…"


# --- Autostart ---------------------------------------------------------------

def autostart_command() -> str:
    return f'wscript.exe "{LAUNCHER}" --autostart'


def is_autostart_enabled() -> bool:
    if sys.platform != "win32":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
            return bool(value)
    except OSError:
        return False


def set_autostart(enabled: bool) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except FileNotFoundError:
                pass
