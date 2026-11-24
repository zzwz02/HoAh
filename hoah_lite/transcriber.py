"""gpt-transcribe client and the merge → split → transcribe → polish pipeline for a session."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests

from . import audio, polisher
from .config import TRACK_LABELS, Settings
from .recorder import read_meta, write_meta

API_URL = "https://api.openai.com/v1/audio/transcriptions"
MODELS_URL = "https://api.openai.com/v1/models"
MODEL = "gpt-transcribe"
MAX_KEYWORDS = 100
CONTEXT_TAIL_CHARS = 600
MAX_RETRIES = 4
POLISH_STATE = "polish_state.json"
POLISHED_MD = "transcript_polished.md"

Emit = Callable[[dict], None]


class TranscribeError(Exception):
    pass


# --- request building ---------------------------------------------------------

def clean_keywords(vocabulary: str) -> list[str]:
    """One keyword per line; drop characters the API rejects (<, >, CR, LF) and duplicates."""
    seen: set[str] = set()
    result = []
    for line in vocabulary.splitlines():
        word = line.replace("<", "").replace(">", "").replace("\r", "").strip()
        key = word.casefold()
        if word and key not in seen:
            seen.add(key)
            result.append(word)
    return result[:MAX_KEYWORDS]


def parse_languages(text: str) -> list[str]:
    return [code.strip().lower() for code in text.replace("，", ",").split(",") if code.strip()]


def build_prompt(context: str, previous_text: str) -> str:
    prompt = context.strip()
    tail = previous_text.strip()[-CONTEXT_TAIL_CHARS:]
    if tail:
        prompt += "\n\n这段录音紧接在下面这段转写之后（仅作上下文，不要重复输出）：\n" + tail
    return prompt


def _error_message(response: requests.Response) -> str:
    try:
        message = response.json().get("error", {}).get("message")
    except ValueError:
        message = None
    return f"HTTP {response.status_code}：{message or response.text[:300] or response.reason}"


def _retry_delay(attempt: int, response: requests.Response | None) -> float:
    if response is not None:
        try:
            return min(float(response.headers.get("Retry-After", "")), 60.0)
        except ValueError:
            pass
    return min(2.0 ** (attempt + 1), 30.0)


def check_api_key(api_key: str, http=requests) -> str:
    """Returns a human-readable result; raises TranscribeError when the key is rejected."""
    try:
        response = http.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
    except requests.RequestException as exc:
        raise TranscribeError(f"网络错误：{exc}") from exc
    if response.status_code != 200:
        raise TranscribeError(_error_message(response))
    return "API Key 有效。"


def transcribe_file(
    path: Path,
    api_key: str,
    *,
    languages: list[str],
    keywords: list[str],
    prompt: str,
    on_delta: Callable[[str], None] | None = None,
    http=requests,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Upload one file with stream=true; returns the final transcript text."""
    data: list[tuple[str, str]] = [("model", MODEL), ("stream", "true")]
    data += [("languages[]", code) for code in languages]
    data += [("keywords[]", word) for word in keywords]
    if prompt:
        data.append(("prompt", prompt))

    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        response = None
        try:
            with open(path, "rb") as fh:
                response = http.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                    files={"file": (path.name, fh, "audio/mpeg")},
                    stream=True,
                    timeout=(15, 900),
                )
            if response.status_code == 200:
                return _read_stream(response, on_delta)
            last_error = _error_message(response)
            if response.status_code != 429 and response.status_code < 500:
                raise TranscribeError(last_error)
        except requests.RequestException as exc:
            last_error = f"网络错误：{exc}"
        except _StreamBroken as exc:
            last_error = str(exc)
        if attempt < MAX_RETRIES:
            sleep(_retry_delay(attempt, response))
    raise TranscribeError(f"重试 {MAX_RETRIES} 次后仍失败。{last_error}")


class _StreamBroken(Exception):
    pass


def _read_stream(response: requests.Response, on_delta: Callable[[str], None] | None) -> str:
    text = ""
    if on_delta:
        on_delta("")
    try:
        for raw in response.iter_lines(decode_unicode=False):
            line = raw.decode("utf-8", "replace").strip() if raw else ""
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            kind = event.get("type", "")
            if kind == "transcript.text.delta":
                text += event.get("delta", "")
                if on_delta:
                    on_delta(text)
            elif kind == "transcript.text.done":
                return event.get("text", text)
            elif kind == "error" or "error" in event:
                err = event.get("error") or {}
                message = err.get("message") if isinstance(err, dict) else str(err)
                raise TranscribeError(message or "OpenAI 返回错误事件。")
    except requests.RequestException as exc:
        raise _StreamBroken(f"连接中断：{exc}") from exc
    if text:
        return text
    raise _StreamBroken("转写结果流意外结束。")


# --- session pipeline ---------------------------------------------------------

def _state_path(session_dir: Path) -> Path:
    return session_dir / "transcript.json"


def load_state(session_dir: Path) -> dict:
    try:
        return json.loads(_state_path(session_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tracks": {}}


def save_state(session_dir: Path, state: dict) -> None:
    path = _state_path(session_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def reset_state(session_dir: Path) -> None:
    for name in ("transcript.json", "transcript.md", POLISH_STATE, POLISHED_MD):
        (session_dir / name).unlink(missing_ok=True)


def session_tracks(session_dir: Path) -> list[str]:
    tracks = read_meta(session_dir).get("tracks")
    if tracks:
        return list(tracks)
    found = [t for t in TRACK_LABELS if (session_dir / f"{t}.mp3").exists()
             or any((session_dir / "chunks").glob(f"{t}-*.mp3"))]
    return found


def merge_session(session_dir: Path, emit: Emit) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    chunks_dir = session_dir / "chunks"
    for track in session_tracks(session_dir):
        target = session_dir / f"{track}.mp3"
        chunks = sorted(chunks_dir.glob(f"{track}-*.mp3")) if chunks_dir.exists() else []
        if chunks:
            emit({"type": "stage", "text": f"正在合并{TRACK_LABELS[track]}录音（{len(chunks)} 个分块）…"})
            if target.exists():  # a previous merge was interrupted; rebuild from the chunks
                target.unlink()
            audio.merge_chunks(chunks, target)
        if target.exists():
            merged[track] = target
    if chunks_dir.exists() and not any(chunks_dir.iterdir()):
        chunks_dir.rmdir()
    if not merged:
        raise TranscribeError("这个会话里没有录音文件。")
    return merged


def prepare_parts(session_dir: Path, merged: dict[str, Path], state: dict, emit: Emit) -> None:
    for track, source in merged.items():
        entry = state["tracks"].get(track)
        mtime = source.stat().st_mtime
        if entry and entry.get("source_mtime") == mtime and all(
            Path(p["path"]).exists() for p in entry["parts"]
        ):
            continue
        label = TRACK_LABELS[track]
        emit({"type": "stage", "text": f"正在检查{label}音频大小，必要时切分…"})
        parts = audio.plan_uploads(source, session_dir / "parts")
        if len(parts) > 1:
            emit({"type": "stage", "text": f"{label}超过 25MB，已在停顿处切成 {len(parts)} 段。"})
        records = []
        for part in parts:
            silent = audio.is_silent(part.path)
            records.append({
                "path": str(part.path),
                "start": round(part.start, 3),
                "end": round(part.end, 3),
                "status": "silent" if silent else "pending",
                "text": "",
            })
        state["tracks"][track] = {"source_mtime": mtime, "parts": records}
        save_state(session_dir, state)


def prepare_session(session_dir: Path, emit: Emit) -> tuple[dict[str, Path], dict]:
    """Merge chunks and split oversized tracks; no API calls. Safe to run repeatedly."""
    merged = merge_session(session_dir, emit)
    state = load_state(session_dir)
    state.setdefault("tracks", {})
    prepare_parts(session_dir, merged, state, emit)
    return merged, state


def describe_parts(state: dict) -> str:
    pieces = []
    for track, entry in state.get("tracks", {}).items():
        parts = entry["parts"]
        silent = sum(p["status"] == "silent" for p in parts)
        text = f"{TRACK_LABELS[track]} {len(parts)} 段" if len(parts) > 1 else f"{TRACK_LABELS[track]} 1 个文件"
        if silent == len(parts):
            text += "（静音）"
        pieces.append(text)
    return "，".join(pieces)


def run_session(session_dir: Path, settings: Settings, api_key: str, emit: Emit, *, http=requests) -> str:
    """Merge chunks, split if needed, transcribe every pending part. Resumable."""
    merged, state = prepare_session(session_dir, emit)

    languages = parse_languages(settings.languages)
    keywords = clean_keywords(settings.vocabulary)
    tracks = [t for t in merged if t in state["tracks"]]
    total = sum(p["end"] - p["start"] for t in tracks for p in state["tracks"][t]["parts"]) or 1.0
    done = sum(p["end"] - p["start"] for t in tracks for p in state["tracks"][t]["parts"]
               if p["status"] in ("done", "silent"))

    for track in tracks:
        parts = state["tracks"][track]["parts"]
        label = TRACK_LABELS[track]
        previous = ""
        for i, part in enumerate(parts, 1):
            where = f"{label} 第 {i}/{len(parts)} 段"
            if part["status"] in ("done", "silent"):
                previous = part["text"] or previous
                continue
            emit({"type": "progress", "phase": "transcribe", "text": f"正在转写{where}…",
                  "done": done, "total": total, "busy": True})
            try:
                text = transcribe_file(
                    Path(part["path"]),
                    api_key,
                    languages=languages,
                    keywords=keywords,
                    prompt=build_prompt(settings.context, previous),
                    on_delta=lambda t, tr=track: emit({"type": "delta", "track": tr, "text": t}),
                    http=http,
                )
            except TranscribeError as exc:
                part["status"] = "error"
                part["error"] = str(exc)
                save_state(session_dir, state)
                raise TranscribeError(f"{where}：{exc}") from exc
            part.update(status="done", text=text.strip(), error=None)
            save_state(session_dir, state)
            previous = part["text"]
            done += part["end"] - part["start"]
            emit({"type": "progress", "phase": "transcribe", "text": f"{where}完成",
                  "done": done, "total": total, "busy": False})

    markdown = render_markdown(session_dir, state, tracks)
    (session_dir / "transcript.md").write_text(markdown, encoding="utf-8")
    meta = read_meta(session_dir)
    meta["status"] = "transcribed"
    write_meta(session_dir, meta)
    return markdown


def polish_only(session_dir: Path, settings: Settings, api_key: str, emit: Emit, *, redo: bool = False,
                http=requests) -> str:
    """Polish an already transcribed session (resumable). redo=True discards the previous polish."""
    state = load_state(session_dir)
    tracks = [t for t, entry in state.get("tracks", {}).items()
              if all(p["status"] in ("done", "silent") for p in entry["parts"])]
    if not tracks or len(tracks) != len(state.get("tracks", {})):
        raise TranscribeError("这个会话还没有转写完成，请先点击“转写 / 继续”。")
    if redo:
        for name in (POLISH_STATE, POLISHED_MD):
            (session_dir / name).unlink(missing_ok=True)
    markdown = polish_session(session_dir, state, tracks, settings, api_key, emit, http=http)
    if not (session_dir / POLISHED_MD).exists():
        raise TranscribeError("转写内容太短（每条音轨不足 200 字），没有可润色的内容。")
    meta = read_meta(session_dir)
    meta["status"] = "transcribed"
    write_meta(session_dir, meta)
    return markdown


def polish_session(session_dir: Path, state: dict, tracks: list[str], settings: Settings, api_key: str,
                   emit: Emit, *, http=requests) -> str:
    """Strict gpt-6-luna polish of every track; cached per chunk in polish_state.json."""
    meta = read_meta(session_dir)
    meta["status"] = "polishing"
    write_meta(session_dir, meta)
    path = session_dir / POLISH_STATE
    try:
        cache_all = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache_all = {}
    lock = threading.Lock()

    def save() -> None:
        with lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache_all, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    vocabulary = clean_keywords(settings.vocabulary)
    results: dict[str, dict] = {}
    for track in tracks:
        text = "\n".join(p["text"] for p in state["tracks"][track]["parts"] if p.get("text")).strip()
        if len(text) < polisher.MIN_CHARS:
            continue
        label = TRACK_LABELS[track]
        cache = cache_all.setdefault(track, {})
        if cache.get("glossary") is None or cache.get("source_hash") != polisher.text_hash(text):
            emit({"type": "stage", "phase": "polish", "text": f"正在用 {polisher.MODEL} 通读{label}原文，生成统一术语表…"})

        def progress(done: int, total: int, paragraphs: list[str], tr=track, lb=label) -> None:
            emit({"type": "progress", "phase": "polish", "text": f"正在润色{lb}：已完成 {done}/{total} 块",
                  "done": done, "total": total or 1, "busy": False})
            emit({"type": "polished_delta", "track": tr, "text": "\n\n".join(paragraphs)})

        try:
            results[track] = polisher.polish_strict(text, api_key, vocabulary, cache, save, progress, http=http)
        except polisher.PolishError as exc:
            raise TranscribeError(f"润色失败（已完成的块已保存，可点“润色 / 继续”只补做剩下的块）：{label} {exc}") from exc

    if not results:
        return (session_dir / "transcript.md").read_text(encoding="utf-8")
    markdown = render_polished_markdown(session_dir, results)
    (session_dir / POLISHED_MD).write_text(markdown, encoding="utf-8")
    return markdown


def render_polished_markdown(session_dir: Path, results: dict[str, dict]) -> str:
    meta = read_meta(session_dir)
    started = meta.get("started_at") or session_dir.name
    try:
        started = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass
    lines = [f"# 录音文字稿 {started}", "",
             f"> 由 {polisher.MODEL} 只做润色（修正错字、标点、去口头禅）和分段，不总结、不改写；原始转写见 transcript.md。", ""]
    multi = len(results) > 1
    for track, r in results.items():
        if multi:
            lines += [f"## {TRACK_LABELS[track]}", ""]
        for p in r["paragraphs"]:
            lines += [p, ""]
    for track, r in results.items():
        if r["corrections"]:
            title = f"附录：错字错词修订（{TRACK_LABELS[track]}）" if multi else "附录：错字错词修订"
            lines += ["---", "", f"## {title}", "", "| 原文 | 修订为 | 理由 |", "|---|---|---|"]
            for c in r["corrections"]:
                cells = [c[k].replace("|", "｜").replace("\n", " ") for k in ("original", "corrected", "reason")]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        if r["uncertain"]:
            lines += ["## 附录：无法确定的地方（保留原文）", ""] + [f"- {u}" for u in r["uncertain"]] + [""]
    return "\n".join(lines).rstrip() + "\n"


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_markdown(session_dir: Path, state: dict, tracks: list[str]) -> str:
    meta = read_meta(session_dir)
    started = meta.get("started_at") or session_dir.name
    try:
        started = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass
    lines = [f"# 录音转写 {started}", ""]
    for track in tracks:
        lines += [f"## {TRACK_LABELS[track]}", ""]
        for part in state["tracks"][track]["parts"]:
            if part["status"] == "silent":
                lines.append(f"[{_hms(part['start'])}] （静音，未转写）")
            else:
                lines.append(f"[{_hms(part['start'])}] {part['text']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
