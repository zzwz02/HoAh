import json

import pytest
import requests

from hoah_lite import transcriber
from hoah_lite.config import Settings
from hoah_lite.recorder import write_meta
from hoah_lite.transcriber import TranscribeError, clean_keywords, transcribe_file

from .conftest import silence, tone


class FakeResponse:
    def __init__(self, status=200, events=None, body=None, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.reason = "x"
        self._events = events or []
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=False):
        for event in self._events:
            yield f"data: {json.dumps(event)}".encode()
            yield b""


def ok(text):
    return FakeResponse(events=[
        {"type": "transcript.text.delta", "delta": text[:2]},
        {"type": "transcript.text.delta", "delta": text[2:]},
        {"type": "transcript.text.done", "text": text},
    ])


class FakeHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_clean_keywords():
    vocab = "Kubernetes\n\n<HPC>\nkubernetes\nPHS Epic\r\n" + "\n".join(f"w{i}" for i in range(200))
    words = clean_keywords(vocab)
    assert words[:3] == ["Kubernetes", "HPC", "PHS Epic"]
    assert len(words) == 100


def test_request_fields_and_streaming(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    http = FakeHttp([ok("你好 world")])
    deltas = []
    text = transcribe_file(f, "sk-test", languages=["zh-cn", "en"], keywords=["HPC"],
                           prompt="ctx", on_delta=deltas.append, http=http)
    assert text == "你好 world"
    assert deltas[-1] == "你好 world"
    data = http.calls[0]["data"]
    assert ("model", "gpt-transcribe") in data and ("stream", "true") in data
    assert ("languages[]", "zh-cn") in data and ("languages[]", "en") in data
    assert ("keywords[]", "HPC") in data and ("prompt", "ctx") in data
    assert not any(k == "language" for k, _ in data)


def test_retries_429_then_succeeds(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    http = FakeHttp([FakeResponse(429, headers={"Retry-After": "1"}),
                     requests.ConnectionError("boom"), ok("done")])
    sleeps = []
    assert transcribe_file(f, "k", languages=[], keywords=[], prompt="", http=http, sleep=sleeps.append) == "done"
    assert sleeps == [1.0, 4.0]


def test_400_fails_immediately_with_message(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    http = FakeHttp([FakeResponse(400, body={"error": {"message": "Invalid keyword"}})])
    with pytest.raises(TranscribeError, match="Invalid keyword"):
        transcribe_file(f, "k", languages=[], keywords=[], prompt="", http=http, sleep=lambda s: None)
    assert len(http.calls) == 1


def test_session_resumes_after_failure(tmp_path, make_mp3, monkeypatch):
    session = tmp_path / "20260101-090000"
    (session / "chunks").mkdir(parents=True)
    write_meta(session, {"tracks": ["microphone", "system"], "status": "recorded"})
    make_mp3("20260101-090000/chunks/microphone-0000.mp3", tone(5))
    make_mp3("20260101-090000/chunks/microphone-0001.mp3", tone(5))
    make_mp3("20260101-090000/chunks/system-0000.mp3", silence(10))
    events = []

    failing = FakeHttp([FakeResponse(401, body={"error": {"message": "Incorrect API key"}})])
    with pytest.raises(TranscribeError, match="麦克风 第 1/1 段.*Incorrect API key"):
        transcriber.run_session(session, Settings(), "bad", events.append, http=failing)
    assert (session / "microphone.mp3").exists() and not (session / "chunks").exists()

    good = FakeHttp([ok("会议内容")])
    md = transcriber.run_session(session, Settings(), "good", events.append, http=good)
    assert len(good.calls) == 1  # silent system track is never uploaded
    assert "## 麦克风" in md and "会议内容" in md and "（静音，未转写）" in md
    assert any(e["type"] == "progress" for e in events)

    again = FakeHttp([])
    transcriber.run_session(session, Settings(), "good", events.append, http=again)
    assert again.calls == []  # everything cached
