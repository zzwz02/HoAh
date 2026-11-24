import json

import pytest

from hoah_lite import polisher, transcriber
from hoah_lite.config import Settings
from hoah_lite.recorder import read_meta, write_meta

from .conftest import tone


class Resp:
    def __init__(self, status=200, payload=None, error=None):
        self.status_code = status
        self.headers = {}
        self.reason = "x"
        self._payload = payload
        self._error = error
        self.text = json.dumps(error or {})

    def json(self):
        if self.status_code != 200:
            return {"error": {"message": self._error}}
        return {"output": [{"type": "message", "content": [{"type": "output_text",
                                                            "text": json.dumps(self._payload, ensure_ascii=False)}]}],
                "usage": {}}


class FakeLuna:
    """Answers glossary/chunk requests; chunks listed in fail_on get an HTTP 400."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.chunk_calls = []
        self.glossary_calls = 0

    def post(self, url, json=None, **kw):
        assert url == polisher.RESPONSES_URL
        name = json["text"]["format"]["name"]
        user = json["input"][1]["content"]
        if name == "glossary":
            self.glossary_calls += 1
            return Resp(payload={"terms": [{"original": "山叔", "corrected": "杉数", "reason": "公司名"}]})
        chunk = user.split("本块原文：\n", 1)[1]
        self.chunk_calls.append(chunk)
        if any(marker in chunk for marker in self.fail_on):
            return Resp(400, error="bad chunk")
        return Resp(payload={"paragraphs": [chunk.replace("山叔", "杉数")], "corrections": [], "uncertain": []})


def long_text(n=6):
    return "".join(f"第{i}段讲的是山叔的项目进展和后续计划。" * 40 for i in range(n))


def test_chunk_text_keeps_sentences_whole():
    text = long_text()
    chunks = polisher.chunk_text(text)
    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(c.endswith("。") for c in chunks)
    assert all(len(c) <= polisher.CHUNK_CHARS + 40 for c in chunks)


def test_failed_chunk_is_resumed_without_redoing_the_rest():
    text = long_text()
    cache, saves = {}, []
    progress = []
    first = FakeLuna(fail_on=["第3段"])
    with pytest.raises(polisher.PolishError, match="bad chunk"):
        polisher.polish_strict(text, "k", [], cache, lambda: saves.append(1), lambda d, n, p: progress.append(d),
                               http=first, sleep=lambda s: None)
    n = len(cache["chunks"])
    done_after_first = sum(c["done"] for c in cache["chunks"])
    assert 0 < done_after_first < n and cache["glossary"]

    second = FakeLuna()
    result = polisher.polish_strict(text, "k", [], cache, lambda: None, None, http=second, sleep=lambda s: None)
    assert second.glossary_calls == 0  # glossary reused from the cache
    assert len(second.chunk_calls) == n - done_after_first  # only the missing chunks
    assert "山叔" not in "".join(result["paragraphs"]) and "杉数" in result["paragraphs"][0]
    assert result["corrections"][0]["corrected"] == "杉数"


def test_changed_source_text_resets_the_cache():
    cache = {}
    polisher.polish_strict(long_text(3), "k", [], cache, lambda: None, http=FakeLuna(), sleep=lambda s: None)
    again = FakeLuna()
    polisher.polish_strict(long_text(4), "k", [], cache, lambda: None, http=again, sleep=lambda s: None)
    assert again.glossary_calls == 1 and len(again.chunk_calls) == len(cache["chunks"])


class PipelineHttp(FakeLuna):
    """Transcription endpoint streams a long text; Responses endpoint polishes."""

    def __init__(self, text, fail_on=()):
        super().__init__(fail_on)
        self.text = text
        self.transcribe_calls = 0

    def post(self, url, **kw):
        if url == transcriber.API_URL:
            self.transcribe_calls += 1
            from .test_transcriber import ok
            return ok(self.text)
        return super().post(url, **kw)


def make_session(tmp_path, make_mp3, name):
    session = tmp_path / name
    (session / "chunks").mkdir(parents=True)
    write_meta(session, {"tracks": ["system"], "status": "recorded"})
    make_mp3(f"{name}/chunks/system-0000.mp3", tone(5))
    return session


def test_transcription_alone_never_polishes(tmp_path, make_mp3):
    session = make_session(tmp_path, make_mp3, "20261104-110000")
    http = PipelineHttp(long_text())
    md = transcriber.run_session(session, Settings(), "k", lambda e: None, http=http)
    assert http.transcribe_calls == 1 and http.chunk_calls == [] and http.glossary_calls == 0
    assert not (session / transcriber.POLISHED_MD).exists()
    assert long_text() in md  # raw gpt-transcribe text, untouched


def test_polish_needs_a_finished_transcript(tmp_path, make_mp3):
    session = make_session(tmp_path, make_mp3, "20261104-120000")
    with pytest.raises(transcriber.TranscribeError, match="还没有转写完成"):
        transcriber.polish_only(session, Settings(), "k", lambda e: None, http=PipelineHttp(long_text()))


def test_separate_polish_resumes_and_redo_restarts(tmp_path, make_mp3):
    session = make_session(tmp_path, make_mp3, "20261104-100000")
    transcriber.run_session(session, Settings(), "k", lambda e: None, http=PipelineHttp(long_text()))
    raw = (session / "transcript.md").read_text(encoding="utf-8")
    events = []

    failing = PipelineHttp(long_text(), fail_on=["第4段"])
    with pytest.raises(transcriber.TranscribeError, match="润色失败"):
        transcriber.polish_only(session, Settings(), "k", events.append, http=failing)
    assert not (session / transcriber.POLISHED_MD).exists()
    assert any(e.get("phase") == "polish" and e["type"] == "progress" for e in events)
    assert any(e["type"] == "polished_delta" for e in events)

    resume = PipelineHttp(long_text())
    md = transcriber.polish_only(session, Settings(), "k", events.append, http=resume)
    assert resume.transcribe_calls == 0 and resume.glossary_calls == 0
    assert 0 < len(resume.chunk_calls) < len(failing.chunk_calls)  # only the missing chunks
    assert "杉数" in md and (session / transcriber.POLISHED_MD).read_text(encoding="utf-8") == md
    assert (session / "transcript.md").read_text(encoding="utf-8") == raw  # raw transcript never touched
    assert read_meta(session)["status"] == "transcribed"

    redo = PipelineHttp(long_text())
    transcriber.polish_only(session, Settings(), "k", lambda e: None, redo=True, http=redo)
    assert redo.glossary_calls == 1 and len(redo.chunk_calls) == len(polisher.chunk_text(long_text()))
