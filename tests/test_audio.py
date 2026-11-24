from hoah_lite import audio

from .conftest import silence, tone


def test_merge_chunks_concatenates_and_removes_chunks(tmp_path, make_mp3):
    chunks = [make_mp3(f"microphone-{i:04d}.mp3", tone(20)) for i in range(3)]
    out = tmp_path / "microphone.mp3"
    total = audio.merge_chunks(chunks, out)
    assert abs(total - 60) < 1.0
    assert abs(audio.duration(out) - 60) < 1.0
    assert not any(c.exists() for c in chunks)


def test_small_file_is_uploaded_whole(tmp_path, make_mp3):
    src = make_mp3("microphone.mp3", tone(30))
    parts = audio.plan_uploads(src, tmp_path / "parts")
    assert len(parts) == 1 and parts[0].path == src
    assert not (tmp_path / "parts").exists()


def test_large_file_is_split_in_pauses_under_limit(tmp_path, make_mp3):
    # 5 x (55 s speech + 2 s pause) at 64 kbps ≈ 2.3 MB; shrink the limits so it must split
    pieces = []
    for _ in range(5):
        pieces += [tone(55), silence(2)]
    src = make_mp3("microphone.mp3", *pieces)
    limit = 800_000  # ~100 s per part
    parts = audio.plan_uploads(src, tmp_path / "parts", direct_limit=1_000_000, part_limit=limit)
    assert len(parts) >= 3
    assert parts[0].start == 0 and abs(parts[-1].end - audio.duration(src)) < 0.5
    for a, b in zip(parts, parts[1:]):
        assert a.end == b.start
        # every cut lands inside a 2 s pause: pauses sit at 55-57, 112-114, 169-171, ...
        offset = a.end % 57
        assert 55 <= offset <= 57, a.end
    for p in parts:
        assert p.path.stat().st_size <= limit


def test_find_cut_falls_back_to_quietest_frame():
    from pydub import AudioSegment
    import numpy as np

    loud = tone(3, amp=0.5)
    quiet = tone(0.3, amp=0.05)
    pcm = np.concatenate([loud, quiet, loud])
    seg = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=16000, channels=1)
    cut = audio.find_cut(seg)
    assert 3000 <= cut <= 3300


def test_silent_detection(make_mp3):
    assert audio.is_silent(make_mp3("system.mp3", silence(10)))
    assert not audio.is_silent(make_mp3("microphone.mp3", tone(10)))
