from hoah_lite import audio
from hoah_lite.recorder import Mp3ChunkWriter, repair_incomplete

from .conftest import tone


def test_writer_rotates_chunks_by_sample_count(tmp_path):
    writer = Mp3ChunkWriter(tmp_path, "microphone", chunk_seconds=10)
    pcm = tone(25)
    for start in range(0, len(pcm), 1600):  # feed 100 ms blocks like the capture thread
        writer.write(pcm[start:start + 1600])
    paths = writer.finish()
    assert [p.name for p in paths] == ["microphone-0000.mp3", "microphone-0001.mp3", "microphone-0002.mp3"]
    durations = [audio.duration(p) for p in paths]
    assert abs(durations[0] - 10) < 0.3 and abs(durations[2] - 5) < 0.3
    assert not list(tmp_path.glob("*.part"))


def test_repair_turns_part_into_mp3(tmp_path):
    chunks = tmp_path / "20260101-000000" / "chunks"
    chunks.mkdir(parents=True)
    good = tmp_path / "src.mp3"
    audio.encode_pcm(tone(5).tobytes(), good)
    data = good.read_bytes()
    (chunks / "microphone-0000.mp3.part").write_bytes(data[: len(data) * 2 // 3])  # truncated by a "crash"
    repaired = repair_incomplete(tmp_path)
    assert len(repaired) == 1
    assert (chunks / "microphone-0000.mp3").exists()
    assert audio.duration(chunks / "microphone-0000.mp3") > 2
