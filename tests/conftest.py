import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hoah_lite import audio  # noqa: E402

RATE = audio.SAMPLE_RATE


def tone(seconds: float, freq: float = 220.0, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    # amplitude-modulated tone so it is not one flat level ("speech-like")
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    return (np.sin(2 * np.pi * freq * t) * env * amp * 32767).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.int16)


@pytest.fixture
def make_mp3(tmp_path):
    def _make(name: str, *pieces: np.ndarray) -> Path:
        path = tmp_path / name
        audio.encode_pcm(np.concatenate(pieces).tobytes(), path)
        return path

    return _make
