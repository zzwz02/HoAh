"""Speaker separation for a HoAh-lite session using gpt-6-luna (semantics) + local acoustic features.

Only current (non-deprecated) OpenAI models are used: gpt-transcribe and gpt-6-luna.
gpt-6-luna accepts only text/images and gpt-transcribe returns no timestamps, so:
  1. Each track is split locally at pauses (PyDub); the cut points are the timestamps.
  2. Every utterance is transcribed with gpt-transcribe (keywords from the vocabulary).
  3. Each utterance gets acoustic features computed locally (pitch, energy, MFCC timbre)
     and an unsupervised voice cluster id.
  4. gpt-6-luna receives the utterances, features, clusters and the full-track
     gpt-transcribe text, and decides who said what.

Usage:
    .venv\\Scripts\\python.exe tools\\diarize_luna.py <session_dir> [--dry-run]

--dry-run only splits and computes features, without calling the API.
--text-only skips all audio work: the existing transcript text goes straight to gpt-6-luna
            (writes speakers_text.json / speakers_text.md).
--render    rebuilds speakers.md from an existing speakers.json without calling the API.

Writes speakers.json and speakers.md into the session folder. The API key comes from
Windows Credential Manager (the one HoAh-lite saved), or OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hoah_lite import audio, config  # noqa: E402
from hoah_lite.transcriber import TranscribeError, clean_keywords, parse_languages, transcribe_file  # noqa: E402

RATE = audio.SAMPLE_RATE
RESPONSES_URL = "https://api.openai.com/v1/responses"
LUNA = "gpt-6-luna"
TRACK_LABELS = {"microphone": "麦克风（本机）", "system": "系统声音（线上/外放）"}
MIN_PAUSE_MS = 350  # a pause this long ends an utterance (speaker changes almost always sit on a pause)
MIN_UTTERANCE_S = 1.2
MAX_UTTERANCE_S = 25.0


# --- 1. utterances split at pauses, transcribed with gpt-transcribe -----------------------

def utterances(pcm: np.ndarray) -> list[tuple[float, float]]:
    seg = AudioSegment((pcm * 32767).astype(np.int16).tobytes(), sample_width=2, frame_rate=RATE, channels=1)
    thresh = max(seg.dBFS - 16, -45)  # relative to the track, but never below -45 dBFS (room noise)
    spans = [(s / 1000, e / 1000) for s, e in
             detect_nonsilent(seg, min_silence_len=MIN_PAUSE_MS, silence_thresh=thresh, seek_step=10)]
    merged: list[list[float]] = []
    for s, e in spans:  # glue very short bursts to their neighbour
        if merged and (e - merged[-1][0] <= MIN_UTTERANCE_S or s - merged[-1][1] < 0.15):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    result = []
    for s, e in merged:  # long monologues: cut at the quietest point so each piece stays short
        while e - s > MAX_UTTERANCE_S:
            window = seg[int((s + MAX_UTTERANCE_S * 0.6) * 1000):int((s + MAX_UTTERANCE_S) * 1000)]
            cut = s + MAX_UTTERANCE_S * 0.6 + audio.find_cut(window) / 1000
            result.append((s, cut))
            s = cut
        result.append((s, e))
    pad = 0.12
    total = len(pcm) / RATE
    return [(max(0.0, s - pad), min(total, e + pad)) for s, e in result if e - s >= 0.4]


def transcribe_utterances(pcm: np.ndarray, spans: list[tuple[float, float]], key: str,
                          settings: config.Settings, workdir: Path, label: str) -> list[str]:
    keywords = clean_keywords(settings.vocabulary)
    languages = parse_languages(settings.languages)
    done = [0]

    def one(i: int) -> str:
        s, e = spans[i]
        clip = pcm[int(s * RATE):int(e * RATE)]
        if np.abs(clip).max() < 0.01:  # below -40 dBFS: nothing worth paying for
            return ""
        path = workdir / f"{label}-{i:04d}.mp3"
        audio.encode_pcm((clip * 32767).astype(np.int16).tobytes(), path)
        try:
            text = transcribe_file(path, key, languages=languages, keywords=keywords, prompt=settings.context)
        except TranscribeError as exc:
            if "HTTP 4" in str(exc) and "429" not in str(exc):
                raise
            text = ""  # stream ended with no speech: treat as silence
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(spans):
            print(f"  gpt-transcribe {done[0]}/{len(spans)}")
        return text.strip()

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, range(len(spans))))


# --- 2. acoustic features -----------------------------------------------------------------

def decode(path: Path) -> np.ndarray:
    proc = audio.run_ffmpeg(["-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(RATE), "pipe:1"])
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _mel_filterbank(n_fft: int, n_mels: int = 26) -> np.ndarray:
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz_to_mel(60), hz_to_mel(RATE / 2 - 200), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / RATE).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        lo, mid, hi = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, lo:mid] = (np.arange(lo, mid) - lo) / max(mid - lo, 1)
        fb[i - 1, mid:hi] = (hi - np.arange(mid, hi)) / max(hi - mid, 1)
    return fb


N_FFT, FRAME, HOP = 512, 400, 160  # 25 ms frames, 10 ms hop
MEL = _mel_filterbank(N_FFT)
DCT = np.cos(np.pi / 26 * (np.arange(26)[None, :] + 0.5) * np.arange(13)[:, None])
WINDOW = np.hamming(FRAME)


def features(pcm: np.ndarray, start: float, end: float) -> dict | None:
    seg = pcm[int(start * RATE):int(end * RATE)]
    if len(seg) < FRAME * 10:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(seg, FRAME)[::HOP]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    voiced = rms > max(np.percentile(rms, 60), 0.003)  # speech frames of this segment
    if voiced.sum() < 10:
        return None
    vf = frames[voiced] * WINDOW
    spec = np.abs(np.fft.rfft(vf, N_FFT)) ** 2
    mfcc = DCT @ np.log(MEL @ spec.T + 1e-10)  # 13 x frames
    freqs = np.fft.rfftfreq(N_FFT, 1 / RATE)
    centroid = (spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-12)
    # pitch by autocorrelation, 70–400 Hz
    f0 = []
    for fr in vf[:: max(1, len(vf) // 120)]:
        ac = np.correlate(fr, fr, "full")[FRAME - 1:]
        lo, hi = RATE // 400, RATE // 70
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[lag] / ac[0] > 0.3:
            f0.append(RATE / lag)
    return {
        "pitch_hz": round(float(np.median(f0)), 1) if f0 else None,
        "loudness_db": round(float(20 * np.log10(rms[voiced].mean())), 1),
        "brightness_hz": round(float(np.median(centroid)), 0),
        "mfcc": mfcc[1:].mean(axis=1),  # drop c0 (loudness); used for clustering only
    }


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = (None, np.inf)
    for _ in range(8):
        c = x[rng.choice(len(x), k, replace=False)]
        for _ in range(iters):
            labels = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[labels == j].mean(0) if (labels == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[labels]) ** 2).sum()
        if inertia < best[1]:
            best = (labels, inertia)
    return best


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() <= 1:
            scores.append(0.0)
            continue
        a = d[i, same].sum() / (same.sum() - 1)
        b = min(d[i, labels == j].mean() for j in set(labels) if j != labels[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def voice_clusters(segments: list[dict]) -> int:
    idx = [i for i, s in enumerate(segments) if s.get("_mfcc") is not None]
    if len(idx) < 6:
        return 0
    x = np.array([np.concatenate([segments[i]["_mfcc"], [segments[i]["pitch_hz"] or 0, segments[i]["brightness_hz"]]])
                  for i in idx])
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    choice = (1, None, -1.0)
    for k in range(2, min(6, len(idx) - 1)):
        labels, _ = kmeans(x, k)
        score = silhouette(x, labels)
        if score > choice[2]:
            choice = (k, labels, score)
    k, labels, score = choice
    for i, label in zip(idx, labels):
        segments[i]["voice_cluster"] = f"V{int(label) + 1}"
    for i in range(1, len(idx)):  # timbre change versus the previous segment of the same track
        a, b = x[i - 1], x[i]
        segments[idx[i]]["timbre_change"] = round(float(np.linalg.norm(a - b) / np.sqrt(len(a))), 2)
    print(f"  声学聚类：{k} 个音色簇（silhouette={score:.2f}）")
    return k


# --- 3. gpt-6-luna -----------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["speakers", "turns", "notes"],
    "properties": {
        "speakers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "role", "evidence"],
            "properties": {
                "id": {"type": "string", "description": "说话人1、说话人2…"},
                "role": {"type": "string", "description": "从内容推断的身份/角色，例如“汇报人”“部门负责人”；不要编造真实姓名"},
                "evidence": {"type": "string", "description": "判断依据：语义线索 + 声学线索"},
            }}},
        "turns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["speaker", "segment_ids", "start", "end", "text", "confidence"],
            "properties": {
                "speaker": {"type": "string"},
                "segment_ids": {"type": "array", "items": {"type": "integer"}},
                "start": {"type": "number"},
                "end": {"type": "number"},
                "text": {"type": "string", "description": "这一轮发言的文字，优先采用参考高精度转写中的对应文字"},
                "confidence": {"type": "string", "enum": ["高", "中", "低"]},
            }}},
        "notes": {"type": "string", "description": "整体判断说明、不确定的地方"},
    },
}

INSTRUCTIONS = """你是会议录音的说话人分离专家。你会拿到：
A. 按停顿切分的语音片段（start/end 是切点时间，文字由 gpt-transcribe 逐段转写；片段很短时上下文少，个别字可能不准），每段带有本地计算的声学特征：
   pitch_hz（基频中位数，男声通常 85–180Hz，女声 165–300Hz）、loudness_db、brightness_hz（频谱质心）、
   voice_cluster（无监督音色聚类结果，可能有误）、timbre_change（与同轨上一段相比的音色变化，越大越可能换人）。
B. 每条音轨整段转写的全文（gpt-transcribe，上下文完整，无时间戳）。片段文字与它冲突时，以全文为准修正 turns 的文字。
音轨说明：麦克风轨是本机用户；系统声音轨是电脑播放的声音（线上会议的其他参会者）。同一句话可能同时出现在两条轨上（外放回声），只保留一次。

请综合语义（话题衔接、提问与回答、称呼、“我”“你们”等人称、汇报与点评的口吻、总结收尾）和声学线索判断说话人：
- 声学聚类只是参考，语义证据明确时以语义为准；两者冲突时在 confidence 里体现。
- 把连续属于同一人的片段合并成一轮发言（turns），按时间顺序输出，覆盖所有片段。
- 不要猜测真实姓名；用“说话人1/2/…”，在 role 中描述身份。
- 一个片段内部如果明显换人，可以把它拆到两轮里（segment_ids 可重复），并在 notes 说明。"""


TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["speakers", "turns", "notes"],
    "properties": {
        "speakers": SCHEMA["properties"]["speakers"],
        "turns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["speaker", "text", "confidence"],
            "properties": {
                "speaker": {"type": "string"},
                "text": {"type": "string", "description": "这一轮发言的原文，按原顺序，不删减、不润色"},
                "confidence": {"type": "string", "enum": ["高", "中", "低"]},
            }}},
        "notes": {"type": "string", "description": "整体判断说明、不确定的地方"},
    },
}

TEXT_INSTRUCTIONS = """你是会议记录的说话人分离专家。你只会拿到会议录音的转写文字（没有时间戳、没有声音信息），按音轨给出：
麦克风轨是本机用户；系统声音轨是电脑播放的声音（线上会议的其他参会者）。同一句话可能同时出现在两条轨上（外放回声），只保留一次。

请只根据语义判断说话人：话题衔接、提问与回答、对他人的回应（“你刚刚讲的”“回到刚刚的问题”）、人称（我/你/我们/你们）、
汇报与点评/布置工作的口吻差异、开场与收尾语、提到的人名与对方身份的关系等。
- 按原文顺序把文字切成若干轮发言（turns），每轮标注说话人；所有原文都要覆盖，不删减、不改写、不总结。
- 一段话中间如果换人（例如一句简短的回应、附和），也要单独拆出来。
- 不要猜测真实姓名；用“说话人1/2/…”，在 role 中描述身份，evidence 写清楚语义依据。
- 说话人数量由你判断，不要为了凑数拆人，也不要把明显不同的人合并。"""


def ask_luna(key: str, payload: dict, instructions: str = INSTRUCTIONS, schema: dict = SCHEMA) -> dict:
    body = {
        "model": LUNA,
        "reasoning": {"effort": "high"},
        "input": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "text": {"format": {"type": "json_schema", "name": "diarization", "schema": schema, "strict": True}},
    }
    for attempt in range(4):
        r = requests.post(RESPONSES_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=(15, 900))
        if r.status_code == 200:
            data = r.json()
            for item in data.get("output", []):
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        return {"result": json.loads(part["text"]), "usage": data.get("usage", {})}
            raise SystemExit(f"{LUNA} 没有返回文本：{json.dumps(data)[:500]}")
        if r.status_code != 429 and r.status_code < 500:
            raise SystemExit(f"{LUNA} 失败：HTTP {r.status_code} {r.text[:500]}")
        time.sleep(2 ** (attempt + 1))
    raise SystemExit(f"{LUNA} 多次重试失败")


# --- main --------------------------------------------------------------------------------------

def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def text_only(session: Path, key: str) -> None:
    """Semantic-only pass: the existing transcript.json text goes straight to gpt-6-luna."""
    state = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
    tracks = {TRACK_LABELS[t]: "\n".join(p["text"] for p in entry["parts"] if p.get("text"))
              for t, entry in state["tracks"].items()}
    payload = {"transcripts": tracks, "vocabulary": clean_keywords(config.Settings.load().vocabulary)}
    print(f"[{LUNA}] 纯文本分析（{sum(len(v) for v in tracks.values())} 字）…")
    answer = ask_luna(key, payload, TEXT_INSTRUCTIONS, TEXT_SCHEMA)
    result, usage = answer["result"], answer["usage"]
    (session / "speakers_text.json").write_text(
        json.dumps({"model": LUNA, **result, "usage": usage}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# 说话人分离（纯文本）：{session.name}", "",
             f"方法：只把已有的 gpt-transcribe 转写文字交给 {LUNA} 做语义判断，不使用声音信息和时间戳。",
             "", "## 说话人", ""]
    for sp in result["speakers"]:
        lines.append(f"- **{sp['id']}**：{sp['role']}。依据：{sp['evidence']}")
    lines += ["", "## 对话", ""]
    for t in result["turns"]:
        mark = "" if t["confidence"] == "高" else f"（置信度：{t['confidence']}）"
        lines += [f"**{t['speaker']}**{mark}：{t['text']}", ""]
    lines += ["## 说明", "", result["notes"], ""]
    (session / "speakers_text.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"完成：{session / 'speakers_text.md'}")
    print(f"token 用量：{usage}")


def main() -> None:
    session = Path(sys.argv[1])
    if "--render" in sys.argv:  # rebuild speakers.md from an existing speakers.json, no API calls
        render_speakers_md(session, json.loads((session / "speakers.json").read_text(encoding="utf-8")))
        print(f"完成：{session / 'speakers.md'}")
        return
    if "--text-only" in sys.argv:
        key, _ = config.get_api_key()
        if not key:
            raise SystemExit("没有找到 API Key（凭据管理器或 OPENAI_API_KEY）。")
        text_only(session, key)
        return
    dry_run = "--dry-run" in sys.argv  # split + features only, no API calls
    key, _ = config.get_api_key()
    if not key and not dry_run:
        raise SystemExit("没有找到 API Key（凭据管理器或 OPENAI_API_KEY）。")
    settings = config.Settings.load()
    vocab = clean_keywords(settings.vocabulary)
    state = json.loads((session / "transcript.json").read_text(encoding="utf-8")) \
        if (session / "transcript.json").exists() else {"tracks": {}}
    workdir = Path(tempfile.mkdtemp(prefix="hoah-diarize-"))

    segments: list[dict] = []
    reference: dict[str, str] = {}
    for track in ("microphone", "system"):
        path = session / f"{track}.mp3"
        if not path.exists() or audio.is_silent(path):
            continue
        pcm = decode(path)
        spans = utterances(pcm)
        print(f"[{TRACK_LABELS[track]}] 按停顿切成 {len(spans)} 段，"
              f"合计 {sum(e - s for s, e in spans) / 60:.1f} 分钟语音")
        texts = [""] * len(spans) if dry_run else transcribe_utterances(pcm, spans, key, settings, workdir, track)
        track_segs = []
        for (start, end), text in zip(spans, texts):
            if not text and not dry_run:
                continue
            f = features(pcm, start, end)
            seg = {"track": track, "start": round(start, 2), "end": round(end, 2), "text": text}
            if f:
                seg.update(pitch_hz=f["pitch_hz"], loudness_db=f["loudness_db"], brightness_hz=f["brightness_hz"],
                           _mfcc=f["mfcc"])
            track_segs.append(seg)
        print(f"  {len(track_segs)} 个片段")
        voice_clusters(track_segs)
        segments += track_segs
        parts = state["tracks"].get(track, {}).get("parts", [])
        reference[track] = "\n".join(p["text"] for p in parts if p.get("text"))

    if dry_run:
        for s in segments[:40]:
            print(f"  {s['track'][:3]} {hms(s['start'])} {s['end'] - s['start']:5.1f}s "
                  f"pitch={s.get('pitch_hz')} bright={s.get('brightness_hz')} {s.get('voice_cluster', '-')}")
        return
    segments.sort(key=lambda s: (s["start"], s["track"]))
    for i, s in enumerate(segments):
        s["id"] = i
    payload = {
        "segments": [{k: v for k, v in s.items() if not k.startswith("_")} for s in segments],
        "reference_transcripts": {TRACK_LABELS[t]: txt for t, txt in reference.items()},
        "vocabulary": vocab,
    }
    print(f"[{LUNA}] 分析 {len(segments)} 个片段…")
    answer = ask_luna(key, payload)
    result, usage = answer["result"], answer["usage"]

    out = {"model": LUNA, "segments": payload["segments"], **result, "usage": usage}
    (session / "speakers.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    render_speakers_md(session, out)
    print(f"完成：{session / 'speakers.md'}")
    print(f"token 用量：{usage}")


def render_speakers_md(session: Path, data: dict) -> None:
    """Turn text is rebuilt from the segments themselves, so nothing the model shortened is lost."""
    segs = {s["id"]: s for s in data["segments"]}
    used: set[int] = set()
    lines = [f"# 说话人分离：{session.name}", "",
             f"方法：本地按停顿分段 + gpt-transcribe 逐段转写 + 本地声学特征（基频/音色聚类）+ {LUNA} 语义判断。"
             "对话文字取自逐段转写原文（未经模型改写）。",
             "", "## 说话人", ""]
    for sp in data["speakers"]:
        lines.append(f"- **{sp['id']}**：{sp['role']}。依据：{sp['evidence']}")
    lines += ["", "## 对话", ""]
    for t in data["turns"]:
        ids = [i for i in t["segment_ids"] if i in segs and i not in used]
        used.update(ids)
        text = "".join(segs[i]["text"] for i in ids) or t["text"]
        start = segs[ids[0]]["start"] if ids else t["start"]
        mark = "" if t["confidence"] == "高" else f"（置信度：{t['confidence']}）"
        lines += [f"**[{hms(start)}] {t['speaker']}**{mark}：{text}", ""]
    lines += ["## 说明", "", data["notes"], ""]
    (session / "speakers.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
