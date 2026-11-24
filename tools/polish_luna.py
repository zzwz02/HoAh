"""Polish an existing HoAh-lite transcript with gpt-6-luna from the command line.

Usage:
    .venv\\Scripts\\python.exe tools\\polish_luna.py <session_dir> [--strict]

--strict  The same strict mode the app runs after transcription (hoah_lite.polisher): fix typos,
          terms, punctuation and filler words, split into paragraphs; no headings, no rewording.
          Writes transcript_polished.md / polish_state.json, so the app shows it as 润色稿, and
          an interrupted run resumes with only the missing chunks.
(default) An editorial version: smoother wording (some words are dropped, ~85% length) and
          topic headings. Writes polished.md / polished.json.

Both modes first build a shared terminology table from the whole text and then work in ~1800
character chunks so the model edits instead of summarising. The API key comes from Windows
Credential Manager (HoAh-lite's) or OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hoah_lite import config, polisher  # noqa: E402
from hoah_lite.config import Settings  # noqa: E402
from hoah_lite.recorder import read_meta, write_meta  # noqa: E402
from hoah_lite.transcriber import TranscribeError, clean_keywords, load_state, polish_session  # noqa: E402

TRACK_LABELS = config.TRACK_LABELS

EDITORIAL_INSTRUCTIONS = """你是中文会议记录的校对和编辑。输入是一段会议录音语音识别原文中的一块（普通话为主，夹杂英文技术术语），
可能有大量同音错字、错词、断句错误、口头禅和重复。请把“本块原文”整理成忠实、通顺、分好段的文字：

1. 修订错字错词：根据上下文和行业常识纠正语音识别错误（同音字、近音词、术语、英文缩写、产品名、人名地名的明显误识别）。
   按统一术语表和词汇表修订；没把握的保留原文，并记入 uncertain。
2. 润色：补全标点、理顺断句，删除无意义的口头禅和机械重复（如“就是说就是说”“然后然后”“那个那个”）。
   这是校对而不是摘要：每一句的信息、细节、举例、数字、观点、问句和语气都要保留，不要合并成概括句，
   不要压缩、不要增加原文没有的内容。正常情况下整理后的字数应为原文的 80%–95%。
3. 分段：按意思自然分段（每段约 100–300 字），顺序与原文一致，覆盖“本块原文”的全部内容。
4. 不要区分说话人，不要加说话人标签。“上文”只用于理解语境，不要输出上文内容。
5. corrections 列出做过的有实质意义的错字/错词修订（原文片段 → 修订后，附简短理由）；纯标点和口头禅删除不用列。"""

OUTLINE_INSTRUCTIONS = """你会拿到一份已整理好的会议文字稿，按顺序编号的段落。请按话题把它们分成若干节，
每节给一个简短的小标题（不超过 16 个字），并给整份会议起一个标题。
每节用起始段落编号表示，第一节必须从 0 开始，编号递增；不要改写段落。"""

OUTLINE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["title", "sections"],
    "properties": {
        "title": {"type": "string"},
        "sections": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["start_paragraph", "heading"],
            "properties": {"start_paragraph": {"type": "integer"}, "heading": {"type": "string"}},
        }},
    },
}


def editorial(session: Path, key: str) -> None:
    vocab = clean_keywords(Settings.load().vocabulary)
    vocab_line = f"词汇表（可能出现的专有名词）：{'、'.join(vocab) if vocab else '（无）'}"
    state = load_state(session)
    lines = []
    for track, entry in state["tracks"].items():
        text = "\n".join(p["text"] for p in entry["parts"] if p.get("text")).strip()
        if len(text) < polisher.MIN_CHARS:
            continue
        print(f"[{TRACK_LABELS[track]}] 编辑版润色（{len(text)} 字）…")
        glossary, _ = polisher.call_luna(key, polisher.GLOSSARY_INSTRUCTIONS, f"{vocab_line}\n\n语音识别原文：\n{text}",
                                         polisher.GLOSSARY_SCHEMA, "glossary")
        terms = glossary["terms"]
        glossary_line = "统一术语表（原文中出现这些写法时必须按此修订）：\n" + (
            "\n".join(f"- {t['original']} → {t['corrected']}" for t in terms) if terms else "（无）")
        chunks = polisher.chunk_text(text)

        def one(i: int) -> dict:
            context = chunks[i - 1][-polisher.CONTEXT_CHARS:] if i else "（无，这是开头）"
            user = (f"{vocab_line}\n\n{glossary_line}\n\n上文（仅供理解语境，不要输出）：\n{context}"
                    f"\n\n本块原文：\n{chunks[i]}")
            result, _ = polisher.call_luna(key, EDITORIAL_INSTRUCTIONS, user, polisher.CHUNK_SCHEMA, "polished_chunk")
            print(f"  块 {i + 1}/{len(chunks)} 完成")
            return result

        with ThreadPoolExecutor(max_workers=polisher.WORKERS) as pool:
            outputs = list(pool.map(one, range(len(chunks))))
        paragraphs = [p.strip() for r in outputs for p in r["paragraphs"] if p.strip()]
        corrections = terms + [c for r in outputs for c in r["corrections"]]
        uncertain = [u for r in outputs for u in r["uncertain"]]
        numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs))
        outline, _ = polisher.call_luna(key, OUTLINE_INSTRUCTIONS, numbered, OUTLINE_SCHEMA, "outline", effort="low")
        starts = sorted({min(max(s["start_paragraph"], 0), len(paragraphs) - 1): s["heading"]
                         for s in outline["sections"]}.items())
        if not starts or starts[0][0] != 0:
            starts.insert(0, (0, "开场"))
        chars = sum(len(p) for p in paragraphs)
        lines += [f"# {outline['title']}", "",
                  f"> 来源：{session.name} {TRACK_LABELS[track]}轨原始转写（{len(text)} 字），"
                  f"由 {polisher.MODEL} 修订错字错词、润色并分段（{chars} 字）；未做说话人区分。", ""]
        for n, (start, heading) in enumerate(starts):
            end = starts[n + 1][0] if n + 1 < len(starts) else len(paragraphs)
            lines += [f"## {heading}", ""] + [x for p in paragraphs[start:end] for x in (p, "")]
        lines += ["---", "", "## 附录：错字错词修订", "", "| 原文 | 修订为 | 理由 |", "|---|---|---|"]
        lines += ["| " + " | ".join(c[k].replace("|", "｜") for k in ("original", "corrected", "reason")) + " |"
                  for c in corrections]
        if uncertain:
            lines += ["", "## 附录：无法确定的地方（保留原文）", ""] + [f"- {u}" for u in uncertain]
        print(f"  原文 {len(text)} 字 → {chars} 字（{100 * chars / len(text):.0f}%）")
    (session / "polished.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"完成：{session / 'polished.md'}")


def strict(session: Path, key: str) -> None:
    state = load_state(session)
    tracks = list(state["tracks"])

    def emit(event: dict) -> None:
        if event["type"] in ("stage", "progress"):
            print("  " + event["text"])

    status = read_meta(session).get("status", "transcribed")
    try:
        polish_session(session, state, tracks, Settings.load(), key, emit)
    except TranscribeError as exc:
        raise SystemExit(str(exc))
    finally:  # polish_session marks the session "polishing" for the app; put the status back
        meta = read_meta(session)
        meta["status"] = status
        write_meta(session, meta)
    print(f"完成：{session / 'transcript_polished.md'}")


def main() -> None:
    session = Path(sys.argv[1])
    key, _ = config.get_api_key()
    if not key:
        raise SystemExit("没有找到 API Key（凭据管理器或 OPENAI_API_KEY）。")
    try:
        (strict if "--strict" in sys.argv else editorial)(session, key)
    except polisher.PolishError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
