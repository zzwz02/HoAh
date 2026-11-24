"""Strict polishing with gpt-6-luna: fix typos/terms, punctuation and filler words, split into
paragraphs. Never summarises or rewords.

The text is processed in ~1800-character chunks (sentence boundaries, previous chunk's tail as
context) after one pass that builds a shared terminology table. Every finished chunk is saved to
a cache, so an interrupted or failed run resumes with only the missing chunks.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import requests

RESPONSES_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-6-luna"
CHUNK_CHARS = 1800
CONTEXT_CHARS = 300
MIN_CHARS = 200  # shorter tracks are echo snippets; not worth polishing
WORKERS = 4
MAX_RETRIES = 4


class PolishError(Exception):
    pass


CORRECTIONS_SCHEMA = {"type": "array", "items": {
    "type": "object", "additionalProperties": False,
    "required": ["original", "corrected", "reason"],
    "properties": {
        "original": {"type": "string"},
        "corrected": {"type": "string"},
        "reason": {"type": "string"},
    }}}

GLOSSARY_INSTRUCTIONS = """你是中文会议记录的校对。下面是一整段会议录音的语音识别原文（普通话为主，夹杂英文技术术语）。
请通读全文，找出被反复或明显误识别的专有名词、术语、英文缩写、产品名、公司/机构名、人名，
给出“原文写法 → 正确写法”的统一术语表，供后续逐段校对时保持一致。
- 同一个词在原文中的不同误写都要列出来（每种写法一条）。
- 只收录有把握的修订；没把握的不要收录。参考词汇表。"""

GLOSSARY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["terms"], "properties": {"terms": CORRECTIONS_SCHEMA},
}

STRICT_INSTRUCTIONS = """你是中文会议记录的校对。输入是一段会议录音语音识别原文中的一块（普通话为主，夹杂英文技术术语）。
你的任务只有“润色”和“分段”，绝对不能总结、概括、压缩或改写内容：

允许做的修改（只限这些）：
- 修正错别字、同音错词和术语误识别（按统一术语表和词汇表；没把握的保留原文并记入 uncertain）；
- 补全、修正标点，在句子边界处断句；
- 删除口吃式的机械重复（如“我们我们”“就是就是”“那个那个”）和纯粹无意义的语气填充（如句首孤立的“呃”“嗯”）。

禁止做的事：
- 不要删除任何有信息的词句，包括问句、附和（“对”“好”“OK”）、口语化表达和不完整的句子；
- 不要合并、重组、调换句子顺序，不要换一种说法重写句子；
- 不要添加标题、小结、说明或原文没有的内容；不要区分说话人。
整理后的字数应为原文的 90% 以上。

分段：按原文顺序，在意思转换处分段（每段约 100–300 字）。“上文”只用于理解语境，不要输出。
corrections 列出做过的有实质意义的错字/错词修订（原文片段 → 修订后，附简短理由）。"""

CHUNK_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["paragraphs", "corrections", "uncertain"],
    "properties": {
        "paragraphs": {"type": "array", "items": {"type": "string"}},
        "corrections": CORRECTIONS_SCHEMA,
        "uncertain": {"type": "array", "items": {"type": "string"},
                      "description": "无法确定正确写法、保留原文的片段及可能的含义"},
    },
}


def _error_message(response: requests.Response) -> str:
    try:
        message = response.json().get("error", {}).get("message")
    except ValueError:
        message = None
    return f"HTTP {response.status_code}：{message or response.text[:300] or response.reason}"


def call_luna(key: str, instructions: str, user: str, schema: dict, name: str, *, effort: str = "high",
              http=requests, sleep: Callable[[float], None] = time.sleep) -> tuple[dict, dict]:
    """Responses API call with strict JSON output; retries network errors, 429 and 5xx."""
    body = {
        "model": MODEL,
        "reasoning": {"effort": effort},
        "input": [{"role": "system", "content": instructions}, {"role": "user", "content": user}],
        "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
    }
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        response = None
        try:
            response = http.post(RESPONSES_URL, headers={"Authorization": f"Bearer {key}"}, json=body,
                                 timeout=(15, 1200))
            if response.status_code == 200:
                data = response.json()
                for item in data.get("output", []):
                    for part in item.get("content", []) or []:
                        if part.get("type") == "output_text":
                            return json.loads(part["text"]), data.get("usage", {})
                raise PolishError(f"{MODEL} 没有返回文本。")
            last_error = _error_message(response)
            if response.status_code != 429 and response.status_code < 500:
                raise PolishError(last_error)
        except requests.RequestException as exc:
            last_error = f"网络错误：{exc}"
        except ValueError as exc:  # malformed JSON in the reply
            last_error = f"返回内容无法解析：{exc}"
        if attempt < MAX_RETRIES:
            delay = 2.0 ** (attempt + 1)
            if response is not None:
                try:
                    delay = min(float(response.headers.get("Retry-After", "")), 60.0)
                except ValueError:
                    pass
            sleep(min(delay, 30.0))
    raise PolishError(f"重试 {MAX_RETRIES} 次后仍失败。{last_error}")


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split at sentence ends so no chunk cuts a sentence in half."""
    sentences = re.findall(r"[^。！？!?\n]*[。！？!?\n]+|[^。！？!?\n]+$", text)
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > size:
            chunks.append(current)
            current = ""
        current += s
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def polish_strict(
    text: str,
    key: str,
    vocabulary: list[str],
    cache: dict,
    save: Callable[[], None],
    on_progress: Callable[[int, int, list[str]], None] | None = None,
    *,
    http=requests,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Polish text, resuming from `cache` (mutated in place; `save()` persists it after every step).

    on_progress(done_chunks, total_chunks, paragraphs_of_leading_finished_chunks) is called as
    chunks finish. Returns {"paragraphs", "corrections", "uncertain"}.
    """
    chunks = chunk_text(text)
    if cache.get("source_hash") != text_hash(text) or len(cache.get("chunks", [])) != len(chunks):
        cache.clear()
        cache.update(source_hash=text_hash(text), glossary=None,
                     chunks=[{"done": False, "paragraphs": [], "corrections": [], "uncertain": []} for _ in chunks])
        save()
    vocab_line = f"词汇表（可能出现的专有名词）：{'、'.join(vocabulary) if vocabulary else '（无）'}"

    if cache["glossary"] is None:
        glossary, _ = call_luna(key, GLOSSARY_INSTRUCTIONS, f"{vocab_line}\n\n语音识别原文：\n{text}",
                                GLOSSARY_SCHEMA, "glossary", http=http, sleep=sleep)
        cache["glossary"] = glossary["terms"]
        save()
    terms = cache["glossary"]
    glossary_line = "统一术语表（原文中出现这些写法时必须按此修订）：\n" + (
        "\n".join(f"- {t['original']} → {t['corrected']}" for t in terms) if terms else "（无）")

    lock = threading.Lock()

    def leading_paragraphs() -> list[str]:
        out = []
        for c in cache["chunks"]:
            if not c["done"]:
                break
            out += c["paragraphs"]
        return out

    def report() -> None:
        if on_progress:
            on_progress(sum(c["done"] for c in cache["chunks"]), len(chunks), leading_paragraphs())

    def one(i: int) -> None:
        context = chunks[i - 1][-CONTEXT_CHARS:] if i else "（无，这是开头）"
        user = (f"{vocab_line}\n\n{glossary_line}\n\n上文（仅供理解语境，不要输出）：\n{context}"
                f"\n\n本块原文：\n{chunks[i]}")
        result, _ = call_luna(key, STRICT_INSTRUCTIONS, user, CHUNK_SCHEMA, "polished_chunk", http=http, sleep=sleep)
        with lock:
            cache["chunks"][i] = {
                "done": True,
                "paragraphs": [p.strip() for p in result["paragraphs"] if p.strip()],
                "corrections": result["corrections"],
                "uncertain": result["uncertain"],
            }
            save()
            report()

    report()
    pending = [i for i, c in enumerate(cache["chunks"]) if not c["done"]]
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(one, i): i for i in pending}
        for future in as_completed(futures):  # let the other chunks finish (and be cached) even if one fails
            try:
                future.result()
            except Exception as exc:
                errors.append(f"第 {futures[future] + 1}/{len(chunks)} 块：{exc}")
    if errors:
        done = sum(c["done"] for c in cache["chunks"])
        raise PolishError(f"已完成 {done}/{len(chunks)} 块，已保存；{errors[0]}")

    corrections, seen = [], set()
    for c in terms + [c for ch in cache["chunks"] for c in ch["corrections"]]:
        pair = (c["original"].strip(), c["corrected"].strip())
        if pair not in seen:
            seen.add(pair)
            corrections.append(c)
    return {
        "paragraphs": [p for ch in cache["chunks"] for p in ch["paragraphs"]],
        "corrections": corrections,
        "uncertain": [u for ch in cache["chunks"] for u in ch["uncertain"]],
    }
