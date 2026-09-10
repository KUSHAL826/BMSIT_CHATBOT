"""
Token-aware smart chunking.

Chunks are sized in TOKENS (target 250-300) rather than characters, because the
embedding model's limits and cost are token-based. Splitting respects structure:
paragraphs first, then sentences, then words, so a chunk rarely cuts a sentence
in half. A small overlap carries context across boundaries.

Token counts are estimated locally (no extra dependency). The estimate is
deliberately conservative - it over-counts slightly - so batches assembled from
these numbers stay inside the API's hard token ceiling.
"""
import re

from app.config import Config

_WORD_RE = re.compile(r"\S+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def count_tokens(text):
    """
    Conservative token estimate for Gemini-family tokenizers.

    Uses the larger of two heuristics: ~4 characters per token, and ~1.3 tokens
    per whitespace word. Taking the maximum keeps the estimate on the safe side
    for text with long words, numbers or punctuation-heavy tables.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    chars = len(stripped)
    words = len(_WORD_RE.findall(stripped))
    return max(1, int(max(chars / 4.0, words * 1.3)) + 1)


def _split_long_sentence(sentence, max_tokens):
    """Splits an over-long sentence on word boundaries."""
    words = _WORD_RE.findall(sentence)
    if not words:
        return []
    pieces, current, current_tokens = [], [], 0
    for word in words:
        word_tokens = count_tokens(word)
        if current and current_tokens + word_tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _atoms(text):
    """
    Breaks text into the smallest units a chunk boundary may fall between:
    sentences, or word-groups when a single sentence is too long.
    """
    max_tokens = Config.CHUNK_MAX_TOKENS
    units = []
    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for line in paragraph.split("\n"):
            line = line.strip()
            if not line:
                continue
            for sentence in _SENTENCE_SPLIT_RE.split(line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if count_tokens(sentence) > max_tokens:
                    units.extend(_split_long_sentence(sentence, max_tokens))
                else:
                    units.append(sentence)
    return units


def split_into_token_chunks(text, target_tokens=None, max_tokens=None, overlap_tokens=None,
                            min_tokens=None):
    """
    Splits text into chunks of roughly `target_tokens` (default 250-300).

    Returns a list of {"text": str, "tokens": int}. The last chunk is merged
    backwards if it is too small to be useful on its own.
    """
    if not text or not text.strip():
        return []

    target = target_tokens or Config.CHUNK_TARGET_TOKENS
    ceiling = max_tokens or Config.CHUNK_MAX_TOKENS
    overlap = overlap_tokens if overlap_tokens is not None else Config.CHUNK_OVERLAP_TOKENS
    floor = min_tokens or Config.CHUNK_MIN_TOKENS

    units = _atoms(text)
    if not units:
        return []

    chunks = []
    current, current_tokens = [], 0

    for unit in units:
        unit_tokens = count_tokens(unit)

        # Closing the chunk here would exceed the ceiling, or we already met the
        # target: emit and start a new one carrying a little overlap.
        if current and (current_tokens + unit_tokens > ceiling or current_tokens >= target):
            chunks.append((list(current), current_tokens))
            carry, carry_tokens = [], 0
            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if carry_tokens + previous_tokens > overlap:
                    break
                carry.insert(0, previous)
                carry_tokens += previous_tokens
            current, current_tokens = carry, carry_tokens

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append((list(current), current_tokens))

    result = []
    for units_in_chunk, tokens in chunks:
        body = " ".join(units_in_chunk).strip()
        if not body:
            continue
        if result and tokens < floor:
            # Too small to stand alone: fold it into the previous chunk.
            merged = result[-1]["text"] + " " + body
            result[-1] = {"text": merged, "tokens": count_tokens(merged)}
            continue
        result.append({"text": body, "tokens": tokens})

    return result
