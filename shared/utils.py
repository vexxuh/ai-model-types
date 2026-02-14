"""Common text preprocessing utilities for all models."""

import re
import string
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "war_of_the_worlds.txt"

# Gutenberg markers
_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def load_raw_text(path: Path = DATA_PATH) -> str:
    """Load the book text, stripping Gutenberg header/footer."""
    text = path.read_text(encoding="utf-8")
    start = text.find(_START_MARKER)
    if start != -1:
        start = text.index("\n", start) + 1
    else:
        start = 0
    end = text.find(_END_MARKER)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def split_chapters(text: str) -> list[dict]:
    """Split text into chapters. Returns list of {book, chapter, title, text}."""
    # Pattern: Roman numeral on its own line, followed by title in caps on next line
    chapter_pattern = re.compile(
        r"^(X{0,3}(?:IX|IV|V?I{0,3}))\.\n([A-Z][A-Z\s\-\"\u201c\u201d\u2019\u2018,.]+)\n",
        re.MULTILINE,
    )

    # Find body book boundaries (standalone "BOOK ONE\n" / "BOOK TWO\n", not TOC entries)
    # The TOC uses "BOOK ONE.—" with a period-dash, the body uses just "BOOK ONE\n"
    book_one_match = re.search(r"^BOOK ONE\n", text, re.MULTILINE)
    book_two_match = re.search(r"^BOOK TWO\n", text, re.MULTILINE)

    book_one_start = book_one_match.start() if book_one_match else 0
    book_two_start = book_two_match.start() if book_two_match else len(text)

    # Only search for chapters after the body starts (skip table of contents)
    text_body = text[book_one_start:]
    body_offset = book_one_start
    book_two_start_rel = book_two_start - body_offset

    chapters = []
    matches = list(chapter_pattern.finditer(text_body))

    for i, match in enumerate(matches):
        book = 1 if match.start() < book_two_start_rel else 2
        roman = match.group(1)
        title = match.group(2).strip().rstrip(".")
        # Chapter text runs from end of this match to start of next match (or end)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_body)
        chapter_text = text_body[start:end].strip()

        # Clean up any trailing BOOK TWO header from last chapter of book one
        for marker in ["BOOK TWO", "BOOK ONE"]:
            idx = chapter_text.rfind(marker)
            if idx != -1 and idx > len(chapter_text) - 200:
                chapter_text = chapter_text[:idx].strip()

        chapters.append({
            "book": book,
            "chapter": roman,
            "title": title,
            "text": chapter_text,
        })

    return chapters


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, split on whitespace and punctuation."""
    text = text.lower()
    # Replace common punctuation with space-separated tokens
    text = re.sub(r"([.!?;:,\"\'\-\(\)])", r" \1 ", text)
    # Collapse whitespace
    tokens = text.split()
    return tokens


def clean_tokens(tokens: list[str], remove_punctuation: bool = False) -> list[str]:
    """Optionally remove punctuation tokens."""
    if remove_punctuation:
        punct = set(string.punctuation) | {"\u201c", "\u201d", "\u2018", "\u2019", "\u2014"}
        return [t for t in tokens if t not in punct]
    return tokens


def build_vocab(tokens: list[str], min_count: int = 1) -> dict[str, int]:
    """Build word-to-index mapping from token list."""
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    vocab = {}
    idx = 0
    for word, count in sorted(freq.items()):
        if count >= min_count:
            vocab[word] = idx
            idx += 1
    return vocab


def get_full_text() -> str:
    """Load and return the full book text (no Gutenberg boilerplate)."""
    return load_raw_text()


def get_chapters() -> list[dict]:
    """Load and return chapters."""
    return split_chapters(load_raw_text())


def get_all_tokens(remove_punctuation: bool = False) -> list[str]:
    """Load text, tokenize, and optionally clean."""
    text = load_raw_text()
    tokens = tokenize(text)
    return clean_tokens(tokens, remove_punctuation=remove_punctuation)
