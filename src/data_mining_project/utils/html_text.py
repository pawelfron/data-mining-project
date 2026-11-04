"""Turn a statement HTML page into plain speech text."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "h1", "h2", "title"}
_SKIP_PARAGRAPH_CLASSES = {"punkt-tytul"}

# Stenographer inserts in Sejm transcripts, e.g. (Oklaski), (Głos z sali: ...).
_STAGE_DIRECTION = re.compile(r"\([^)]*\)")
_MULTI_SPACE = re.compile(r" {2,}")


class _ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._ignore_paragraph = False
        self._in_paragraph = False
        self._buffer: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "br" and self._in_paragraph and self._skip_depth == 0:
            self._buffer.append(" ")
            return
        if tag == "p" and self._skip_depth == 0:
            if _has_skip_class(attrs):
                self._ignore_paragraph = True
                return
            self._in_paragraph = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag != "p":
            return
        if self._ignore_paragraph:
            self._ignore_paragraph = False
            return
        if self._in_paragraph:
            text = " ".join("".join(self._buffer).split())
            if text:
                self.paragraphs.append(text)
            self._buffer = []
            self._in_paragraph = False

    def handle_data(self, data: str) -> None:
        if (
            self._skip_depth == 0
            and not self._ignore_paragraph
            and self._in_paragraph
        ):
            self._buffer.append(data)


def _has_skip_class(attrs: list[tuple[str, str | None]]) -> bool:
    for name, value in attrs:
        if name.lower() != "class" or not value:
            continue
        classes = set(value.lower().split())
        if classes & _SKIP_PARAGRAPH_CLASSES:
            return True
    return False


def _strip_stage_directions(text: str) -> str:
    cleaned = _STAGE_DIRECTION.sub(" ", text)
    return _MULTI_SPACE.sub(" ", cleaned).strip()


def html_to_text(html: str) -> str:
    """Keep speech paragraphs, skip titles, and drop stenographer inserts."""
    parser = _ParagraphExtractor()
    parser.feed(html)
    parser.close()
    text = " ".join(parser.paragraphs).strip()
    return _strip_stage_directions(text)
