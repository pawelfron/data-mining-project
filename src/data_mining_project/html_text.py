"""Turn a statement HTML page into plain speech text."""

from __future__ import annotations

from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "h1", "h2", "title"}


class _ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_paragraph = False
        self._buffer: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "br" and self._in_paragraph and self._skip_depth == 0:
            self._buffer.append(" ")
        elif tag == "p" and self._skip_depth == 0:
            self._in_paragraph = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "p" and self._in_paragraph:
            text = " ".join("".join(self._buffer).split())
            if text:
                self.paragraphs.append(text)
            self._buffer = []
            self._in_paragraph = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._in_paragraph:
            self._buffer.append(data)


def html_to_text(html: str) -> str:
    """Keep paragraph text and collapse whitespace inside each paragraph."""
    parser = _ParagraphExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.paragraphs).strip()
