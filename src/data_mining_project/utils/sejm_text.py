"""Sejm-specific text cleaning and procedural stop-words."""

from __future__ import annotations

import re

# Pure courtesy / procedural tokens (ideological nouns like "państwo" stay!).
SEJM_STOP_WORDS: frozenset[str] = frozenset(
    {
        "panie",
        "pani",
        "panowie",
        "panom",
        "panami",
        "szanowni",
        "szanowny",
        "szanowna",
        "szanowne",
        "marszałku",
        "marszałek",
        "marszałka",
        "marszałkowi",
        "wicemarszałek",
        "wicemarszałku",
        "wysoka",
        "wysoki",
        "wysokie",
        "izbo",
        "izbie",
        "izby",
        "dziękuję",
        "dziekuje",
        "bardzo",
        "serdecznie",
        "proszę",
        "prosze",
        "posłowie",
        "posłanka",
        "poseł",
        "posła",
        "pośle",
        "czcigodna",
        "czcigodny",
        "czcigodni",
    }
)

# Stenographer speaker labels glued into transcripts, e.g. "Wicemarszałek X: ...".
_SPEAKER_LABEL = re.compile(
    r"(?:^|\s)(?:(?:Wice)?Marszałek|Poseł|Posłanka|Minister|Prezes Rady Ministrów)"
    r"[^:]{0,80}:\s*",
    re.IGNORECASE,
)

# Agenda / point-of-order openers sometimes prepended before the speech body.
_TITLE_PREFIX = re.compile(
    r"^(?:"
    r"Punkt\s+\d+[a-zA-Z]?\.?|"
    r"(?:Pierwsze|Drugie|Trzecie)\s+czytanie|"
    r"Sprawozdanie(?:\s+Komisji)?|"
    r"Informacja(?:\s+bieżąca)?|"
    r"Zmiana\s+(?:w\s+składzie|porządku)|"
    r"Poselski\s+projekt|"
    r"Rządowy\s+projekt|"
    r"Oświadczenia?\s+poselskie|"
    r"Pytania\s+w\s+sprawach\s+bieżących|"
    r"Wniosek\s+formalny"
    r")\b[^.!?]{0,220}",
    re.IGNORECASE,
)

# Canonical chamber greetings used to locate where preamble ends and to strip courtesy headers.
_GREETING_START = re.compile(
    r"(?:"
    r"(?:(?:Szanown[ay]|Czcigodn[ay])\s+)?"
    r"(?:Panie|Pani)\s+Marszałk\w*!"
    r"|Wysoka\s+Izbo!"
    r"|Wysoki\s+Sejmie!"
    r"|Szanowni\s+Państwo!"
    r")",
    re.IGNORECASE,
)

_LEADING_COURTESY = re.compile(
    r"^(?:"
    r"(?:Bardzo\s+)?Dziękuję(?:\s+bardzo|\s+serdecznie)?(?:\s*,?\s*pan(?:ie|i)\s+marszałk\w*)?[.,!]?|"
    r"(?:(?:Szanown[ayie]+|Czcigodn[ayie]+|Drogi|Droga|Drodzy)\s+)?"
    r"(?:Panie\s+i\s+Panowie\s+Posłowie|Panie\s+Marszałk\w*|Pani\s+Marszałk\w*|"
    r"Panie\s+Premierze|Pani\s+Premier|Panie\s+Ministrze|Pani\s+Minister|"
    r"Panie\s+Prezydencie|Wysoka\s+Izbo|Wysoki\s+Sejmie|Szanowni\s+Państwo|"
    r"Szanowni\s+Posłowie|Panie\s+Pośle|Pani\s+Poseł)[.,!]?"
    r")\s*",
    re.IGNORECASE,
)

_TRAILING_THANKS = re.compile(
    r"(?:\s*(?:Bardzo\s+)?Dziękuję(?:\s+bardzo|\s+serdecznie)?(?:\s+za\s+uwagę)?[.,!]?)+\s*$",
    re.IGNORECASE,
)

_MULTI_SPACE = re.compile(r" {2,}")


def clean_speech_text(text: str) -> str:
    """Strip agenda preambles, stenographer labels, and chamber courtesy formulas."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    cleaned = _SPEAKER_LABEL.sub(" ", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()

    title_match = _TITLE_PREFIX.match(cleaned)
    if title_match:
        remainder = cleaned[title_match.end() :].lstrip(" .–—-:;")
        greeting = _GREETING_START.search(remainder)
        if greeting and greeting.start() <= 40:
            cleaned = remainder[greeting.start() :]
        elif remainder:
            cleaned = remainder

    greeting = _GREETING_START.search(cleaned)
    if greeting and 15 <= greeting.start() <= 220:
        preamble = cleaned[: greeting.start()]
        if _looks_like_preamble(preamble):
            cleaned = cleaned[greeting.start() :]

    while True:
        updated = _LEADING_COURTESY.sub("", cleaned).strip()
        if updated == cleaned:
            break
        cleaned = updated

    cleaned = _TRAILING_THANKS.sub("", cleaned).strip()
    return _MULTI_SPACE.sub(" ", cleaned).strip()


def _looks_like_preamble(preamble: str) -> bool:
    stripped = preamble.strip()
    if not stripped:
        return False
    if _TITLE_PREFIX.match(stripped):
        return True
    if ":" in stripped or len(stripped.split()) <= 25:
        return True
    if not re.search(r"[.!?]$", stripped) and len(stripped) <= 180:
        return True
    return False
