"""Is every number in the answer found in what the question and the tools gave?

The brief asks for an answer "grounded in the actual numbers". This makes that a check the code
runs: take each number in the answer and look for it among the numbers the user wrote, the
arguments the model sent and the results the tools returned.

A number counts as found when a source number equals it at the answer's precision, within half
a unit of its last digit: 737 matches 736.8, 0.52 matches 0.521, 40.818 matches 40.81796. A
percentage also matches a share (62 % and 0.62). Signs are ignored, so "a fall of 0.35" matches
-0.349.
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)  # "1. " at the start of a line: not data

# Numbers written as words, two to ninety-nine ("one" is left out: it is too often not a number).
SMALL = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
WORD_VALUE = {w: i + 2 for i, w in enumerate(SMALL)} | {w: 20 + 10 * i for i, w in enumerate(TENS)}
NUMBER_WORD = re.compile(
    rf"\b(?:({'|'.join(TENS)})(?:[- ]({'|'.join(DIGITS)}))?|({'|'.join(SMALL)}))\b", re.IGNORECASE
)


def numbers(text: str) -> list[tuple[str, float, int]]:
    """Every number in ``text`` as (as written, value, decimals), list markers left out, numbers
    written as words included."""
    found = []
    for match in NUMBER.finditer(LIST_MARKER.sub(" ", text)):
        whole, fraction = match.group(1).replace(",", ""), match.group(2)
        value = float(f"{whole}.{fraction}") if fraction else float(whole)
        found.append((match.group(0), value, len(fraction) if fraction else 0))
    for match in NUMBER_WORD.finditer(text):
        tens, unit, small = (g.lower() if g else None for g in match.groups())
        value = WORD_VALUE[small] if small else WORD_VALUE[tens] + (DIGITS.index(unit) + 1 if unit else 0)
        found.append((match.group(0), float(value), 0))
    return found


def is_found(value: float, decimals: int, sources: set[float]) -> bool:
    """True when some source number equals ``value`` at its precision, as itself or as a share."""
    tolerance = 0.5 * 10**-decimals + 1e-9
    return any(abs(value - s) <= tolerance or abs(value - 100 * s) <= tolerance for s in sources)


def check(answer: str, sources: list[str]) -> dict:
    """The numbers of ``answer`` and which of them no source contains."""
    known = {abs(value) for text in sources for _, value, _ in numbers(text)}
    written = numbers(answer)
    missing = [shown for shown, value, decimals in written if not is_found(value, decimals, known)]
    return {"numbers_checked": len(written), "not_found": missing}
