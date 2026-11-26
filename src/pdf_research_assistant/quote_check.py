"""Check that text an answer puts in quote marks is really in the passages, and unquote it when it is not.

The answer model sometimes puts its own wording in quote marks, so a reader takes a paraphrase for a direct
quote from the PDF (#149). Each quote of at least MIN_WORDS words is looked for, word for word, in the full
text of the chunks the question read. One that is not found loses its quote marks and gets PARAPHRASE_MARK
after it, so the words stay but no longer look quoted.

"Word for word" ignores what a PDF's text layer and the answer model change without changing the words:
case, spacing, line breaks, words hyphenated across lines, curly or straight quote marks and apostrophes,
other punctuation, and ligatures such as "ﬁ". An ellipsis or a [bracketed insertion] in a quote is a gap:
the parts either side must be found in that order in one chunk.

Shorter quotes are left alone: they are usually terms or titles ("small teaching"), not quoted passages.
"""

import re
import unicodedata

MIN_WORDS = 4
PARAPHRASE_MARK = "(paraphrase, not a direct quote)"

# A quote in curly double, straight double or curly single marks. It never runs across a blank line, so a
# quote mark left open in one paragraph cannot pair with one in the next. A curly apostrophe followed by a
# letter (isn’t) does not close a single-quoted quote.
QUOTE_PATTERN = re.compile(
    r"“(?P<curly>(?:[^“”\n]|\n(?!\n))*)”"
    r'|"(?P<straight>(?:[^"\n]|\n(?!\n))*)"'
    r"|‘(?P<single>(?:[^‘’\n]|\n(?!\n)|’(?=\w))*)’(?!\w)"
)
# An ellipsis (... or . . . once the text is normalised) or a [bracketed insertion] inside a quote.
GAP_PATTERN = re.compile(r"(?:\.\s?){2}\.|\[[^\]]*\]")
WORD_PATTERN = re.compile(r"\w+")
# Everything but letters and digits: spaces, line breaks, hyphens, quote marks and other punctuation.
NOT_A_WORD_CHARACTER = re.compile(r"[\W_]+")


def compare_form(text: str) -> str:
    """Return the text as quotes are compared: letters and digits only, lower case, ligatures spelled out."""
    return NOT_A_WORD_CHARACTER.sub("", unicodedata.normalize("NFKC", text).casefold())


def quote_parts(quote: str) -> list[str]:
    """Return the parts of a quote that must be found, in order, each in compare form; gaps are dropped."""
    pieces = GAP_PATTERN.split(unicodedata.normalize("NFKC", quote))
    return [part for part in (compare_form(piece) for piece in pieces) if part]


def word_count(quote: str) -> int:
    return len(WORD_PATTERN.findall(GAP_PATTERN.sub(" ", unicodedata.normalize("NFKC", quote))))


def found_in(parts: list[str], chunk: str) -> bool:
    """Return whether every part is in the chunk (already in compare form), in order."""
    position = 0
    for part in parts:
        found = chunk.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def mark_unverified_quotes(answer: str, chunks: list[str]) -> tuple[str, list[str]]:
    """Return the answer with every long quote not found in any chunk unquoted and marked, and those quotes."""
    compared = [compare_form(chunk) for chunk in chunks]
    flagged = []

    def check(match: re.Match) -> str:
        quote = next(group for group in match.groups() if group is not None)
        if word_count(quote) < MIN_WORDS:
            return match.group(0)
        parts = quote_parts(quote)
        if any(found_in(parts, chunk) for chunk in compared):
            return match.group(0)
        flagged.append(quote)
        return f"{quote} {PARAPHRASE_MARK}"

    return QUOTE_PATTERN.sub(check, answer), flagged
