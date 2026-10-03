"""Character spans of scalar JSON leaves in a raw model output.

The stdlib json module drops positions, so this is a small recursive-descent
scanner. Each leaf gets a path (``total``, ``line_items[2].amount``) and the
[start, end) range of its value text (string contents without quotes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_WS = re.compile(r"\s*")


@dataclass(frozen=True)
class Leaf:
    path: str
    start: int
    end: int
    is_null: bool

    @property
    def group(self) -> str:
        """``line_items[3].amount`` -> ``line_items.amount``; top-level keys unchanged."""
        return re.sub(r"\[\d+\]", "", self.path)


class _Scanner:
    def __init__(self, text: str) -> None:
        self.s = text
        self.leaves: list[Leaf] = []

    def ws(self, i: int) -> int:
        return _WS.match(self.s, i).end()

    def string(self, i: int) -> tuple[int, int, int]:
        """i at opening quote -> (content_start, content_end, index after closing quote)."""
        j = i + 1
        while j < len(self.s):
            ch = self.s[j]
            if ch == "\\":
                j += 2
                continue
            if ch == '"':
                return i + 1, j, j + 1
            j += 1
        return i + 1, len(self.s), len(self.s)  # truncated generation

    def value(self, i: int, path: str) -> int:
        i = self.ws(i)
        if i >= len(self.s):
            return i
        ch = self.s[i]
        if ch == "{":
            return self.obj(i, path)
        if ch == "[":
            return self.arr(i, path)
        if ch == '"':
            a, b, j = self.string(i)
            self.leaves.append(Leaf(path, a, b, False))
            return j
        for word, null in (("null", True), ("true", False), ("false", False)):
            if self.s.startswith(word, i):
                self.leaves.append(Leaf(path, i, i + len(word), null))
                return i + len(word)
        m = _NUM.match(self.s, i)
        if m:
            self.leaves.append(Leaf(path, m.start(), m.end(), False))
            return m.end()
        raise ValueError(f"unexpected {ch!r} at {i}")

    def obj(self, i: int, path: str) -> int:
        i = self.ws(i + 1)
        while i < len(self.s) and self.s[i] != "}":
            a, b, i = self.string(i)
            key = self.s[a:b]
            i = self.ws(i)
            i = self.ws(i + 1)  # ':'
            i = self.value(i, f"{path}.{key}" if path else key)
            i = self.ws(i)
            if i < len(self.s) and self.s[i] == ",":
                i = self.ws(i + 1)
        return i + 1

    def arr(self, i: int, path: str) -> int:
        i = self.ws(i + 1)
        k = 0
        while i < len(self.s) and self.s[i] != "]":
            i = self.value(i, f"{path}[{k}]")
            k += 1
            i = self.ws(i)
            if i < len(self.s) and self.s[i] == ",":
                i = self.ws(i + 1)
        return i + 1


def json_leaves(text: str) -> list[Leaf]:
    """Scalar leaves of the first JSON object in ``text`` (tolerates fences and truncation)."""
    start = text.find("{")
    if start < 0:
        return []
    sc = _Scanner(text)
    try:
        sc.value(start, "")
    except (ValueError, IndexError):
        pass  # keep what was scanned before the damage
    return sc.leaves


def leaf_token_rows(leaf: Leaf, offsets: list[tuple[int, int]]) -> list[int]:
    """Output-token indices whose character range overlaps the leaf value."""
    return [k for k, (a, b) in enumerate(offsets) if b > leaf.start and a < leaf.end]
