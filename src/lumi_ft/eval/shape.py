"""Forma da saída (A-08): comprimento e idioma do traço.

O idioma é uma heurística de palavras funcionais, não um detector de idioma: serve para ver se o traço
sai em inglês (risco do PRD), não para medir com precisão. Traços de matemática têm poucas palavras
funcionais, então muitos caem em `unk`.
"""

from __future__ import annotations

import re
import statistics

_PT = frozenset(
    "de que não uma para com os as na no por mais mas foi ao ele das tem seu sua ou ser quando muito nos já "
    "está também só pelo pela até isso ela entre era depois sem mesmo ter quem nas esse eles você então "
    "cada porque são vezes temos".split()
)
_EN = frozenset(
    "the of and to is that it was for are with his they be at have this from or had by but what there we can "
    "were your when which their if will about then them would so these her has he you i each first".split()
)
_WORD = re.compile(r"[a-zà-ÿ]+")


def trace_language(text: str, min_hits: int = 3, ratio: float = 2.0) -> str:
    """'pt', 'en', 'mixed' (as duas contagens são próximas) ou 'unk' (poucas palavras funcionais)."""
    words = _WORD.findall(text.lower())
    pt = sum(w in _PT for w in words)
    en = sum(w in _EN for w in words)
    if pt + en < min_hits:
        return "unk"
    if pt >= ratio * en:
        return "pt"
    if en >= ratio * pt:
        return "en"
    return "mixed"


def length_stats(values: list[int]) -> dict:
    """Média, mediana, p95, p99 e máximo (índice do vizinho mais próximo, sem interpolação)."""
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]  # noqa: E731
    return {"n": len(v), "mean": round(statistics.fmean(v), 1), "median": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": v[-1]}
