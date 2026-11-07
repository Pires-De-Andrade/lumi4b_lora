"""Descontaminação por k-gramas de palavras (D-05): remove do treino quem divide k-gramas com as provas."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Iterator

_WORD = re.compile(r"\w+", re.UNICODE)


def ngram_hashes(text: str, n: int = 8) -> Iterator[int]:
    """Hash (8 bytes) de cada n-grama de palavras normalizadas (minúsculas, sem pontuação)."""
    words = _WORD.findall(text.lower())
    for i in range(len(words) - n + 1):
        yield int.from_bytes(hashlib.blake2b(" ".join(words[i : i + n]).encode(), digest_size=8).digest(), "big")


class NgramIndex:
    def __init__(self, n: int = 8):
        self.n = n
        self.hashes: set[int] = set()
        self.n_texts = 0

    def add(self, text: str) -> None:
        self.hashes.update(ngram_hashes(text, self.n))
        self.n_texts += 1

    def hits(self, text: str) -> bool:
        return any(h in self.hashes for h in ngram_hashes(text, self.n))

    def __len__(self) -> int:
        return len(self.hashes)


def flatten_strings(obj) -> Iterator[str]:
    """Todas as strings de um item de prova (pergunta, alternativas, resposta), sem depender do esquema."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from flatten_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from flatten_strings(v)


def build_index(rows: Iterable, n: int = 8) -> NgramIndex:
    idx = NgramIndex(n)
    for row in rows:
        idx.add("\n".join(s for s in flatten_strings(row) if len(s) > 3))
    return idx
