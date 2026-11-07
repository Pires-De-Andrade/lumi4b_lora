"""Questões de ciência do `open-r1/Mixture-of-Thoughts` (múltipla escolha, raciocínio do R1 em inglês) para traduzir.

Cada questão aparece 2 a 4 vezes, com traços diferentes e a resposta em `\\boxed{X}`. O dataset não traz o gabarito: usamos a
letra majoritária entre os traços da mesma questão (como a maioria de 75% do adalberto). Fica só o traço que fecha
`</think>` uma vez, termina em `\\boxed{letra}` igual à maioria e cabe no limite de tokens.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

LETTERS = "ABCDEFGHIJ"
BOXED = re.compile(r"\\boxed\{\s*([A-J])\s*\}")


def split_options(text: str) -> tuple[str, list[str]] | None:
    """Separa o enunciado das alternativas `A: ...`, `B: ...` (cada letra começa linha, exceto a A, que pode vir colada
    ao enunciado). Devolve (enunciado, alternativas) ou None se não achar ao menos 3 alternativas em ordem."""
    for m in re.finditer(r"A:[ \t]", text):
        if m.start() > 0 and text[m.start() - 1].isalnum():
            continue
        starts = [m.start()]
        for letter in LETTERS[1:]:
            mm = re.compile(rf"\n{letter}:[ \t]").search(text, starts[-1] + 1)
            if not mm:
                break
            starts.append(mm.start() + 1)
        if len(starts) < 3:
            continue
        bounds = starts + [len(text)]
        opts = [text[bounds[i] + 3: bounds[i + 1]].strip() for i in range(len(starts))]
        stem = text[: starts[0]].strip()
        if stem and all(opts):
            return stem, opts
    return None


def parse_trace(content: str) -> tuple[str, str, str] | None:
    """(raciocínio, texto final sem o `\\boxed`, letra) de uma resposta `<think>...</think>...\\boxed{X}`; None se o formato falha."""
    if content.count("</think>") != 1 or not content.lstrip().startswith("<think>"):
        return None
    head, _, tail = content.partition("</think>")
    thought = head.lstrip().removeprefix("<think>").strip()
    letters = BOXED.findall(tail)
    if not thought or not letters:
        return None
    final = BOXED.sub("", tail).strip()
    return thought, final, letters[-1]


def select(rows: list[dict], *, min_traces: int = 3, min_agree: float = 0.75, max_tokens: int = 3000, per_question: int = 2) -> tuple[list[dict], dict]:
    """`rows`: registros do dataset ({messages, num_tokens}). Devolve (registros, contagens)."""
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["messages"][0]["content"]].append(r)
    st = Counter(perguntas=len(by_q))
    out = []
    for qi, (q, rs) in enumerate(by_q.items()):
        sp = split_options(q)
        if sp is None:
            st["sem_alternativas"] += 1
            continue
        parsed = [(r, parse_trace(r["messages"][1]["content"])) for r in rs]
        votes = Counter(p[2] for _, p in parsed if p)
        if sum(votes.values()) < min_traces:
            st["poucos_tracos_validos"] += 1
            continue
        top, n_top = votes.most_common(1)[0]
        if n_top / len(rs) < min_agree:
            st["sem_maioria"] += 1
            continue
        if LETTERS.index(top) >= len(sp[1]):
            st["letra_fora_das_alternativas"] += 1
            continue
        kept = 0
        for k, (r, p) in enumerate(parsed):
            if p is None or p[2] != top:
                continue
            if r["num_tokens"] > max_tokens:
                st["traco_longo"] += 1
                continue
            if kept >= per_question:
                break
            kept += 1
            out.append({"id": f"mot_sci:{qi}:{k}", "problem_id": f"mot_sci:{qi}", "question": sp[0], "options": sp[1], "letter": top,
                        "thought": p[0], "final": p[1], "num_tokens": r["num_tokens"], "source": "mot_science"})
        st["perguntas_usadas" if kept else "sem_traco_aproveitavel"] += 1
    st["registros"] = len(out)
    return out, dict(st)
