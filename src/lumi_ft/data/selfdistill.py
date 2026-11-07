"""Dados do próprio Lumi: aceita uma saída gerada só se for bem formada, certa e em português.

Com raciocínio: um único `</think>`, traço e resposta não vazios, traço em português, `#### N` na resposta igual à
referência (leitura numérica brasileira ou inglesa), terminou sozinha (não truncou). Sem raciocínio: terminou, sem tags
`<think>`, `#### N` igual à referência.
"""

from __future__ import annotations

from lumi_ft.eval import parse as P
from lumi_ft.eval.shape import trace_language


def same_number(got: str | None, ref: str) -> bool:
    if got is None:
        return False
    a, b = P.to_decimal_ptbr(got), P.to_decimal_ptbr(ref)
    return a is not None and b is not None and a == b


def accept_think(text: str, finished: bool, ref: str) -> tuple[dict | None, str]:
    """(registro ou None, motivo). `text` já inclui o começo forçado."""
    if not finished:
        return None, "truncado"
    p = P.split_response(text, thinking=True)
    if not p.closed or p.n_close != 1:
        return None, "sem_um_unico_close"
    if not p.thought or not p.answer:
        return None, "traco_ou_resposta_vazios"
    if not same_number(P.strict_number(p.answer), ref):
        return None, "resposta_errada_ou_sem_marcador"
    if trace_language(p.thought) != "pt":
        return None, "traco_nao_pt"
    return {"thought": p.thought, "answer": p.answer}, "ok"


def accept_nothink(text: str, finished: bool, ref: str) -> tuple[dict | None, str]:
    if not finished:
        return None, "truncado"
    if P.THINK_OPEN in text or P.THINK_CLOSE in text:
        return None, "tem_tags_think"
    answer = text.split(P.IM_END, 1)[0].strip()
    if not answer:
        return None, "vazio"
    if not same_number(P.strict_number(answer), ref):
        return None, "resposta_errada_ou_sem_marcador"
    return {"answer": answer}, "ok"
