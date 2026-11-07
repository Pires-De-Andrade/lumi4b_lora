"""Tradução inglês -> português de exemplos de raciocínio (para o SFT): prompt, divisão em blocos e conferência da saída.

Traduz-se em blocos de parágrafos (cabe na janela de um modelo pequeno e limita o estrago de um erro); cada bloco só
é aceito se terminar, não trouxer marcadores de raciocínio e tiver tamanho compatível com o original.
"""

from __future__ import annotations

from lumi_ft.eval.shape import trace_language

INSTR = ("Traduza o texto abaixo do inglês para o português do Brasil. Preserve o sentido, a ordem das frases, os números, "
         "as fórmulas, os símbolos e as quebras de linha. Não resuma, não explique e não acrescente nada: escreva só a tradução.\n\nTexto:\n")
MIN_RATIO, MAX_RATIO = 0.8, 1.7  # caracteres da tradução / caracteres do original (pt costuma ficar ~1,1 a 1,25x maior)


def make_prompt(text: str) -> str:
    return INSTR + text


def chunk_paragraphs(text: str, max_chars: int = 1800) -> list[str]:
    """Junta parágrafos (separados por linha em branco) em blocos de até `max_chars`; um parágrafo maior que isso fica sozinho."""
    out: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        if cur and len(cur) + 2 + len(para) > max_chars:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def check_chunk(src: str, out: str, finished: bool) -> str | None:
    """None se o bloco traduzido serve; senão o motivo."""
    out = out.strip()
    if not finished:
        return "nao_terminou"
    if not out:
        return "vazio"
    if "<think>" in out or "</think>" in out:
        return "marcador_de_raciocinio"
    r = len(out) / max(1, len(src.strip()))
    if r < MIN_RATIO or r > MAX_RATIO:
        return "tamanho_incompativel"
    return None


def join_chunks(parts: list[str]) -> str:
    return "\n\n".join(p.strip() for p in parts)


def is_portuguese(text: str) -> bool:
    return trace_language(text) == "pt"
