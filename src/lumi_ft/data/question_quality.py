"""Marcadores automáticos de qualidade de uma questão de múltipla escolha traduzida (enunciado e alternativas em inglês e em português).

Servem para achar traduções que podem ter **trocado o sentido** e, com isso, tornado o gabarito errado: a negação do inglês
("which is NOT / except / incorrect") que sumiu no português, o enunciado que ficou sem tradução, alternativas repetidas. São
heurísticas de palavras: apontam suspeitas (há falsos positivos), não provam erro. Não julgam o conteúdo da questão.
"""
import re

from lumi_ft.eval.shape import trace_language

NEG_EN = re.compile(r"\b(not|except|incorrect|false|least|never|cannot|can't|isn't|doesn't|won't|unlikely|nor)\b", re.I)
NEG_PT = re.compile(r"\b(não|nao|exce\w+|incorret\w*|errad\w+|inválid\w+|fals[ao]s?|menos|nunca|nenhum\w*|jamais|nem|improv\w*|impossível|sem|ausên\w+|inexist\w+)\b", re.I)
REF_EN = re.compile(r"\b(all of the above|none of the above|both [a-j]\b|[a-j] and [a-j]\b|[a-j], [a-j]|neither)", re.I)
REF_PT = re.compile(r"\b(todas as anteriores|todos os anteriores|nenhuma das anteriores|nenhum dos anteriores|todas as alternativas|nenhuma das alternativas|nenhum dos acima|ambas|ambos|tanto|[a-j] e [a-j]\b|[a-j], [a-j]|nem)", re.I)

# marcadores que tiram a questão do RL (a tradução pode ter invertido o gabarito ou deixado o enunciado em inglês)
EXCLUDE = frozenset({"negacao_do_ingles_sem_correspondente", "enunciado_nao_traduzido", "enunciado_em_ingles", "alternativas_repetidas"})


def _raw(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def quality_flags(rec: dict) -> list[str]:
    """Marcadores de uma questão traduzida (`question_en/pt`, `options_en/pt`, `ref`). Lista vazia = nada a apontar."""
    out = []
    q_pt, q_en = str(rec["question_pt"]).strip(), str(rec["question_en"]).strip()
    opts = [str(o).strip() for o in rec["options_pt"]]
    if any(not o for o in opts) or not q_pt:
        out.append("texto_vazio")
    if "ABCDEFGHIJ".find(str(rec["ref"]).strip().upper()) not in range(len(opts)):
        out.append("gabarito_fora_do_intervalo")
    if len({_raw(o) for o in opts}) < len(opts):  # comparação do texto como está: "1%" e "-1%" não são iguais
        out.append("alternativas_repetidas")
    if NEG_EN.search(q_en) and not NEG_PT.search(q_pt):
        out.append("negacao_do_ingles_sem_correspondente")
    if q_pt and _raw(q_pt) == _raw(q_en) and re.search(r"[a-z]{4,}", q_en, re.I):
        out.append("enunciado_nao_traduzido")
    if len(q_pt) >= 40 and trace_language(q_pt) == "en":  # enunciado (quase) todo em inglês, mesmo sem ser idêntico ao original
        out.append("enunciado_em_ingles")
    if REF_EN.search(" | ".join(str(o) for o in rec["options_en"])):
        out.append("alternativa_remete_a_outras" + ("" if REF_PT.search(" | ".join(opts)) else "_sem_correspondente_no_pt"))
    if len(q_pt) < 15:
        out.append("enunciado_muito_curto")
    return out


def excluded_reason(rec: dict) -> str | None:
    """O primeiro marcador de `EXCLUDE` da questão, ou None se ela fica."""
    return next((f for f in quality_flags(rec) if f in EXCLUDE), None)
