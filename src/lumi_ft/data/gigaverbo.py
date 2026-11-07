"""GigaVerbo-v2 SFT -> exemplos de treino do Lumi: filtros, renderização e máscara (D-01 a D-11 do PRD).

Cada exemplo vira `input_ids` e `labels` (tokenizados com o template do Lumi). O prompt é prefixo exato do
exemplo (em texto e em ids); a loss cobre traço, `</think>`, resposta e `<|im_end|>`, e nada mais.
"""

from __future__ import annotations

import hashlib
import re

from lumi_ft.eval.parse import THINK_CLOSE, THINK_OPEN, last_number
from lumi_ft.eval.shape import trace_language

IM_END_ID = 11
MAX_LEN = 1024  # D-08: o máximo dos dados é 1.000 tokens; acima disso descarta, nunca trunca
MIN_SCORE_NONREASONING = 4.0
# Resíduo do pipeline de geração (D-06): expressões que citam um contexto que o prompt não tem.
RESIDUE = re.compile(r"contexto fornecido|resposta fornecida", re.IGNORECASE)
DROP_LANGUAGES = frozenset({"en", "mixed"})  # D-09, com shape.trace_language (heurística fixada, versão v1)


def norm_prompt(text: str) -> str:
    return " ".join(text.lower().split())


def bucket(example_id: str, n: int = 10_000) -> int:
    """Posição estável do exemplo em [0, n): define os subconjuntos de 20% (bucket < 2.000) e a validação."""
    return int(hashlib.sha256(example_id.encode()).hexdigest()[:12], 16) % n


# ----------------------------------------------------------------------------- mensagens
def parse_messages(messages: list[dict]) -> dict | None:
    """Só conversas de um turno: [system?, user, assistant]. Devolve system, user e o texto do assistente."""
    roles = [m["role"] for m in messages]
    if roles not in (["system", "user", "assistant"], ["user", "assistant"]):
        return None
    system = messages[0]["content"] if roles[0] == "system" else None
    return {"system": system, "user": messages[-2]["content"], "assistant": messages[-1]["content"]}


def split_trace(content: str) -> tuple[str, str] | None:
    """D-02: `<think>\\n...\\n</think>\\n\\nresposta` -> (traço, resposta). None se o formato não for esse."""
    if not content.startswith(THINK_OPEN) or content.count(THINK_OPEN) != 1 or content.count(THINK_CLOSE) != 1:
        return None
    head, _, tail = content.partition(THINK_CLOSE)
    thought, answer = head[len(THINK_OPEN):].strip(), tail.strip()
    if not thought or not answer:
        return None
    return thought, answer


def has_think_tags(text: str) -> bool:
    return THINK_OPEN in text or THINK_CLOSE in text


# ------------------------------------------------------------------------------ filtros
def _num_forms(n: str) -> set[str]:
    n = n.strip().rstrip(".").replace("$", "")
    forms = {n, n.replace(".", ""), n.replace(",", ""), n.replace(".", "").replace(",", ".")}
    return {f for f in forms if re.search(r"\d", f)}


def last_trace_number_in_answer(thought: str, answer: str) -> bool:
    """D-07: o último número do traço aparece na resposta (aceita `.` e `,` de milhar ou decimal)."""
    ln = last_number(thought)
    if ln is None:
        return False
    forms = _num_forms(ln)
    if not forms:
        return False
    return any(forms & _num_forms(m.group()) for m in re.finditer(r"-?[$0-9.,]*\d", answer))


def has_residue(thought: str, answer: str) -> bool:
    return bool(RESIDUE.search(thought) or RESIDUE.search(answer))


def trace_dropped_by_language(thought: str) -> bool:
    return trace_language(thought) in DROP_LANGUAGES


# -------------------------------------------------------------------------- renderização
def render_example(tok, system: str | None, user: str, answer: str, thought: str | None = None) -> dict:
    """Tokeniza um exemplo e monta os `labels`. `thought=None` é o exemplo sem raciocínio.

    Com raciocínio o prompt termina em `<think>\\n`; sem raciocínio, em `<think></think>`. Levanta
    AssertionError se o prompt não for prefixo exato do exemplo (texto ou ids).
    """
    thinking = thought is not None
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    final = {"role": "assistant", "content": answer}
    if thinking:
        final["reasoning_content"] = thought
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    text = tok.apply_chat_template(msgs + [final], tokenize=False, enable_thinking=thinking)
    assert text.startswith(prompt), "o prompt não é prefixo do exemplo renderizado"
    assert prompt.endswith("<think>\n" if thinking else "<think></think>"), prompt[-30:]
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert ids[: len(p_ids)] == p_ids, "o prompt não é prefixo dos ids do exemplo"
    end = len(ids) - 1 - ids[::-1].index(IM_END_ID)  # último <|im_end|>; o `\n` depois dele fica de fora
    labels = [-100] * len(p_ids) + ids[len(p_ids): end + 1] + [-100] * (len(ids) - end - 1)
    assert len(labels) == len(ids) and any(x != -100 for x in labels)
    return {"input_ids": ids, "labels": labels, "n_tokens": len(ids), "n_prompt": len(p_ids), "loss_end": end + 1}
