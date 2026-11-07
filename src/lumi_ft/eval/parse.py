"""Extração de resposta da saída do Lumi (A-05 e A-06 do PRD).

Regras:
- A resposta é o texto depois do primeiro `</think>`. Em `enable_thinking=True` a saída gerada
  começa dentro do traço (o prompt termina em `<think>\\n`), então sem `</think>` não há resposta.
- Em `enable_thinking=False` o prompt já termina em `<think></think>`; a saída é a própria resposta.
- Nada dentro do traço vale como resposta, mesmo que o traço contenha `Resposta: B` ou `#### 7`.
- Cada falha tem um status próprio, para que a taxa de cada caso vá para o relatório.

Notas de pontuação:
- Múltipla escolha: a nota é a forma estrita (linha final `Resposta: <letra>`); a forma flexível
  é registrada ao lado, para expor erro de formato do harness (risco do BELEBELE do Qwen3-4B).
- GSM8K-PT: `strict-match` e `flexible-extract` replicam o `gsm8k.yaml` do harness (o fork
  Polygl0t/lm-evaluation-harness usa o mesmo), inclusive as limitações, para a nota ser comparável.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
IM_END = "<|im_end|>"

# Status de uma extração (A-05).
OK = "ok"
NO_CLOSE = "no_close"  # modo com raciocínio sem `</think>`, ou `<think>` aberto e nunca fechado
EMPTY_ANSWER = "empty_answer"  # nada depois de `</think>`
NO_OPTION = "no_option"  # múltipla escolha sem alternativa no formato pedido
INVALID_OPTION = "invalid_option"  # letra fora das alternativas da questão
NO_NUMBER = "no_number"  # GSM8K-PT sem `#### <número>`
STATUSES = (OK, NO_CLOSE, EMPTY_ANSWER, NO_OPTION, INVALID_OPTION, NO_NUMBER)


@dataclass(frozen=True)
class Parsed:
    thought: str | None  # texto antes do `</think>`; None se a saída não tem `</think>`
    answer: str | None  # texto depois do primeiro `</think>`; None se não há `</think>`
    closed: bool
    n_close: int  # quantos `</think>` a saída tem (mais de 1 vira alerta no relatório)

    @property
    def answer_or_empty(self) -> str:
        """Texto para avaliadores que não distinguem erro de formato, como o do IFEval."""
        return self.answer or ""


def split_response(text: str, *, thinking: bool) -> Parsed:
    """Separa traço e resposta. `text` é a saída gerada, sem o prompt."""
    text = text.split(IM_END, 1)[0]
    n_close = text.count(THINK_CLOSE)
    if n_close:
        before, _, after = text.partition(THINK_CLOSE)
        thought = before.removeprefix(THINK_OPEN).strip()
        return Parsed(thought, after.strip(), True, n_close)
    if thinking or THINK_OPEN in text:
        # Com raciocínio, a saída começa dentro do traço; sem `</think>` o traço não terminou.
        return Parsed(None, None, False, 0)
    return Parsed(None, text.strip(), False, 0)


def _status_of(parsed: Parsed) -> str | None:
    if parsed.answer is None:
        return NO_CLOSE
    if not parsed.answer:
        return EMPTY_ANSWER
    return None


def rates(statuses: Iterable[str]) -> dict[str, float | int]:
    """Taxa de cada status, para o relatório (A-05). Inclui `n`."""
    statuses = list(statuses)
    n = len(statuses)
    c = Counter(statuses)
    unknown = set(c) - set(STATUSES)
    if unknown:
        raise ValueError(f"status desconhecido: {sorted(unknown)}")
    return {"n": n, **{s: (c[s] / n if n else 0.0) for s in STATUSES}}


# ---------------------------------------------------------------- múltipla escolha

# Forma estrita: linha própria `Resposta: X`, com negrito, parênteses e ponto finais opcionais.
_STRICT_CHOICE = re.compile(
    r"^[ \t]*\**[ \t]*resposta[ \t]*\**[ \t]*:[ \t]*\**[ \t]*\(?([A-Za-z])\)?[ \t]*\**[ \t]*\.?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Forma flexível: afirma a alternativa em frase (`a resposta correta é a letra B`, `gabarito: C`).
# A palavra-chave ignora caixa; a letra tem que ser maiúscula, porque `a` e `o` são artigos e `E` é
# conjunção em português: sem a palavra-chave, nenhuma letra solta vale.
_FLEX_CHOICE = re.compile(
    r"(?i:(?:resposta|gabarito|alternativa|letra|op[çc][ãa]o)(?:\s+(?:correta|certa|final))?"
    r"\s*(?::|é|seria|=)\s*(?:(?:a|o)\s+)?(?:(?:alternativa|letra|op[çc][ãa]o)\s+)?)"
    r"\**\(?([A-Z])\)?\**(?![\wÀ-ÿ])"
)
_BARE_CHOICE = re.compile(r"^\W*([A-Z])\W*$")


def strict_choice(answer: str) -> str | None:
    """Última linha `Resposta: X`, em maiúscula; None se não houver."""
    found = _STRICT_CHOICE.findall(answer)
    return found[-1].upper() if found else None


def flexible_choice(answer: str) -> str | None:
    """Última afirmação de alternativa em frase, ou a resposta inteira sendo só a letra."""
    found = _FLEX_CHOICE.findall(answer)
    if found:
        return found[-1]
    bare = _BARE_CHOICE.match(answer.strip())
    return bare.group(1) if bare else None


@dataclass(frozen=True)
class ChoiceExtraction:
    status: str  # um de STATUSES; só `ok` pontua
    pred: str | None  # forma estrita, validada contra `letters`; é a nota
    flexible: str | None  # forma flexível, validada contra `letters`; só reportada
    parsed: Parsed


def extract_choice(text: str, *, thinking: bool, letters: str) -> ChoiceExtraction:
    """Extrai a alternativa de uma saída. `letters` são as alternativas válidas da questão."""
    parsed = split_response(text, thinking=thinking)
    status = _status_of(parsed)
    if status:
        return ChoiceExtraction(status, None, None, parsed)
    assert parsed.answer is not None
    ok = lambda x: x if x is not None and x in letters else None  # noqa: E731
    flexible = ok(flexible_choice(parsed.answer))
    raw = strict_choice(parsed.answer)
    if raw is None:
        return ChoiceExtraction(NO_OPTION, None, flexible, parsed)
    if raw not in letters:
        return ChoiceExtraction(INVALID_OPTION, None, flexible, parsed)
    return ChoiceExtraction(OK, raw, flexible, parsed)


# ---------------------------------------------------------------------------- GSM8K-PT

# Idênticos ao `gsm8k.yaml` do harness (`filter_list`) e às `regexes_to_ignore` da métrica.
_GSM_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_GSM_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_GSM_IGNORE = (",", r"\$", r"(?s).*#### ", r"\.$")


def normalize_number(s: str) -> str:
    """`exact_match` do harness: aplica `regexes_to_ignore` e ignora caixa; não mexe em `.`."""
    for pattern in _GSM_IGNORE:
        s = re.sub(pattern, "", s)
    return s.lower()


def gsm8k_target(answer: str) -> str:
    """Alvo normalizado a partir do campo `answer` do dataset (`... #### 70000`)."""
    return normalize_number(answer)


def strict_number(answer: str) -> str | None:
    """`strict-match`: o primeiro `#### <número>`; None se não houver (o harness grava `[invalid]`)."""
    m = _GSM_STRICT.search(answer)
    return m.group(1).strip() if m else None


def last_number(text: str) -> str | None:
    """`flexible-extract`: o último número do texto, com a regex do harness (limitações incluídas)."""
    found = _GSM_FLEX.findall(text)
    if not found:
        return None
    return next(g for g in found[-1] if g).strip()


def gsm8k_correct(pred: str | None, target: str) -> bool:
    return pred is not None and normalize_number(pred) == normalize_number(target)


def to_decimal_ptbr(s: str) -> Decimal | None:
    """Lê um número em formato brasileiro ou inglês. Só para diagnóstico, nunca para a nota.

    `70.000` é 70000; `3,5` é 3,5; `1.234,56` é 1234,56; `3.5` é 3,5. `1.500` é lido como 1500,
    porque no GSM8K-PT os passos usam `.` como separador de milhar.
    """
    s = re.sub(r"\.$", "", s.replace("$", "").strip())  # sem `normalize_number`, que apaga a vírgula
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def gsm8k_correct_ptbr(pred: str | None, target: str) -> bool:
    """Igual a `gsm8k_correct`, mas aceita `70.000` para 70000. Diagnóstico de A-06."""
    if pred is None:
        return False
    if gsm8k_correct(pred, target):
        return True
    a, b = to_decimal_ptbr(pred), to_decimal_ptbr(target)
    return a is not None and b is not None and a == b


@dataclass(frozen=True)
class NumberExtraction:
    status: str  # um de STATUSES; `ok` só diz que `strict-match` achou um número
    strict: str | None  # `strict-match`: é a nota (A-06)
    flexible: str | None  # `flexible-extract`: só reportada
    parsed: Parsed

    def trace_last_number(self) -> str | None:
        """Último número do traço, para a métrica de A-08 (o traço termina na resposta certa?)."""
        return last_number(self.parsed.thought) if self.parsed.thought else None


def extract_number(text: str, *, thinking: bool) -> NumberExtraction:
    """Extrai a resposta numérica do GSM8K-PT de uma saída, só depois do `</think>`."""
    parsed = split_response(text, thinking=thinking)
    status = _status_of(parsed)
    if status:
        return NumberExtraction(status, None, None, parsed)
    assert parsed.answer is not None
    strict = strict_number(parsed.answer)
    flexible = last_number(parsed.answer)
    return NumberExtraction(OK if strict is not None else NO_NUMBER, strict, flexible, parsed)
