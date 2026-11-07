"""Provas de múltipla escolha: carregadores, prompt zero-shot e pontuação (A-04, A-05, A-08).

Todos os carregadores devolvem itens `{id, question, options, gold}`; `gold` é a letra da alternativa certa.
A nota é a forma estrita (`Resposta: <letra>` na última linha, depois do `</think>`); a flexível é registrada ao lado.
"""

from __future__ import annotations

from collections import Counter

from lumi_ft.eval import parse as P
from lumi_ft.eval.gsm8k import _rate
from lumi_ft.eval.shape import length_stats, trace_language

LETTERS = "ABCDEFGHIJ"

INSTRUCTION = (
    "Responda à questão de múltipla escolha abaixo. Mostre o raciocínio e termine a resposta com uma única linha no "
    "formato `Resposta: <letra>`, em que <letra> é a alternativa escolhida."
)


INSTRUCTION_EN = (
    "Answer the multiple-choice question below. Show your reasoning and end your answer with a single line in the "
    "format `Resposta: <letter>`, where <letter> is the chosen option."
)  # só para o diagnóstico português x inglês; o marcador `Resposta:` fica igual para a extração ser a mesma


def build_prompt(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return f"{INSTRUCTION}\n\nQuestão: {item['question'].strip()}\n\n{opts}"


def build_prompt_en(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return f"{INSTRUCTION_EN}\n\nQuestion: {item['question'].strip()}\n\n{opts}"


def _clean(opts: list) -> list[str]:
    return [str(o).strip() for o in opts]


# -------------------------------------------------------------------------------- carregadores
def _load(name, cfg, split):
    from datasets import load_dataset

    ds = load_dataset(name, cfg, split=split) if split else load_dataset(name, cfg)
    return ds if split else [r for d in ds.values() for r in d]


def load_mmlu_prox(split: str, lang: str = "pt") -> list[dict]:
    items = []
    tag = f"mmlu_prox_{split}" if lang == "pt" else f"mmlu_prox_{lang}_{split}"
    for r in _load("li-lab/MMLU-ProX", lang, split):
        opts = [r[f"option_{i}"] for i in range(10)]
        while opts and (opts[-1] is None or str(opts[-1]).strip() in ("", "N/A")):
            opts.pop()  # só alternativas ausentes no fim; as letras não deslocam
        if any(o is None for o in opts) or r["answer_index"] >= len(opts):
            continue
        items.append({"id": f"{tag}:{r['question_id']}", "question": r["question"], "options": _clean(opts), "gold": r["answer"]})
    return items


def load_mmlu_prox_diag(lang: str, n: int = 1200, seed: int = 0) -> list[dict]:
    """Diagnóstico português x inglês: as mesmas `n` questões do teste do MMLU-ProX nos dois idiomas (ids iguais nos dois lados,
    para o pareamento). Só entram questões válidas nos dois idiomas."""
    import random

    pt = {i["id"].split(":", 1)[1]: i for i in load_mmlu_prox("test", "pt")}
    en = {i["id"].split(":", 1)[1]: i for i in load_mmlu_prox("test", "en")}
    ids = sorted(set(pt) & set(en), key=int)
    keep = random.Random(seed).sample(ids, min(n, len(ids)))
    src = pt if lang == "pt" else en
    return [{**src[q], "id": f"mmlu_prox_diag:{q}"} for q in keep]


def load_global_mmlu(split: str) -> list[dict]:
    return [{"id": f"global_mmlu_{split}:{r['sample_id']}", "question": r["question"],
             "options": _clean([r["option_a"], r["option_b"], r["option_c"], r["option_d"]]), "gold": r["answer"]}
            for r in _load("CohereLabs/Global-MMLU", "pt", split)]


def load_mmmlu() -> list[dict]:
    return [{"id": f"mmmlu:{i}", "question": r["Question"], "options": _clean([r["A"], r["B"], r["C"], r["D"]]), "gold": r["Answer"]}
            for i, r in enumerate(_load("openai/MMMLU", "PT_BR", "test"))]


def load_include(split: str) -> list[dict]:
    return [{"id": f"include_{split}:{i}", "question": r["question"],
             "options": _clean([r["option_a"], r["option_b"], r["option_c"], r["option_d"]]), "gold": LETTERS[int(r["answer"])]}
            for i, r in enumerate(_load("CohereLabs/include-base-44", "Portuguese", split))]


def _load_choices(name: str, tag: str) -> list[dict]:
    items = []
    for r in _load(name, None, None):
        texts, labels = r["choices"]["text"], r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        # as letras do prompt são A, B, C, ... na ordem; `gold` vira a letra da posição do rótulo certo
        items.append({"id": f"{tag}:{r['id']}", "question": r["question"], "options": _clean(texts),
                      "gold": LETTERS[labels.index(r["answerKey"])]})
    return items


LOADERS = {
    "mmlu_prox_test": lambda: load_mmlu_prox("test"),
    "mmlu_prox_val": lambda: load_mmlu_prox("validation"),
    "mmlu_prox_diag_pt": lambda: load_mmlu_prox_diag("pt"),
    "mmlu_prox_diag_en": lambda: load_mmlu_prox_diag("en"),
    "global_mmlu_test": lambda: load_global_mmlu("test"),
    "global_mmlu_dev": lambda: load_global_mmlu("dev"),
    "mmmlu_test": load_mmmlu,
    "include_test": lambda: load_include("test"),
    "include_val": lambda: load_include("validation"),
    "enem": lambda: _load_choices("eduagarcia/enem_challenge", "enem"),
    "bluex": lambda: _load_choices("nicholasKluge/BLUEX_without_images", "bluex"),
    "oab": lambda: _load_choices("eduagarcia/oab_exams", "oab"),
}


# ----------------------------------------------------------------------------- pontuação
def score_row(gen: dict, item: dict, *, thinking: bool) -> dict:
    letters = LETTERS[: len(item["options"])]
    ex = P.extract_choice(gen["text"], thinking=thinking, letters=letters)
    t = ex.parsed.thought
    return {
        "id": gen["id"], "status": ex.status, "pred": ex.pred, "flexible": ex.flexible, "gold": item["gold"],
        "strict_ok": ex.pred == item["gold"], "flexible_ok": ex.flexible == item["gold"],
        "finished": gen["finished"], "n_gen": gen["n_gen"], "closed": ex.parsed.closed, "n_close": ex.parsed.n_close,
        "answer_nonempty": bool(ex.parsed.answer), "trace_nonempty": bool(t),
        "trace_language": trace_language(t) if t else None,
    }


def summarize(rows: list[dict], *, thinking: bool) -> dict:
    out = {
        "n": len(rows), "thinking": thinking,
        "strict_acc": _rate([r["strict_ok"] for r in rows]),
        "flexible_acc": _rate([r["flexible_ok"] for r in rows]),
        "status_rates": P.rates(r["status"] for r in rows),
        "truncated_rate": _rate([not r["finished"] for r in rows]),
        "n_gen": length_stats([r["n_gen"] for r in rows]),
    }
    if thinking:
        out |= {"closed_rate": _rate([r["closed"] for r in rows]),
                "closed_and_answer_rate": _rate([r["closed"] and r["answer_nonempty"] for r in rows]),
                "trace_language": dict(Counter(r["trace_language"] for r in rows if r["trace_language"]))}
    return out
