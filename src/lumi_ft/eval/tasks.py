"""Registro de tarefas de avaliação: cada uma sabe carregar seus itens, montar o prompt e pontuar uma saída."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lumi_ft.eval import gsm8k as K
from lumi_ft.eval import mcq as M

VAL_GSM8K = "data/val_gsm8k.jsonl"
VAL_MCQ = "data/val_mcq.jsonl"


@dataclass
class Task:
    name: str
    items: list[dict]  # cada item tem `id`
    prompt: Callable[[dict], str]  # texto do usuário
    score: Callable[[dict, dict, bool], dict]  # (gen, item, thinking) -> linha pontuada
    summarize: Callable[[list[dict], bool], dict]


def _read_jsonl(path: str) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} não existe; rode scripts/build_validation.py")
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _gsm8k(name: str, items: list[dict]) -> Task:
    return Task(name, items, lambda it: K.build_prompt(it["question"]),
                lambda gen, it, thinking: K.score_row(gen, it["answer"], thinking=thinking),
                lambda rows, thinking: K.summarize(rows, [], thinking=thinking))


def _mcq(name: str, items: list[dict]) -> Task:
    return Task(name, items, M.build_prompt, lambda gen, it, thinking: M.score_row(gen, it, thinking=thinking),
                lambda rows, thinking: M.summarize(rows, thinking=thinking))


# nome -> função que devolve a Task
SUITE = ["gsm8k_test", "mmlu_prox_test", "enem", "bluex", "oab"]  # a Nota de raciocínio é a média destas
CARD = ["global_mmlu_test", "mmmlu_test", "include_test"]  # notas do cartão disponíveis (faltam MMLU PT e IFEval PT)
VALIDATION = ["gsm8k_val", "mcq_val"]


def get_task(name: str, limit: int = 0) -> Task:
    if name == "gsm8k_test":
        items = [{"id": it["id"], "question": it["question"], "answer": it["answer"]} for it in K.load_items(None)[0]]
        t = _gsm8k(name, items)
    elif name == "gsm8k_val":
        t = _gsm8k(name, _read_jsonl(VAL_GSM8K))
    elif name == "mcq_val":
        t = _mcq(name, _read_jsonl(VAL_MCQ))
    elif name in M.LOADERS:
        t = _mcq(name, M.LOADERS[name]())
    else:
        raise KeyError(f"tarefa desconhecida: {name}")
    if limit:
        t.items = t.items[:limit]
    return t
