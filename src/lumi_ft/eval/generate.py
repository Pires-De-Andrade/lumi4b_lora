"""Geração em lotes para a avaliação (A-04): lotes fixos, ordem fixa, retomável por lote.

- `enable_thinking` é sempre declarado (regra 7 do PRD) e o prompt vem de `apply_chat_template`.
- Os lotes dependem só dos prompts (não do modelo nem do que já foi gerado): os itens são ordenados
  por comprimento do prompt, do maior para o menor (se faltar memória, falha no primeiro lote), e o
  tamanho do lote é `min(max_batch, token_budget // maior_prompt_do_lote)`. Com isso, retomar uma
  execução interrompida mantém a composição de cada lote, que muda o top-1 em ~1,6% das posições.
- Cada lote vira uma linha do JSONL (uma escrita só); um lote ou está inteiro no arquivo ou não está.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from lumi_ft.gpu import append_jsonl, read_jsonl
from lumi_ft.modeling.load import EOS_IDS

MAX_NEW_TOKENS = 4096  # A-04, nos dois modos
MAX_BATCH = 32  # discovery.md §11
TOKEN_BUDGET = 32_768  # prefill medido: 32 x 1.024 cabe (16,9 GiB), 64 x 1.024 não


def render_prompt(tok, user_text: str, *, thinking: bool) -> str:
    """Prompt de geração: termina em `<think>\\n` com raciocínio e em `<think></think>` sem."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def plan_batches(lengths: list[int], max_batch: int = MAX_BATCH, token_budget: int = TOKEN_BUDGET) -> list[list[int]]:
    """Índices dos itens por lote. Determinístico: só depende de `lengths`."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, pos = [], 0
    while pos < len(order):
        size = max(1, min(max_batch, token_budget // max(1, lengths[order[pos]])))
        batches.append(order[pos : pos + size])
        pos += size
    return batches


def cut_at_eos(ids: list[int], eos_ids: list[int] = EOS_IDS) -> tuple[list[int], bool]:
    """Corta a geração no primeiro `eos`. Devolve (ids sem o eos, terminou)."""
    for i, t in enumerate(ids):
        if t in eos_ids:
            return ids[:i], True
    return ids, False


def make_generate_fn(model, tok, max_new_tokens: int = MAX_NEW_TOKENS) -> Callable[[list[str]], list[dict]]:
    """Função `prompts -> [{text, n_gen, finished, gen_len}]` com greedy, sem penalidade de repetição."""
    import torch

    def fn(prompts: list[str]) -> list[dict]:
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                eos_token_id=EOS_IDS,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        gen = out[:, enc["input_ids"].shape[1] :].cpu().tolist()
        rows = []
        for ids in gen:
            cut, finished = cut_at_eos(ids)
            rows.append(
                {
                    "text": tok.decode(cut, skip_special_tokens=False),
                    "n_gen": len(cut),
                    "finished": finished,
                    "gen_len": len(ids),  # passos que o lote inteiro rodou
                }
            )
        return rows

    return fn


def _header(items: list[dict], max_batch: int, token_budget: int, max_new: int, meta: dict) -> dict:
    ids = hashlib.sha256(json.dumps([[it["id"], it["n_prompt"]] for it in items]).encode()).hexdigest()[:16]
    return {"n_items": len(items), "items_hash": ids, "max_batch": max_batch, "token_budget": token_budget,
            "max_new_tokens": max_new, **meta}


def run_batches(
    items: list[dict],
    gen_fn: Callable[[list[str]], list[dict]],
    out: Path,
    *,
    max_batch: int = MAX_BATCH,
    token_budget: int = TOKEN_BUDGET,
    max_new_tokens: int = MAX_NEW_TOKENS,
    meta: dict | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Gera todos os itens, lote a lote, e devolve as linhas de itens (ordem dos lotes).

    `items`: `[{id, prompt, n_prompt}]`. Retomável: lotes já gravados em `out` são pulados, desde que o
    cabeçalho (itens, tamanho de lote, orçamento, limite de tokens, `meta`) seja idêntico.
    """
    header = _header(items, max_batch, token_budget, max_new_tokens, meta or {})
    rows = read_jsonl(out)
    if rows:
        assert rows[0].get("header") == header, f"{out} foi gerado com outra configuração:\n{rows[0].get('header')}\n{header}"
    else:
        append_jsonl(out, {"header": header})
    done = {r["batch"]: r for r in rows if "batch" in r}
    batches = plan_batches([it["n_prompt"] for it in items], max_batch, token_budget)
    for k, idxs in enumerate(batches):
        if k in done:
            continue
        t0 = time.perf_counter()
        res = gen_fn([items[i]["prompt"] for i in idxs])
        assert len(res) == len(idxs)
        seconds = round(time.perf_counter() - t0, 2)
        line = {
            "batch": k,
            "size": len(idxs),
            "seconds": seconds,
            "gen_len": max(r["gen_len"] for r in res),
            "items": [{"id": items[i]["id"], "n_prompt": items[i]["n_prompt"], **r} for i, r in zip(idxs, res)],
        }
        append_jsonl(out, line)
        done[k] = line
        log(f"lote {k + 1}/{len(batches)}: {len(idxs)} itens, {line['gen_len']} passos, {seconds:.0f} s")
    return [it for k in sorted(done) for it in done[k]["items"]]
