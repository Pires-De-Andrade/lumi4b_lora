"""Fase anti-laço, no estilo do Antidoom/FTPO (Liquid AI, https://liquid.ai/blog/antidoom): preferência aplicada só ao token em que o modelo decide repetir.

Para cada resposta gulosa que não termina e acaba em laço, o exemplo é (prefixo até o início da repetição, token rejeitado = o primeiro token da
primeira repetição, escolhidos = alternativas plausíveis do próprio modelo de referência naquela posição). A perda age nos logits (sem softmax sobre o
vocabulário): sobe a média dos escolhidos e desce o rejeitado, e segura o resto do vocabulário perto da referência, ponderado pela probabilidade de
referência. Este módulo tem só a lógica (sem vLLM e sem modelo): detector de laço, escolha dos substitutos e perda.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def find_loop_start(ids: list[int], *, k_min: int = 4, k_max: int = 800, min_repeats: int = 4, min_chars: int = 0, decode=None) -> tuple[int, int] | None:
    """Laço no fim da resposta: um bloco de `k` tokens que se repete em sequência, com `min_repeats` cópias ou mais (a última pode estar cortada).

    Devolve (s, k): `s` é a posição do primeiro token da primeira repetição (o token em que o modelo repetiu em vez de seguir) e `k` o período, o menor
    que serve. Com `decode` e `min_chars`, exige que o bloco tenha pelo menos esse tamanho em caracteres (o Antidoom usa 60). None se não há laço."""
    a = np.asarray(ids, dtype=np.int64)
    n = len(a)
    for k in range(k_min, min(k_max, n // min_repeats) + 1):
        eq = a[k:] == a[:-k]  # eq[i] compara a[i + k] com a[i]
        if not eq[-1]:
            continue
        falses = np.flatnonzero(~eq)
        run = len(eq) if len(falses) == 0 else len(eq) - 1 - int(falses[-1])
        if run < (min_repeats - 1) * k:
            continue
        s = k if len(falses) == 0 else int(falses[-1]) + 1 + k
        if min_chars and decode is not None and len(decode(ids[s:s + k])) < min_chars:
            continue
        return s, k
    return None


def choose_candidates(ref_logits: torch.Tensor, rejected: int, decode, *, top_k: int = 64, max_chosen: int = 20, min_alnum: int = 2,
                      exclude: set[int] | None = None) -> list[int]:
    """Até `max_chosen` substitutos plausíveis para `rejected`: os de maior logit de referência cujo texto tem pelo menos `min_alnum` caracteres
    alfanuméricos (tira pontuação, espaços e pedaços curtos) e que não são tokens especiais (`exclude`)."""
    exclude = exclude or set()
    out: list[int] = []
    for t in torch.topk(ref_logits.float(), min(top_k, ref_logits.numel())).indices.tolist():
        if t == rejected or t in exclude:
            continue
        if sum(ch.isalnum() for ch in decode([t])) < min_alnum:
            continue
        out.append(t)
        if len(out) == max_chosen:
            break
    return out


def ftpo_loss(z: torch.Tensor, z0: torch.Tensor, chosen: list[int], rejected: int, *, beta: float = 1.0, lam_rest: float = 5.0,
              lam_cr: float = 1.0, tau: float = 3.0) -> tuple[torch.Tensor, dict]:
    """Perda de um exemplo. `z`: logits do aluno (V,); `z0`: logits da referência (V,), sem gradiente; `chosen`: ids escolhidos; `rejected`: id rejeitado.

    - preferência: softplus(-beta * m), com m = média(z_c - z0_c) - (z_r - z0_r): sobe os escolhidos e desce o rejeitado em relação à referência;
    - resto do vocabulário: lam_rest * soma_v p0_v * (z_v - z0_v)^2, com p0 = softmax(z0) (o que importa fica perto da referência);
    - escolhidos e rejeitado: lam_cr * média relu(|z_v - z0_v| - tau)^2 (podem se mover até `tau` sem custo).
    Devolve (perda, métricas)."""
    d = (z - z0).float()
    ch = torch.as_tensor(chosen, device=z.device, dtype=torch.long)
    m = d[ch].mean() - d[rejected]
    pref = F.softplus(-beta * m)
    mask = torch.ones_like(d, dtype=torch.bool)
    mask[ch] = False
    mask[rejected] = False
    p0 = torch.softmax(z0.float(), dim=-1)
    rest = (p0 * d ** 2 * mask).sum()
    moved = torch.cat([d[ch], d[rejected].view(1)])
    cr = (torch.relu(moved.abs() - tau) ** 2).mean()
    loss = pref + lam_rest * rest + lam_cr * cr
    with torch.no_grad():
        zz = z.float()
        metrics = {"chosen_win": float(zz[ch].max() > zz[rejected]), "rejected_is_argmax": float(int(zz.argmax()) == rejected),
                   "margin": float(m), "pref": float(pref), "rest": float(rest), "cr": float(cr)}
    return loss, metrics
