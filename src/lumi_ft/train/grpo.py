"""GRPO (RL com recompensa verificável) sobre o adaptador do SFT: recompensas, vantagens, loss e montagem dos exemplos.

Fluxo de uma iteração (processos separados, porque a 4090 não comporta o vLLM e o treino juntos):
  1. funde o adaptador atual e gera `G` respostas por prompt com o vLLM (`scripts/grpo_rollout.py`), já com a recompensa;
  2. treina o adaptador com a vantagem de cada resposta em relação ao grupo (`scripts/grpo_train.py`).

Recompensa: 1 só se a resposta fecha um único `</think>`, termina sozinha (não trunca), o traço está em português e a
resposta final bate com o gabarito no formato estrito do prompt (`#### N` ou `Resposta: <letra>`); senão 0. Resposta truncada
(laço) vale 0 e entra no treino com vantagem negativa, que é o que ensina a parar.
"""

from __future__ import annotations

import random
from collections import Counter

from lumi_ft.data import selfdistill as S
from lumi_ft.eval import mcq as M
from lumi_ft.modeling.load import EOS_IDS


# ------------------------------------------------------------------------------ recompensas
def reward_math(text: str, finished: bool, ref: str) -> tuple[float, str]:
    rec, why = S.accept_think(text, finished, ref)
    return (1.0 if rec else 0.0), why


def reward_mcq(text: str, finished: bool, gold: str, n_options: int) -> tuple[float, str]:
    if not finished:
        return 0.0, "truncado"
    row = M.score_row({"id": "", "text": text, "finished": finished, "n_gen": 0}, {"options": [None] * n_options, "gold": gold}, thinking=True)
    if not row["closed"] or row["n_close"] != 1:
        return 0.0, "sem_um_unico_close"
    if not row["answer_nonempty"] or not row["trace_nonempty"]:
        return 0.0, "traco_ou_resposta_vazios"
    if not row["strict_ok"]:
        return 0.0, "resposta_errada_ou_sem_marcador"
    if row["trace_language"] != "pt":
        return 0.0, "traco_nao_pt"
    return 1.0, "ok"


def reward_for(prompt: dict, text: str, finished: bool) -> tuple[float, str]:
    if prompt["kind"] == "math":
        return reward_math(text, finished, prompt["ref"])
    return reward_mcq(text, finished, prompt["gold"], prompt["n_options"])


# ------------------------------------------------------------------------------- vantagens
def group_advantages(rewards: list[float], mode: str = "std", eps: float = 1e-4) -> list[float]:
    """Vantagem de cada resposta de um grupo: recompensa menos a média do grupo (`mean`, Dr. GRPO) ou, além disso,
    dividida pelo desvio-padrão (`std`, GRPO original). Grupo em que todas as recompensas são iguais não ensina nada: zeros."""
    n = len(rewards)
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if all(abs(c) < 1e-9 for c in centered):
        return [0.0] * n
    if mode == "mean":
        return centered
    if mode != "std":
        raise ValueError(f"modo de vantagem desconhecido: {mode}")
    std = (sum(c * c for c in centered) / n) ** 0.5
    return [c / (std + eps) for c in centered]


# ----------------------------------------------------------------------------- prompts
def select_prompts(prompts: list[dict], it: int, n: int, frac_mcq: float, seed: int = 0) -> list[dict]:
    """Os `n` prompts da iteração `it`: `frac_mcq` de múltipla escolha e o resto de matemática. Cada tipo anda por uma
    ordem embaralhada fixa (depende só de `seed`), sem repetir até acabar o conjunto. Determinístico em (it, seed)."""
    out = []
    n_mcq = round(n * frac_mcq)
    for kind, k in (("mcq", n_mcq), ("math", n - n_mcq)):
        pool = sorted((p for p in prompts if p["kind"] == kind), key=lambda p: p["id"])
        if not pool or k == 0:
            continue
        random.Random(seed).shuffle(pool)
        start = (it * k) % len(pool)
        take = [pool[(start + j) % len(pool)] for j in range(k)]
        out += take
    return out


# ------------------------------------------------------------------------------ rollouts
def complete_ids(token_ids: list[int], finish_reason: str | None, stop_reason) -> tuple[list[int], bool]:
    """Garante que a resposta que parou sozinha termine no token de parada (para o treino ensinar a parar).
    Devolve (ids, se acrescentou o token). `stop_reason` é o id do token de parada que o vLLM informa, ou None."""
    ids = list(token_ids)
    if finish_reason == "stop" and (not ids or ids[-1] not in EOS_IDS):
        ids.append(stop_reason if isinstance(stop_reason, int) else EOS_IDS[-1])
        return ids, True
    return ids, False


def build_samples(prompt: dict, prompt_ids: list[int], outputs: list[dict], adv_mode: str = "std") -> tuple[list[dict], dict]:
    """`outputs`: [{ids, text, finished}] de um prompt. Devolve (amostras com recompensa e vantagem, contagem de motivos)."""
    rs, whys, samples = [], Counter(), []
    for o in outputs:
        r, why = reward_for(prompt, o["text"], o["finished"])
        rs.append(r)
        whys[why] += 1
    adv = group_advantages(rs, adv_mode)
    for o, r, a in zip(outputs, rs, adv):
        samples.append({"prompt_id": prompt["id"], "kind": prompt["kind"], "prompt_ids": prompt_ids, "completion_ids": o["ids"],
                        "reward": r, "advantage": a, "finished": o["finished"]})
    return samples, dict(whys)


# ---------------------------------------------------------------------------------- loss
def completion_logprobs(logits, input_ids, n_prompt: int, n_total: int):
    """Log-probabilidade de cada token da resposta (posições n_prompt .. n_total-1) de UMA sequência.
    `logits`: [L, V] da sequência (sem padding à direita contado além de n_total). Devolve tensor [n_total - n_prompt] em FP32."""
    import torch

    lg = logits[n_prompt - 1: n_total - 1].float()
    tgt = input_ids[n_prompt:n_total]
    return torch.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def ppo_clip_loss_sum(logp_new, logp_old, adv: float, eps_low: float = 0.2, eps_high: float = 0.28):
    """Soma, sobre os tokens da resposta, de -min(r*A, clip(r, 1-eps_low, 1+eps_high)*A). Quem divide pelo total de
    tokens do minibatch é o laço de treino (assim a acumulação de gradiente dá o mesmo gradiente de um batch único)."""
    import torch

    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    return -torch.minimum(unclipped, clipped).sum()
