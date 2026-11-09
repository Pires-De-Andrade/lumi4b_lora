"""GRPO (RL com recompensa verificável) sobre o adaptador do SFT: recompensas, vantagens, loss e montagem dos exemplos.

Fluxo de uma iteração (processos separados, porque a 4090 não comporta o vLLM e o treino juntos):
  1. funde o adaptador atual e gera `G` respostas por prompt com o vLLM (`scripts/grpo_rollout.py`), já com a recompensa;
  2. treina o adaptador com a vantagem de cada resposta em relação ao grupo (`scripts/grpo_train.py`).

Recompensa: 1 só se a resposta fecha um único `</think>`, termina sozinha (não trunca), o traço está em português e a
resposta final bate com o gabarito no formato estrito do prompt (`#### N` ou `Resposta: <letra>`); senão 0. Resposta truncada
(laço) vale 0 e entra no treino com vantagem negativa, que é o que ensina a parar.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from lumi_ft.data import selfdistill as S
from lumi_ft.eval import mcq as M
from lumi_ft.eval import parse as P
from lumi_ft.eval.shape import trace_language
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


PARTS = ("finished", "closed_once", "nonempty", "correct", "pt")


def reward_parts(prompt: dict, text: str, finished: bool) -> dict[str, int]:
    """Os cinco requisitos da recompensa como indicadores 0/1 **independentes** (sem ordem de prioridade): terminou sozinha,
    fechou exatamente um `</think>`, traço e resposta não vazios, acertou no formato estrito, traço em português. A recompensa é 1
    se e só se os cinco valem; os `whys` guardam só o PRIMEIRO motivo de falha, e isto guarda todos (só registro: a recompensa não muda)."""
    parsed = P.split_response(text, thinking=True)
    # O traço só existe depois do `</think>`; numa resposta que não fechou (truncada) o texto inteiro é o traço até ali, e o idioma dele
    # é o que interessa para saber se o laço é em português. A recompensa não depende disso (sem fechar, ela já é 0).
    trace = parsed.thought or ("" if parsed.closed else text)
    pt = bool(trace) and trace_language(trace) == "pt"
    if prompt["kind"] == "mcq":
        row = M.score_row({"id": "", "text": text, "finished": finished, "n_gen": 0},
                          {"options": [None] * prompt["n_options"], "gold": prompt["gold"]}, thinking=True)
        closed_once = bool(row["closed"]) and row["n_close"] == 1
        nonempty = bool(row["answer_nonempty"]) and bool(row["trace_nonempty"])
        correct = bool(row["strict_ok"])
    else:
        closed_once = bool(parsed.closed) and parsed.n_close == 1
        nonempty = bool(parsed.thought) and bool(parsed.answer)
        correct = bool(parsed.answer) and S.same_number(P.strict_number(parsed.answer), prompt["ref"])
    return {"finished": int(bool(finished)), "closed_once": int(closed_once), "nonempty": int(nonempty), "correct": int(bool(correct)), "pt": int(pt)}


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
                        "reward": r, "advantage": a, "finished": o["finished"],
                        "source": prompt.get("source"), "parts": reward_parts(prompt, o["text"], o["finished"])})
    return samples, dict(whys)


LENGTH_BINS = ((256, "<256"), (512, "256-511"), (1024, "512-1023"), (2048, "1024-2047"), (10**9, ">=2048"))


def _length_bin(n: int) -> str:
    return next(label for edge, label in LENGTH_BINS if n < edge)


def summarize_rollouts(samples: list[dict]) -> dict:
    """Resumo de um ciclo de rollouts (amostras de `build_samples`), por tipo: taxa de cada requisito da recompensa; grupos (todos certos,
    todos zero, mistos) e, nos todos-zero, a causa; e recompensa e truncamento por fonte e por faixa de comprimento da resposta (tokens)."""
    rate = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    groups: dict = defaultdict(list)
    for s in samples:
        groups[(s["kind"], s["prompt_id"])].append(s)
    out: dict = {"flags": {}, "groups": {}, "by_source": {}, "by_length": {}}
    for kind in sorted({s["kind"] for s in samples}):
        ss = [s for s in samples if s["kind"] == kind]
        out["flags"][kind] = {"n": len(ss), **{f: rate([s["parts"][f] for s in ss]) for f in PARTS}}
        gs = [g for (k, _), g in groups.items() if k == kind]
        ones = [g for g in gs if all(x["reward"] == 1 for x in g)]
        zeros = [g for g in gs if all(x["reward"] == 0 for x in g)]
        causes: Counter = Counter()
        for g in zeros:  # todas truncadas; ou alguma acertou o conteúdo mas falhou em idioma/fechamento/formato; ou todas terminaram erradas
            causes["todas_truncadas" if all(not x["parts"]["finished"] for x in g)
                   else "alguma_correta_mas_reprovada" if any(x["parts"]["correct"] for x in g) else "erradas"] += 1
        out["groups"][kind] = {"n": len(gs), "todos_certos": len(ones), "todos_zero": len(zeros), "mistos": len(gs) - len(ones) - len(zeros),
                               "causas_todos_zero": dict(causes)}
        by_len: dict = defaultdict(list)
        for s in ss:
            by_len[_length_bin(len(s["completion_ids"]))].append(s)
        out["by_length"][kind] = {label: {"n": len(by_len[label]), "reward": rate([x["reward"] for x in by_len[label]]),
                                          "truncated": rate([not x["finished"] for x in by_len[label]])}
                                  for _, label in LENGTH_BINS if by_len[label]}
    by_src: dict = defaultdict(list)
    for s in samples:
        by_src[f"{s['kind']}:{s.get('source') or '-'}"].append(s)
    out["by_source"] = {k: {"n": len(v), "reward": rate([x["reward"] for x in v]), "truncated": rate([not x["finished"] for x in v]),
                            "correct": rate([x["parts"]["correct"] for x in v])} for k, v in sorted(by_src.items())}
    return out


# ---------------------------------------------------------------------------------- loss
def minibatch_indices(n: int, mb_seqs: int, seed: int) -> list[list[int]]:
    """Os minibatches do otimizador: índices embaralhados pela semente e repartidos em blocos de `mb_seqs`.
    A mesma partição serve para a passada das log-probabilidades antigas (`--old-same-batches`) e para o treino."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return [order[s0: s0 + mb_seqs] for s0 in range(0, n, mb_seqs)]


def completion_logprobs(logits, input_ids, n_prompt: int, n_total: int, temperature: float = 1.0):
    """Log-probabilidade de cada token da resposta (posições n_prompt .. n_total-1) de UMA sequência.
    `logits`: [L, V] da sequência (sem padding à direita contado além de n_total). Devolve tensor [n_total - n_prompt] em FP32.
    `temperature` != 1: log-probabilidade da distribuição amostrada (logits / T), a mesma do gerador; o padrão 1,0 não muda nada."""
    import torch

    lg = logits[n_prompt - 1: n_total - 1].float()
    if temperature != 1.0:
        lg = lg / temperature
    tgt = input_ids[n_prompt:n_total]
    return torch.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def completion_entropy(logits, n_prompt: int, n_total: int, chunk: int = 512):
    """Entropia (em nats, temperatura 1) da distribuição do modelo em cada posição da resposta de UMA sequência: o mesmo recorte
    de `completion_logprobs`. FP32, por blocos de `chunk` posições (o vocabulário é grande). Devolve tensor [n_total - n_prompt]."""
    import torch

    lg = logits[n_prompt - 1: n_total - 1]
    out = []
    for s in range(0, lg.shape[0], chunk):
        lp = torch.log_softmax(lg[s: s + chunk].float(), dim=-1)
        out.append(-(lp.exp() * lp).sum(-1))
    return torch.cat(out) if out else lg.new_zeros(0).float()


def noclose_share(stats: dict) -> dict[str, float]:
    """Por tipo (`math`, `mcq`), a fração das respostas que terminaram sozinhas sem exatamente um `</think>` (sinal de degeneração
    na amostragem). `stats` é o `rollouts.stats.json`: `whys` com chaves `tipo:motivo`."""
    tot, bad = Counter(), Counter()
    for k, v in stats["whys"].items():
        kind, why = k.split(":", 1)
        tot[kind] += v
        if why == "sem_um_unico_close":
            bad[kind] += v
    return {kind: bad[kind] / tot[kind] for kind in tot}


def ppo_clip_loss_sum(logp_new, logp_old, adv: float, eps_low: float = 0.2, eps_high: float = 0.28):
    """Soma, sobre os tokens da resposta, de -min(r*A, clip(r, 1-eps_low, 1+eps_high)*A). Quem divide pelo total de
    tokens do minibatch é o laço de treino (assim a acumulação de gradiente dá o mesmo gradiente de um batch único)."""
    import torch

    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    return -torch.minimum(unclipped, clipped).sum()


def gspo_loss_sum(logp_new, logp_old, adv: float, eps_low: float = 3e-4, eps_high: float = 4e-4):
    """Variante GSPO (Group Sequence Policy Optimization): a razão de importância e o corte são da SEQUÊNCIA inteira (média
    geométrica das razões dos tokens), não de cada token. Para manter a normalização por tokens do minibatch do laço de treino,
    o termo da sequência é multiplicado pelo número de tokens: o gradiente de cada token fica `-A * s` (com `s` a razão da
    sequência) e é cortado quando `s` sai de [1 - eps_low, 1 + eps_high]. Cortes bem menores que os do PPO por token, porque a
    razão da sequência varia mais que a de um token (valores do paper do GSPO: 3e-4 e 4e-4)."""
    import torch

    n = logp_new.numel()
    s = torch.exp((logp_new - logp_old).mean())
    unclipped = s * adv
    clipped = torch.clamp(s, 1.0 - eps_low, 1.0 + eps_high) * adv
    return -torch.minimum(unclipped, clipped) * n


def seq_clip_stats(logp_new, logp_old, adv: float, eps_low: float, eps_high: float) -> dict:
    """Diagnóstico do corte do GSPO para UMA sequência, igual ao da perda (`gspo_loss_sum`): `s = exp(média dos tokens de logp_novo - logp_antigo)`.
    `outside`: s fora de [1-eps_low, 1+eps_high]. `active`: o ramo cortado vale e o gradiente da sequência é zero, isto é, A > 0 e
    s > 1+eps_high, ou A < 0 e s < 1-eps_low (fora do intervalo no sentido contrário, o gradiente passa)."""
    d = (logp_new.detach() - logp_old.detach()).float()
    n = d.numel()
    log_s = float(d.mean()) if n else 0.0
    s = math.exp(log_s)
    return {"n": n, "log_s": log_s, "outside": bool(s < 1 - eps_low or s > 1 + eps_high),
            "active": bool((adv > 0 and s > 1 + eps_high) or (adv < 0 and s < 1 - eps_low))}


def token_clip_active(logp_new, logp_old, adv: float, eps_low: float, eps_high: float) -> tuple[int, int]:
    """(tokens com o corte do PPO ativo, tokens) de UMA sequência no modo por token: A > 0 com r > 1+eps_high, ou A < 0 com r < 1-eps_low
    (o gradiente do token é zero). Não serve para o modo GSPO, em que o corte é da sequência."""
    import torch

    r = torch.exp(logp_new.detach() - logp_old.detach())
    if adv > 0:
        act = r > 1 + eps_high
    elif adv < 0:
        act = r < 1 - eps_low
    else:
        act = torch.zeros_like(r, dtype=torch.bool)
    return int(act.sum()), r.numel()


def kl_k3_sum(logp, logp_ref):
    """Soma, sobre os tokens da resposta, do estimador k3 da KL(política || referência): exp(r) - r - 1 com r = logp_ref - logp
    (sempre >= 0, zero quando as duas coincidem). O gradiente em `logp` é 1 - exp(r): puxa a política de volta para a referência
    e cresce rápido quando ela se afasta (a razão de probabilidades fica pequena)."""
    import torch

    r = logp_ref - logp
    return (torch.exp(r) - r - 1).sum()
