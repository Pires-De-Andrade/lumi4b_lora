"""Receita oficial do Antidoom (FTPO), portada para o pipeline do Lumi.

Derivado de https://github.com/Liquid4All/antidoom (commit bd6a126, Apache-2.0; licença e lista de mudanças em third_party/antidoom/).
Cada função indica o arquivo de origem. A lógica foi mantida; as adaptações estão em third_party/antidoom/NOTICE.md:
  - detector de repetição em caracteres, em qualquer trecho da resposta (`find_inner_repetition`, de repetition.py);
  - token rejeitado = primeiro token legível da primeira repetição, com ajuste por período em tokens (`select_rejected_token_index_token_aware`);
  - escolhidos = alternativas do topo renormalizadas na temperatura da geração, com corte min-p relativo ao maior (`select_chosen_tokens`);
  - regularização de frequência dos rejeitados (`sample_rows`) e dos escolhidos (`regularise_chosen`), de ftpo_data.py;
  - perda FTPO por evento (`ftpo_loss_oficial`), a mesma conta de reference_ftpo_trainer.py (`compute_loss`).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- tokens.py

def _build_u2b_full() -> dict[int, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {ord(chr(c)): b for b, c in zip(bs, cs)}
    for b in range(256):
        u2b[0x2500 + b] = b
    return u2b


_U2B = _build_u2b_full()


def fix_mojibake(text: str) -> str:
    buf = bytearray()
    changed = False
    for ch in text:
        cp = ord(ch)
        if cp in _U2B:
            buf.append(_U2B[cp])
            changed = True
        else:
            buf.extend(ch.encode("utf-8"))
    if not changed:
        return text
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return text


def decode_token(token: str) -> str:
    if not token:
        return token
    token = fix_mojibake(token)
    return token.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")


@dataclass
class TokenState:
    """Texto da resposta montado token a token (tokens.py, sem as log-probabilidades)."""
    token_strings: list[str] = field(default_factory=list)
    decoded_lens: list[int] = field(default_factory=list)
    text: str = ""

    def append(self, token_strings: list[str]) -> None:
        for raw in token_strings:
            decoded = decode_token(raw)
            self.token_strings.append(raw)
            self.decoded_lens.append(len(decoded))
            self.text += decoded

    def char_to_token_index(self, char_pos: int) -> int | None:
        if char_pos < 0 or char_pos >= len(self.text):
            return None
        running = 0
        for idx, length in enumerate(self.decoded_lens):
            running += length
            if char_pos < running:
                return idx
        return None


# ---------------------------------------------------------------- repetition.py

@dataclass(frozen=True)
class RepeatHit:
    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self) -> int:
        return self.start + self.period


def _verify_repetition_at(text: str, start_pos: int, period: int, min_repeats: int, min_total_repeated: int):
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return False, None
    pattern = text[start_pos:start_pos + period]
    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos:pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos
    pos = start_pos - period
    while pos >= 0 and text[pos:pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period
    total = reps * period
    if reps >= min_repeats and total >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return True, RepeatHit(start_pos, end_pos, period, reps, snippet)
    return False, None


def find_inner_repetition(text: str, *, min_repeats: int = 4, max_period: int = 1024, min_period: int = 1, min_total_repeated: int = 60,
                          sample_len: int = 16, sample_interval: int = 128):
    if not text or len(text) < min_total_repeated:
        return False, None
    n = len(text)
    for sample_pos in range(0, n - sample_len, sample_interval):
        fingerprint = text[sample_pos:sample_pos + sample_len]
        other_pos = text.find(fingerprint, sample_pos + sample_len)
        if other_pos != -1:
            candidate_period = other_pos - sample_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(text, sample_pos, candidate_period, min_repeats, min_total_repeated)
                if found:
                    return True, hit
        other_pos = text.rfind(fingerprint, 0, sample_pos)
        if other_pos != -1:
            candidate_period = sample_pos - other_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(text, other_pos, candidate_period, min_repeats, min_total_repeated)
                if found:
                    return True, hit
    return False, None


# ---------------------------------------------------------------- generate.py (escolha do rejeitado)

_RESTART_WORDS = {
    "actually", "after", "also", "alternatively", "because", "but", "finally", "first", "given", "hmm", "however", "in", "let", "looking",
    "maybe", "now", "okay", "perhaps", "second", "since", "so", "the", "then", "therefore", "this", "thus", "wait",
}
# adaptação: equivalentes em português (os traços do Lumi treinado são em português)
_RESTART_WORDS_PT = {
    "na", "também", "alternativamente", "porque", "mas", "finalmente", "primeiro", "dado", "hum", "hmm", "porém", "contudo", "no", "em",
    "vamos", "olhando", "talvez", "agora", "ok", "certo", "segundo", "como", "então", "o", "a", "depois", "portanto", "isso", "assim",
    "espere", "logo", "entretanto", "aliás",
}
RESTART_WORDS = _RESTART_WORDS | _RESTART_WORDS_PT


def select_rejected_token_index(state: TokenState, repeat_start: int, repeat_end: int) -> int | None:
    reject_char = repeat_start
    while reject_char < repeat_end and state.text[reject_char].isspace():
        reject_char += 1
    if reject_char >= repeat_end:
        return None
    reject_idx = state.char_to_token_index(reject_char)
    while reject_idx is not None and reject_idx < len(state.token_strings):
        if decode_token(state.token_strings[reject_idx]).strip():
            return reject_idx
        reject_idx += 1
    return None


def _token_key_sequence(token_ids: list[int], token_strings: list[str]) -> list:
    return token_ids if len(token_ids) == len(token_strings) else token_strings


def _is_boundary_only(text: str) -> bool:
    return bool(text) and not any(ch.isalnum() or ch == "_" for ch in text)


def _skip_boundary_only_tokens(token_strings: list[str], token_idx: int, *, max_tokens: int = 4, max_chars: int = 12) -> int:
    idx = token_idx
    chars = 0
    while idx < len(token_strings) and idx - token_idx < max_tokens:
        text = decode_token(token_strings[idx])
        chars += len(text)
        if chars > max_chars or not _is_boundary_only(text):
            break
        idx += 1
    return idx


def _normalised_word(text: str) -> str:
    return re.sub(r"^[^\w_]+|[^\w_]+$", "", text.strip().lower())


def _lstrip_token_key_sequence(token_strings: list[str]) -> list[str]:
    return [decode_token(raw).lstrip() for raw in token_strings]


def _slide_token_period_left(token_keys: list, seed_idx: int, period_tokens: int) -> int:
    idx = seed_idx
    while idx > 0:
        prev_start = idx - 1
        prev_repeat = prev_start + period_tokens
        if prev_repeat >= len(token_keys):
            break
        if token_keys[prev_start] != token_keys[prev_repeat]:
            break
        idx = prev_start
    return idx


def select_rejected_token_index_token_aware(state: TokenState, seed_start: int, repeat_start: int, repeat_end: int,
                                            token_ids: list[int]) -> int | None:
    seed_idx = state.char_to_token_index(seed_start)
    repeat_idx = state.char_to_token_index(repeat_start)
    if seed_idx is None or repeat_idx is None or repeat_idx <= seed_idx:
        return select_rejected_token_index(state, repeat_start, repeat_end)
    token_keys = _lstrip_token_key_sequence(state.token_strings)
    exact_token_keys = _token_key_sequence(token_ids, state.token_strings)
    period_tokens = repeat_idx - seed_idx
    seed_idx = _slide_token_period_left(token_keys, seed_idx, period_tokens)
    target_idx = seed_idx + period_tokens
    if target_idx >= len(state.token_strings):
        return None
    target_idx = _skip_boundary_only_tokens(state.token_strings, target_idx)
    if target_idx >= len(state.token_strings):
        return None
    word = _normalised_word(decode_token(state.token_strings[target_idx]))
    if word and any(ch.isalnum() or ch == "_" for ch in word):
        punct_start = target_idx + 1
        punct_end = _skip_boundary_only_tokens(state.token_strings, punct_start, max_tokens=3, max_chars=8)
        punct_text = "".join(decode_token(tok) for tok in state.token_strings[punct_start:punct_end])
        next_word = _normalised_word(decode_token(state.token_strings[punct_end])) if punct_end < len(state.token_strings) else ""
        seed_prev = seed_idx - 1
        repeat_prev = seed_idx + period_tokens - 1
        if (punct_end > punct_start and any(ch in punct_text for ch in ".!?") and seed_prev >= 0 and repeat_prev >= 0
                and repeat_prev < len(exact_token_keys) and exact_token_keys[seed_prev] != exact_token_keys[repeat_prev]
                and next_word in RESTART_WORDS):
            target_idx = punct_end
    while target_idx < len(state.token_strings):
        if decode_token(state.token_strings[target_idx]).strip():
            return target_idx
        target_idx += 1
    return None


DETECTOR_DEFAULTS = {"min_repeats": 4, "max_period": 1024, "min_period": 1, "min_total_repeated": 60, "sample_len": 16, "sample_interval": 128}


def find_loop_event(token_ids: list[int], token_strings: list[str], **detector) -> dict | None:
    """Detector oficial sobre uma resposta (ids e strings de token, sem o prompt). Devolve o índice do token rejeitado e os dados da repetição."""
    state = TokenState()
    state.append(token_strings)
    found, hit = find_inner_repetition(state.text, **{**DETECTOR_DEFAULTS, **detector})
    if not found or hit is None:
        return None
    idx = select_rejected_token_index_token_aware(state, hit.start, hit.repeat_start, hit.end, token_ids)
    if idx is None:
        return None
    return {"reject_idx": idx, "start_char": hit.start, "repeat_start_char": hit.repeat_start, "end_char": hit.end, "period": hit.period,
            "repeats": hit.repeats, "snippet": hit.snippet}


# ---------------------------------------------------------------- sampling.py (escolhidos)

def select_chosen_tokens(logprob_items: list[tuple[int, str, float]], *, rejected_id: int, rejected_token: str, temperature: float = 0.1,
                         min_p: float | None = 0.01, top_k: int | None = 50, max_tokens: int = 20, min_decoded_chars: int = 1,
                         require_alnum: bool = False) -> list[int]:
    """`logprob_items` = (id, string do token, log-probabilidade) das maiores alternativas no ponto do rejeitado. Devolve os ids escolhidos,
    na ordem do original (da menor para a maior probabilidade depois da renormalização)."""
    temp = max(float(temperature), 1e-6)
    vals = [(tid, tok, math.exp(lp) ** (1.0 / temp)) for tid, tok, lp in logprob_items if lp is not None]
    total = sum(v for _, _, v in vals)
    if total <= 0:
        return []
    pairs = [(tid, tok, v / total) for tid, tok, v in vals]
    rejected_decoded = decode_token(rejected_token).lower()
    pairs = [(tid, tok, p) for tid, tok, p in pairs if not (tid == rejected_id or tok == rejected_token or decode_token(tok).lower() == rejected_decoded)]
    if not pairs:
        return []
    if min_p is not None:
        floor = min_p * max(p for _, _, p in pairs)
        pairs = [x for x in pairs if x[2] >= floor]
    if top_k is not None and len(pairs) > top_k:
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_k]
    pairs.sort(key=lambda x: x[2])
    out: list[int] = []
    for tid, tok, _ in pairs:
        surf = decode_token(tok)
        if len(surf.strip()) < min_decoded_chars:
            continue
        if require_alnum and not any(c.isalnum() for c in surf):
            continue
        out.append(tid)
        if len(out) >= max_tokens:
            break
    return out


# ---------------------------------------------------------------- ftpo_data.py (regularização de frequência)

CHOSEN_REGULARISATION_MIN_REF_COUNT = 50
CHOSEN_REGULARISATION_REF_PERCENTILE = 95


def normalise_chosen_key(text: str) -> str:
    stripped = text.strip().lower()
    word = re.sub(r"^[^\w]+|[^\w]+$", "", stripped, flags=re.UNICODE)
    return word or stripped or text


def _regularised_ratios(counts: Counter, strength: float) -> dict:
    if not counts:
        return {}
    median = float(np.median(list(counts.values())))
    weights = {tok: 1.0 if count <= median or strength <= 0 else (median / count) ** strength for tok, count in counts.items()}
    total = sum(weights[tok] * count for tok, count in counts.items())
    return {tok: (weights[tok] * count) / total for tok, count in counts.items()}


def sample_rows(rows: list[dict[str, Any]], *, max_train_examples: int | None, rejected_strength: float, seed: int,
                rejected_key: str = "rejected_decoded", source_key: str = "kind") -> list[dict[str, Any]]:
    """`_rejected_normalisation` do original (sem o balanceamento por fonte, desligado na configuração padrão). Com menos linhas que
    `max_train_examples`, todas entram (o corte só age acima do teto)."""
    if not rows:
        return []
    target_n = min(max_train_examples or len(rows), len(rows))
    ratios = _regularised_ratios(Counter(r[rejected_key] for r in rows), rejected_strength)
    targets = {tok: int(round(ratio * target_n)) for tok, ratio in ratios.items()}
    pool = Counter(r.get(source_key, "__unknown__") for r in rows)
    pool_total = sum(pool.values()) or 1
    share = {s: c / pool_total for s, c in pool.items()}
    buckets: defaultdict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[row[rejected_key]][row.get(source_key, "__unknown__")].append(row)
    rng = np.random.default_rng(seed)
    for tok in buckets:
        for source in buckets[tok]:
            rng.shuffle(buckets[tok][source])
    selected: list[dict[str, Any]] = []
    seen_tok: defaultdict = defaultdict(int)
    selected_src: Counter = Counter()

    def best_choice(respect_cap: bool):
        denom = max(len(selected), 1)
        best, best_key = None, None
        for tok, src_buckets in buckets.items():
            if respect_cap and seen_tok[tok] >= targets.get(tok, 0):
                continue
            tok_gap = ratios.get(tok, 0.0) - seen_tok[tok] / denom
            for source, available in src_buckets.items():
                if not available:
                    continue
                key = (tok_gap, share.get(source, 0.0) - selected_src[source] / denom)
                if best_key is None or key > best_key:
                    best_key, best = key, (tok, source)
        return best

    for respect in (True, False):
        while len(selected) < target_n:
            choice = best_choice(respect_cap=respect)
            if choice is None:
                break
            tok, source = choice
            selected.append(buckets[tok][source].pop())
            seen_tok[tok] += 1
            selected_src[source] += 1
    rng.shuffle(selected)
    return selected


def _count_percentile(counts: Counter, percentile: int) -> int:
    if not counts:
        return 1
    values = sorted(counts.values())
    return max(1, values[int((len(values) - 1) * percentile / 100)])


def _regularised_chosen_targets(counts: Counter, strength: float) -> dict:
    ref_count = max(CHOSEN_REGULARISATION_MIN_REF_COUNT, _count_percentile(counts, CHOSEN_REGULARISATION_REF_PERCENTILE))
    return {key: (max(1, int(round(count * (ref_count / count) ** strength))) if strength > 0 and count > ref_count else count)
            for key, count in counts.items()}


def regularise_chosen(examples: list[dict[str, Any]], strength: float, filter_counts: Counter | None = None) -> list[dict[str, Any]]:
    """`_regularise_chosen_examples` do original. Cada exemplo tem `chosen_ids` e `chosen_keys` (chave normalizada de cada escolhido); devolve
    os exemplos com escolhidos podados (e sem os que ficaram vazios), mantendo os outros campos."""
    filter_counts = filter_counts if filter_counts is not None else Counter()
    deduped: list[dict[str, Any]] = []
    key_counts: Counter = Counter()
    for ex in examples:
        seen: set = set()
        ids, keys = [], []
        for tid, key in zip(ex["chosen_ids"], ex["chosen_keys"], strict=True):
            if key in seen:
                filter_counts["chosen_regularised"] += 1
                continue
            seen.add(key)
            ids.append(tid)
            keys.append(key)
        deduped.append({**ex, "chosen_ids": ids, "chosen_keys": keys})
        key_counts.update(keys)
    if strength <= 0:
        return [ex for ex in deduped if ex["chosen_ids"]]
    targets = _regularised_chosen_targets(key_counts, strength)
    current = Counter(key_counts)
    by_key: defaultdict = defaultdict(set)
    for i, ex in enumerate(deduped):
        for key in ex["chosen_keys"]:
            by_key[key].add(i)
    for key in sorted(targets, key=lambda k: current[k] - targets[k], reverse=True):
        while current[key] > targets[key]:
            cands = [i for i in by_key[key] if key in deduped[i]["chosen_keys"] and len(deduped[i]["chosen_keys"]) > 1]
            if not cands:
                break
            i = max(cands, key=lambda j: (len(deduped[j]["chosen_keys"]), -j))
            at = deduped[i]["chosen_keys"].index(key)
            del deduped[i]["chosen_keys"][at]
            del deduped[i]["chosen_ids"][at]
            current[key] -= 1
            filter_counts["chosen_regularised"] += 1
    for key in sorted(targets, key=lambda k: current[k] - targets[k], reverse=True):
        while current[key] > targets[key]:
            cands = [i for i in by_key[key] if deduped[i]["chosen_keys"] == [key]]
            if not cands:
                break
            i = min(cands)
            deduped[i]["chosen_keys"] = []
            deduped[i]["chosen_ids"] = []
            current[key] -= 1
            filter_counts["chosen_regularised"] += 1
            filter_counts["no_chosen_after_chosen_regularisation"] += 1
    return [ex for ex in deduped if ex["chosen_ids"]]


# ---------------------------------------------------------------- reference_ftpo_trainer.py (perda)

def ftpo_loss_oficial(z: torch.Tensor, z_ref: torch.Tensor, chosen: list[int] | torch.Tensor, rejected: int, *, eps: float = 2.0,
                      lambda_mse: float = 0.4, lambda_mse_target: float = 0.05, tau_mse_target: float = 0.5):
    """Perda FTPO de um evento (logits do último token, 1-D). Mesma conta do `compute_loss` oficial com lote de 1:
    preferência = média sobre os escolhidos de softplus(eps - Δ) · clamp((eps - Δ)/eps, 0, 1), com Δ = z[c] - z[rejeitado] (desliga quando o
    escolhido passa o rejeitado por `eps` logits); amarra = lambda_mse · média de (z - z_ref)² no resto do vocabulário + lambda_mse_target ·
    média de relu(|z - z_ref| - tau)² nos escolhidos e no rejeitado. `chosen_win` = fração dos escolhidos acima do rejeitado."""
    ch = torch.as_tensor(chosen, device=z.device, dtype=torch.long)
    delta = z[ch] - z[rejected]
    weights = torch.clamp((eps - delta) / eps, 0.0, 1.0)
    pref = (F.softplus(eps - delta) * weights).sum() / max(len(ch), 1)
    diff = z - z_ref
    tgt = torch.zeros_like(z, dtype=torch.bool)
    tgt[ch] = True
    tgt[rejected] = True
    mse_elem = (diff.pow(2) * (~tgt)).sum() / (~tgt).sum()
    excess = torch.clamp(diff.abs() - tau_mse_target, min=0.0) * tgt
    mse_tgt = excess.pow(2).sum() / tgt.sum() if lambda_mse_target else z.new_tensor(0.0)
    loss = pref + lambda_mse * mse_elem + lambda_mse_target * mse_tgt
    with torch.no_grad():
        metrics = {"loss": float(loss), "pref_loss": float(pref), "mse_elem": float(mse_elem), "mse_tgt": float(mse_tgt),
                   "chosen_win": float((delta > 0).float().mean()), "margin_win": float((delta >= eps).float().mean()),
                   "mean_delta": float(delta.mean()), "active_weight": float(weights.mean()),
                   "rejected_is_argmax": float(int(z.argmax()) == int(rejected))}
    return loss, metrics
