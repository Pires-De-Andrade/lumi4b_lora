"""Alvos e configuração do adaptador (LoRA ou DoRA), conforme o PRD, mais as variações do estudo de capacidade.

Variações (todas com padrão igual ao do PRD, então as rodadas antigas não mudam):
- `targets`: "all" (os 71 módulos do PRD) ou uma lista de tipos de bloco separados por vírgula: `mamba` (in_proj),
  `mlp` (up_proj e down_proj) e `attn` (q, k, v e o). Ex.: "mlp", "mamba,attn".
- `layers`: faixa fechada de camadas, "0-20" (vazio = todas).
- `rslora`: escala alpha/raiz(r) em vez de alpha/r.
- `rank_pattern`, `alpha_pattern`: rank e alpha por tipo de módulo (chave = regex do final do nome, como no PEFT).
"""
import re

import torch
from peft import LoraConfig, get_peft_model

from lumi_ft.modeling import peft_patch

TARGET_REGEX = r".*\.mixer\.(in_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)"
EXPECTED_TARGETS = {"in_proj": 21, "up_proj": 17, "down_proj": 17,
                    "q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4}  # 71 no total

# Tipos de bloco: módulos com adaptador e contagem esperada no Lumi (42 camadas: 21 Mamba, 17 MLP, 4 atenção).
BLOCKS = {
    "mamba": ("in_proj",),
    "mlp": ("up_proj", "down_proj"),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
}
FORBIDDEN = ("out_proj", "conv1d", "embed", "lm_head")


def parse_targets(spec: str) -> tuple[str, ...]:
    """'all' -> os três tipos; 'mlp,attn' -> ('mlp', 'attn') na ordem canônica. Falha em nome desconhecido."""
    if spec == "all":
        return tuple(BLOCKS)
    asked = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [s for s in asked if s not in BLOCKS]
    assert asked and not bad, f"tipo de bloco desconhecido em {spec!r}: {bad} (válidos: all, {', '.join(BLOCKS)})"
    return tuple(b for b in BLOCKS if b in asked)


def parse_layers(spec: str) -> tuple[int, int] | None:
    """'' -> None (todas); '0-20' -> (0, 20), faixa fechada."""
    if not spec:
        return None
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    assert m and int(m.group(1)) <= int(m.group(2)), f"faixa de camadas inválida: {spec!r} (esperado 'A-B', A <= B)"
    return int(m.group(1)), int(m.group(2))


def module_types(targets: str) -> tuple[str, ...]:
    return tuple(m for b in parse_targets(targets) for m in BLOCKS[b])


def target_regex(targets: str = "all", layers: str = "") -> str:
    """Regex (casamento total) dos módulos-alvo. Sem `layers` e com todos os tipos é exatamente o do PRD."""
    types = module_types(targets)
    rng = parse_layers(layers)
    alt = "|".join(types)
    if rng is None:
        return TARGET_REGEX if set(types) == set(EXPECTED_TARGETS) else rf".*\.mixer\.({alt})"
    idx = "|".join(str(i) for i in range(rng[0], rng[1] + 1))
    return rf".*\.layers\.(?:{idx})\.mixer\.({alt})"


def expected_counts(targets: str = "all") -> dict[str, int]:
    """Contagem esperada por tipo de módulo no Lumi inteiro (sem faixa de camadas)."""
    return {k: EXPECTED_TARGETS[k] for k in module_types(targets)}


def lora_config(r: int = 16, use_dora: bool = False, alpha: float | None = None, targets: str = "all", layers: str = "",
                rslora: bool = False, rank_pattern: dict | None = None, alpha_pattern: dict | None = None) -> LoraConfig:
    return LoraConfig(
        r=r, lora_alpha=alpha if alpha is not None else 2 * r, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=target_regex(targets, layers), use_dora=use_dora,
        use_rslora=rslora, rank_pattern=rank_pattern or {}, alpha_pattern=alpha_pattern or {},
    )


def resolve_alpha_pattern(r: int, alpha: float | None, rank_pattern: dict | None, alpha_pattern: dict | None,
                          rslora: bool) -> dict:
    """Se há rank por tipo e o alpha por tipo não foi dado, mantém a escala alpha/r do módulo igual à do base
    (alpha = (alpha_base / r_base) * r_tipo), para que a diferença entre tipos seja só de capacidade. Com RSLoRA o
    alpha fica o mesmo em todos (é a convenção do método) e nada é derivado."""
    if alpha_pattern:
        return dict(alpha_pattern)
    if rank_pattern and not rslora:
        ratio = (alpha if alpha is not None else 2 * r) / r
        return {k: ratio * rk for k, rk in rank_pattern.items()}
    return {}


def target_names(model, targets: str = "all", layers: str = "") -> list[str]:
    rx = re.compile(target_regex(targets, layers))
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and rx.fullmatch(n)]


def count_by_type(names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in names:
        k = n.rsplit(".", 1)[-1]
        out[k] = out.get(k, 0) + 1
    return out


def layer_index(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def check_targets(model, targets: str = "all", layers: str = "") -> None:
    """Falha se o regex não casar exatamente com os módulos esperados, ou se casar com os proibidos.

    Sem faixa de camadas, a contagem esperada é a fixa do Lumi (71 no total). Com faixa, a contagem esperada vem de uma
    varredura independente do regex: todo `Linear` em `.mixer.` de um dos tipos, com índice de camada dentro da faixa."""
    names = target_names(model, targets, layers)
    rng = parse_layers(layers)
    if rng is None:
        want = expected_counts(targets)
    else:
        types = set(module_types(targets))
        want = {}
        for n, m in model.named_modules():
            i = layer_index(n)
            if isinstance(m, torch.nn.Linear) and ".mixer." in n and n.rsplit(".", 1)[-1] in types and i is not None \
                    and rng[0] <= i <= rng[1]:
                k = n.rsplit(".", 1)[-1]
                want[k] = want.get(k, 0) + 1
        assert want, f"nenhum módulo na faixa {layers!r}"
    got = count_by_type(names)
    assert got == want, (got, want)
    for n in names:
        assert not any(x in n for x in FORBIDDEN), n


def add_adapter(model, r: int = 16, use_dora: bool = False, targets: str = "all", alpha: float | None = None,
                layers: str = "", rslora: bool = False, rank_pattern: dict | None = None,
                alpha_pattern: dict | None = None):
    check_targets(model, targets, layers)
    if use_dora:
        peft_patch.apply()  # linhas zeradas da in_proj: ver peft_patch.py
    ap = resolve_alpha_pattern(r, alpha, rank_pattern, alpha_pattern, rslora)
    return get_peft_model(model, lora_config(r=r, use_dora=use_dora, alpha=alpha, targets=targets, layers=layers,
                                             rslora=rslora, rank_pattern=rank_pattern, alpha_pattern=ap))
