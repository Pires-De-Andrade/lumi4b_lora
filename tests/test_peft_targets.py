"""Conjuntos de alvos do adaptador (all = 71 módulos do PRD, mlp = só up_proj e down_proj), sem carregar o Lumi."""

import re

import pytest
import torch.nn as nn

from lumi_ft.modeling import peft_setup as P


def fake_lumi() -> nn.Module:
    """Mesma contagem de módulos do Lumi: 21 Mamba (in_proj, out_proj), 17 MLP (up, down) e 4 atenção (q, k, v, o)."""
    layers = nn.ModuleDict()
    for i in range(21):
        layers[f"m{i}"] = nn.ModuleDict({"mixer": nn.ModuleDict({"in_proj": nn.Linear(2, 2), "out_proj": nn.Linear(2, 2)})})
    for i in range(17):
        layers[f"f{i}"] = nn.ModuleDict({"mixer": nn.ModuleDict({"up_proj": nn.Linear(2, 2), "down_proj": nn.Linear(2, 2)})})
    for i in range(4):
        layers[f"a{i}"] = nn.ModuleDict({"mixer": nn.ModuleDict({k: nn.Linear(2, 2) for k in ("q_proj", "k_proj", "v_proj", "o_proj")})})
    return nn.ModuleDict({"backbone": nn.ModuleDict({"layers": layers})})


def test_all_targets_unchanged():
    m = fake_lumi()
    assert len(P.target_names(m)) == 71 and P.count_by_type(P.target_names(m)) == P.EXPECTED_TARGETS
    P.check_targets(m)
    assert P.lora_config().target_modules == P.TARGET_REGEX


def test_mlp_targets_are_only_up_and_down():
    m = fake_lumi()
    names = P.target_names(m, "mlp")
    assert len(names) == 34 and P.count_by_type(names) == {"up_proj": 17, "down_proj": 17}
    P.check_targets(m, "mlp")
    rx = re.compile(P.lora_config(targets="mlp").target_modules)
    assert rx.fullmatch("backbone.layers.3.mixer.up_proj") and rx.fullmatch("backbone.layers.3.mixer.down_proj")
    assert not any(rx.fullmatch(f"backbone.layers.3.mixer.{k}") for k in ("in_proj", "out_proj", "q_proj", "o_proj"))


def test_wrong_counts_fail():
    m = fake_lumi()
    del m["backbone"]["layers"]["f0"]
    with pytest.raises(AssertionError):
        P.check_targets(m, "mlp")
    with pytest.raises(AssertionError):
        P.check_targets(m)


def test_train_config_and_cli_carry_targets():
    from lumi_ft.train import sft

    assert sft.TrainConfig().targets == "all" and sft.TrainConfig(targets="mlp").targets == "mlp"


# ------------------------------------------------------------- estudo de capacidade: tipos, faixa de camadas, rank e lr
PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"  # padrão real do Lumi: M = Mamba, - = MLP, * = atenção


def numbered_lumi() -> nn.Module:
    """Os nomes reais (`backbone.layers.<i>.mixer.<módulo>`), um bloco por camada, no padrão do Lumi."""
    assert len(PATTERN) == 42 and PATTERN.count("M") == 21 and PATTERN.count("-") == 17 and PATTERN.count("*") == 4
    layers = nn.ModuleDict()
    for i, c in enumerate(PATTERN):
        names = {"M": ("in_proj", "out_proj"), "-": ("up_proj", "down_proj"), "*": ("q_proj", "k_proj", "v_proj", "o_proj")}[c]
        layers[str(i)] = nn.ModuleDict({"mixer": nn.ModuleDict({k: nn.Linear(8, 8) for k in names})})
    return nn.ModuleDict({"backbone": nn.ModuleDict({"layers": layers})})


@pytest.mark.parametrize("spec,n,types", [
    ("all", 71, {"in_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"}),
    ("mlp", 34, {"up_proj", "down_proj"}),
    ("mamba", 21, {"in_proj"}),
    ("attn", 16, {"q_proj", "k_proj", "v_proj", "o_proj"}),
    ("mamba,attn", 37, {"in_proj", "q_proj", "k_proj", "v_proj", "o_proj"}),
    ("attn,mlp", 50, {"up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"}),
    ("mlp,mamba", 55, {"in_proj", "up_proj", "down_proj"}),
])
def test_block_type_subsets(spec, n, types):
    m = numbered_lumi()
    names = P.target_names(m, spec)
    assert len(names) == n and set(P.count_by_type(names)) == types
    P.check_targets(m, spec)
    assert not any("out_proj" in x for x in names)


def test_all_regex_is_the_prd_regex_and_order_does_not_matter():
    assert P.target_regex("all") == P.TARGET_REGEX
    assert P.target_regex("mamba,mlp,attn") == P.TARGET_REGEX
    assert P.parse_targets("attn,mamba") == ("mamba", "attn")


@pytest.mark.parametrize("bad", ["", "mlp,foo", "out_proj", "mlpp"])
def test_unknown_block_type_fails(bad):
    with pytest.raises(AssertionError):
        P.parse_targets(bad)


@pytest.mark.parametrize("rng,n,mamba,mlp,attn", [("0-20", 36, 10, 18, 8), ("21-41", 35, 11, 16, 8), ("0-41", 71, 21, 34, 16)])
def test_layer_ranges_split_the_71_modules(rng, n, mamba, mlp, attn):
    m = numbered_lumi()
    names = P.target_names(m, "all", rng)
    c = P.count_by_type(names)
    assert len(names) == n and c.get("in_proj", 0) == mamba and c.get("up_proj", 0) + c.get("down_proj", 0) == mlp
    assert sum(c.get(k, 0) for k in ("q_proj", "k_proj", "v_proj", "o_proj")) == attn
    P.check_targets(m, "all", rng)
    lo, hi = map(int, rng.split("-"))
    assert all(lo <= P.layer_index(x) <= hi for x in names)


def test_halves_are_disjoint_and_cover_everything():
    m = numbered_lumi()
    a, b = set(P.target_names(m, "all", "0-20")), set(P.target_names(m, "all", "21-41"))
    assert not a & b and a | b == set(P.target_names(m, "all"))


def test_layers_combined_with_type():
    m = numbered_lumi()
    names = P.target_names(m, "mlp", "21-41")
    assert len(names) == 16 and all(P.layer_index(x) >= 21 for x in names)
    P.check_targets(m, "mlp", "21-41")


@pytest.mark.parametrize("bad", ["20-0", "a-b", "5", "0-", "-5"])
def test_bad_layer_range_fails(bad):
    with pytest.raises(AssertionError):
        P.parse_layers(bad)


def test_range_without_modules_fails():
    with pytest.raises(AssertionError):
        P.check_targets(numbered_lumi(), "all", "50-60")


def test_alpha_pattern_keeps_scale_per_module_unless_rslora():
    # base r=16, alpha=32 (escala 2): in_proj com r=4 -> alpha 8 (escala 2); MLP com r=32 -> alpha 64 (escala 2)
    ap = P.resolve_alpha_pattern(16, None, {"in_proj": 4, "up_proj|down_proj": 32}, None, rslora=False)
    assert ap == {"in_proj": 8.0, "up_proj|down_proj": 64.0}
    assert P.resolve_alpha_pattern(16, 32, {"in_proj": 4}, None, rslora=True) == {}
    assert P.resolve_alpha_pattern(16, None, None, None, rslora=False) == {}
    assert P.resolve_alpha_pattern(16, None, {"in_proj": 4}, {"in_proj": 99}, rslora=False) == {"in_proj": 99}


def test_rank_alpha_and_rslora_reach_the_peft_modules():
    """Injeta o adaptador num modelo de brinquedo (sem carregar o Lumi) e confere rank, alpha e escala de verdade."""
    import math

    from peft import inject_adapter_in_model

    def build(**kw):
        m = numbered_lumi()
        cfg = P.lora_config(r=16, **kw)
        inject_adapter_in_model(cfg, m)
        return m

    def mod(m, i, k):
        return m.backbone.layers[str(i)].mixer[k]

    m = build()  # padrão do PRD: r=16, alpha=32, escala 2
    assert mod(m, 0, "in_proj").lora_A["default"].weight.shape[0] == 16 and mod(m, 0, "in_proj").scaling["default"] == 2.0
    m = build(rslora=True, alpha=8)  # alpha fixo 8, escala 8/raiz(16) = 2: o mesmo ponto do PRD
    assert math.isclose(mod(m, 0, "in_proj").scaling["default"], 2.0)
    cfg = P.lora_config(r=64, rslora=True, alpha=8)  # rank 64 com rslora e alpha 8: escala 8/raiz(64) = 1
    m = numbered_lumi()
    inject_adapter_in_model(cfg, m)
    assert math.isclose(mod(m, 0, "in_proj").scaling["default"], 1.0) and mod(m, 0, "in_proj").lora_A["default"].weight.shape[0] == 64
    rp = {"in_proj": 4, "up_proj|down_proj": 32}
    m = numbered_lumi()
    inject_adapter_in_model(P.lora_config(r=16, rank_pattern=rp, alpha_pattern=P.resolve_alpha_pattern(16, None, rp, None, False)), m)
    assert mod(m, 0, "in_proj").lora_A["default"].weight.shape[0] == 4 and mod(m, 0, "in_proj").scaling["default"] == 2.0
    assert mod(m, 1, "up_proj").lora_A["default"].weight.shape[0] == 32 and mod(m, 1, "down_proj").scaling["default"] == 2.0
    assert mod(m, 12, "q_proj").lora_A["default"].weight.shape[0] == 16  # sem regra: rank base


def test_lr_multiplier_first_match_wins_and_default_is_one():
    from lumi_ft.train.sft import lr_multiplier, param_groups

    mult = {"in_proj": 0.2, "lora_B": 4.0}
    assert lr_multiplier("x.layers.0.mixer.in_proj.lora_B.default.weight", mult) == 0.2  # primeiro que casa
    assert lr_multiplier("x.layers.1.mixer.up_proj.lora_B.default.weight", mult) == 4.0
    assert lr_multiplier("x.layers.1.mixer.up_proj.lora_A.default.weight", mult) == 1.0
    assert lr_multiplier("qualquer", {}) == 1.0
    params = {"a.in_proj.lora_A": 1, "a.up_proj.lora_A": 2, "b.in_proj.lora_B": 3, "b.up_proj.lora_B": 4}
    g = param_groups(params, 1e-4, mult)
    assert [(x["mult"], x["params"]) for x in g] == [(0.2, [1, 3]), (1.0, [2]), (4.0, [4])]
    assert [round(x["lr"], 10) for x in g] == [2e-5, 1e-4, 4e-4]
    assert len(param_groups(params, 1e-4, {})) == 1  # sem multiplicadores: um grupo só, como antes


def test_train_config_new_fields_default_to_the_prd():
    from lumi_ft.train.sft import TrainConfig

    c = TrainConfig()
    assert (c.alpha, c.rslora, c.layers, c.rank_pattern, c.alpha_pattern, c.lr_mult) == (None, False, "", "", "", "")
