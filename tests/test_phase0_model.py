"""Fase 0: provas sobre o modelo, o adaptador e a geração. Rodam na GPU."""
import re

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from conftest import IMPL, assert_within_noise, equivalence_stats
from lumi_ft.modeling.load import EOS_IDS, MODEL_ID, REVISION, is_fast_path
from lumi_ft.modeling.peft_setup import (EXPECTED_TARGETS, TARGET_REGEX, add_adapter,
                                         check_targets, count_by_type, target_names)

ARMS = [pytest.param(False, id="lora"), pytest.param(True, id="dora")]
ARMS_MERGE = ARMS


@torch.no_grad()
def logits_of(model, batch):
    model.eval()
    return model(**batch).logits.float()


def test_target_regex_on_meta_device():
    remote = IMPL == "remote"
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=remote)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=remote, dtype=torch.bfloat16)
    check_targets(model)
    names = target_names(model)
    assert len(names) == 71 and count_by_type(names) == EXPECTED_TARGETS
    assert re.compile(TARGET_REGEX).fullmatch("backbone.layers.0.mixer.out_proj") is None


def test_fast_kernels_are_used(fresh_model, batch):
    """Prova pelos kernels que aparecem no profiler: o caminho PyTorch puro não os usa."""
    assert type(fresh_model).__name__ == "NemotronHForCausalLM"
    flag = is_fast_path(fresh_model)  # None na implementação nativa, que não tem a flag
    assert flag in (True, None)
    fresh_model.eval()
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fresh_model(**batch)
        torch.cuda.synchronize()
    names = [e.key.lower() for e in prof.key_averages()]
    scan = [n for n in names if "chunk_scan" in n or "state_passing" in n]
    conv = [n for n in names if "causal_conv1d" in n]
    print(f"\n[kernels] impl={IMPL} flag={flag} | scan={len(scan)} ex={scan[:1]} | conv1d={len(conv)} ex={conv[:1]}")
    assert scan and conv, "kernels do mamba-ssm / causal-conv1d não foram executados (caminho lento?)"


@pytest.mark.parametrize("use_dora", ARMS)
def test_fresh_adapter_does_not_change_logits(fresh_model, big_batch, noise_floor, use_dora):
    base = logits_of(fresh_model, big_batch)
    peft_model = add_adapter(fresh_model, r=16, use_dora=use_dora)
    new = logits_of(peft_model, big_batch)
    stats = equivalence_stats(base, new, big_batch["attention_mask"].bool())
    print(f"\n[fresh adapter dora={use_dora}] {stats}")
    if use_dora:  # DoRA recalcula a norma em BF16: aceitar até 2x o ruído natural do modelo
        assert_within_noise(stats, noise_floor, "DoRA recém-criado")
    else:  # B = 0 soma zero: nada pode mudar
        assert (new - base).abs().max().item() == 0.0
        assert stats["top1_disagree"] == 0.0


@pytest.mark.parametrize("use_dora", ARMS)
def test_all_trainable_lora_params_get_nonzero_grad(fresh_model, batch, use_dora):
    """Com B=0 o gradiente de A é zero no primeiro passo (esperado). Por isso: 1 passo de
    otimizador e, no passo seguinte, todo parâmetro treinável tem de receber gradiente não nulo."""
    model = add_adapter(fresh_model, r=16, use_dora=use_dora)
    model.train()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert params and all("lora_" in n for n in params), "só parâmetros lora_* devem ser treináveis"
    opt = torch.optim.AdamW(params.values(), lr=1e-3)
    labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(**batch, labels=labels).loss
        loss.backward()
        return loss.item()

    step()
    first_zero = [n for n, p in params.items() if p.grad is None or p.grad.abs().max() == 0]
    opt.step()
    step()
    bad = [n for n, p in params.items() if p.grad is None or not torch.isfinite(p.grad).all()
           or p.grad.abs().max() == 0]
    print(f"\n[grad dora={use_dora}] treináveis={len(params)} | grad zero no 1º passo={len(first_zero)} "
          f"| grad nulo/inválido no 2º={len(bad)}")
    assert not bad, bad[:5]


@pytest.mark.parametrize("use_dora", ARMS_MERGE)
def test_merged_model_matches_active_adapter(fresh_model, big_batch, noise_floor, use_dora):
    batch = big_batch
    model = add_adapter(fresh_model, r=16, use_dora=use_dora)
    torch.manual_seed(0)
    with torch.no_grad():  # B != 0, para o adaptador ter efeito real no teste
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    active = logits_of(model, batch)
    with model.disable_adapter():
        off = logits_of(model, batch)
    merged = logits_of(model.merge_and_unload(), batch)
    mask = batch["attention_mask"].bool()
    effect = equivalence_stats(off, active, mask)
    stats = equivalence_stats(active, merged, mask)
    print(f"\n[merge dora={use_dora}] efeito do adaptador: {effect}\n   merged x ativo: {stats}")
    assert effect["mean_abs"] > 5 * stats["mean_abs"], "efeito do adaptador não supera o ruído da fusão: teste inconclusivo"
    assert_within_noise(stats, noise_floor, "modelo fundido x adaptador ativo")


def test_generation_stops_at_im_end(fresh_model, tok):
    model = fresh_model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": "Diga apenas: olá."}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to("cuda")
    n_prompt = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=128, do_sample=False, eos_token_id=EOS_IDS,
                         pad_token_id=tok.pad_token_id)[0, n_prompt:]
    default = model.generate(**enc, max_new_tokens=128, do_sample=False, eos_token_id=2,
                             pad_token_id=tok.pad_token_id)[0, n_prompt:]
    print(f"\n[stop] eos=[2,11]: {len(out)} tokens, último id={out[-1].item()} | só eos=2: {len(default)} tokens")
    print("   texto:", repr(tok.decode(out, skip_special_tokens=False)[:160]))
    assert out[-1].item() in EOS_IDS and len(out) < 128
