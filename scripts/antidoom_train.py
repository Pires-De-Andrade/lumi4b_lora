"""Fase anti-laço, passo 2: treina a preferência no token que inicia o laço (roda no ambiente do treino).

Duas receitas:
  --recipe oficial (padrão): a do repositório do Antidoom (Liquid4All/antidoom, commit bd6a126; conta portada em
      `lumi_ft.train.antidoom_oficial`). LoRA NOVO (r=256, alpha=128, dropout 0, em todas as projeções lineares que o PEFT aceita no NemotronH, isto é, menos o `out_proj` do Mamba, e no `lm_head`) sobre o modelo de
      partida já fundido (`--base-model`); escolhidos = alternativas do topo da distribuição do modelo de partida no ponto do rejeitado
      (20 maiores, renormalizadas na temperatura 0,1, corte min-p 0,01); regularização de frequência dos rejeitados (0,3) e dos escolhidos (0,5);
      perda FTPO com margem de 2 logits e amarras MSE (0,4 no resto do vocabulário, 0,05 com folga de 0,5 nos escolhidos e no rejeitado);
      AdamW lr 1,5e-5, aquecimento de 10% e queda linear, decaimento de peso 0,01, corte de gradiente 2,5, lote efetivo de 16, uma época;
      para quando a média do `chosen_win` de treino (fração dos escolhidos acima do rejeitado) nos últimos 5 passos chega a 0,4.
  --recipe propria: a primeira versão (continua o adaptador r=16 de `--adapter-in`; perda softplus da média dos escolhidos; parada no `chosen_win`
      do melhor escolhido na validação). Mantida para comparação.
Diferenças da receita oficial que ficam (third_party/antidoom/NOTICE.md): os logits são calculados em FP32, um evento por passada (sem
preenchimento até 6.000 tokens), e a referência é calculada uma vez antes do treino (o adaptador novo começa nulo, então é o modelo de partida).
  PYTHONPATH=src python scripts/antidoom_train.py --events runs/antidoom/thor_pt/mine/events.jsonl --base-model runs/merged/thor_pt --out runs/antidoom/thor_pt/train
Saída em --out: adapter/ (LoRA sobre o --base-model; fundir com scripts/merge_on_base.py), train_log.jsonl e summary.json.
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

TARGETS_OFICIAL = r"(.*\.mixer\.(in_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)|lm_head)"  # o PEFT recusa LoRA no out_proj do Mamba (NemotronH)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="oficial", choices=["oficial", "propria"])
    ap.add_argument("--events", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="teste de fumaça: 6 eventos, 2 passos, sem parada antecipada")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda", help="oficial: 'cpu' só para o teste de lógica com um modelo minúsculo")
    # receita oficial
    ap.add_argument("--base-model", help="oficial: pasta do modelo de partida fundido")
    ap.add_argument("--lora-r", type=int, default=256)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--targets", default=TARGETS_OFICIAL)
    ap.add_argument("--lr", type=float, default=None, help="padrão: 1,5e-5 (oficial) ou 1e-5 (própria)")
    ap.add_argument("--batch", type=int, default=None, help="eventos por passo do otimizador; padrão: 16 (oficial) ou 8 (própria)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=2.5)
    ap.add_argument("--eps", type=float, default=2.0, help="margem em logits (clip_epsilon_logits)")
    ap.add_argument("--lambda-mse", type=float, default=0.4)
    ap.add_argument("--lambda-mse-target", type=float, default=0.05)
    ap.add_argument("--tau-mse-target", type=float, default=0.5)
    ap.add_argument("--early-stop-win", type=float, default=0.4)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--max-train-examples", type=int, default=12000)
    ap.add_argument("--rejected-reg", type=float, default=0.3)
    ap.add_argument("--chosen-reg", type=float, default=0.5)
    ap.add_argument("--min-chosen", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=6000)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.0, help="fração de eventos fora do treino, só para relatório (a oficial treina em todos)")
    # receita própria
    ap.add_argument("--adapter-in", help="própria: adaptador a continuar treinando")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--target-win", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam-rest", type=float, default=5.0)
    ap.add_argument("--lam-cr", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-chosen", type=int, default=20)
    ap.add_argument("--max-rank", type=int, default=3)
    a = ap.parse_args()
    if a.recipe == "oficial":
        if not a.base_model:
            ap.error("--recipe oficial exige --base-model")
        return train_oficial(a)
    if not a.adapter_in:
        ap.error("--recipe propria exige --adapter-in")
    return train_propria(a)


def decoder_of(model):
    """Tronco do NemotronH (camadas + norma final, sem a cabeça): `model` na implementação nativa do Transformers 5, `backbone` no código remoto."""
    for name in ("model", "backbone"):
        sub = getattr(model, name, None)
        if sub is not None and hasattr(sub, "layers"):
            return sub
    raise AttributeError(f"tronco não encontrado em {type(model).__name__}")


def load_events(paths: list[str], max_len: int) -> list[dict]:
    ev = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    return [e for e in ev if len(e["prompt_ids"]) + len(e["prefix_ids"]) + 1 <= max_len]


# ======================================================================================= receita oficial
def train_oficial(a) -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

    from lumi_ft import gpu
    from lumi_ft.modeling.load import load_tokenizer
    from lumi_ft.train import antidoom_oficial as O

    lr = a.lr if a.lr is not None else 1.5e-5
    batch = a.batch if a.batch is not None else 16
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "train_log.jsonl"
    log.write_text("")
    tok = load_tokenizer()
    events = load_events(a.events, a.max_seq_len)
    if a.smoke:
        events = events[:6]
    print({"eventos": len(events)}, flush=True)
    if len(events) < 4:
        print("poucos eventos de laço; nada a treinar")
        return 1
    tstr = lambda i: tok.convert_ids_to_tokens(int(i))
    for e in events:
        e["rejected_decoded"] = O.decode_token(tstr(e["rejected"]))
    # 1) regularização dos rejeitados (só age acima de max_train_examples, como no original)
    rows = O.sample_rows(events, max_train_examples=a.max_train_examples, rejected_strength=a.rejected_reg, seed=a.seed)

    # 2) modelo de partida + LoRA novo
    torch.manual_seed(a.seed)
    dev = a.device
    model = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16, trust_remote_code=False, device_map={"": dev})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none", target_modules=a.targets)
    pm = get_peft_model(model, lcfg)
    params = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    assert params and all(p.dtype == torch.float32 for p in params.values()), "o adaptador deve estar em FP32"
    n_params = sum(p.numel() for p in params.values())
    targeted = sorted({n.split(".lora_")[0] for n in params})
    head = pm.base_model.model.lm_head
    head_lora = hasattr(head, "lora_A") and "default" in head.lora_A
    w32 = (head.base_layer.weight if head_lora else head.weight).detach().float()
    backbone = decoder_of(pm.base_model.model)
    print({"lora_params": n_params, "modulos": len(targeted), "lm_head_com_lora": head_lora}, flush=True)
    if dev == "cuda":
        gpu.reset_peak()

    def last_logits(ids: list[int]):
        x = torch.tensor([ids], device=dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            h = backbone(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)[0]
        h = h[0, -1].float()
        z = h @ w32.T
        if head_lora:
            A = head.lora_A["default"].weight
            B = head.lora_B["default"].weight
            z = z + head.scaling["default"] * ((h @ A.T) @ B.T)
        return z

    # 3) referência (adaptador desligado = modelo de partida) e escolhidos pela regra oficial
    pm.eval()
    data, drops = [], Counter()
    t0 = time.time()
    check = None
    with torch.no_grad(), pm.disable_adapter():
        for i, e in enumerate(rows):
            ids = e["prompt_ids"] + e["prefix_ids"]
            z0 = last_logits(ids)
            if i == 0:  # conferência da cabeça manual contra a passada completa do modelo
                xx = torch.tensor([ids], device=dev)
                try:
                    full = pm(input_ids=xx, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                except TypeError:
                    full = pm(input_ids=xx, use_cache=False).logits[0, -1].float()
                check = {"max_abs_diff_logits": round(float((full - z0).abs().max()), 4), "argmax_igual": int(full.argmax()) == int(z0.argmax())}
                print({"conferencia_cabeca": check}, flush=True)
                assert check["max_abs_diff_logits"] < 0.5 and check["argmax_igual"], check
            lp = torch.log_softmax(z0, -1)
            top = torch.topk(lp, a.top_logprobs)
            items = [(int(t), tstr(t), float(v)) for v, t in zip(top.values.tolist(), top.indices.tolist())]
            chosen = O.select_chosen_tokens(items, rejected_id=int(e["rejected"]), rejected_token=tstr(e["rejected"]), temperature=0.1,
                                            min_p=0.01, top_k=50, max_tokens=20, min_decoded_chars=1, require_alnum=False)
            if len(chosen) < a.min_chosen:
                drops["sem_escolhidos"] += 1
                continue
            rank = int((z0 > z0[int(e["rejected"])]).sum())
            data.append({"ids": ids, "rejected": int(e["rejected"]), "chosen_ids": chosen,
                         "chosen_keys": [O.normalise_chosen_key(O.decode_token(tstr(c))) for c in chosen],
                         "z0": z0.cpu(), "rank": rank, "kind": e.get("kind"), "rejected_decoded": e["rejected_decoded"]})
    fc = Counter()
    data = O.regularise_chosen(data, a.chosen_reg, fc)
    print({"referencia_s": round(time.time() - t0), "usados": len(data), "descartados": dict(drops), "regularizacao": dict(fc)}, flush=True)
    if len(data) < 4:
        print("eventos de menos depois dos filtros")
        return 1
    rng = random.Random(a.seed)
    rng.shuffle(data)
    n_val = int(len(data) * a.val_frac) if not a.smoke else 0
    val, train = data[:n_val], data[n_val:]
    if a.smoke:
        batch = 3
    steps_per_epoch = math.ceil(len(train) / batch)
    total_steps = 2 if a.smoke else max(1, math.ceil(steps_per_epoch * a.epochs))
    opt = torch.optim.AdamW(list(params.values()), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(math.ceil(a.warmup * total_steps)), num_training_steps=total_steps)

    def loss_of(d):
        z = last_logits(d["ids"])
        return O.ftpo_loss_oficial(z, d["z0"].to(dev), d["chosen_ids"], d["rejected"], eps=a.eps, lambda_mse=a.lambda_mse,
                                   lambda_mse_target=a.lambda_mse_target, tau_mse_target=a.tau_mse_target)

    def mean(ms):
        return {k: round(sum(m[k] for m in ms) / len(ms), 4) for k in ms[0]} if ms else {}

    def evaluate(rows_):
        if not rows_:
            return {}
        pm.eval()
        with torch.no_grad():
            ms = [loss_of(d)[1] for d in rows_]
        pm.train()
        return mean(ms)

    info = {"receita": "oficial", "base_model": a.base_model, "eventos": len(events), "linhas_amostradas": len(rows), "treino": len(train),
            "validacao": len(val), "lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "lora_params": n_params, "modulos": len(targeted),
            "lm_head_com_lora": head_lora, "lr": lr, "lote": batch, "passos_previstos": total_steps, "eps": a.eps, "lambda_mse": a.lambda_mse,
            "lambda_mse_target": a.lambda_mse_target, "tau_mse_target": a.tau_mse_target, "early_stop_win": a.early_stop_win,
            "conferencia_cabeca": check, "descartados": dict(drops), "regularizacao": dict(fc),
            "escolhidos_por_evento": round(sum(len(d["chosen_ids"]) for d in data) / len(data), 2),
            "rejeitados_mais_comuns": Counter(d["rejected_decoded"] for d in data).most_common(15),
            "rank_do_rejeitado": dict(Counter(min(d["rank"], 5) for d in data)),
            "por_tipo": dict(Counter(d["kind"] for d in data)), "inicio_validacao": evaluate(val)}
    gpu.append_jsonl(log, {"event": "inicio", **info})
    print(json.dumps(info, ensure_ascii=False), flush=True)

    pm.train()
    order = list(range(len(train)))
    rng.shuffle(order)
    batches: list[list[int]] = []  # uma época = cada evento uma vez (o último lote pode ser menor, como no Trainer com drop_last=False)
    while len(batches) < total_steps:
        for s0 in range(0, len(order), batch):
            batches.append(order[s0:s0 + batch])
            if len(batches) >= total_steps:
                break
        rng.shuffle(order)
    step, stopped, since_log, hist = 0, None, [], []
    t1 = time.time()
    while step < total_steps and stopped is None:
        idx = batches[step]
        opt.zero_grad(set_to_none=True)
        ms = []
        for i in idx:
            loss, m = loss_of(train[i])
            (loss / len(idx)).backward()
            ms.append(m)
        gn = float(torch.nn.utils.clip_grad_norm_(list(params.values()), a.max_grad_norm))
        opt.step()
        sched.step()
        step += 1
        since_log.extend(ms)
        row = {"step": step, "lr": sched.get_last_lr()[0], "grad_norm": round(gn, 4), **mean(ms)}
        if step % a.log_every == 0 or a.smoke:
            lg = mean(since_log)
            since_log = []
            row["log"] = lg
            if val:
                row["val"] = evaluate(val)
            hist.append((step, lg.get("chosen_win"), row.get("val", {}).get("chosen_win")))
            print(row, flush=True)
            if not a.smoke and lg["chosen_win"] >= a.early_stop_win:
                stopped = step
                print(f"[parada] chosen_win de treino {lg['chosen_win']:.4f} >= {a.early_stop_win} no passo {step}", flush=True)
        gpu.append_jsonl(log, row)
    pm.save_pretrained(out / "adapter", save_embedding_layers=False)  # o lm_head só tem LoRA: não grava cópias inteiras das camadas de embedding
    summary = {**info, "passos": step, "parou_em": stopped, "treino_s": round(time.time() - t1), "historico_chosen_win": hist,
               "fim_validacao": evaluate(val), "pico_gib": gpu.peak_gib() if dev == "cuda" else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("passos", "parou_em", "treino_s", "pico_gib")}), flush=True)
    return 0


# ======================================================================================= receita própria (primeira versão)
def train_propria(a) -> int:
    import torch

    from lumi_ft import gpu
    from lumi_ft.modeling import peft_patch
    from lumi_ft.modeling.load import EOS_IDS, load_model, load_tokenizer
    from lumi_ft.train import antidoom as A

    lr = a.lr if a.lr is not None else 1e-5
    batch = a.batch if a.batch is not None else 8
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "train_log.jsonl"
    events = load_events(a.events, 3584)
    if a.smoke:
        events = events[:6]
        a.max_steps, a.eval_every, batch = 2, 1, 3
    print({"eventos": len(events)}, flush=True)
    if len(events) < 4:
        print("poucos eventos de laço; nada a treinar")
        return 1
    cfg = json.loads((Path(a.adapter_in) / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        peft_patch.apply()
    from peft import PeftModel

    torch.manual_seed(a.seed)
    tok = load_tokenizer()
    model = load_model("native")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm = PeftModel.from_pretrained(model, a.adapter_in, is_trainable=True)
    params = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    assert params and all(p.dtype == torch.float32 for p in params.values()), "o adaptador deve estar em FP32"
    opt = torch.optim.AdamW(list(params.values()), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    w32 = model.lm_head.weight.detach().float()
    gpu.reset_peak()

    def last_logits(ids):
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = decoder_of(model)(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)[0]
        return h[0, -1].float() @ w32.T

    special = set(EOS_IDS) | set(tok.all_special_ids)
    pm.eval()
    data, dropped = [], {"rejeitado_fora_do_topo": 0, "sem_escolhidos": 0}
    with torch.no_grad():
        for e in events:
            ids = e["prompt_ids"] + e["prefix_ids"]
            z0 = last_logits(ids)
            rank = int((z0 > z0[e["rejected"]]).sum())
            if rank >= a.max_rank:
                dropped["rejeitado_fora_do_topo"] += 1
                continue
            chosen = A.choose_candidates(z0, e["rejected"], tok.decode, top_k=a.top_k, max_chosen=a.max_chosen, exclude=special)
            if not chosen:
                dropped["sem_escolhidos"] += 1
                continue
            data.append({"ids": ids, "rejected": e["rejected"], "chosen": chosen, "z0": z0.half().cpu(), "rank": rank})
    if len(data) < 4:
        print("eventos de menos depois do filtro")
        return 1
    rng = random.Random(a.seed)
    rng.shuffle(data)
    n_val = 2 if a.smoke else max(8, int(len(data) * 0.15))
    val, train = data[:n_val], data[n_val:]

    def loss_of(d):
        z = last_logits(d["ids"])
        return A.ftpo_loss(z, d["z0"].cuda().float(), d["chosen"], d["rejected"], beta=a.beta, lam_rest=a.lam_rest, lam_cr=a.lam_cr, tau=a.tau)

    def mean_metrics(ms):
        return {k: round(sum(m[k] for m in ms) / len(ms), 4) for k in ms[0]}

    def evaluate():
        pm.eval()
        with torch.no_grad():
            ms = [loss_of(d)[1] for d in val]
        pm.train()
        return mean_metrics(ms)

    info = {"receita": "propria", "events": len(events), "treino": len(train), "validacao": len(val), "descartados": dropped, "inicio_validacao": evaluate()}
    gpu.append_jsonl(log, {"event": "inicio", **info})
    pm.train()
    order: list[int] = []
    step, stopped, hist = 0, None, []
    t1 = time.time()
    while step < a.max_steps and stopped is None:
        if len(order) < batch:
            order = list(range(len(train)))
            rng.shuffle(order)
        idx, order = order[:batch], order[batch:]
        opt.zero_grad(set_to_none=True)
        ms = []
        for i in idx:
            loss, m = loss_of(train[i])
            (loss / len(idx)).backward()
            ms.append(m)
        gn = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0))
        opt.step()
        step += 1
        row = {"step": step, "grad_norm": round(gn, 4), **mean_metrics(ms)}
        if step % a.eval_every == 0:
            v = evaluate()
            row["val"] = v
            hist.append((step, v))
            if not a.smoke and v["chosen_win"] >= a.target_win:
                stopped = step
        gpu.append_jsonl(log, row)
    pm.save_pretrained(out / "adapter")
    summary = {**info, "passos": step, "parou_em": stopped, "treino_s": round(time.time() - t1), "validacao_por_passo": hist,
               "fim_validacao": evaluate(), "pico_gib": gpu.peak_gib()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
