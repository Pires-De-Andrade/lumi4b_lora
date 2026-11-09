"""Braço de conferência: treina com o TREINADOR do repositório oficial do Antidoom (Liquid4All/antidoom, clonado em ../antidoom_oficial, commit
bd6a126, Apache-2.0), sobre os mesmos eventos da nossa mineração.

Usa sem mudança as classes `FTPOTrainer` (perda e coleta), `FTPODataset` (filtros e regularização dos escolhidos), `sample_rows` e
`ThresholdStop`, e a escolha dos escolhidos `select_chosen_tokens` do original. Adaptações, todas por limite da máquina ou do modelo:
  - modelo carregado na implementação nativa do Transformers (o NemotronH do Lumi), LoRA r=256/alpha 128 nos alvos que o PEFT aceita no NemotronH
    (sem o `out_proj` do Mamba) e no `lm_head`;
  - lote de 1 com acumulação de 16 (o original usa 4 x 4) e checkpoint de gradiente, para caber nos 24 GiB da 4090;
  - a coleta preenche até o maior exemplo do lote, não até 6.000 tokens (com lote 1, sem preenchimento; a conta é a mesma);
  - `datasets.Dataset` em vez do `FTPODataset` como entrada do treinador (o TRL instalado, 1.17, exige; o original travou o 1.2);
  - a checagem do TRL 1.17 que recusa LoRA no `lm_head` (válida para a perda DPO, não para a do FTPO) é contornada na criação do treinador;
  - otimizador `adamw_torch` (o do arquivo de configuração para AMD do original; o `paged_adamw_32bit` exige bitsandbytes, que não está instalado);
  - os escolhidos vêm das 20 maiores log-probabilidades do modelo de partida no ponto do rejeitado (no original, do vLLM durante a geração).
  PYTHONPATH=src python scripts/antidoom_oficial_train.py --events runs/antidoom/thor_pt/mine/events.jsonl --base-model runs/merged/thor_pt --out runs/antidoom/thor_pt_of
Saída em --out: adapter/ (fundir com scripts/merge_on_base.py), train_log.jsonl (métricas registradas pelo treinador) e summary.json.
"""
import argparse
import inspect
import json
import sys
import time
import types
from collections import Counter
from pathlib import Path

OFICIAL = Path(__file__).resolve().parents[2] / "antidoom_oficial" / "src"
TARGETS = r"(.*\.mixer\.(in_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)|lm_head)"


def import_oficial():
    if not OFICIAL.is_dir():
        raise SystemExit(f"repositório oficial não encontrado em {OFICIAL}")
    sys.path.insert(0, str(OFICIAL))
    if "antidoom.client" not in sys.modules:  # o cliente HTTP do original não é usado aqui
        fake = types.ModuleType("antidoom.client")
        fake.CompletionChunk = type("CompletionChunk", (), {})
        fake.VllmClient = type("VllmClient", (), {})
        sys.modules["antidoom.client"] = fake
    from antidoom import ftpo_data, reference_ftpo_trainer, sampling, tokens
    return ftpo_data, reference_ftpo_trainer, sampling, tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, nargs="+")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-r", type=int, default=256)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=6000)
    ap.add_argument("--max-train-examples", type=int, default=12000)
    ap.add_argument("--device", default="cuda", help="'cpu' só para o teste de lógica com um modelo minúsculo")
    ap.add_argument("--smoke", action="store_true", help="6 eventos, 2 passos de 3")
    a = ap.parse_args()

    ftpo_data, ref_trainer, sampling, tokens = import_oficial()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import DPOConfig

    from lumi_ft.modeling.load import load_tokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    events = [json.loads(line) for p in a.events for line in Path(p).read_text().splitlines() if line.strip()]
    events = [e for e in events if len(e["prompt_ids"]) + len(e["prefix_ids"]) + 1 <= a.max_seq_len]
    if a.smoke:
        events = events[:6]
    dev = a.device
    model = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16, trust_remote_code=False, device_map={"": dev})
    model.config.use_cache = False

    # linhas no formato do original: contexto em texto, rejeitado e escolhidos como strings decodificadas
    rows, drops = [], Counter()
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for e in events:
            ids = e["prompt_ids"] + e["prefix_ids"]
            x = torch.tensor([ids], device=dev)
            try:
                z = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            except TypeError:
                z = model(input_ids=x, use_cache=False).logits[0, -1].float()
            lp = torch.log_softmax(z, -1)
            top = torch.topk(lp, 20)
            items = [(tok.convert_ids_to_tokens(int(t)), float(v)) for v, t in zip(top.values.tolist(), top.indices.tolist())]
            rej_raw = tok.convert_ids_to_tokens(int(e["rejected"]))
            chosen_raw, chosen_dec = sampling.select_chosen_tokens(items, rejected_token=rej_raw, temperature=0.1, min_p=0.01, top_k=50,
                                                                   max_tokens=20, min_decoded_chars=1, require_alnum=False)
            if not chosen_raw:
                drops["sem_escolhidos"] += 1
                continue
            rows.append({"context_with_chat_template": tok.decode(ids, skip_special_tokens=False), "rejected_decoded": tokens.decode_token(rej_raw),
                         "rejected_raw": rej_raw, "multi_chosen_decoded": chosen_dec, "multi_chosen_raw": chosen_raw,
                         "source_dataset": e.get("kind", "mcq"), "prompt_id": e.get("id")})
    print({"referencia_s": round(time.time() - t0), "linhas": len(rows), "descartados": dict(drops)}, flush=True)
    sampled = ftpo_data.sample_rows(rows, max_train_examples=a.max_train_examples, rejected_strength=0.3, min_chosen_tokens=1, seed=3407,
                                    source_balance_mode="off")
    dataset = ftpo_data.FTPODataset(sampled, tok, max_seq_length=a.max_seq_len, filter_rejected_stop_words=False, chosen_regularisation_strength=0.5)
    print({"exemplos": len(dataset), "filtros": dict(dataset.filter_counts)}, flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none", target_modules=TARGETS))
    pm.train()

    class FTPOTrainerLote(ref_trainer.FTPOTrainer):
        """Igual ao original, mas a coleta preenche só até o maior exemplo do lote."""
        def ftpo_collator(self, features):
            self.ftpo_max_length = max(len(f["prompt_ids"]) for f in features) + 1
            return super().ftpo_collator(features)

    log_path = out / "train_log.jsonl"
    log_path.write_text("")

    class Registro(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": state.global_step, **{k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in logs.items()}}) + "\n")

    cfg = {"per_device_train_batch_size": 1 if not a.smoke else 3, "gradient_accumulation_steps": 16 if not a.smoke else 1, "warmup_ratio": 0.1,
           "num_train_epochs": 1, "learning_rate": 1.5e-5, "logging_steps": 5 if not a.smoke else 1, "optim": "adamw_torch", "seed": 3407,
           "output_dir": str(out / "trainer"), "max_length": a.max_seq_len, "beta": 0.1, "weight_decay": 0.01, "report_to": "none",
           "lr_scheduler_type": "linear", "bf16": dev == "cuda", "fp16": False, "remove_unused_columns": False, "max_grad_norm": 2.5,
           "save_strategy": "no", "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
           "use_cpu": dev == "cpu", "max_steps": 2 if a.smoke else -1}
    ps = inspect.signature(DPOConfig).parameters
    if "warmup_ratio" not in ps:  # Transformers 5: a fração de aquecimento vai em `warmup_steps` (float em [0, 1))
        cfg["warmup_steps"] = cfg.pop("warmup_ratio")
    dropped = sorted(k for k in cfg if k not in ps)
    args = DPOConfig(**{k: v for k, v in cfg.items() if k in ps})
    import datasets  # o TRL instalado (1.17) só aceita `datasets.Dataset`; o original travou o TRL 1.2. Mesmo conteúdo do FTPODataset.
    hf_ds = datasets.Dataset.from_list([dict(ex) for ex in dataset.examples])
    # O TRL 1.17 recusa LoRA no lm_head porque a perda DPO dele lê `lm_head.weight` direto; a perda do FTPO (compute_loss do original) usa os
    # logits da passada do modelo, com o LoRA. A checagem é contornada só durante a criação do treinador.
    pm.get_output_embeddings = lambda: pm.base_model.model.lm_head.base_layer
    trainer = FTPOTrainerLote(model=pm, ref_model=None, train_dataset=hf_ds, processing_class=tok, args=args)
    del pm.get_output_embeddings
    trainer.lambda_mse_target, trainer.tau_mse_target, trainer.lambda_mse, trainer.clip_epsilon_logits = 0.05, 0.5, 0.4, 2.0
    if not a.smoke:
        trainer.add_callback(ref_trainer.ThresholdStop("chosen_win", threshold=0.4, higher_is_better=True))
    trainer.add_callback(Registro())
    t1 = time.time()
    trainer.train()
    pm.save_pretrained(out / "adapter", save_embedding_layers=False)
    summary = {"receita": "treinador oficial", "base_model": a.base_model, "eventos": len(events), "linhas": len(rows), "exemplos": len(dataset),
               "filtros": dict(dataset.filter_counts), "descartados": dict(drops), "args_ignorados_pelo_DPOConfig": dropped,
               "passos": trainer.state.global_step, "treino_s": round(time.time() - t1), "lora_r": a.lora_r, "lora_alpha": a.lora_alpha}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
