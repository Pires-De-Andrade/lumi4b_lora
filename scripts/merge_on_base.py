"""Funde um adaptador LoRA treinado sobre um modelo JÁ fundido (por exemplo, o da fase anti-laço na receita oficial, sobre runs/merged/thor_pt).

O `scripts/merge_adapter.py` funde no Lumi original e confere os 71 módulos do nosso LoRA; este aceita qualquer alvo (inclusive `lm_head`).
Carrega o modelo de partida na implementação nativa em FP32 na CPU, aplica o adaptador, funde, grava em BF16 e copia `config.json`, tokenizer e
`.py` do modelo de partida (que vieram do repositório original). Confere que chaves e formas são as do modelo de partida e conta os tensores
alterados e os que não mudaram por arredondamento do BF16.
  PYTHONPATH=src python scripts/merge_on_base.py --base runs/merged/thor_pt --adapter runs/antidoom/thor_pt/train/adapter --out runs/merged/thor_pt_antidoom
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

COPY = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "configuration_nemotron_h.py", "modeling_nemotron_h.py", "__init__.py", "special_tokens_map.json"]


def tensors(d: str) -> dict:
    out = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32, trust_remote_code=False, device_map={"": "cpu"})
    pm = PeftModel.from_pretrained(base, a.adapter)
    merged = pm.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="20GB")
    for f in COPY:
        if (Path(a.base) / f).exists():
            shutil.copy(Path(a.base) / f, out / f)
    # conferência contra o modelo de partida
    t0, t1 = tensors(a.base), tensors(str(out))
    assert set(t0) == set(t1), f"chaves diferentes: {sorted(set(t0) ^ set(t1))[:10]}"
    bad = [k for k in t0 if t0[k].shape != t1[k].shape or t0[k].dtype != t1[k].dtype]
    assert not bad, f"forma ou tipo diferentes: {bad[:10]}"
    changed = [k for k in t0 if not torch.equal(t0[k], t1[k])]
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    rep = {"base": a.base, "adapter": a.adapter, "out": a.out, "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
           "target_modules": cfg.get("target_modules"), "keys": len(t0), "changed_tensors": len(changed),
           "max_abs_change": max((float((t1[k].float() - t0[k].float()).abs().max()) for k in changed), default=0.0),
           "config_from": "modelo de partida"}
    (out / "merge_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if changed else 1


if __name__ == "__main__":
    sys.exit(main())
