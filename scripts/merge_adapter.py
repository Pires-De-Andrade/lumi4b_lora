"""Funde um adaptador (LoRA ou DoRA) no Lumi em BF16 e grava o modelo pronto para o vLLM (A-11).

Os pesos vêm da implementação nativa; `config.json`, tokenizer e `.py` vêm do repositório original, porque o
`config.json` que a nativa salva é outro (discovery.md §10). Confere que as chaves e formas são idênticas às
do Lumi original e que só mudaram os 71 módulos-alvo. Roda na CPU (não disputa a GPU com o treino).

Uso: scripts/detach.sh merge-viab PYTHONPATH=src python scripts/merge_adapter.py --adapter runs/train/viab/adapter --out runs/merged/viab
"""
import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open

from lumi_ft.modeling import peft_patch
from lumi_ft.modeling.lora_scale import scale_lora
from lumi_ft.modeling.load import MODEL_ID, REVISION, load_model

COPY = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "configuration_nemotron_h.py", "modeling_nemotron_h.py", "__init__.py", "special_tokens_map.json"]


def header(d: str) -> dict:
    h = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                s = sf.get_slice(k)
                h[k] = (tuple(s.get_shape()), s.get_dtype(), f)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fast", action="store_true", help="não confere chaves e tensores (para as iterações do GRPO; a primeira fusão confere)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplica o efeito do adaptador LoRA antes de fundir (1 = como treinado)")
    a = ap.parse_args()
    out = Path(a.out)
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        peft_patch.apply()  # linhas zeradas da in_proj (peft_patch.py)
    model = load_model("native", device="cpu")
    peft_model = PeftModel.from_pretrained(model, a.adapter)
    if a.scale != 1.0:
        assert not cfg.get("use_dora"), "--scale só vale para LoRA"
        assert scale_lora(peft_model, a.scale) == 71, "esperava 71 módulos LoRA"
    merged = peft_model.merge_and_unload()
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(out, safe_serialization=True)
    orig_dir = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    for f in COPY:
        if (Path(orig_dir) / f).exists():
            shutil.copy(Path(orig_dir) / f, out / f)  # sobrescreve o config.json salvo pela nativa
    if a.fast:
        (out / "merge_report.json").write_text(json.dumps({"adapter": a.adapter, "scale": a.scale, "fast": True, "out": str(out)}, indent=1))
        print("EXIT_OK (rápido: sem conferência)")
        return 0
    o, s = header(orig_dir), header(str(out))
    assert set(o) == set(s), (sorted(set(o) ^ set(s))[:5])
    assert all(o[k][:2] == s[k][:2] for k in o), "forma ou dtype diferente do Lumi original"
    assert not [k for k in s if "lora_" in k or "base_layer" in k]
    changed = []
    with safe_open(next(iter(o.values()))[2], "pt") as so:
        files = {}
        for k in o:
            f = s[k][2]
            files.setdefault(f, safe_open(f, "pt"))
            if not torch.equal(files[f].get_tensor(k), so.get_tensor(k)):
                changed.append(k)
    rx = re.compile(r"backbone\.layers\.\d+\.mixer\.(in_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)\.weight")
    targets = {k for k in o if rx.fullmatch(k)}
    assert len(targets) == 71, len(targets)
    assert set(changed) <= targets, f"tensores fora dos alvos mudaram: {sorted(set(changed) - targets)[:5]}"
    assert changed, "nenhum tensor mudou: o adaptador não foi fundido"
    report = {"adapter": a.adapter, "use_dora": bool(cfg.get("use_dora")), "r": cfg.get("r"), "scale": a.scale, "keys": len(s), "changed_tensors": len(changed), "targets_unchanged_by_bf16_rounding": len(targets) - len(changed),
              "config_from": "repositório original", "out": str(out)}
    (out / "merge_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("EXIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
