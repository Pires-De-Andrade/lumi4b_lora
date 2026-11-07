"""Fase 0: o modelo fundido salvo pela implementação nativa mantém o formato de chaves do Lumi?

Carrega a nativa na CPU, injeta pesos LoRA aleatórios, funde, salva e compara com o checkpoint
original (nomes, formas, dtypes e valores). Só lê arquivos locais. Saída: runs/phase0/merged_keys.json
Uso: scripts/detach.sh merged-keys python scripts/check_merged_keys.py
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from lumi_ft.modeling.load import MODEL_ID, REVISION
from lumi_ft.modeling.peft_setup import add_adapter

OUT = Path("runs/phase0")
SAVE = OUT / "merged_model"


def header(dirpath: str) -> dict[str, tuple[tuple[int, ...], str, str]]:
    """nome -> (forma, dtype, arquivo), lendo só o cabeçalho dos .safetensors."""
    h = {}
    for f in sorted(glob.glob(f"{dirpath}/*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                s = sf.get_slice(k)
                h[k] = (tuple(s.get_shape()), s.get_dtype(), os.path.basename(f))
    return h


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    orig_dir = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    orig = header(orig_dir)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=False, device_map={"": "cpu"}
    )
    peft_model = add_adapter(model, r=16, use_dora=False)
    g = torch.Generator().manual_seed(0)
    n_b = 0
    for n, p in peft_model.named_parameters():
        if "lora_B" in n:
            p.data = (torch.randn(p.shape, generator=g) * 0.02).to(p.dtype)
            n_b += 1
    merged = peft_model.merge_and_unload()
    in_memory = {k: v.detach().clone() for k, v in merged.state_dict().items()}

    if SAVE.exists():
        shutil.rmtree(SAVE)
    merged.save_pretrained(SAVE, safe_serialization=True)
    saved = header(str(SAVE))

    only_orig = sorted(set(orig) - set(saved))
    only_saved = sorted(set(saved) - set(orig))
    common = sorted(set(orig) & set(saved))
    shape_dtype_diff = [k for k in common if orig[k][:2] != saved[k][:2]]
    leftovers = [k for k in saved if "lora_" in k or "base_layer" in k]

    # Valores: o salvo é igual ao fundido em memória? E quais tensores mudaram em relação ao original?
    saved_vs_memory_bad, changed = [], []
    with safe_open(f"{orig_dir}/model.safetensors", "pt") as so:
        files = {}
        for k in common:
            f = saved[k][2]
            files.setdefault(f, safe_open(str(SAVE / f), "pt"))
            t_saved = files[f].get_tensor(k)
            if k in in_memory and not torch.equal(t_saved, in_memory[k]):
                saved_vs_memory_bad.append(k)
            if not torch.equal(t_saved, so.get_tensor(k)):
                changed.append(k)

    # A nativa recarrega o diretório salvo sem chaves faltando ou sobrando?
    _, info = AutoModelForCausalLM.from_pretrained(
        str(SAVE), dtype=torch.bfloat16, trust_remote_code=False, device_map={"": "cpu"}, output_loading_info=True
    )
    cfg_o = json.load(open(f"{orig_dir}/config.json"))
    cfg_s = json.load(open(SAVE / "config.json"))
    cfg_diff = {k: (cfg_o.get(k, "<ausente>"), cfg_s.get(k, "<ausente>"))
                for k in sorted(set(cfg_o) | set(cfg_s)) if cfg_o.get(k, "<ausente>") != cfg_s.get(k, "<ausente>")}

    prefixes = lambda ks: sorted({k.split(".")[0] for k in ks})  # noqa: E731
    result = {
        "lora_B_randomized": n_b,
        "n_keys_original": len(orig), "n_keys_saved": len(saved),
        "prefixes_original": prefixes(orig), "prefixes_saved": prefixes(saved),
        "only_in_original": only_orig[:20], "n_only_in_original": len(only_orig),
        "only_in_saved": only_saved[:20], "n_only_in_saved": len(only_saved),
        "shape_or_dtype_differs": shape_dtype_diff[:20],
        "lora_or_base_layer_leftovers": leftovers[:20],
        "saved_differs_from_in_memory_merged": saved_vs_memory_bad[:20],
        "n_tensors_changed_vs_original": len(changed),
        "changed_suffixes": sorted({k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1] for k in changed}),
        "reload_missing_keys": sorted(info.get("missing_keys", []))[:20],
        "reload_unexpected_keys": sorted(info.get("unexpected_keys", []))[:20],
        "config_diff": cfg_diff,
        "saved_files": sorted(os.listdir(SAVE)),
    }
    (OUT / "merged_keys.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    shutil.rmtree(SAVE)  # 8 GB; o resultado está no JSON
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())
