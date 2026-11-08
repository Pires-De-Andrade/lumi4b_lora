"""Funde só uma parte do adaptador LoRA (triagem de "em quais camadas o adaptador ajuda ou atrapalha"), sem treinar de novo.

Os módulos fora da seleção ficam com efeito zero (volta o peso do Lumi original). Seleção = faixa de camadas E tipos de bloco:
  mamba = in_proj (21 camadas) | mlp = up_proj e down_proj (17) | attn = q, k, v e o_proj (4).
Só LoRA. Saída pronta para o vLLM, como em merge_adapter.py (modo rápido). Roda na CPU.
Uso: PYTHONPATH=src python scripts/merge_adapter_layers.py --adapter runs/train/adalberto/adapter --out runs/merged/adalberto_L21_41 --layers 21-41 --types mamba,mlp,attn
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
import merge_adapter as M  # noqa: E402

from lumi_ft.modeling.load import MODEL_ID, REVISION, load_model  # noqa: E402

TYPES = {"mamba": {"in_proj"}, "mlp": {"up_proj", "down_proj"}, "attn": {"q_proj", "k_proj", "v_proj", "o_proj"}}
NAME = re.compile(r"layers\.(\d+)\.mixer\.(\w+)$")


def keep(name: str, lo: int, hi: int, projs: set[str]) -> bool | None:
    """True/False se `name` é um módulo-alvo (dentro/fora da seleção); None se não for um módulo-alvo."""
    m = NAME.search(name)
    if not m:
        return None
    return lo <= int(m.group(1)) <= hi and m.group(2) in projs


def mask_lora(model, lo: int, hi: int, types: list[str]) -> tuple[int, int]:
    """Zera o `scaling` dos módulos LoRA fora da seleção. Devolve (mantidos, zerados)."""
    projs = set().union(*(TYPES[t] for t in types))
    kept = zeroed = 0
    for name, m in model.named_modules():
        sc = getattr(m, "scaling", None)
        if not (isinstance(sc, dict) and hasattr(m, "lora_A") and sc):
            continue
        k = keep(name, lo, hi, projs)
        assert k is not None, name
        if k:
            kept += 1
        else:
            for key in sc:
                sc[key] = 0.0
            zeroed += 1
    return kept, zeroed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-41", help="faixa de camadas mantidas, ex.: 21-41")
    ap.add_argument("--types", default="mamba,mlp,attn")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-"))
    types = a.types.split(",")
    assert all(t in TYPES for t in types), types
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    assert not cfg.get("use_dora"), "só LoRA"
    model = load_model("native", device="cpu")
    pm = PeftModel.from_pretrained(model, a.adapter)
    kept, zeroed = mask_lora(pm, lo, hi, types)
    assert kept + zeroed == 71, (kept, zeroed)
    merged = pm.merge_and_unload()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(out, safe_serialization=True)
    orig = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    for f in M.COPY:
        if (Path(orig) / f).exists():
            shutil.copy(Path(orig) / f, out / f)
    (out / "merge_report.json").write_text(json.dumps({"adapter": a.adapter, "layers": a.layers, "types": types, "kept": kept, "zeroed": zeroed, "out": str(out)}, indent=1))
    print(json.dumps({"layers": a.layers, "types": types, "kept": kept, "zeroed": zeroed}))
    print("EXIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
