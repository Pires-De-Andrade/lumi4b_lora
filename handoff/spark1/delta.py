"""Diferença exata entre os pesos do T6-Lumi e os do Lumi (sem perda), para enviar 2 GB em vez de 8 GB.

O T6-Lumi (thor_pt_antidoom_of) difere do Lumi em 72 dos 263 tensores. `pack` grava, para cada um, a diferença dos bits (aritmética modular sobre os
inteiros de mesmo tamanho; em BF16, a diferença em unidades de arredondamento), com os bytes separados por plano e comprimidos com zlib. `unpack` baixa o
Lumi (revisão fixada), soma a diferença e grava o `model.safetensors` do T6-Lumi, idêntico byte a byte ao original (o sha256 é conferido depois).
Só precisa de torch, numpy, safetensors e huggingface_hub (o ambiente do vLLM tem tudo).

  python handoff/spark1/delta.py pack   --base <model.safetensors do Lumi> --model <model.safetensors do T6-Lumi> --out handoff/spark1/t6_delta.bin
  python handoff/spark1/delta.py unpack --delta handoff/spark1/t6_delta.bin --out runs/merged/thor_pt_antidoom_of/model.safetensors \\
         [--base <model.safetensors do Lumi> | --repo ceia-br/lumi-4b --revision <hash>]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MAGIC = b"T6DELTA1\n"
UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
SINT = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(t: torch.Tensor) -> np.ndarray:
    """Os bits do tensor como inteiros sem sinal do mesmo tamanho (vista, sem cópia)."""
    t = t.contiguous()
    return t.view(SINT[t.element_size()]).numpy().view(UINT[t.element_size()])


def encode(base: torch.Tensor, model: torch.Tensor) -> bytes:
    w = base.element_size()
    d = np.subtract(_bits(model), _bits(base))  # modular em inteiros sem sinal
    raw = np.ascontiguousarray(d).view(np.uint8).reshape(-1, w)
    return zlib.compress(np.ascontiguousarray(raw.T).tobytes(), 6)  # planos de bytes: os bytes altos (quase sempre 0) comprimem muito


def decode(base: torch.Tensor, comp: bytes) -> torch.Tensor:
    w = base.element_size()
    n = base.numel()
    planos = np.frombuffer(zlib.decompress(comp), dtype=np.uint8).reshape(w, n)
    d = np.ascontiguousarray(planos.T).reshape(-1).view(UINT[w])
    out = np.add(_bits(base).reshape(-1), d)  # modular
    return torch.from_numpy(out.copy()).view(SINT[w]).view(base.dtype).reshape(base.shape)


def pack(base_path: str, model_path: str, out: str) -> dict:
    n_changed = n_total = 0
    with safe_open(base_path, "pt") as fb, safe_open(model_path, "pt") as fm, open(out, "wb") as o:
        if set(fb.keys()) != set(fm.keys()):
            raise SystemExit("o Lumi e o T6-Lumi não têm as mesmas chaves")
        o.write(MAGIC)
        for k in sorted(fm.keys()):
            b, m = fb.get_tensor(k), fm.get_tensor(k)
            if b.shape != m.shape or b.dtype != m.dtype:
                raise SystemExit(f"{k}: forma ou tipo diferentes")
            n_total += 1
            if torch.equal(b.view(SINT[b.element_size()]), m.view(SINT[m.element_size()])):
                continue
            comp = encode(b, m)
            o.write((json.dumps({"k": k, "n": len(comp)}) + "\n").encode())
            o.write(comp)
            n_changed += 1
        o.write((json.dumps({"fim": True, "alterados": n_changed, "total": n_total}) + "\n").encode())
    return {"alterados": n_changed, "total": n_total, "bytes": Path(out).stat().st_size}


def unpack(delta: str, out: str, base_path: str) -> dict:
    tensores = load_file(base_path)
    n = 0
    with open(delta, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise SystemExit("arquivo de diferença inválido (cabeçalho)")
        while True:
            h = json.loads(f.readline())
            if h.get("fim"):
                if h["alterados"] != n:
                    raise SystemExit(f"arquivo de diferença truncado: {n} de {h['alterados']} tensores")
                break
            tensores[h["k"]] = decode(tensores[h["k"]], f.read(h["n"]))
            n += 1
    tmp = out + ".tmp"
    save_file(tensores, tmp, metadata={"format": "pt"})
    Path(tmp).replace(out)
    return {"alterados": n, "total": len(tensores)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["pack", "unpack"])
    ap.add_argument("--base", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--delta", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="ceia-br/lumi-4b")
    ap.add_argument("--revision", default="")
    a = ap.parse_args()
    if a.modo == "pack":
        print(pack(a.base, a.model, a.out))
        return 0
    base = a.base
    if not base:
        from huggingface_hub import hf_hub_download

        print(f"baixando o Lumi ({a.repo} @ {a.revision[:8]}) do Hugging Face (~8 GB, só na primeira vez)...", flush=True)
        base = hf_hub_download(a.repo, "model.safetensors", revision=a.revision or None)
    print(unpack(a.delta, a.out, base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
