"""Interpolação de pesos entre dois modelos com as mesmas chaves e formas (WiSE-FT): out = alpha * A + (1 - alpha) * B, tensor a tensor,
em float32, gravado no tipo de A. Uso típico: A = o modelo nosso fundido, B = o Lumi original; alpha = 1 devolve A, alpha = 0 devolve B.
Copia de A os arquivos de configuração e do tokenizador e grava `interp_report.json` (alpha, sha256 dos arquivos de entrada, tensores
alterados, maior diferença absoluta contra A e contra B).
  PYTHONPATH=src python scripts/interpolate_models.py --a runs/merged/thor_pt_antidoom_of --b lumi --alpha 0.75 --out runs/merged/t6_interp075
`--b lumi` resolve o Lumi original na revisão fixada (cache local, sem baixar).
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def resolve(path: str) -> Path:
    if path == "lumi":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from huggingface_hub import snapshot_download

        from lumi_ft.modeling.load import MODEL_ID, REVISION
        return Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    return Path(path)


def tensors(d: Path) -> dict[str, tuple[Path, str]]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                assert k not in out, f"chave repetida {k}"
                out[k] = (f, k)
    if not out:
        raise SystemExit(f"nenhum .safetensors em {d}")
    return out


def interpolate(ta: torch.Tensor, tb: torch.Tensor, alpha: float) -> torch.Tensor:
    if ta.shape != tb.shape:
        raise ValueError(f"formas diferentes: {tuple(ta.shape)} e {tuple(tb.shape)}")
    if not ta.is_floating_point():
        if not torch.equal(ta, tb):
            raise ValueError("tensor não flutuante difere entre os modelos")
        return ta.clone()
    return (alpha * ta.float() + (1.0 - alpha) * tb.float()).to(ta.dtype)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True, help="pasta ou 'lumi' (o Lumi original, revisão fixada)")
    ap.add_argument("--alpha", type=float, required=True, help="peso de A, em [0, 1]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sha", action="store_true", help="não calcula o sha256 das entradas (teste)")
    a = ap.parse_args()
    if not 0.0 <= a.alpha <= 1.0:
        raise SystemExit("--alpha fora de [0, 1]")
    da, db, out = resolve(a.a), resolve(a.b), Path(a.out)
    ka, kb = tensors(da), tensors(db)
    if set(ka) != set(kb):
        raise SystemExit(f"chaves diferentes: só A {sorted(set(ka) - set(kb))[:5]}, só B {sorted(set(kb) - set(ka))[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    res, changed, max_da, max_db = {}, 0, 0.0, 0.0
    handles = {}
    get = lambda f, k: (handles.setdefault(f, safe_open(str(f), "pt"))).get_tensor(k)
    for k in sorted(ka):
        ta, tb = get(*ka[k]), get(*kb[k])
        t = interpolate(ta, tb, a.alpha)
        res[k] = t
        if t.is_floating_point() and not torch.equal(ta, tb):
            changed += 1
            max_da = max(max_da, float((t.float() - ta.float()).abs().max()))
            max_db = max(max_db, float((t.float() - tb.float()).abs().max()))
    save_file(res, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in da.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.endswith(".index.json") and f.name not in ("merge_report.json", "interp_report.json"):
            shutil.copy2(f, out / f.name)
    rep = {"a": str(da), "b": str(db), "alpha": a.alpha, "keys": len(res), "tensores_diferentes_entre_a_e_b": changed,
           "max_abs_contra_a": max_da, "max_abs_contra_b": max_db}
    if not a.no_sha:
        rep["sha256"] = {str(f): sha(f) for f in sorted({p for p, _ in list(ka.values()) + list(kb.values())})}
    (out / "interp_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "sha256"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
