"""Importa a devolução do pacote handoff/thor5 (avaliações do Lumi e do Nemotron rodadas na Thor) para runs/eval/, nos nomes que as tabelas usam.

  base_novas_thor        -> base_thor        (junta às provas da suíte de 09/11 que já estão lá)
  nemotron4b_novas_thor  -> nemotron4b_thor  (provas novas)
  nemotron4b_suite_thor  -> nemotron4b_thor  (suíte do PRD)
Nada é sobrescrito: um arquivo que já existe no destino precisa ser idêntico, e uma chave do `summary.json` que já existe precisa ter o mesmo
resultado; senão para com erro. Os campos `_engine` (modelo, memória, decodificação) das origens ficam em `_engine_<origem>` no destino.
  python scripts/import_thor_eval.py --tar novas_thor_saida.tar.gz [--sha256 <hash>]
  python scripts/import_thor_eval.py --src runs/_thor_in/runs/eval/base_novas_thor --dst runs/eval/base_thor   (uma pasta só)
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

MAPA = {"base_novas_thor": "base_thor", "nemotron4b_novas_thor": "nemotron4b_thor", "nemotron4b_suite_thor": "nemotron4b_thor"}


def merge_eval_dir(src: Path, dst: Path) -> dict:
    """Copia as saídas de `src` para `dst` e junta o summary.json. Devolve {arquivos, chaves} novos."""
    dst.mkdir(parents=True, exist_ok=True)
    copiados = 0
    for f in sorted(src.iterdir()):
        if f.name == "summary.json" or not f.is_file():
            continue
        alvo = dst / f.name
        if alvo.exists():
            if alvo.read_bytes() != f.read_bytes():
                raise SystemExit(f"{alvo} já existe e é diferente de {f}: nada foi sobrescrito")
            continue
        shutil.copy2(f, alvo)
        copiados += 1
    novo = json.loads((src / "summary.json").read_text())
    p = dst / "summary.json"
    atual = json.loads(p.read_text()) if p.exists() else {}
    chaves = 0
    for k, v in novo.items():
        if k == "_engine":
            atual[f"_engine_{src.name}"] = v
            atual.setdefault("_engine", v)
        elif k in atual:
            if atual[k] != v:
                raise SystemExit(f"{p}: a chave {k} já existe com outro resultado")
        else:
            atual[k] = v
            chaves += 1
    p.write_text(json.dumps(atual, ensure_ascii=False, indent=1))
    return {"arquivos": copiados, "chaves": chaves}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar")
    ap.add_argument("--sha256", default="")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--root", default="runs/eval")
    a = ap.parse_args()
    if a.src:
        print(merge_eval_dir(Path(a.src), Path(a.dst)))
        return 0
    if not a.tar:
        raise SystemExit("use --tar ou --src/--dst")
    tar = Path(a.tar)
    if a.sha256 and sha256(tar) != a.sha256:
        raise SystemExit(f"sha256 de {tar} não confere")
    tmp = Path("runs/_thor_in") / tar.stem
    tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(tmp, filter="data")
    for origem, destino in MAPA.items():
        achados = sorted(tmp.rglob(origem))
        if not achados:
            print(f"{origem}: não veio na devolução")
            continue
        print(origem, "->", destino, merge_eval_dir(achados[0], Path(a.root) / destino))
    return 0


if __name__ == "__main__":
    sys.exit(main())
