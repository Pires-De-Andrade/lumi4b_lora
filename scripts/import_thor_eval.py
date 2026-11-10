"""Importa a devolução de um pacote de avaliação rodado em outra máquina para runs/eval/, nos nomes que as tabelas usam.

Thor (handoff/thor5, o padrão; Lumi e Nemotron):
  base_novas_thor        -> base_thor        (junta às provas da suíte de 09/11 que já estão lá)
  nemotron4b_novas_thor  -> nemotron4b_thor  (provas novas)
  nemotron4b_suite_thor  -> nemotron4b_thor  (suíte do PRD)
Spark (handoff/spark1, `--origem spark`): toda pasta runs/eval/<nome>_spark da devolução entra com o mesmo nome (base_spark,
thor_pt_antidoom_of_spark, base_8k_spark, ...); a devolução parcial pode ser importada de novo, e o que já está lá só é aceito se for idêntico.
Nada é sobrescrito: um arquivo que já existe no destino precisa ser idêntico, e uma chave do `summary.json` que já existe precisa ter o mesmo
resultado; senão para com erro. Os campos `_engine` (modelo, memória, decodificação) das origens ficam em `_engine_<origem>` no destino.
  python scripts/import_thor_eval.py --tar novas_thor_saida.tar.gz [--sha256 <hash>]
  python scripts/import_thor_eval.py --origem spark --tar spark1_saida.tar.gz [--sha256 <hash>]
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


def destinos(origem: str, tmp: Path) -> list[tuple[str, Path]]:
    """Pares (nome no destino, pasta de origem) da devolução extraída em `tmp`."""
    if origem == "spark":
        return [(p.name, p) for p in sorted(tmp.glob("**/runs/eval/*_spark")) if p.is_dir()]
    out = []
    for nome, destino in MAPA.items():
        achados = sorted(tmp.rglob(nome))
        if achados:
            out.append((destino, achados[0]))
        else:
            print(f"{nome}: não veio na devolução")
    return out


def chave_do_arquivo(nome: str) -> str | None:
    """`<tarefa>_<modo>.jsonl` ou `<tarefa>_<modo>_scores.json` -> `<tarefa>|<modo>` (None se o nome não segue o padrão)."""
    base = nome.removesuffix("_scores.json") if nome.endswith("_scores.json") else nome.removesuffix(".jsonl") if nome.endswith(".jsonl") else None
    for modo in ("nothinking", "thinking"):
        if base and base.endswith("_" + modo):
            return f"{base[: -len(modo) - 1]}|{modo}"
    return None


def merge_eval_dir(src: Path, dst: Path) -> dict:
    """Copia as saídas de `src` para `dst` e junta o summary.json. Devolve {arquivos, chaves} novos.
    Só entra o que está completo: um arquivo de tarefa e modo que não consta no summary.json da origem (avaliação em andamento numa entrega
    parcial) é ignorado, para a entrega seguinte poder trazê-lo inteiro."""
    dst.mkdir(parents=True, exist_ok=True)
    copiados = 0
    prontos = {k for k in json.loads((src / "summary.json").read_text()) if "|" in k}
    for f in sorted(src.iterdir()):
        if f.name == "summary.json" or not f.is_file():
            continue
        chave = chave_do_arquivo(f.name)
        if chave is not None and chave not in prontos:
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
    ap.add_argument("--origem", choices=["thor", "spark"], default="thor")
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
    pares = destinos(a.origem, tmp)
    if not pares:
        print("nada para importar na devolução")
    for destino, origem in pares:
        print(origem.name, "->", destino, merge_eval_dir(origem, Path(a.root) / destino))
    return 0


if __name__ == "__main__":
    sys.exit(main())
