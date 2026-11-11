"""Monta (e, com --push, envia) o repositório do T6-Lumi no Hugging Face.

Só vai o que o modelo precisa para rodar, mais o card e as figuras: os pesos, a configuração, o tokenizador, o chat template, o código do modelo (copiados
do Lumi-4B na revisão fixada) e o README.md com as figuras. Nada de pacotes (`handoff/`), scripts, estudos ou dados: a lista permitida está em ALLOWED e a
montagem recusa qualquer arquivo fora dela.

Sem --push, só monta a pasta e confere (não envia nada). Com --push, cria o repositório **privado** e envia; tornar público é um passo à parte, depois
de olhar a página.

  # na máquina que tem os pesos (a 4090): precisa de huggingface_hub; o token vem do login do Hugging Face
  python hf/t6-lumi/publicar.py --weights runs/merged/thor_pt_antidoom_of/model.safetensors --repo-id ORG/NOME
  python hf/t6-lumi/publicar.py --weights runs/merged/thor_pt_antidoom_of/model.safetensors --repo-id ORG/NOME --push
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

LUMI_REPO = "ceia-br/lumi-4b"
LUMI_REVISION = "27b30f40b930fd05cee5e1c1f8790f70fa5b816d"  # Lumi-4B v0.2
WEIGHTS_SHA256 = "1ea6bea57a56a67ebb70c1702fa0ae4f7dbde6967830c5d3961af3db43e944e5"  # handoff/spark1/PESOS.sha256
FROM_LUMI = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "configuration_nemotron_h.py",
             "modeling_nemotron_h.py", "nano_v3_reasoning_parser.py", "__init__.py"]
CARD = "README.md"
FIGURES = ["fig1_benchmarks.svg", "fig2_paired_difference.svg", "fig3_closure.svg", "fig4_decomposition.svg", "fig5_recipe.svg",
           "fig6_new_exams.svg", "fig7_new_exams_paired_difference.svg", "fig8_new_exams_closure.svg", "fig9_new_exams_decomposition.svg"]
ALLOWED = {"model.safetensors", CARD, *FROM_LUMI, *FIGURES}
PENDING = re.compile(r"DECISION|TODO")


def plan(card_dir: Path) -> dict[str, str]:
    """Nome no repositório -> de onde vem. Falha se faltar o card ou uma figura."""
    out = {"model.safetensors": "weights", **{n: f"{LUMI_REPO}@{LUMI_REVISION[:8]}" for n in FROM_LUMI}}
    for n in (CARD, *FIGURES):
        if not (card_dir / n).is_file():
            raise SystemExit(f"falta {card_dir / n}")
        out[n] = str(card_dir / n)
    extra = set(out) - ALLOWED
    if extra:
        raise SystemExit(f"fora da lista permitida: {sorted(extra)}")
    return out


def pending_markers(card_text: str) -> list[str]:
    """Linhas do card que ainda pedem decisão (DECISION) ou preenchimento (TODO)."""
    return [f"{i}: {line.strip()[:110]}" for i, line in enumerate(card_text.splitlines(), 1) if PENDING.search(line)]


def render_card(card_text: str, repo_id: str) -> str:
    return card_text.replace("<repo-id>", repo_id)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def stage(card_dir: Path, weights: Path, repo_id: str, staging: Path) -> Path:
    from huggingface_hub import hf_hub_download

    plan(card_dir)
    got = sha256_of(weights)
    if got != WEIGHTS_SHA256:
        raise SystemExit(f"sha256 dos pesos diferente do esperado ({got} em vez de {WEIGHTS_SHA256}): não é o T6-Lumi")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for n in FROM_LUMI:
        shutil.copy(hf_hub_download(LUMI_REPO, n, revision=LUMI_REVISION), staging / n)
    shutil.copy(weights, staging / "model.safetensors")
    for n in FIGURES:
        shutil.copy(card_dir / n, staging / n)
    (staging / CARD).write_text(render_card((card_dir / CARD).read_text(encoding="utf-8"), repo_id), encoding="utf-8")
    names = {p.name for p in staging.iterdir()}
    if names != ALLOWED:
        raise SystemExit(f"pasta montada difere da lista permitida: {sorted(names ^ ALLOWED)}")
    return staging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True, help="model.safetensors do T6-Lumi (sha256 conferido)")
    ap.add_argument("--repo-id", required=True, help="ORG/NOME no Hugging Face")
    ap.add_argument("--staging", default="runs/hf_stage/t6-lumi", help="pasta de montagem (fora do Git)")
    ap.add_argument("--push", action="store_true", help="criar o repositório privado e enviar")
    a = ap.parse_args()
    card_dir = Path(__file__).resolve().parent
    pend = pending_markers((card_dir / CARD).read_text(encoding="utf-8"))
    if pend:
        print(f"{len(pend)} pendência(s) no card:")
        print("\n".join("  " + p for p in pend))
    out = stage(card_dir, Path(a.weights), a.repo_id, Path(a.staging))
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"montado em {out}: {len(list(out.iterdir()))} arquivos, {total / 1e9:.2f} GB (só a lista permitida)")
    if not a.push:
        print("sem --push: nada foi enviado")
        return 0
    if pend:
        raise SystemExit("recusado: resolva as pendências do card antes de enviar")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo_id, repo_type="model", private=True, exist_ok=False)
    api.upload_folder(folder_path=str(out), repo_id=a.repo_id, repo_type="model", commit_message="Add T6-Lumi")
    print(f"enviado (privado): https://huggingface.co/{a.repo_id}  | confira a página e só depois torne público")
    return 0


if __name__ == "__main__":
    sys.exit(main())
