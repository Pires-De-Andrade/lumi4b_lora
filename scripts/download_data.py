"""Baixa os dados da fase 1: shards do GigaVerbo-v2 SFT (exemplos sem raciocínio) e as provas de avaliação.

As provas servem à descontaminação (D-05) e ao harness de múltipla escolha. Falhas não interrompem:
cada conjunto grava o resultado em runs/data/download_report.json (tamanho, colunas, erro).
Uso: scripts/detach.sh download-data PYTHONPATH=src python scripts/download_data.py
"""
import json
import sys
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

GV = "Polygl0t/gigaverbo-v2-sft"
# Um shard por subset basta: a amostra sem raciocínio é ~11,6 M de tokens (25% do total).
GV_FILES = {
    "general": ["general/train-00000-of-00003.parquet"],
    "structured": ["structured/train-00000-of-00001.parquet"],
    "system_prompts": ["system_prompts/train-00000-of-00001.parquet"],
    "rewriting": ["rewriting/train-00000-of-00001.parquet"],
    "summarization": ["summarization/train-00000-of-00001.parquet"],
    "math": ["math/train-00000-of-00001.parquet"],
    "retrieval": ["retrieval/train-00000-of-00004.parquet"],
}
# (id, config, split): conjuntos de teste/validação do PRD
EVALS = [
    ("Polygl0t/gsm8k-pt", None, "test"),
    ("Polygl0t/gsm8k-pt", None, "train"),
    ("li-lab/MMLU-ProX", "pt", "test"),
    ("li-lab/MMLU-ProX", "pt", "validation"),
    ("eduagarcia/enem_challenge", None, None),
    ("nicholasKluge/BLUEX_without_images", None, None),
    ("eduagarcia/oab_exams", None, None),
    ("CohereLabs/Global-MMLU", "pt", "test"),
    ("CohereLabs/Global-MMLU", "pt", "dev"),
    ("openai/MMMLU", "PT_BR", "test"),
    ("CohereLabs/include-base-44", "Portuguese", "test"),
    ("CohereLabs/include-base-44", "Portuguese", "validation"),
]


def main() -> None:
    report = {"gigaverbo": {}, "evals": {}}
    out = Path("runs/data/download_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for cfg, files in GV_FILES.items():
        for f in files:
            t = time.time()
            try:
                p = hf_hub_download(GV, f, repo_type="dataset")
                report["gigaverbo"][f] = {"path": p, "mb": round(Path(p).stat().st_size / 1e6), "s": round(time.time() - t)}
            except Exception as e:  # noqa: BLE001
                report["gigaverbo"][f] = {"error": repr(e)[:300]}
            print(f, report["gigaverbo"][f], flush=True)
    for name, cfg, split in EVALS:
        key = f"{name}|{cfg}|{split}"
        try:
            ds = load_dataset(name, cfg, split=split) if split else load_dataset(name, cfg)
            if split:
                report["evals"][key] = {"rows": len(ds), "columns": list(ds.features)}
            else:
                report["evals"][key] = {sp: {"rows": len(d), "columns": list(d.features)} for sp, d in ds.items()}
        except Exception as e:  # noqa: BLE001
            report["evals"][key] = {"error": repr(e)[:300]}
        print(key, json.dumps(report["evals"][key], ensure_ascii=False)[:300], flush=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())
