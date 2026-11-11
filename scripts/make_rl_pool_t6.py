"""Pool de prompts do RL a partir do T6-Lumi: o pool com gabarito oficial (`data/grpo_prompts_gold.jsonl`) sem os itens que o SFT do T6-Lumi já viu.

O SFT do `thor_pt` (a base do T6-Lumi) treinou com 1.719 problemas do GSM8K-PT de treino, matemática do GigaVerbo e ciência traduzida; como o pool de RL
tem o GSM8K-PT de treino inteiro, parte dos prompts de RL já tem raciocínio decorado (por esse motivo a taxa de acerto deles não mede dificuldade).
Critério (o mesmo da descontaminação da H74): um item do pool sai se >= 10% dos seus 8-gramas de palavras (e >= 3) estiverem em algum prompt do SFT, depois
de tirar os 8-gramas de molde de instrução (presentes em mais de 150 prompts). Confere também que nenhum item restante divide 8-gramas com as provas novas.

  PYTHONPATH=src python scripts/make_rl_pool_t6.py        # grava data/grpo_prompts_t6.jsonl e data/grpo_prompts_t6.report.json
"""
import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from lumi_ft.data.decontaminate import flatten_strings, ngram_hashes
from lumi_ft.modeling.load import load_tokenizer

MOLDE = 150  # 8-grama que aparece em mais prompts que isto é molde de instrução, não conteúdo
NOVAS = ["oab_novas", "enem_2024", "mmlu_pt_exatas", "mmlu_pt_enade", "mmlu_pt_concursos"]


def sft_prompts(parquet: str) -> list[tuple[str, str]]:
    """(grupo, texto do prompt) de cada exemplo do SFT: o parquet guarda só os ids, e `n_prompt` diz onde o prompt termina."""
    tok = load_tokenizer()
    df = pd.read_parquet(parquet)
    return [(g, tok.decode(list(ids)[: int(n)], skip_special_tokens=False)) for ids, n, g in zip(df["input_ids"], df["n_prompt"], df["group"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/grpo_prompts_gold.jsonl")
    ap.add_argument("--sft", default="data/thor_pt.parquet", help="dados de SFT do T6-Lumi (o `thor_pt`)")
    ap.add_argument("--out", default="data/grpo_prompts_t6.jsonl")
    ap.add_argument("--min-frac", type=float, default=0.10)
    ap.add_argument("--min-hits", type=int, default=3)
    a = ap.parse_args()

    pool = [json.loads(line) for line in Path(a.pool).read_text().splitlines() if line.strip()]
    hp = [set(ngram_hashes(p["prompt"])) for p in pool]
    molde = {h for h, c in collections.Counter(h for s in hp for h in s).items() if c > MOLDE}
    prompts = sft_prompts(a.sft)
    por_grupo: dict[str, set[int]] = collections.defaultdict(set)
    freq: collections.Counter = collections.Counter()
    for g, t in prompts:
        hs = set(ngram_hashes(t))
        freq.update(hs)
        por_grupo[g] |= hs
    molde |= {h for h, c in freq.items() if c > MOLDE}

    kept, saiu = [], collections.Counter()
    for p, s in zip(pool, hp):
        s = s - molde
        vistos = None
        for g, hs in por_grupo.items():
            k = len(s & (hs - molde))
            if s and k >= a.min_hits and k / len(s) >= a.min_frac:
                vistos = g
                break
        if vistos:
            saiu[f"{p['kind']}:{vistos}"] += 1
        else:
            kept.append(p)

    # verificação: nenhum item restante divide 8-gramas com as provas novas (checagem externa preservada)
    novas = set()
    for t in NOVAS:
        for line in Path(f"data/novas_provas/{t}.jsonl").read_text().splitlines():
            novas.update(ngram_hashes("\n".join(s for s in flatten_strings(json.loads(line)) if len(s) > 3)))
    cruzam = 0
    for p in kept:
        s = set(ngram_hashes(p["prompt"])) - molde
        k = len(s & novas)
        cruzam += bool(s and k >= a.min_hits and k / len(s) >= a.min_frac)

    Path(a.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept), encoding="utf-8")
    rel = {
        "pool_de_origem": a.pool, "itens_de_origem": len(pool), "por_tipo_origem": dict(collections.Counter(p["kind"] for p in pool)),
        "retirados_por_grupo_do_sft": dict(saiu), "retirados": sum(saiu.values()),
        "itens_restantes": len(kept), "por_tipo_restante": dict(collections.Counter(p["kind"] for p in kept)),
        "restantes_que_cruzam_com_provas_novas": cruzam, "criterio": f">= {a.min_frac:.0%} dos 8-gramas (e >= {a.min_hits}) em algum prompt do SFT; molde > {MOLDE} prompts",
        "saida": a.out,
    }
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rel, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
