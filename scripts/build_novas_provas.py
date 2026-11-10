"""Monta as provas novas do plano de 10/11 (checagem externa): `oab_novas`, `enem_2024`, `mmlu_pt_exatas`, `mmlu_pt_enade`, `mmlu_pt_concursos`.

Fontes (todas para avaliação; nada disto entra em treino): russ7/oab_exams_2011_2025_combined (MIT), maritaca-ai/enem config 2024 (Apache-2.0),
Portuguese-MMLU/mmlu-pt-knowledge-annotated (licença não declarada). Para cada prova: converte as linhas (src/lumi_ft/eval/novas.py), tira o que
divide 10% ou mais dos 8-gramas (e pelo menos 3) com a suíte de teste e a validação, e o mesmo com os dados de treino do candidato `thor_pt_antidoom_of`
(SFT `thor_pt`, prompts do Antidoom e do RL), tira os prompts que não cabem em 4.096 tokens (o orçamento de geração não pode encolher), e, nas
fatias do MMLU-PT, sorteia 1.000 itens com a semente 0 depois da descontaminação. Grava em data/novas_provas/: <tarefa>.jsonl, <tarefa>_ids.json e
report.json (contagens de cada etapa, revisão de cada fonte e o tamanho dos prompts).
  source env.sh; PYTHONPATH=src python scripts/build_novas_provas.py
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from lumi_ft.data.decontaminate import NgramIndex, build_index
from lumi_ft.eval import generate as G
from lumi_ft.eval import gsm8k as K
from lumi_ft.eval import mcq as M
from lumi_ft.eval import novas as N
from lumi_ft.modeling.load import load_tokenizer


def revision(repo: str) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception as e:  # sem rede: registra e segue com o cache
        return f"indisponível ({type(e).__name__})"


def suite_index() -> tuple[NgramIndex, set[str], int]:
    rows, oab_q = [], set()
    for name in ("oab", "enem", "bluex", "mmlu_prox_test"):
        its = M.LOADERS[name]()
        rows += its
        if name == "oab":
            oab_q = {N.norm_text(i["question"]) for i in its}
    for f in ("data/val_mcq.jsonl", "data/val_gsm8k.jsonl"):
        rows += [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
    rows += [{"q": it["question"]} for it in K.load_items(None)[0]]
    return build_index(rows), oab_q, len(rows)


def training_index(tok) -> tuple[NgramIndex, dict]:
    import pandas as pd

    idx, counts = NgramIndex(), {}
    df = pd.read_parquet("data/thor_pt.parquet", columns=["input_ids"])
    for ids in df["input_ids"]:
        idx.add(tok.decode([int(i) for i in ids], skip_special_tokens=True))
    counts["thor_pt (SFT, sequências)"] = len(df)
    for f in ("data/antidoom_prompts.jsonl", "data/grpo_prompts.jsonl", "data/grpo_prompts_gold.jsonl"):
        n = 0
        for line in Path(f).read_text().splitlines():
            if line.strip():
                idx.add(json.loads(line)["prompt"])
                n += 1
        counts[f] = n
    return idx, counts


def prompt_tokens(tok, its: list[dict]) -> list[int]:
    return [len(tok(G.render_prompt(tok, M.build_prompt(it), thinking=True), add_special_tokens=False)["input_ids"]) for it in its]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=N.OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-mmlu", type=int, default=1000)
    ap.add_argument("--max-prompt", type=int, default=4096)
    a = ap.parse_args()
    from datasets import load_dataset

    t0 = time.time()
    tok = load_tokenizer()
    suite, suite_oab_q, n_suite = suite_index()
    train, train_counts = training_index(tok)
    print(f"índices: suíte+validação {len(suite)} 8-gramas ({n_suite} itens); treino {len(train)} 8-gramas ({train_counts}); {time.time() - t0:.0f} s", flush=True)
    indexes = {"suite_teste_e_validacao": suite, "treino_do_candidato": train}
    report = {"seed": a.seed, "max_prompt_tokens": a.max_prompt, "treino_indexado": train_counts, "tarefas": {}}

    oab_rows = list(load_dataset("russ7/oab_exams_2011_2025_combined")["train"])
    enem_rows = list(load_dataset("maritaca-ai/enem", "2024")["train"])
    mmlu_rows = list(load_dataset("Portuguese-MMLU/mmlu-pt-knowledge-annotated")["train"])
    old_exams = {i["id"].split(":", 1)[1].rsplit("_", 1)[0] for i in M.LOADERS["oab"]()}

    jobs = [("oab_novas", "russ7/oab_exams_2011_2025_combined", len(oab_rows), N.parse_oab_novas(oab_rows, old_exams), None),
            ("enem_2024", "maritaca-ai/enem (2024)", len(enem_rows), N.parse_enem_2024(enem_rows), None)]
    for name, exams in N.MMLU_PT_SLICES.items():
        jobs.append((name, "Portuguese-MMLU/mmlu-pt-knowledge-annotated", sum(r["exam"] in exams for r in mmlu_rows),
                     N.parse_mmlu_pt(mmlu_rows, exams, name), a.n_mmlu))
    for name, fonte, n_raw, (items, parse_drops), n_sample in jobs:
        rep = {"fonte": fonte, "revisao": revision(fonte.split(" ")[0]), "n_bruto": n_raw, "descartes_de_formato": parse_drops, "n_convertido": len(items)}
        if name == "oab_novas":
            items, n_same = N.drop_same_text(items, suite_oab_q)
            rep["texto_igual_ao_teste_oab"] = n_same
        items, removed, strict_ids = N.decontaminate(items, indexes)
        rep["removidos_por_8grama"] = removed
        rep["criterio"] = "remove se >= 10% dos 8-gramas do item (e >= 3) estão no índice; o conjunto estrito (nenhum 8-grama em comum) vai em <tarefa>_estrito_ids.json"
        ntok = prompt_tokens(tok, items)
        longos = {it["id"] for it, n in zip(items, ntok) if n > a.max_prompt}
        items = [it for it in items if it["id"] not in longos]
        rep["prompt_acima_do_limite"] = len(longos)
        rep["n_pool"] = len(items)
        if n_sample:
            items = M.subsample(items, n_sample, a.seed)
            rep["amostra"] = {"n": len(items), "seed": a.seed}
        ntok = sorted(prompt_tokens(tok, items))
        rep["n_final"] = len(items)
        rep["tokens_do_prompt"] = {"media": round(sum(ntok) / len(ntok)), "p99": ntok[int(0.99 * len(ntok)) - 1], "max": ntok[-1]}
        rep["gabaritos"] = dict(sorted(Counter(it["gold"] for it in items).items()))
        rep["alternativas"] = dict(sorted(Counter(len(it["options"]) for it in items).items()))
        rep["menciona_figura_grafico_tabela"] = sum(bool(N.FIGURE_WORDS.search(it["question"])) for it in items)
        N.write_task(a.out, name, items)
        estrito = [it["id"] for it in items if it["id"] in strict_ids]
        (Path(a.out) / f"{name}_estrito_ids.json").write_text(json.dumps(estrito))
        rep["n_estrito"] = len(estrito)
        report["tarefas"][name] = rep
        print(name, json.dumps(rep, ensure_ascii=False), flush=True)
    (Path(a.out) / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"pronto em {time.time() - t0:.0f} s: {a.out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
