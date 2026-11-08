"""Auditoria ESTRUTURAL das questões de múltipla escolha do `gspo_gold` (sem julgar o conteúdo, que exige especialista):
os IDs que os 8 ciclos usam (determinístico: `select_prompts`), a distribuição do que é usado contra o pool, e marcadores automáticos
de problema em cada questão (alternativas repetidas, enunciado repetido, negação do inglês que sumiu no português, remissões do tipo
"A e B" ou "todas as anteriores" que dependem da ordem das alternativas, texto vazio ou gabarito fora do intervalo).
Escreve <out>/{used_ids.json, report.json, sample.md}; `sample.md` é uma amostra estratificada (fonte x nº de alternativas x letra)
com inglês e português lado a lado para quem for revisar à mão.

Uso: PYTHONPATH=src python scripts/audit_gold_questions.py --questions data/thor3/qfull/out/questions_pt.jsonl --prompts data/grpo_prompts_gold.jsonl --out data/audit_gold
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lumi_ft.data.question_quality import EXCLUDE, quality_flags

def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="out/questions_pt.jsonl da Thor")
    ap.add_argument("--prompts", default="data/grpo_prompts_gold.jsonl")
    ap.add_argument("--out", default="data/audit_gold")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--frac-mcq", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=60)
    a = ap.parse_args()
    from lumi_ft.train import grpo as G

    byid = {}
    for line in Path(a.questions).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            byid[r["id"]] = r
    prompts = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    pool = [p for p in prompts if p["kind"] == "mcq"]
    used_ids = []
    for it in range(a.cycles):
        used_ids += [p["id"] for p in G.select_prompts(prompts, it, a.n_prompts, a.frac_mcq, a.seed) if p["kind"] == "mcq"]
    used = set(used_ids)
    qnorm = Counter(norm(byid[p["id"][4:]]["question_pt"]) for p in pool)

    def row(p):
        r = byid[p["id"][4:]]
        fl = quality_flags(r)
        if qnorm[norm(r["question_pt"])] > 1:
            fl.append("enunciado_repetido_no_pool")
        return r, fl

    rep = {"pool": len(pool), "usadas_em_%d_ciclos" % a.cycles: len(used), "ids_unicos_usados": len(set(used_ids)), "marcadores": {}}
    per = {"pool": Counter(), "usadas": Counter()}
    examples = defaultdict(list)
    dist = {"pool": defaultdict(Counter), "usadas": defaultdict(Counter)}
    for p in pool:
        r, fl = row(p)
        for name, sel in (("pool", True), ("usadas", p["id"] in used)):
            if not sel:
                continue
            per[name]["total"] += 1
            dist[name]["fonte"][r["source"]] += 1
            dist[name]["letra"][p["gold"]] += 1
            dist[name]["n_alternativas"][p["n_options"]] += 1
            dist[name]["dominio"][r["category"]] += 1
            for f in fl:
                per[name][f] += 1
                if name == "usadas" and len(examples[f]) < 8:
                    examples[f].append(p["id"])
            per[name]["com_algum_marcador"] += bool(fl)
            per[name]["seria_excluida_pelo_filtro"] += any(f in EXCLUDE for f in fl)
    rep["marcadores"] = {k: dict(v) for k, v in per.items()}
    rep["distribuicao"] = {n: {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in d.items()} for n, d in dist.items()}
    rep["exemplos_de_ids_usados_por_marcador"] = dict(examples)

    # amostra estratificada (fonte x faixa de alternativas x letra) das questões USADAS, com inglês e português lado a lado
    strata = defaultdict(list)
    for p in pool:
        if p["id"] in used:
            r = byid[p["id"][4:]]
            strata[(r["source"], min(p["n_options"], 5), p["gold"] if p["gold"] in "ABCD" else "E+")].append(p)
    rnd = random.Random(1)
    keys = sorted(strata)
    pick = []
    while len(pick) < a.sample and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(pick) < a.sample:
                pick.append(strata[k].pop(rnd.randrange(len(strata[k]))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Amostra estratificada das questões usadas no gspo_gold (inglês e português lado a lado)\n",
          "Marcadores automáticos entre colchetes. O gabarito é o do dataset (MedMCQA: de provas; WebInstruct-verified: resposta da web verificada por modelo).\n"]
    for n, p in enumerate(pick, 1):
        r, fl = row(p)
        L = "ABCDEFGHIJ"
        md.append(f"\n## {n}. `{p['id']}` · {r['source']} · {r['category']} · gabarito **{p['gold']}** {fl if fl else ''}\n")
        md.append(f"**EN:** {r['question_en'].strip()}\n")
        md += [f"- {L[i]}) {o}" for i, o in enumerate(r["options_en"])]
        md.append(f"\n**PT:** {r['question_pt'].strip()}\n")
        md += [f"- {L[i]}) {o}" for i, o in enumerate(r["options_pt"])]
    (out / "sample.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "used_ids.json").write_text(json.dumps({"mcq_ids": sorted(used), "ciclos": a.cycles, "n_prompts": a.n_prompts, "frac_mcq": a.frac_mcq, "seed": a.seed}, indent=1))
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: rep[k] for k in ("pool", "marcadores")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
