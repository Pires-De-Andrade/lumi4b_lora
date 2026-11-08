"""Gera estudos/resultados_auto.md: a tabela de todas as rodadas avaliadas, direto de runs/eval/*/ (sem copiar números à mão).

Cada linha: nota estrita na validação (GSM8K limpo, 676 itens; múltipla escolha, 385 itens), com e sem thinking, e a diferença contra
o Lumi com IC 95% (bootstrap pareado por item, 10.000 reamostras); depois a forma da saída (fecha `</think>`, traço em português,
truncadas, mediana de tokens). Linhas sem a avaliação completa aparecem com "–". Rode de novo quando houver rodada nova:
  PYTHONPATH=src python scripts/gen_estudos_resultados.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_runs import paired_bootstrap  # noqa: E402

ROOT = Path("runs/eval")
CLEAN = set(json.loads(Path("data/val_gsm8k_clean_ids.json").read_text()))
REF = "base"

# (grupo, nome da avaliação, descrição curta)
RUNS = [
    ("Referências (sem treino nosso)", [("base", "Lumi original"), ("nemotron4b", "Nemotron 3 Nano 4B (pai do Lumi)")]),
    ("SFT: dados e professor", [("viab", "viab: 10 mil traços do GigaVerbo, 1 época"), ("adalberto", "adalberto: traços do Qwen3-30B (GSM8K), lr 1e-4"),
                                ("adalberto_lr5e5", "adalberto com lr 5e-5"), ("adalberto_L2048", "adalberto com limite de 2.048 tokens"),
                                ("selfdist", "selfdist: traços do próprio Lumi com prefill em português"), ("adalberto_dora", "adalberto com DoRA")]),
    ("SFT: tamanho do efeito do adaptador", [("adalberto_s050", "adalberto, efeito do adaptador x0,5"), ("adalberto_s075", "adalberto, efeito do adaptador x0,75")]),
    ("SFT: só os MLPs", [("adalberto_mlp", "adalberto_mlp: LoRA só em up_proj e down_proj (34 módulos), treinado do zero")]),
    ("RL (a partir do adalberto)", [("grpo1", "grpo1: 8 ciclos, T 0,8"), ("grpo2", "grpo2: 4 ciclos, T 0,6, top-p 0,95, truncadas fora da perda"),
                                    ("grpo3", "grpo3: Dr. GRPO + 70% múltipla escolha"), ("grpo4", "grpo4: KL 0,04"),
                                    ("raft1", "raft1: só respostas certas"), ("gspo1_c4", "gspo1: razão por sequência, 4 ciclos"), ("gspo1", "gspo1: 8 ciclos")]),
]
LAYERS = [("adalberto", "adaptador completo (71 módulos)"), ("adalberto_L00_20", "só camadas 0 a 20"), ("adalberto_L21_41", "só camadas 21 a 41"),
          ("adalberto_noMLP", "sem MLP (Mamba e atenção)"), ("adalberto_MLPonly", "só MLP (do adaptador completo, sem retreinar)")]


def load(name: str, task: str, mode: str):
    p = ROOT / name / f"{task}_{mode}_scores.json"
    if not p.exists():
        return None
    d = {r["id"]: r for r in json.loads(p.read_text())}
    return {k: v for k, v in d.items() if k in CLEAN} if task == "gsm8k_val" else d


def pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",")


def cell(name: str, task: str, mode: str) -> str:
    d, r = load(name, task, mode), load(REF, task, mode)
    if d is None:
        return "–"
    ids = sorted(d)
    acc = sum(int(d[i]["strict_ok"]) for i in ids) / len(ids)
    if name == REF or r is None or set(r) != set(d):
        return pct(acc)
    diff, lo, hi = paired_bootstrap([int(d[i]["strict_ok"]) for i in ids], [int(r[i]["strict_ok"]) for i in ids])
    mark = " ▲" if lo > 0 else (" ▼" if hi < 0 else "")
    return f"{pct(acc)} ({100 * diff:+.1f}){mark}".replace(".", ",")


def form(name: str, task: str) -> str:
    d = load(name, task, "thinking")
    if d is None:
        return "–"
    rows = list(d.values())
    n = len(rows)
    langs = [r["trace_language"] for r in rows if r["trace_language"]]
    closed = sum(r["closed"] and r["answer_nonempty"] for r in rows) / n
    pt = langs.count("pt") / len(langs) if langs else 0
    trunc = sum(not r["finished"] for r in rows) / n
    med = sorted(r["n_gen"] for r in rows)[n // 2]
    return f"{pct(closed)} / {pct(pt)} / {pct(trunc)} / {med}"


def minutes(name: str) -> str:
    sm = ROOT / name / "summary.json"
    if not sm.exists():
        return "–"
    j = json.loads(sm.read_text())
    return f"{sum(v['run']['wall_seconds'] for k, v in j.items() if not k.startswith('_')) / 60:.0f}"


def main() -> None:
    out = ["# Resultados de todas as rodadas (gerado por `scripts/gen_estudos_resultados.py`)\n",
           "Validação limpa, nota **estrita**, decodificação gulosa, 4.096 tokens. GSM8K = 676 itens (sem duplicatas por tradução); múltipla escolha = 385 itens. "
           "Entre parênteses, a diferença em pontos contra o Lumi no mesmo modo, com a marca ▲ (IC 95% acima de 0) ou ▼ (IC abaixo de 0); sem marca, empate estatístico. "
           "O ruído de 385 itens é de ±4 a 5 pontos.\n"]
    for grupo, runs in RUNS:
        out += [f"## {grupo}\n", "| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, desc in runs:
            if not (ROOT / name / "summary.json").exists():
                out.append(f"| {desc} (`{name}`) | – | – | – | – | – | – |")
                continue
            out.append(f"| {desc} (`{name}`) | {cell(name, 'gsm8k_val', 'thinking')} | {cell(name, 'gsm8k_val', 'nothinking')} | "
                       f"{cell(name, 'mcq_val', 'thinking')} | {cell(name, 'mcq_val', 'nothinking')} | {form(name, 'mcq_val')} | {minutes(name)} |")
        out.append("")
    out += ["## Triagem por camadas (adaptador `adalberto` fundido só em parte; só com thinking)\n",
            "| Parte mantida | GSM8K | Múlt. escolha | Forma da múlt. escolha: fecha / PT / trunca / mediana tok |", "| --- | --- | --- | --- |"]
    for name, desc in LAYERS:
        out.append(f"| {desc} | {cell(name, 'gsm8k_val', 'thinking')} | {cell(name, 'mcq_val', 'thinking')} | {form(name, 'mcq_val')} |")
    out.append("\n'PT' é a fração dos traços classificados em português pela heurística de palavras funcionais (`lumi_ft/eval/shape.py`); 'fecha' é a fração que fecha um `</think>` e escreve a resposta depois.\n")
    Path("estudos").mkdir(exist_ok=True)
    Path("estudos/resultados_auto.md").write_text("\n".join(out))
    print("estudos/resultados_auto.md escrito")


if __name__ == "__main__":
    main()
