"""Escolhe o ÚNICO modelo que vai para a suíte de teste, pela regra escrita antes dos resultados (estudos/07, H58 e H59).

Um candidato da fase anti-laço só substitui o `gspo1` se, na validação com thinking, tiver múltipla escolha >= 69,6% (empate com o Lumi),
fechar o `</think>` em >= 93% da múltipla escolha, GSM8K limpo >= 83% e traço em português >= 90% nas duas provas. Entre os que passam, vence
a maior múltipla escolha (empate: maior GSM8K limpo). Se nenhum passar, o candidato é o `gspo1` (8 ciclos).
Imprime o diagnóstico na saída de erro e, na última linha da saída padrão, `NOME|pasta_do_modelo` (o formato do scripts/run_suite_models.sh).
"""
import json
import sys
from pathlib import Path

CANDIDATOS = ["thor_pt_antidoom", "gspo_thor_antidoom", "thor_pt_antidoom_of", "gspo1_antidoom"]
PADRAO = ("gspo1", "runs/grpo/gspo1/merged")
LIMIAR = {"mcq": 0.696, "fecha": 0.93, "gsm8k_limpo": 0.83, "pt": 0.90}


def medidas(nome: str, root: Path = Path("runs/eval"), clean: Path = Path("data/val_gsm8k_clean_ids.json")) -> dict | None:
    try:
        s = json.loads((root / nome / "summary.json").read_text())
        mcq, gsm = s["mcq_val|thinking"], s["gsm8k_val|thinking"]
        ids = set(json.loads(clean.read_text()))
        rows = [r for r in json.loads((root / nome / "gsm8k_val_thinking_scores.json").read_text()) if r["id"] in ids]
    except (OSError, KeyError, ValueError):
        return None

    def pt(t: dict) -> float:
        lang = t.get("trace_language") or {}
        tot = sum(lang.values())
        return lang.get("pt", 0) / tot if tot else 0.0

    return {"mcq": mcq["strict_acc"], "fecha": mcq["closed_rate"], "gsm8k_limpo": sum(bool(r["strict_ok"]) for r in rows) / len(rows),
            "pt": min(pt(mcq), pt(gsm))}


def escolher(cands: list[str] = CANDIDATOS, **kw) -> tuple[str, str, list]:
    passam, diag = [], []
    for nome in cands:
        m = medidas(nome, **kw)
        if m is None:
            diag.append((nome, "sem avaliação completa"))
            continue
        ok = all(m[k] >= LIMIAR[k] for k in LIMIAR)
        diag.append((nome, {k: round(v, 4) for k, v in m.items()}, "passa" if ok else "não passa"))
        if ok:
            passam.append((m["mcq"], m["gsm8k_limpo"], nome))
    if passam:
        _, _, nome = max(passam)
        return nome, f"runs/merged/{nome}", diag
    return PADRAO[0], PADRAO[1], diag


def main() -> int:
    nome, pasta, diag = escolher()
    for d in diag:
        print(f"candidato à suíte: {d}", file=sys.stderr)
    print(f"escolhido para a suíte: {nome}", file=sys.stderr)
    print(f"{nome}|{pasta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
