"""Prepara os registros do dataset `adalberto-temp/gsm8k-sft-thinking-pt` (Qwen3-30B-A3B-Thinking, GSM8K em português).

Seleção: problema com resposta majoritária de ao menos 75% das respostas; resposta igual à majoritária; um único
`</think>`; traço em português; última linha `Resposta final: X` convertida em `#### X`. Exclui qualquer problema com
8-grama das provas de teste/validação ou com a mesma assinatura de números de um item da nossa validação do GSM8K
(o dataset usa outra tradução dos mesmos problemas do GSM8K `train`, e 805 dos 1.000 itens da validação têm 8-grama
em comum com ele).

Saída: data/adalberto_think.jsonl (com traço) e data/adalberto_nothink.jsonl (só resposta, para o modo sem raciocínio).
Sem licença declarada no dataset: só para experimento. Uso: PYTHONPATH=src python scripts/prepare_adalberto.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, "scripts")
import build_dataset as bd  # noqa: E402

from lumi_ft.data import gigaverbo as GV  # noqa: E402
from lumi_ft.data import teacher as T  # noqa: E402
from lumi_ft.data.decontaminate import NgramIndex  # noqa: E402
from lumi_ft.eval import parse as P  # noqa: E402
from lumi_ft.eval.shape import trace_language  # noqa: E402


def main() -> None:
    ds = load_dataset("adalberto-temp/gsm8k-sft-thinking-pt", split="train")
    by = defaultdict(list)
    for r in ds:
        by[r["problem_id"]].append(r)
    test_idx, _ = bd.build_decontam_index(8)
    val = [json.loads(line) for line in Path("data/val_gsm8k.jsonl").read_text().splitlines()]
    val_idx = NgramIndex(8)
    for v in val:
        val_idx.add(v["question"] + "\n" + v["answer"])
    for line in Path("data/val_mcq.jsonl").read_text().splitlines():
        m = json.loads(line)
        val_idx.add(m["question"] + "\n" + "\n".join(m["options"]))
    val_sigs = {T.number_signature(v["question"]) for v in val}
    val_sigs = {s for s in val_sigs if len(s) >= 2}

    st = Counter()
    think, nothink = [], []
    for pid, rs in by.items():
        st["problemas"] += 1
        q = rs[0]["prompt"].split("Problema:\n", 1)[-1].strip()
        if test_idx.hits(q):
            st["prob_removido_8grama_teste"] += 1
            continue
        if val_idx.hits(q):
            st["prob_removido_8grama_validacao"] += 1
            continue
        if T.number_signature(q) in val_sigs:
            st["prob_removido_mesmos_numeros_da_validacao"] += 1
            continue
        finals = [T.plain_number(r["final_extracted"]) for r in rs]
        top, n_top = Counter(f for f in finals if f).most_common(1)[0] if any(finals) else (None, 0)
        if top is None or n_top / len(rs) < 0.75:
            st["prob_sem_maioria_75"] += 1
            continue
        st["problemas_usados"] += 1
        for k, (r, f) in enumerate(zip(rs, finals)):
            st["respostas"] += 1
            if f != top:
                st["resp_diferente_da_maioria"] += 1
                continue
            resp = r["response"]
            if resp.count("</think>") != 1 or not resp.lstrip().startswith("<think>"):
                st["resp_sem_um_think"] += 1
                continue
            head, _, tail = resp.partition("</think>")
            thought, answer = head.lstrip().removeprefix("<think>").strip(), tail.strip()
            if not thought or not answer:
                st["resp_vazia"] += 1
                continue
            if trace_language(thought) != "pt":
                st["resp_traco_nao_pt"] += 1
                continue
            conv = T.to_hash_format(answer, top)
            if conv is None:
                st["resp_sem_linha_final_no_fim"] += 1
                continue
            st["respostas_boas"] += 1
            base = {"problem_id": f"adalberto:{pid}", "question": q, "source": "adalberto"}
            think.append({**base, "id": f"adalberto:{pid}:{k}", "thought": thought, "answer": conv})
            nothink.append({**base, "id": f"adalberto_nt:{pid}:{k}", "answer": conv})
    Path("data/adalberto_think.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in think) + "\n")
    Path("data/adalberto_nothink.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in nothink) + "\n")
    print(json.dumps(dict(st), indent=1))
    print("exemplo da resposta convertida:\n", think[0]["answer"][-300:])


if __name__ == "__main__":
    main()
