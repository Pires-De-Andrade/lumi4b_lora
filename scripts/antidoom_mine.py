"""Fase anti-laço, passo 1: gera com o vLLM e minera os laços (roda no ambiente do vLLM).

Para cada prompt de `data/antidoom_prompts.jsonl` (campo `split`), gera uma resposta no modo thinking, sem começo forçado. Dois detectores:
  - `exato` (próprio): bloco de >= `--min-chars` caracteres repetido >= 4 vezes até o FIM de uma resposta que não terminou
    (`lumi_ft.train.antidoom.find_loop_start`); o rejeitado é o primeiro token da primeira repetição;
  - `oficial` (receita do Antidoom, `lumi_ft.train.antidoom_oficial.find_loop_event`): repetição em caracteres em QUALQUER trecho da resposta,
    >= 4 repetições somando >= 60 caracteres, terminada ou não; o rejeitado é o primeiro token legível da primeira repetição.
`--detector` escolhe qual gera os eventos; `stats.json` traz as taxas dos dois por tipo de prompt (para medir a taxa de laço de um modelo, use
`--split heldout` com a decodificação gulosa padrão, a da avaliação). A receita oficial minera com temperatura 0,1, top-k 50, min-p 0,01 e 4.000 tokens.
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/antidoom_mine.py --model runs/merged/thor_pt --split mine --out runs/antidoom/thor_pt/mine \
      --temperature 0.1 --top-k 50 --min-p 0.01 --max-new 4000 --detector oficial
Saída em --out: events.jsonl (um por laço do detector escolhido), samples.jsonl (uma linha por prompt, sem ids) e stats.json.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from lumi_ft.eval import generate as G
from lumi_ft.modeling.load import EOS_IDS, load_tokenizer
from lumi_ft.train import antidoom as A
from lumi_ft.train import antidoom_oficial as O


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pasta do modelo fundido")
    ap.add_argument("--prompts", default="data/antidoom_prompts.jsonl")
    ap.add_argument("--split", default="mine", choices=["mine", "heldout"])
    ap.add_argument("--limit", type=int, default=0, help="usa só os N primeiros prompts (0 = todos)")
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = gulosa (a da avaliação)")
    ap.add_argument("--top-k", type=int, default=0, help="0 = sem corte")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--detector", default="exato", choices=["exato", "oficial"], help="qual detector gera os eventos")
    ap.add_argument("--min-chars", type=int, default=40, help="detector exato: tamanho mínimo, em caracteres, do bloco que se repete")
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["split"] == a.split]
    if a.limit:
        rows = rows[:a.limit]
    tok = load_tokenizer()
    texts = [G.render_prompt(tok, r["prompt"], thinking=True) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_new + 1536, gpu_memory_utilization=a.mem, max_num_seqs=a.max_seqs,
              enable_prefix_caching=True, seed=a.seed, generation_config="vllm")
    kw = {"n": 1, "temperature": a.temperature, "max_tokens": a.max_new, "stop_token_ids": EOS_IDS, "skip_special_tokens": False}
    if a.temperature > 0:
        kw["seed"] = a.seed
        if a.top_k > 0:
            kw["top_k"] = a.top_k
        if a.min_p > 0:
            kw["min_p"] = a.min_p
    sp = SamplingParams(**kw)
    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cnt, n_tok, n_ev = Counter(), 0, 0
    with (out / "events.jsonl").open("w") as fe, (out / "samples.jsonl").open("w") as fs:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            n_tok += len(ids)
            finished = c.finish_reason == "stop"
            hit = None if finished else A.find_loop_start(ids, min_chars=a.min_chars, decode=tok.decode)
            strs = tok.convert_ids_to_tokens(ids)
            ev_of = O.find_loop_event(ids, strs)
            k = r["kind"]
            cnt[f"{k}:total"] += 1
            cnt[f"{k}:nao_termina"] += int(not finished)
            cnt[f"{k}:laco"] += int(hit is not None)
            cnt[f"{k}:laco_oficial"] += int(ev_of is not None)
            cnt[f"{k}:laco_oficial_nao_termina"] += int(ev_of is not None and not finished)
            fs.write(json.dumps({"id": r["id"], "kind": k, "n_gen": len(ids), "finished": finished, "loop": hit is not None,
                                 "k": hit[1] if hit else None, "loop_oficial": ev_of is not None,
                                 "periodo_oficial": ev_of["period"] if ev_of else None}) + "\n")
            if a.detector == "exato" and hit is not None:
                s, kk = hit
                fe.write(json.dumps({"id": r["id"], "kind": k, "prompt_ids": list(o.prompt_token_ids), "prefix_ids": ids[:s], "rejected": ids[s],
                                     "k": kk, "s": s, "n_gen": len(ids), "finished": finished, "detector": "exato"}) + "\n")
                n_ev += 1
            elif a.detector == "oficial" and ev_of is not None:
                s = ev_of["reject_idx"]
                fe.write(json.dumps({"id": r["id"], "kind": k, "prompt_ids": list(o.prompt_token_ids), "prefix_ids": ids[:s], "rejected": ids[s],
                                     "s": s, "n_gen": len(ids), "finished": finished, "detector": "oficial",
                                     "repeticao": {x: ev_of[x] for x in ("period", "repeats", "start_char", "repeat_start_char", "end_char", "snippet")}},
                                    ensure_ascii=False) + "\n")
                n_ev += 1
    st = {"model": a.model, "split": a.split, "max_new": a.max_new, "temperature": a.temperature, "top_k": a.top_k, "min_p": a.min_p,
          "detector_eventos": a.detector, "eventos": n_ev, "min_chars": a.min_chars, "prompts": len(rows), "gen_seconds": round(dt),
          "gen_tokens": n_tok, "tok_per_s": round(n_tok / dt), "contagens": dict(cnt), "taxas": {}}
    for kind in sorted({r["kind"] for r in rows}):
        tot = cnt[f"{kind}:total"]
        st["taxas"][kind] = {x: round(cnt[f"{kind}:{x}"] / tot, 4) for x in ("nao_termina", "laco", "laco_oficial", "laco_oficial_nao_termina")}
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
