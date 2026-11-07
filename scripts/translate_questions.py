"""Traduz só os enunciados e alternativas (sem traços) das questões de ciência para o português, com o Lumi no vLLM.

Serve de prompts de múltipla escolha para o GRPO: cada questão distinta vira {id, question, options, prompt, gold, n_options},
com `gold` = letra majoritária dos traços do R1 (indicador, não verdade). Tradução gulosa, mesmo modo e validações de
`translate_traces.py` (terminou, tamanho compatível, mesmas alternativas). A descontaminação (8-grama contra as provas) é
feita depois por `scripts/filter_translated.py <dir>`, que grava questions_clean.jsonl.
  scripts/detach.sh --gpu trans-q "VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/translate_questions.py --n 4000 --out data/mot_questions"
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
from translate_traces import pick, question_block  # noqa: E402

from lumi_ft.data import mot_science as M  # noqa: E402
from lumi_ft.data import translate as TR  # noqa: E402
from lumi_ft.eval import generate as G  # noqa: E402
from lumi_ft.eval import mcq  # noqa: E402
from lumi_ft.modeling.load import EOS_IDS, MODEL_ID, REVISION, load_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mot_science_en.jsonl")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=900)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(a.src).read_text().splitlines() if line.strip()]
    chosen = pick(records, a.n, 1, a.seed)
    kept_f, raw_f = out / "questions.jsonl", out / "raw.jsonl"
    done = {json.loads(line)["id"] for line in raw_f.read_text().splitlines()} if raw_f.exists() else set()
    todo = [r for r in chosen if r["problem_id"] not in done]
    print(f"{len(chosen)} questões escolhidas, {len(done)} já feitas", flush=True)

    from vllm import LLM, SamplingParams

    tok = load_tokenizer()
    llm = LLM(model=MODEL_ID, revision=REVISION, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=a.mem,
              max_num_seqs=a.max_seqs, enable_prefix_caching=True, seed=0, generation_config="vllm")
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=EOS_IDS, skip_special_tokens=False)
    report = Counter()
    for i in range(0, len(todo), a.chunk):
        block = todo[i: i + a.chunk]
        prompts = [G.render_prompt(tok, TR.make_prompt(question_block(r)), thinking=False) for r in block]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        with raw_f.open("a") as fr, kept_f.open("a") as fk:
            for r, o in zip(block, outs):
                c = o.outputs[0]
                why = TR.check_chunk(question_block(r), c.text, c.finish_reason == "stop")
                sp_q = M.split_options(c.text.strip()) if why is None else None
                if why is None and (sp_q is None or len(sp_q[1]) != len(r["options"])):
                    why = "alternativas_nao_voltaram"
                report[why or "ok"] += 1
                fr.write(json.dumps({"id": r["problem_id"], "verdict": why or "ok", "pt": c.text.strip()}, ensure_ascii=False) + "\n")
                if why is None:
                    fk.write(json.dumps({"id": r["problem_id"], "question": sp_q[0], "options": sp_q[1], "gold": r["letter"], "n_options": len(sp_q[1]),
                                         "prompt": mcq.build_prompt({"question": sp_q[0], "options": sp_q[1]}), "source": "mot_science_pt"}, ensure_ascii=False) + "\n")
        print(f"bloco {i // a.chunk + 1}: {len(block)} questões em {dt:.0f} s | {dict(report)}", flush=True)
    (out / "report.json").write_text(json.dumps({"args": vars(a), "verdicts": dict(report)}, ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())
