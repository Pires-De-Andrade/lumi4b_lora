"""Fase 0 (A-14): maior batch de geração no pior caso (prompt longo + 4.096 tokens novos) e tokens/s.

Cada linha da escada gera exatamente `--new` tokens (min_new_tokens = max_new_tokens, o EOS não encerra
antes), então o cache chega ao tamanho máximo. O prompt tem `--prompt-len` tokens: o gabarito de chat do
Lumi com raciocínio ligado (termina em `<think>\\n`) em volta de ids aleatórios. O conteúdo não muda o
custo; a medida é de memória e velocidade. Primeiro o prefill sozinho (tempo e OOM do prefill), depois a
geração inteira. Greedy, `eos_token_id=[2, 11]`, implementação nativa, modelo BF16 sem adaptador (o
modelo fundido tem a mesma forma, A-11). Retomável por linha em runs/phase0/gen_resources.jsonl.

Uso: scripts/detach.sh --gpu measure-gen PYTHONPATH=src python scripts/measure_generate.py
     (smoke: ... --prompt-len 128 --new 32 --ladder 2,4 --out runs/phase0/smoke_gen.jsonl)
"""
import argparse
import gc
import time
from pathlib import Path

import torch

from lumi_ft import gpu
from lumi_ft.modeling.load import EOS_IDS, free_gpu, load_model, load_tokenizer


def build_prompts(tok, b: int, prompt_len: int, seed: int = 0) -> torch.Tensor:
    marker = "@@"
    templ = tok.apply_chat_template([{"role": "user", "content": marker}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
    pre, post = templ.split(marker)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    n_body = prompt_len - len(pre_ids) - len(post_ids)
    assert n_body > 0, "prompt-len menor que o gabarito"
    g = torch.Generator().manual_seed(seed)
    body = torch.randint(1000, 100000, (b, n_body), generator=g)
    pre_t = torch.tensor(pre_ids).expand(b, -1)
    post_t = torch.tensor(post_ids).expand(b, -1)
    return torch.cat([pre_t, body, post_t], dim=1)


def timed_generate(model, ids, n_new: int) -> tuple[float, torch.Tensor]:
    mask = torch.ones_like(ids)
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n_new, min_new_tokens=n_new,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         eos_token_id=EOS_IDS, pad_token_id=0, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--new", type=int, default=4096)
    ap.add_argument("--ladder", default="8,16,32,64,96,128,192,256,384")
    ap.add_argument("--out", default="runs/phase0/gen_resources.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    ladder = [int(x) for x in a.ladder.split(",")]
    done = {(r["P"], r["N"], r["b"]): r for r in gpu.read_jsonl(out) if "b" in r}
    if all((a.prompt_len, a.new, b) in done for b in ladder):
        print("nada a fazer")
        return
    gpu.append_jsonl(out, {"event": "start", **gpu.snapshot(), "torch": torch.__version__})
    tok = load_tokenizer()
    model = load_model("native").eval()
    model.config.use_cache = True
    oom = False
    for b in ladder:
        key = (a.prompt_len, a.new, b)
        if key in done:
            oom = oom or done[key]["status"].startswith("oom")
            continue
        if oom:
            break
        row = {"P": a.prompt_len, "N": a.new, "b": b}
        ids = build_prompts(tok, b, a.prompt_len).to("cuda")
        try:
            with torch.inference_mode():
                gpu.reset_peak()
                try:
                    timed_generate(model, ids, 1)  # aquecimento: autotune dos kernels no primeiro uso
                    gpu.reset_peak()
                    t_pre, _ = timed_generate(model, ids, 1)
                except torch.OutOfMemoryError:
                    raise RuntimeError("oom_prefill")
                row["prefill_s"] = round(t_pre, 2)
                row["peak_after_prefill_gib"] = gpu.peak_gib()["peak_allocated_gib"]
                gc.collect(); torch.cuda.empty_cache()
                gpu.reset_peak()
                t_all, seqs = timed_generate(model, ids, a.new)
            n_gen = seqs.shape[1] - ids.shape[1]
            assert n_gen == a.new, f"gerou {n_gen} tokens, esperado {a.new}"
            decode_s = max(t_all - t_pre, 1e-6)
            row |= {"status": "ok", "total_s": round(t_all, 1), "decode_s": round(decode_s, 1),
                    "ms_per_step": round(1000 * decode_s / (a.new - 1), 1),
                    "decode_tok_per_s": round(b * (a.new - 1) / decode_s),
                    "tok_per_s_total": round(b * a.new / t_all), **gpu.peak_gib()}
        except RuntimeError as e:
            if str(e) != "oom_prefill":
                raise
            row |= {"status": "oom_prefill", **gpu.peak_gib()}
            oom = True
        except torch.OutOfMemoryError:
            row |= {"status": "oom_decode", **gpu.peak_gib()}
            oom = True
        del ids
        gc.collect(); torch.cuda.empty_cache()
        gpu.append_jsonl(out, row)
        print(row, flush=True)
    del model
    free_gpu()
    gpu.append_jsonl(out, {"event": "end", **gpu.snapshot()})
    print("EXIT_OK")


if __name__ == "__main__":
    main()
