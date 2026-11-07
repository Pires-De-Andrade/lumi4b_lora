#!/usr/bin/env python3
"""Gera traços de raciocínio em português com um professor grande, falando com um servidor compatível com OpenAI.

Funciona com qualquer motor que exponha POST /v1/completions (vLLM, SGLang, TensorRT-LLM, llama.cpp server).
Só usa a biblioteca padrão e o tokenizer do `transformers` (para o template de chat do professor).
Retomável: ids já gravados em --out (sem erro) são pulados. Pode parar com Ctrl-C ou com --time-budget-min.

Exemplo:
  python gen_traces.py --tokenizer Qwen/Qwen3-30B-A3B-Thinking-2507 --model-name Qwen/Qwen3-30B-A3B-Thinking-2507 \
      --prompts data/prompts.jsonl --out out/traces.jsonl --limit 300 --concurrency 64
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

BACKOFF_S = 2  # espera entre tentativas: 2, 4, 6 s


def render(tok, prompt: str, prefill: str) -> str:
    """Prompt do professor: template de chat, garantido terminando em `<think>\\n`, mais o começo forçado."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if not text.endswith("<think>\n"):
        text = text.rstrip("\n").removesuffix("<think>") + "<think>\n"
    return text + prefill


def post(endpoint: str, payload: dict, api_key: str, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def generate_one(item: dict, text: str, a) -> dict:
    payload = {"model": a.model_name, "prompt": text, "max_tokens": a.max_tokens, "temperature": a.temperature,
               "top_p": a.top_p, "stop": ["<|im_end|>"]}
    if a.top_k:
        payload["top_k"] = a.top_k
    last = ""
    for attempt in range(3):
        t0 = time.time()
        try:
            r = post(a.endpoint, payload, a.api_key, a.timeout)
            c = r["choices"][0]
            return {"id": item["id"], "source": item["source"], "ref": item["ref"], "prefill": a.prefill,
                    "text": a.prefill + c["text"], "finish_reason": c.get("finish_reason"),
                    "completion_tokens": (r.get("usage") or {}).get("completion_tokens"), "seconds": round(time.time() - t0, 1)}
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError) as e:
            last = repr(e)[:200]
            time.sleep(BACKOFF_S * (attempt + 1))
    return {"id": item["id"], "source": item["source"], "ref": item["ref"], "error": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="out/traces.jsonl")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model-name", required=True, help="nome do modelo como o servidor o expõe")
    ap.add_argument("--tokenizer", required=True, help="id do Hugging Face ou pasta, para o template de chat")
    ap.add_argument("--prefill", default="Vou raciocinar em português.")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20, help="0 desliga (alguns servidores não aceitam top_k)")
    ap.add_argument("--limit", type=int, default=0, help="só os N primeiros problemas (depois do --start)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=0, help="para de enviar novos pedidos depois de N minutos")
    ap.add_argument("--timeout", type=float, default=900, help="segundos por pedido")
    ap.add_argument("--api-key", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    items = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    items = items[a.start:]
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} problemas, {len(done)} já feitos, {len(todo)} a fazer; concorrência {a.concurrency}", flush=True)
    if not todo:
        return 0

    lock = threading.Lock()
    stats = {"done": 0, "errors": 0, "tokens": 0, "closed": 0}
    t_start = time.time()
    deadline = t_start + a.time_budget_min * 60 if a.time_budget_min else None
    stop = threading.Event()

    def progress():
        while not stop.wait(30):
            with lock:
                el = time.time() - t_start
                n, tk = stats["done"], stats["tokens"]
                print(f"[{el / 60:5.1f} min] feitos {n} (erros {stats['errors']}) | {n / el * 60:.1f} problemas/min | "
                      f"{tk / el:.0f} tokens/s | fecham </think>: {stats['closed']}/{max(n, 1)}", flush=True)

    threading.Thread(target=progress, daemon=True).start()
    ex = ThreadPoolExecutor(max_workers=a.concurrency)
    pending, it_iter = set(), iter(todo)
    try:
        with out.open("a") as fout:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < a.concurrency * 2 and not (deadline and time.time() > deadline):
                    item = next(it_iter, None)
                    if item is None:
                        exhausted = True
                        break
                    pending.add(ex.submit(generate_one, item, render(tok, item["prompt"], a.prefill), a))
                if deadline and time.time() > deadline:
                    exhausted = True
                if not pending:
                    break
                fin, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for f in fin:
                    r = f.result()
                    with lock:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                        fout.flush()
                        if "error" in r:
                            stats["errors"] += 1
                        else:
                            stats["done"] += 1
                            stats["tokens"] += r.get("completion_tokens") or 0
                            stats["closed"] += r["text"].count("</think>") == 1
    except KeyboardInterrupt:
        print("interrompido; o que já foi gravado fica em", out, flush=True)
    finally:
        stop.set()
        ex.shutdown(wait=False, cancel_futures=True)
    print(f"fim: {stats['done']} gerados, {stats['errors']} erros, {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 1 if stats["errors"] and not stats["done"] else 0


if __name__ == "__main__":
    sys.exit(main())
