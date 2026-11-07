"""Conversa interativa com o Lumi v0.2 (sem adaptador), com a geração configurável.

Uso (interativo; rode num terminal ou numa sessão tmux sua):
  source env.sh && PYTHONPATH=src python scripts/chat.py
  PYTHONPATH=src python scripts/chat.py --no-think --temperature 0.7 --top-p 0.95
  PYTHONPATH=src python scripts/chat.py -p "Quanto é 17 vezes 23?"      # um prompt só e sai

Dentro do REPL, /help lista os comandos (/think, /temp, /top_p, /top_k, /max, /rep, /seed, /system,
/prefill, /chat, /retry, /both, /prompt, /show). Enquanto estiver aberto, o processo segura ~9 a 10 GiB
da GPU, e o `scripts/detach.sh --gpu` se recusa a lançar jobs; saia com /quit para liberar.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from lumi_ft import gpu
from lumi_ft.chat import Config, Session, make_generate, run_repl
from lumi_ft.modeling.load import load_model, load_tokenizer

MIN_FREE_GIB = 10.0  # pesos BF16 (~7,5 GiB) + cache e ativações de um prompt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-p", "--prompt", help="envia só este prompt e sai")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=True, help="enable_thinking (padrão: ligado)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (padrão)")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 = desligado")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--system", default=None)
    ap.add_argument("--prefill", default="", help="começo forçado da saída do modelo")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--chat", action="store_true", help="multi-turno: guarda o histórico")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--force", action="store_true", help="carrega mesmo com a GPU ocupada")
    a = ap.parse_args()

    free = torch.cuda.mem_get_info()[0] / 1024**3
    if free < MIN_FREE_GIB and not a.force:
        print(f"GPU ocupada: só {free:.1f} GiB livres (preciso de ~{MIN_FREE_GIB:.0f}). Processos:", file=sys.stderr)
        for line in gpu.snapshot()["compute_apps"]:
            print("  ", line, file=sys.stderr)
        print("Espere o job terminar ou use --force (pode dar falta de memória e atrapalhar o outro job).", file=sys.stderr)
        return 3

    cfg = Config(think=a.think, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_new=a.max_new_tokens,
                 rep=a.repetition_penalty, system=a.system, seed=a.seed, chat=a.chat,
                 prefill=a.prefill.replace("\\n", "\n"))
    color = sys.stdout.isatty() and not a.no_color
    t0 = time.time()
    print("carregando o Lumi…", flush=True)
    tok = load_tokenizer()
    model = load_model("native").eval()
    print(f"pronto em {time.time() - t0:.0f} s", flush=True)
    session = Session(tok, make_generate(model, tok, color=color), cfg, streaming=True)

    if a.prompt:
        session.ask(a.prompt)
        return 0
    try:
        import readline

        hist = Path(".cache/chat_history")
        hist.parent.mkdir(exist_ok=True)
        try:
            readline.read_history_file(hist)
        except FileNotFoundError:
            pass
    except ImportError:
        readline = None
    try:
        run_repl(session)
    finally:
        if readline is not None:
            readline.write_history_file(hist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
