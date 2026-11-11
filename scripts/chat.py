"""Conversa interativa com o Lumi v0.2 (sem adaptador, o padrão) ou, com --model, com um modelo fundido como o T6-Lumi, com a geração configurável.

Uso (interativo; rode num terminal ou numa sessão tmux sua):
  source env.sh && PYTHONPATH=src python scripts/chat.py
  PYTHONPATH=src python scripts/chat.py --model runs/merged/thor_pt_antidoom_of    # o T6-Lumi (use com thinking, o padrão)
  PYTHONPATH=src python scripts/chat.py --device cpu --threads 8 --model runs/merged/thor_pt_antidoom_of   # sem GPU (lento: ~1 token/s)
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

# Na CPU a implementação nativa não pode usar os kernels CUDA do causal_conv1d e do mamba_ssm (se estiverem instalados ela os escolhe e falha com
# "Expected x.is_cuda()"): escondê-los faz o Transformers cair na versão em PyTorch. Tem de ser antes de qualquer import do Transformers.
if "--device=cpu" in sys.argv or ("--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1 : sys.argv.index("--device") + 2] == ["cpu"]):
    sys.modules["causal_conv1d"] = None
    sys.modules["mamba_ssm"] = None

import torch

from lumi_ft import gpu
from lumi_ft.chat import Config, Session, make_generate, run_repl
from lumi_ft.modeling.load import load_model, load_tokenizer

MIN_FREE_GIB = 10.0  # pesos BF16 (~7,5 GiB) + cache e ativações de um prompt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-p", "--prompt", help="envia só este prompt e sai")
    ap.add_argument("--model", default=None, help="pasta de um modelo fundido (ex.: runs/merged/thor_pt_antidoom_of, o T6-Lumi) ou id do Hub; sem isso, o Lumi v0.2 original")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cpu: sem GPU, em float32 (lento); serve para conferir o modelo quando a GPU está ocupada")
    ap.add_argument("--threads", type=int, default=8, help="só com --device cpu: quantas threads usar (deixe folga para outros jobs)")
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

    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    free = torch.cuda.mem_get_info()[0] / 1024**3 if a.device == "cuda" else float("inf")
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
    print(f"carregando {a.model or 'o Lumi'}…", flush=True)
    tok = load_tokenizer()  # o T6-Lumi usa o tokenizador do Lumi, sem alteração
    if a.model or a.device == "cpu":  # mesma carga da implementação nativa, só que de outro lugar e/ou na CPU (float32: a CPU não tem BF16 nativo)
        from transformers import AutoModelForCausalLM

        from lumi_ft.modeling.load import MODEL_ID, REVISION

        origem, rev = (a.model, None) if a.model else (MODEL_ID, REVISION)
        dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(origem, revision=rev, dtype=dtype, trust_remote_code=False, device_map={"": a.device}).eval()
    else:
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
