"""REPL para conversar com o Lumi e mexer na configuração de geração.

A lógica (comandos, prompt, histórico, estatísticas) não depende da GPU: `Session` recebe uma função
`generate(prompt, cfg, in_trace) -> {text, n_gen, finished, seconds}`. `make_generate` monta a real.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from typing import Callable

from lumi_ft.eval import parse as P
from lumi_ft.eval.shape import trace_language

DIM, RESET = "\033[2m", "\033[0m"


@dataclass
class Config:
    think: bool = True  # enable_thinking
    temperature: float = 0.0  # 0 = greedy (o protocolo da baseline)
    top_p: float = 1.0
    top_k: int = 0  # 0 = desligado
    max_new: int = 4096
    rep: float = 1.0  # repetition_penalty
    system: str | None = None
    seed: int | None = None
    chat: bool = False  # guarda o histórico (multi-turno); desligado, cada prompt é independente
    prefill: str = ""  # texto com que a saída do modelo começa (depois de `<think>\n` no modo com raciocínio)


# --------------------------------------------------------------------------- comandos de ajuste
def _bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("on", "1", "true", "sim", "s", "yes", "y"):
        return True
    if v in ("off", "0", "false", "nao", "não", "n", "no"):
        return False
    raise ValueError("use on ou off")


def _num(cast, lo, hi):
    def f(s: str):
        v = cast(s.strip())
        if not lo <= v <= hi:
            raise ValueError(f"fora do intervalo [{lo}, {hi}]")
        return v

    return f


def _opt(cast):
    def f(s: str):
        return None if s.strip().lower() in ("off", "none", "nenhum", "") else cast(s)

    return f


def _text(s: str) -> str:
    return s.replace("\\n", "\n")


SETTERS: dict[str, tuple[str, Callable, str]] = {
    "think": ("think", _bool, "on|off: modo com raciocínio (enable_thinking)"),
    "temp": ("temperature", _num(float, 0.0, 5.0), "0 = greedy; acima, amostragem"),
    "top_p": ("top_p", _num(float, 0.01, 1.0), "só vale com temp > 0"),
    "top_k": ("top_k", _num(int, 0, 100_000), "0 = desligado; só vale com temp > 0"),
    "max": ("max_new", _num(int, 1, 32_768), "máximo de tokens novos"),
    "rep": ("rep", _num(float, 0.5, 3.0), "penalidade de repetição (1.0 = nenhuma)"),
    "seed": ("seed", _opt(int), "N|off: semente da amostragem"),
    "chat": ("chat", _bool, "on|off: guardar o histórico (multi-turno)"),
    "system": ("system", _opt(_text), "texto|off: mensagem de sistema"),
    "prefill": ("prefill", lambda s: _opt(_text)(s) or "", "texto|off: começo forçado da saída (use \\n para quebra)"),
}

HELP = """Comandos (qualquer outra linha é enviada ao modelo; \"\"\" abre e fecha um prompt de várias linhas):
  /think on|off      modo com raciocínio (enable_thinking)
  /temp X            0 = greedy (padrão); acima de 0, amostragem
  /top_p X  /top_k N só valem com temp > 0 (top_k 0 = desligado)
  /max N             máximo de tokens novos (padrão 4096)
  /rep X             penalidade de repetição (padrão 1.0 = nenhuma)
  /seed N|off        semente da amostragem
  /system TEXTO|off  mensagem de sistema
  /prefill TEXTO|off começo forçado da saída, ex.: /prefill Vou raciocinar em português. (\\n = quebra de linha)
  /chat on|off       multi-turno: guarda o histórico (padrão off: cada prompt é independente)
  /reset             apaga o histórico
  /retry             repete o último prompt com a configuração atual
  /both              roda o último prompt com e sem raciocínio, lado a lado
  /prompt            mostra o prompt exato enviado da última vez (com os tokens especiais)
  /show              mostra a configuração
  /help  /quit"""


class Session:
    def __init__(self, tok, generate: Callable[[str, Config, bool], dict], cfg: Config | None = None,
                 out: Callable[[str], None] = print, streaming: bool = False):
        self.tok, self.generate, self.cfg, self.out = tok, generate, cfg or Config(), out
        self.streaming = streaming  # True: `generate` já imprime a saída; não imprimir de novo
        self.history: list[dict] = []
        self.last_user: str | None = None
        self.last_prompt: str | None = None
        self._stored_last = False

    # ------------------------------------------------------------------------------ prompt
    def render(self, user_text: str, cfg: Config) -> str:
        msgs = []
        if cfg.system:
            msgs.append({"role": "system", "content": cfg.system})
        if cfg.chat:
            msgs += self.history
        msgs.append({"role": "user", "content": user_text})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=cfg.think) + cfg.prefill

    # ------------------------------------------------------------------------------- turno
    def ask(self, user_text: str, cfg: Config | None = None, *, store: bool = True) -> dict | None:
        cfg = cfg or self.cfg
        prompt = self.render(user_text, cfg)
        self.last_user, self.last_prompt = user_text, prompt
        in_trace = cfg.think and P.THINK_CLOSE not in cfg.prefill
        try:
            gen = self.generate(prompt, cfg, in_trace)
        except KeyboardInterrupt:
            self.out("\n[interrompido]")
            return None
        text = cfg.prefill + gen["text"]
        parsed = P.split_response(text, thinking=cfg.think)
        if not self.streaming:
            self.out(text)
        self.out(self._stats(gen, parsed, cfg))
        self._stored_last = False
        if store and cfg.chat:
            answer = parsed.answer if (parsed.closed or not cfg.think) else None
            if answer:
                self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
                self._stored_last = True
            else:
                self.out("(turno não guardado no histórico: o raciocínio não fechou `</think>`)")
        return {**gen, "parsed": parsed, "full_text": text}

    def _stats(self, gen: dict, parsed: P.Parsed, cfg: Config) -> str:
        sec = max(gen["seconds"], 1e-9)
        parts = [f"{gen['n_gen']} tokens", f"{gen['seconds']:.1f} s", f"{gen['n_gen'] / sec:.0f} tok/s",
                 "terminou" if gen["finished"] else f"TRUNCADO em {cfg.max_new}"]
        if cfg.think:
            parts.append("fechou </think>" if parsed.closed else "NÃO fechou </think>")
            if parsed.thought:
                parts.append(f"traço: {trace_language(parsed.thought)}")
            if parsed.n_close > 1:
                parts.append(f"{parsed.n_close} </think>")
        return "[" + " | ".join(parts) + "]"

    # ----------------------------------------------------------------------------- comandos
    def show(self) -> str:
        c = self.cfg
        rows = [("raciocínio (think)", "on" if c.think else "off"),
                ("temperatura", f"{c.temperature}" + (" (greedy)" if c.temperature == 0 else "")),
                ("top_p / top_k", f"{c.top_p} / {c.top_k}" + ("" if c.temperature > 0 else "  (ignorados: greedy)")),
                ("max tokens novos", c.max_new), ("repetition_penalty", c.rep),
                ("semente", c.seed if c.seed is not None else "aleatória"),
                ("multi-turno (chat)", f"{'on' if c.chat else 'off'} ({len(self.history) // 2} turnos guardados)"),
                ("system", repr(c.system) if c.system else "(nenhum)"),
                ("prefill", repr(c.prefill) if c.prefill else "(nenhum)"),
                ("fixo", "eos_token_id=[2, 11], pad=<unk>, bf16, implementação nativa")]
        return "\n".join(f"  {k:<20} {v}" for k, v in rows)

    def handle(self, line: str) -> bool:
        """Trata um comando `/...`. Devolve False para sair."""
        name, _, arg = line[1:].partition(" ")
        name = name.strip().lower()
        if name in ("quit", "exit", "q"):
            return False
        if name == "help":
            self.out(HELP)
        elif name == "show":
            self.out(self.show())
        elif name == "reset":
            self.history.clear()
            self.out("histórico apagado")
        elif name == "prompt":
            self.out(self.last_prompt if self.last_prompt is not None else "(nenhum prompt enviado ainda)")
        elif name in ("retry", "both"):
            if self.last_user is None:
                self.out("nenhum prompt anterior")
            elif name == "retry":
                if self._stored_last:
                    del self.history[-2:]
                self.ask(self.last_user)
            else:
                for think in (True, False):
                    self.out(f"\n── {'com' if think else 'sem'} raciocínio ──")
                    self.ask(self.last_user, dataclasses.replace(self.cfg, think=think), store=False)
        elif name in SETTERS:
            attr, parse, desc = SETTERS[name]
            if not arg.strip():
                self.out(f"/{name}: {desc}. Valor atual: {getattr(self.cfg, attr)!r}")
            else:
                try:
                    setattr(self.cfg, attr, parse(arg))
                    self.out(f"{attr} = {getattr(self.cfg, attr)!r}")
                except ValueError as e:
                    self.out(f"valor inválido para /{name}: {e}. Configuração mantida.")
        else:
            self.out(f"comando desconhecido: /{name}. /help lista os comandos.")
        return True

    def prompt_label(self) -> str:
        c = self.cfg
        t = "greedy" if c.temperature == 0 else f"T={c.temperature}"
        return f"[think:{'on' if c.think else 'off'} {t}{' chat' if c.chat else ''}] > "


def run_repl(session: Session, read: Callable[[str], str] = input) -> None:
    session.out("Lumi pronto. /help lista os comandos; Ctrl-C interrompe a geração; /quit sai.")
    while True:
        try:
            line = read(session.prompt_label())
        except (EOFError, KeyboardInterrupt):
            session.out("")
            return
        line = line.strip()
        if not line:
            continue
        if line == '"""':
            buf = []
            while True:
                try:
                    nxt = read("... ")
                except (EOFError, KeyboardInterrupt):
                    break
                if nxt.strip() == '"""':
                    break
                buf.append(nxt)
            line = "\n".join(buf).strip()
            if not line:
                continue
        elif line.startswith("/"):
            if not session.handle(line):
                return
            continue
        session.ask(line)


# ---------------------------------------------------------------------------------- GPU
def make_generate(model, tok, color: bool = False):
    """Função de geração real, com streaming. O traço sai esmaecido (cor) até o `</think>`."""
    import time

    import torch
    from transformers import TextStreamer

    from lumi_ft.eval.generate import cut_at_eos
    from lumi_ft.modeling.load import EOS_IDS

    class TraceStreamer(TextStreamer):
        def __init__(self, in_trace: bool):
            super().__init__(tok, skip_prompt=True, skip_special_tokens=False)
            self.in_trace = in_trace

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if color and self.in_trace:
                if P.THINK_CLOSE in text:
                    a, b = text.split(P.THINK_CLOSE, 1)
                    text, self.in_trace = f"{DIM}{a}{P.THINK_CLOSE}{RESET}{b}", False
                else:
                    text = f"{DIM}{text}{RESET}"
            print(text, flush=True, end="\n" if stream_end else "")

    def generate(prompt: str, cfg: Config, in_trace: bool) -> dict:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=cfg.max_new, eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id,
                  repetition_penalty=cfg.rep, use_cache=True)
        if cfg.temperature > 0:
            kw |= dict(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)  # top_k=0 desliga o 50 padrão
        else:
            kw |= dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
        if cfg.prefill:
            print(f"{DIM}{cfg.prefill}{RESET}" if (color and in_trace) else cfg.prefill, end="", flush=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, streamer=TraceStreamer(in_trace), **kw)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        ids = out[0, enc["input_ids"].shape[1]:].tolist()
        cut, finished = cut_at_eos(ids)
        return {"text": tok.decode(cut, skip_special_tokens=False), "n_gen": len(cut), "finished": finished, "seconds": seconds}

    return generate
