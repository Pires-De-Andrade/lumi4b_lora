"""Testes do REPL do Lumi (CPU, com gerador falso): comandos, prompt, histórico, /both, /retry."""

import pytest

from lumi_ft.chat import Config, Session, run_repl


@pytest.fixture(scope="module")
def tok():
    from lumi_ft.modeling.load import load_tokenizer

    return load_tokenizer()


class Fake:
    """Gerador falso: grava (prompt, cfg, in_trace) e devolve o texto da fila."""

    def __init__(self, *texts, finished=True):
        self.texts, self.calls, self.finished = list(texts), [], finished

    def __call__(self, prompt, cfg, in_trace):
        self.calls.append((prompt, cfg, in_trace))
        text = self.texts.pop(0) if self.texts else "ok"
        return {"text": text, "n_gen": 10, "finished": self.finished, "seconds": 0.5}


def make(tok, *texts, cfg=None, finished=True):
    out: list[str] = []
    fake = Fake(*texts, finished=finished)
    return Session(tok, fake, cfg or Config(), out=out.append), fake, out


# ------------------------------------------------------------------------------- ajustes
@pytest.mark.parametrize(
    "line, attr, expected",
    [
        ("/think off", "think", False),
        ("/think on", "think", True),
        ("/temp 0.7", "temperature", 0.7),
        ("/top_p 0.95", "top_p", 0.95),
        ("/top_k 40", "top_k", 40),
        ("/max 256", "max_new", 256),
        ("/rep 1.1", "rep", 1.1),
        ("/seed 7", "seed", 7),
        ("/chat on", "chat", True),
        ("/system Responda em português.", "system", "Responda em português."),
        ("/prefill Vou raciocinar em português.\\n", "prefill", "Vou raciocinar em português.\n"),
    ],
)
def test_settings(tok, line, attr, expected):
    s, _, _ = make(tok)
    assert s.handle(line) is True
    assert getattr(s.cfg, attr) == expected


def test_off_clears_optional_settings(tok):
    s, _, _ = make(tok, cfg=Config(system="x", seed=3, prefill="abc"))
    for line in ("/system off", "/seed off", "/prefill off"):
        s.handle(line)
    assert (s.cfg.system, s.cfg.seed, s.cfg.prefill) == (None, None, "")


@pytest.mark.parametrize("line", ["/temp abc", "/temp -1", "/temp 99", "/max 0", "/think talvez", "/top_p 0", "/rep 0", "/seed x"])
def test_invalid_values_keep_config(tok, line):
    s, _, out = make(tok)
    before = vars(s.cfg).copy()
    s.handle(line)
    assert vars(s.cfg) == before and "inválido" in out[-1]


def test_setting_without_argument_shows_current_value(tok):
    s, _, out = make(tok, cfg=Config(system="abc"))
    s.handle("/system")
    assert s.cfg.system == "abc" and "abc" in out[-1]


def test_unknown_command_and_quit(tok):
    s, _, out = make(tok)
    assert s.handle("/xyz") is True and "desconhecido" in out[-1]
    assert s.handle("/quit") is False and s.handle("/q") is False


def test_show_lists_everything(tok):
    s, _, out = make(tok, cfg=Config(temperature=0.6, system="sys"))
    s.handle("/show")
    txt = out[-1]
    for needle in ("raciocínio", "0.6", "max tokens", "repetition_penalty", "sys", "eos_token_id=[2, 11]"):
        assert needle in txt


# --------------------------------------------------------------------------------- prompt
def test_render_ends_per_mode(tok):
    s, _, _ = make(tok)
    assert s.render("oi", Config(think=True)).endswith("<think>\n")
    assert s.render("oi", Config(think=False)).endswith("<think></think>")


def test_render_system_prefill_and_history(tok):
    s, _, _ = make(tok)
    s.history = [{"role": "user", "content": "primeira"}, {"role": "assistant", "content": "resposta1"}]
    off = s.render("segunda", Config(system="SISTEMA"))
    assert "SISTEMA" in off and "primeira" not in off  # chat desligado: sem histórico
    on = s.render("segunda", Config(chat=True, prefill="Vou pensar"))
    assert "primeira" in on and "resposta1" in on and on.endswith("<think>\nVou pensar")


# ------------------------------------------------------------------------------------ turno
def test_ask_prints_text_and_stats(tok):
    s, fake, out = make(tok, "pensando…</think>\nResposta: 4")
    r = s.ask("Quanto é 2+2?")
    assert out[0] == "pensando…</think>\nResposta: 4"
    assert "10 tokens" in out[1] and "fechou </think>" in out[1] and "terminou" in out[1]
    assert r["parsed"].answer == "Resposta: 4"
    assert fake.calls[0][2] is True  # in_trace: modo com raciocínio


def test_ask_reports_truncated_and_unclosed(tok):
    s, _, out = make(tok, "raciocínio sem fim", finished=False)
    s.ask("oi")
    assert "TRUNCADO em 4096" in out[1] and "NÃO fechou </think>" in out[1]


def test_prefill_goes_into_prompt_and_output(tok):
    s, fake, out = make(tok, " depois</think>fim", cfg=Config(prefill="Vou raciocinar em português."))
    s.ask("oi")
    assert fake.calls[0][0].endswith("<think>\nVou raciocinar em português.")
    assert out[0] == "Vou raciocinar em português. depois</think>fim"


def test_prefill_with_close_means_not_in_trace(tok):
    s, fake, _ = make(tok, "resposta", cfg=Config(prefill="pensei.</think>"))
    s.ask("oi")
    assert fake.calls[0][2] is False


def test_no_think_not_in_trace(tok):
    s, fake, _ = make(tok, "Resposta: 4", cfg=Config(think=False))
    s.ask("oi")
    assert fake.calls[0][2] is False


# ------------------------------------------------------------------------ histórico e /retry
def test_chat_off_does_not_store(tok):
    s, _, _ = make(tok, "t</think>a")
    s.ask("oi")
    assert s.history == []


def test_chat_on_stores_answer_only_and_next_prompt_has_it(tok):
    s, fake, _ = make(tok, "t</think>Paris", "t2</think>Sim", cfg=Config(chat=True))
    s.ask("Capital da França?")
    assert s.history == [{"role": "user", "content": "Capital da França?"}, {"role": "assistant", "content": "Paris"}]
    s.ask("E da Itália?")
    assert "Paris" in fake.calls[1][0] and "t</think>" not in fake.calls[1][0]  # o traço não vai para o histórico


def test_chat_on_does_not_store_unclosed_thinking(tok):
    s, _, out = make(tok, "raciocínio sem fim", cfg=Config(chat=True), finished=False)
    s.ask("oi")
    assert s.history == [] and "não guardado" in out[-1]


def test_retry_replaces_last_turn_instead_of_duplicating(tok):
    s, fake, _ = make(tok, "t</think>A", "t</think>B", cfg=Config(chat=True))
    s.ask("pergunta")
    s.handle("/retry")
    assert [m["content"] for m in s.history] == ["pergunta", "B"]
    assert fake.calls[1][0].count("pergunta") == 1


def test_retry_uses_current_config(tok):
    s, fake, _ = make(tok, "t</think>A", "B")
    s.ask("pergunta")
    s.handle("/think off")
    s.handle("/retry")
    assert fake.calls[1][0].endswith("<think></think>") and fake.calls[1][1].think is False


def test_retry_without_previous_prompt(tok):
    s, fake, out = make(tok)
    s.handle("/retry")
    assert fake.calls == [] and "nenhum prompt" in out[-1]


def test_both_runs_with_and_without_thinking_and_keeps_history(tok):
    s, fake, out = make(tok, "t</think>A", "B", "t</think>X", "Y", cfg=Config(chat=True))
    s.ask("pergunta")
    n = len(s.history)
    s.handle("/both")
    assert [c[1].think for c in fake.calls[1:]] == [True, False]
    assert s.cfg.think is True  # a configuração do usuário não muda
    assert len(s.history) == n and any("sem raciocínio" in o for o in out)


def test_prompt_command_shows_exact_last_prompt(tok):
    s, _, out = make(tok)
    s.handle("/prompt")
    assert "nenhum" in out[-1]
    s.ask("olá")
    s.handle("/prompt")
    assert out[-1].endswith("<think>\n") and "olá" in out[-1] and "<|im_start|>" in out[-1]


def test_keyboard_interrupt_during_generation_is_survivable(tok):
    out: list[str] = []

    def boom(prompt, cfg, in_trace):
        raise KeyboardInterrupt

    s = Session(tok, boom, Config(), out=out.append)
    assert s.ask("oi") is None and "[interrompido]" in out[-1]


# ------------------------------------------------------------------------------------- REPL
def test_repl_loop_with_commands_multiline_and_quit(tok):
    s, fake, out = make(tok, "r1", "r2")
    lines = iter(["/think off", "", "primeira linha", '"""', "linha A", "linha B", '"""', "/quit", "nunca lida"])
    run_repl(s, read=lambda prompt: next(lines))
    assert len(fake.calls) == 2
    assert "linha A\nlinha B" in fake.calls[1][0]
    assert s.cfg.think is False


def test_repl_ends_on_eof(tok):
    s, _, _ = make(tok)

    def eof(prompt):
        raise EOFError

    run_repl(s, read=eof)
