"""Diagnósticos do GRPO (só registro, sem mudar a perda nem a recompensa): corte ativo por sequência e por token, requisitos da
recompensa como indicadores independentes e resumo dos rollouts por grupo (CPU, sem o Lumi)."""

import pytest
import torch

from lumi_ft.train import grpo as G

PT = "Vou raciocinar em português. Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5 porque ela já comprou"
EN = "First, we can calculate the total of the apples that she has for each day, then it is 5 and the answer is"
EPS = (3e-4, 4e-4)


# ------------------------------------------------------------------------------- corte ativo
def _grad_is_zero(loss_fn, lp_new, lp_old, adv, *eps):
    x = lp_new.clone().requires_grad_(True)
    loss_fn(x, lp_old, adv, *eps).backward()
    return bool((x.grad == 0).all())


@pytest.mark.parametrize("delta,adv,active,outside", [
    (+1e-3, +1.0, True, True),    # A > 0 e s acima do limite: corte ativo, gradiente zero
    (+1e-3, -1.0, False, True),   # A < 0 e s acima: fora do intervalo, mas o gradiente passa
    (-1e-3, -1.0, True, True),    # A < 0 e s abaixo do limite: corte ativo
    (-1e-3, +1.0, False, True),   # A > 0 e s abaixo: fora, mas o gradiente passa
    (0.0, +1.0, False, False),    # a razão começa em 1: nada cortado
    (1e-4, +1.0, False, False),   # dentro de [1-3e-4, 1+4e-4]
])
def test_seq_clip_stats_bate_com_o_gradiente_da_perda(delta, adv, active, outside):
    old = torch.zeros(50)
    new = old + delta
    cs = G.seq_clip_stats(new, old, adv, *EPS)
    assert cs["active"] is active and cs["outside"] is outside and cs["n"] == 50
    assert cs["log_s"] == pytest.approx(delta, abs=1e-6)
    assert cs["active"] == _grad_is_zero(G.gspo_loss_sum, new, old, adv, *EPS)  # "ativo" é exatamente "gradiente zero"


def test_seq_clip_stats_usa_a_media_da_sequencia_e_nao_o_token():
    old = torch.zeros(4)
    new = torch.tensor([0.5, -0.5, 0.0, 0.0])  # tokens muito fora, mas a média é zero: s = 1
    cs = G.seq_clip_stats(new, old, +1.0, *EPS)
    assert not cs["outside"] and not cs["active"]
    # o diagnóstico antigo (por token contra os limites do GSPO) veria 50% dos tokens "cortados"
    r = torch.exp(new - old)
    assert float(((r < 1 - EPS[0]) | (r > 1 + EPS[1])).float().mean()) == 0.5


def test_token_clip_active_bate_com_o_gradiente_do_ppo():
    old = torch.zeros(4)
    new = torch.tensor([0.5, 0.1, -0.1, -0.5])  # r = 1,65; 1,11; 0,90; 0,61 (limites 0,8 e 1,28)
    for adv, idx in ((+1.0, 0), (-1.0, 3)):
        assert G.token_clip_active(new, old, adv, 0.2, 0.28) == (1, 4)
        x = new.clone().requires_grad_(True)
        G.ppo_clip_loss_sum(x, old, adv, 0.2, 0.28).backward()
        assert [bool(g == 0) for g in x.grad] == [k == idx for k in range(4)]


# ------------------------------------------------------------------ requisitos da recompensa
MATH = {"id": "m", "kind": "math", "ref": "42"}
MCQ = {"id": "q", "kind": "mcq", "gold": "B", "n_options": 4}
CASES = [
    # (prompt, texto, terminou, partes esperadas finished/closed_once/nonempty/correct/pt)
    (MATH, f"{PT}.\n</think>\nTotal 42.\n\n#### 42", True, (1, 1, 1, 1, 1)),
    (MATH, f"{PT}.\n</think>\nTotal.\n\n#### 41", True, (1, 1, 1, 0, 1)),                   # errou
    (MATH, f"{EN}.\n</think>\nTotal.\n\n#### 42", True, (1, 1, 1, 1, 0)),                   # acertou em inglês: o "whys" diria só "não português"
    (MATH, f"{EN}.\n</think>\nTotal.\n\n#### 41", True, (1, 1, 1, 0, 0)),                   # errou em inglês: o "whys" diria só "errada"
    (MATH, f"{PT}. mais mais mais", False, (0, 0, 0, 0, 1)),                                # truncada, sem fechar
    (MATH, f"{PT}.\n</think>\nTotal.\n\n#### 42\n</think>\nmais", True, (1, 0, 1, 1, 1)),   # dois </think>
    (MATH, f"{PT}.\n</think>\n", True, (1, 1, 0, 0, 1)),                                    # resposta vazia
    (MCQ, f"{PT}.\n</think>\n\nA alternativa B está certa.\n\nResposta: B", True, (1, 1, 1, 1, 1)),
    (MCQ, f"{PT}.\n</think>\n\nA alternativa C está certa.\n\nResposta: C", True, (1, 1, 1, 0, 1)),
    (MCQ, f"{EN}.\n</think>\n\nThe option B is right.\n\nResposta: B", True, (1, 1, 1, 1, 0)),
    (MCQ, f"{PT}. mais mais mais", False, (0, 0, 0, 0, 1)),
]


@pytest.mark.parametrize("prompt,text,finished,expected", CASES)
def test_reward_parts_independentes_e_consistentes_com_a_recompensa(prompt, text, finished, expected):
    parts = G.reward_parts(prompt, text, finished)
    assert tuple(parts[k] for k in G.PARTS) == expected
    reward, _ = G.reward_for(prompt, text, finished)
    assert (reward == 1.0) == all(parts.values())  # a recompensa é 1 se e só se os cinco requisitos valem


def test_build_samples_guarda_as_partes_sem_mudar_a_recompensa():
    outs = [{"ids": [1, 2], "text": t, "finished": f} for _, t, f, _ in CASES[:4]]
    prompt = {**MATH, "source": "gsm8k_train"}
    samples, whys = G.build_samples(prompt, [9], outs)
    assert [s["reward"] for s in samples] == [1.0, 0.0, 0.0, 0.0]
    assert samples[0]["parts"] == {"finished": 1, "closed_once": 1, "nonempty": 1, "correct": 1, "pt": 1} and samples[0]["source"] == "gsm8k_train"
    assert whys == {"ok": 1, "resposta_errada_ou_sem_marcador": 2, "traco_nao_pt": 1}  # o "whys" continua igual


# ------------------------------------------------------------------------- resumo dos grupos
def _s(kind, pid, reward, n_tokens, finished=1, correct=None, pt=1, source="medmcqa"):
    correct = int(reward == 1) if correct is None else correct
    return {"kind": kind, "prompt_id": pid, "reward": reward, "completion_ids": [0] * n_tokens, "finished": bool(finished), "source": source,
            "parts": {"finished": finished, "closed_once": 1, "nonempty": 1, "correct": correct, "pt": pt}}


def test_summarize_rollouts_particiona_os_grupos_e_explica_os_zeros():
    s = []
    s += [_s("mcq", "a", 1.0, 300) for _ in range(4)]                                        # todos certos
    s += [_s("mcq", "b", 0.0, 2048, finished=0, correct=0) for _ in range(4)]                # todos zero: truncadas
    s += [_s("mcq", "c", 0.0, 700, correct=1, pt=0) for _ in range(4)]                       # todos zero: acertou o conteúdo, falhou no idioma
    s += [_s("mcq", "d", 0.0, 700, correct=0) for _ in range(4)]                             # todos zero: erradas
    s += [_s("mcq", "e", 1.0, 100), _s("mcq", "e", 0.0, 100), _s("mcq", "e", 0.0, 100), _s("mcq", "e", 1.0, 100)]  # misto
    s += [_s("math", "f", 1.0, 50, source=None) for _ in range(4)]
    out = G.summarize_rollouts(s)
    g = out["groups"]["mcq"]
    assert (g["n"], g["todos_certos"], g["todos_zero"], g["mistos"]) == (5, 1, 3, 1)
    assert g["causas_todos_zero"] == {"todas_truncadas": 1, "alguma_correta_mas_reprovada": 1, "erradas": 1}
    assert out["groups"]["math"] == {"n": 1, "todos_certos": 1, "todos_zero": 0, "mistos": 0, "causas_todos_zero": {}}
    f = out["flags"]["mcq"]
    assert f["n"] == 20 and f["finished"] == 0.8 and f["pt"] == 0.8 and f["correct"] == pytest.approx(0.5)
    assert out["by_length"]["mcq"]["<256"] == {"n": 4, "reward": 0.5, "truncated": 0.0}  # o grupo misto (100 tokens)
    assert out["by_length"]["mcq"]["256-511"] == {"n": 4, "reward": 1.0, "truncated": 0.0}  # os todos certos (300 tokens)
    assert out["by_length"]["mcq"][">=2048"]["truncated"] == 1.0 and out["by_length"]["mcq"]["512-1023"]["n"] == 8
    assert out["by_source"]["mcq:medmcqa"]["n"] == 20 and out["by_source"]["math:-"]["reward"] == 1.0


def test_length_bin_nos_limites():
    assert [G._length_bin(n) for n in (0, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 5000)] == [
        "<256", "<256", "256-511", "256-511", "512-1023", "512-1023", "1024-2047", "1024-2047", ">=2048", ">=2048"]
