"""GRPO: recompensas, vantagens, seleção de prompts, token de parada e loss (CPU, sem o Lumi)."""

import math

import pytest
import torch

from lumi_ft.train import grpo as G

PT = "Vou raciocinar em português. Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5 porque ela já comprou"
EN = "First, we can calculate the total of the apples that she has for each day, then it is 5 and the answer is"


def test_reward_math():
    ok = f"{PT}.\n</think>\nTotal 42.\n\n#### 42"
    assert G.reward_math(ok, True, "42") == (1.0, "ok")
    assert G.reward_math(f"{PT}.\n</think>\nTotal.\n#### 41", True, "42") == (0.0, "resposta_errada_ou_sem_marcador")
    assert G.reward_math(f"{PT}. mais mais", False, "42") == (0.0, "truncado")
    assert G.reward_math(f"{EN}.\n</think>\nTotal.\n#### 42", True, "42") == (0.0, "traco_nao_pt")


def test_reward_mcq():
    ok = f"{PT}.\n</think>\n\nA alternativa B está certa.\n\nResposta: B"
    assert G.reward_mcq(ok, True, "B", 4) == (1.0, "ok")
    assert G.reward_mcq(ok, True, "C", 4) == (0.0, "resposta_errada_ou_sem_marcador")
    assert G.reward_mcq(ok, False, "B", 4) == (0.0, "truncado")
    assert G.reward_mcq(f"{PT}. sem fechar\n\nResposta: B", True, "B", 4)[1] == "sem_um_unico_close"
    assert G.reward_mcq(f"{EN}.\n</think>\n\nÉ a B.\n\nResposta: B", True, "B", 4) == (0.0, "traco_nao_pt")
    assert G.reward_mcq(f"{PT}.\n</think>\n\nÉ a E.\n\nResposta: E", True, "E", 4)[0] == 0.0  # letra fora das alternativas


def test_reward_for_dispatch():
    ok = f"{PT}.\n</think>\n\nÉ a B.\n\nResposta: B"
    assert G.reward_for({"kind": "mcq", "gold": "B", "n_options": 4}, ok, True)[0] == 1.0
    assert G.reward_for({"kind": "math", "ref": "42"}, f"{PT}.\n</think>\nx\n\n#### 42", True)[0] == 1.0


def test_group_advantages():
    a = G.group_advantages([1, 0, 0, 0], "mean")
    assert a == pytest.approx([0.75, -0.25, -0.25, -0.25])
    s = G.group_advantages([1, 0, 0, 0], "std")
    assert sum(s) == pytest.approx(0, abs=1e-9) and s[0] > 0 > s[1]
    assert (sum(x * x for x in s) / 4) == pytest.approx(1.0, abs=1e-3)  # variância ~1
    assert G.group_advantages([1, 1, 1, 1]) == [0.0] * 4 and G.group_advantages([0, 0]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        G.group_advantages([1, 0], "x")


def test_select_prompts_deterministic_and_mixed():
    ps = [{"id": f"m{i}", "kind": "math"} for i in range(10)] + [{"id": f"q{i}", "kind": "mcq"} for i in range(6)]
    a = G.select_prompts(ps, 0, 8, 0.5)
    assert a == G.select_prompts(ps, 0, 8, 0.5)
    assert sum(p["kind"] == "mcq" for p in a) == 4 and len({p["id"] for p in a}) == 8
    b = G.select_prompts(ps, 1, 8, 0.5)
    assert not ({p["id"] for p in a} & {p["id"] for p in b if p["kind"] == "math"})  # sem repetir antes de acabar
    assert all(p["kind"] == "math" for p in G.select_prompts(ps, 0, 5, 0.0))
    assert G.select_prompts([p for p in ps if p["kind"] == "math"], 0, 4, 0.5) and len(G.select_prompts([p for p in ps if p["kind"] == "math"], 0, 4, 0.5)) == 2


def test_complete_ids():
    assert G.complete_ids([5, 6, 11], "stop", 11) == ([5, 6, 11], False)  # o vLLM já incluiu o token de parada
    assert G.complete_ids([5, 6], "stop", 11) == ([5, 6, 11], True)
    assert G.complete_ids([5, 6], "stop", None) == ([5, 6, 11], True)
    assert G.complete_ids([5, 6], "length", None) == ([5, 6], False)  # truncada: não acrescenta


def test_build_samples_rewards_and_advantages():
    pr = {"id": "p", "kind": "math", "ref": "42"}
    good = f"{PT}.\n</think>\nx\n\n#### 42"
    bad = f"{PT}.\n</think>\nx\n\n#### 7"
    outs = [{"ids": [1, 11], "text": good, "finished": True}, {"ids": [2, 11], "text": bad, "finished": True},
            {"ids": [3, 11], "text": bad, "finished": True}, {"ids": [4], "text": PT, "finished": False}]
    samples, whys = G.build_samples(pr, [9, 9], outs, "mean")
    assert [s["reward"] for s in samples] == [1.0, 0.0, 0.0, 0.0] and samples[0]["advantage"] == pytest.approx(0.75)
    assert whys == {"ok": 1, "resposta_errada_ou_sem_marcador": 2, "truncado": 1} and samples[0]["prompt_ids"] == [9, 9]


def test_completion_logprobs_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(7, 11)
    ids = torch.randint(0, 11, (7,))
    lp = G.completion_logprobs(logits, ids, n_prompt=3, n_total=6)  # tokens nas posições 3, 4, 5, previstos pelas posições 2, 3, 4
    ref = [torch.log_softmax(logits[p - 1], -1)[ids[p]] for p in (3, 4, 5)]
    assert torch.allclose(lp, torch.stack(ref), atol=1e-6) and lp.shape == (3,)


def test_ppo_loss_gradient_unclipped_and_clipped():
    lo = torch.zeros(4)
    ln = torch.zeros(4, requires_grad=True)
    G.ppo_clip_loss_sum(ln, lo, 2.0).backward()
    assert torch.allclose(ln.grad, torch.full((4,), -2.0))  # ratio 1: gradiente -A por token
    # A > 0 e razão acima de 1 + eps_high: cortado, gradiente zero
    ln = torch.full((3,), math.log(1.5), requires_grad=True)
    G.ppo_clip_loss_sum(ln, torch.zeros(3), 1.0).backward()
    assert torch.allclose(ln.grad, torch.zeros(3))
    # A < 0 e razão abaixo de 1 - eps_low: cortado
    ln = torch.full((3,), math.log(0.5), requires_grad=True)
    G.ppo_clip_loss_sum(ln, torch.zeros(3), -1.0).backward()
    assert torch.allclose(ln.grad, torch.zeros(3))
    # A > 0 e razão abaixo de 1: continua com gradiente (o corte só protege o lado que já melhorou)
    ln = torch.full((3,), math.log(0.5), requires_grad=True)
    G.ppo_clip_loss_sum(ln, torch.zeros(3), 1.0).backward()
    assert (ln.grad < 0).all()


def test_completion_entropy_known_values():
    V, n_prompt, n_total = 8, 2, 6
    uniform = torch.zeros(n_total + 1, V)
    assert torch.allclose(G.completion_entropy(uniform, n_prompt, n_total), torch.full((4,), math.log(V)), atol=1e-6)
    peaked = torch.full((n_total + 1, V), -1e4)
    peaked[:, 3] = 0.0
    assert torch.allclose(G.completion_entropy(peaked, n_prompt, n_total), torch.zeros(4), atol=1e-6)
    torch.manual_seed(0)
    lg = torch.randn(n_total + 1, V)
    ref = [-(torch.softmax(lg[p - 1], -1) * torch.log_softmax(lg[p - 1], -1)).sum() for p in range(n_prompt, n_total)]
    assert torch.allclose(G.completion_entropy(lg, n_prompt, n_total, chunk=3), torch.stack(ref), atol=1e-6)  # blocos pequenos dão o mesmo


def test_gspo_loss_gradient():
    lo = torch.zeros(5)
    ln = torch.zeros(5, requires_grad=True)
    G.gspo_loss_sum(ln, lo, 2.0).backward()
    assert torch.allclose(ln.grad, torch.full((5,), -2.0))  # razão 1: gradiente -A por token, como no PPO por token
    ln = torch.full((5,), 0.1, requires_grad=True)  # razão da sequência e^0,1 acima de 1 + 4e-4, A > 0: cortado
    G.gspo_loss_sum(ln, lo, 1.0).backward()
    assert torch.allclose(ln.grad, torch.zeros(5))
    ln = torch.full((5,), -0.1, requires_grad=True)  # razão abaixo de 1 - 3e-4, A < 0: cortado
    G.gspo_loss_sum(ln, lo, -1.0).backward()
    assert torch.allclose(ln.grad, torch.zeros(5))
    ln = torch.full((5,), -0.1, requires_grad=True)  # razão abaixo de 1 e A > 0: continua com gradiente
    G.gspo_loss_sum(ln, lo, 1.0).backward()
    assert (ln.grad < 0).all()
    # a razão é da sequência: um token muito acima e outro muito abaixo se compensam na média geométrica
    ln = torch.tensor([0.5, -0.5], requires_grad=True)
    loss = G.gspo_loss_sum(ln, torch.zeros(2), 1.0)
    assert float(loss) == pytest.approx(-2.0)


def test_noclose_share_and_guard(tmp_path, monkeypatch):
    stats = {"whys": {"mcq:ok": 80, "mcq:sem_um_unico_close": 10, "mcq:truncado": 10, "math:ok": 98, "math:sem_um_unico_close": 2}}
    share = G.noclose_share(stats)
    assert share == pytest.approx({"mcq": 0.10, "math": 0.02})
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("grpo_guard", Path(__file__).resolve().parent.parent / "scripts" / "grpo_guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    import json

    f = tmp_path / "s.json"
    f.write_text(json.dumps(stats))
    monkeypatch.setattr(sys, "argv", ["x", str(f), "--max-noclose", "0.05"])
    assert guard.main() == 3  # mcq passa de 5%
    monkeypatch.setattr(sys, "argv", ["x", str(f), "--max-noclose", "0.15"])
    assert guard.main() == 0


def test_kl_k3_sum():
    lp = torch.tensor([-0.5, -1.0, -2.0])
    assert float(G.kl_k3_sum(lp, lp)) == pytest.approx(0.0)
    ref = torch.tensor([-0.4, -1.2, -0.5])
    r = ref - lp
    assert float(G.kl_k3_sum(lp, ref)) == pytest.approx(float((torch.exp(r) - r - 1).sum()))
    assert float(G.kl_k3_sum(lp, ref)) > 0
    x = lp.clone().requires_grad_(True)
    G.kl_k3_sum(x, ref).backward()
    assert torch.allclose(x.grad, 1 - torch.exp(r))  # puxa a política de volta para a referência
    assert x.grad[2] < x.grad[0]  # e cresce em módulo quando ela se afasta (a razão ref/pi fica grande)


def test_layer_mask_selection():
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("merge_adapter_layers", Path(__file__).resolve().parent.parent / "scripts" / "merge_adapter_layers.py")
    L = importlib.util.module_from_spec(spec)
    sys.modules["merge_adapter_layers"] = L
    spec.loader.exec_module(L)
    mlp = L.TYPES["mlp"]
    assert L.keep("backbone.layers.5.mixer.up_proj", 0, 20, mlp) is True
    assert L.keep("backbone.layers.30.mixer.up_proj", 0, 20, mlp) is False  # fora da faixa
    assert L.keep("backbone.layers.5.mixer.in_proj", 0, 20, mlp) is False  # outro tipo de bloco
    assert L.keep("backbone.embeddings", 0, 41, mlp) is None
    assert sum(len(v) for v in L.TYPES.values()) == 7
