"""Fase anti-laço (src/lumi_ft/train/antidoom.py): detector de laço, escolha dos substitutos e perda."""
import pytest
import torch

from lumi_ft.train import antidoom as A


# ---------------------------------------------------------------- detector de laço
def test_laco_simples_no_fim():
    pre = [100, 101, 102, 103, 104]
    block = [1, 2, 3, 4, 5, 6]
    ids = pre + block * 5  # o bloco aparece 5 vezes
    s, k = A.find_loop_start(ids)
    assert k == 6
    assert s == len(pre) + 6  # primeiro token da primeira repetição (a segunda cópia)


def test_laco_com_ultima_copia_cortada():
    block = [7, 8, 9, 10, 11]
    ids = [1, 2] + block * 4 + block[:3]
    s, k = A.find_loop_start(ids)
    assert (s, k) == (2 + 5, 5)


def test_sem_laco_e_laco_curto_demais():
    assert A.find_loop_start(list(range(300))) is None
    assert A.find_loop_start([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) is None  # nenhuma repetição
    assert A.find_loop_start([5, 6, 7, 8] * 3) is None  # só 3 cópias: abaixo do mínimo de 4


def test_laco_so_vale_se_vai_ate_o_fim():
    ids = [1, 2, 3, 4, 5] * 5 + [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # saiu do laço
    assert A.find_loop_start(ids) is None


def test_menor_periodo_e_bloco_minimo_em_caracteres():
    block = [3, 4, 5, 6]
    ids = [1] * 3 + block * 10
    assert A.find_loop_start(ids) == (3 + 4, 4)
    decode = lambda toks: "xxxxx" * len(toks)  # cada token vale 5 caracteres: o bloco de 4 tokens tem 20
    assert A.find_loop_start(ids, min_chars=10, decode=decode) == (3 + 4, 4)
    assert A.find_loop_start(ids, min_chars=1000, decode=decode) is None  # bloco pequeno demais em caracteres
    assert A.find_loop_start(ids, min_chars=10, decode=lambda toks: "x" * len(toks)) is None  # 4 caracteres < 10


# ---------------------------------------------------------------- escolha dos substitutos
VOCAB = {0: "<eos>", 1: " the", 2: ",", 3: "Wait", 4: "So", 5: " ", 6: "Então", 7: "a", 8: "porque", 9: ".\n"}


def _decode(toks):
    return "".join(VOCAB[t] for t in toks)


def test_candidatos_filtram_curtos_pontuacao_especiais_e_o_rejeitado():
    logits = torch.tensor([9.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0])
    out = A.choose_candidates(logits, rejected=3, decode=_decode, top_k=10, max_chosen=3, exclude={0})
    assert out == [1, 4, 6]  # 0 é especial, 2 e 5 e 9 não têm 2 alfanuméricos ou são pontuação, 3 é o rejeitado; 7 ("a") é curto demais
    assert A.choose_candidates(logits, 3, _decode, top_k=10, max_chosen=20, exclude={0}) == [1, 4, 6, 8]


# ---------------------------------------------------------------- perda
def _setup(V=50):
    torch.manual_seed(0)
    z0 = torch.randn(V) * 2
    rejected = int(z0.argmax())
    chosen = [i for i in torch.topk(z0, 6).indices.tolist() if i != rejected][:4]
    return z0, chosen, rejected


def test_perda_zero_movimento_so_tem_a_preferencia():
    z0, chosen, rejected = _setup()
    loss, m = A.ftpo_loss(z0.clone().requires_grad_(True), z0, chosen, rejected)
    assert m["rest"] == 0.0 and m["cr"] == 0.0 and m["margin"] == 0.0
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(2.0))), abs=1e-6)  # softplus(0)
    assert m["rejected_is_argmax"] == 1.0 and m["chosen_win"] == 0.0


def test_gradiente_sobe_escolhidos_desce_rejeitado_e_segura_o_resto():
    z0, chosen, rejected = _setup()
    z = z0.clone().requires_grad_(True)
    loss, _ = A.ftpo_loss(z, z0, chosen, rejected, lam_rest=5.0)
    loss.backward()
    g = z.grad
    assert g[rejected] > 0  # descer o logit rejeitado = gradiente positivo da perda
    assert all(g[c] < 0 for c in chosen)
    others = [i for i in range(len(z0)) if i not in chosen and i != rejected]
    assert float(g[others].abs().max()) == 0.0  # no ponto de referência o resto não recebe gradiente


def test_o_resto_do_vocabulario_e_penalizado_quando_se_afasta():
    z0, chosen, rejected = _setup()
    others = [i for i in range(len(z0)) if i not in chosen and i != rejected]
    z = z0.clone()
    z[others] += 1.0
    loss0, m0 = A.ftpo_loss(z0.clone(), z0, chosen, rejected, lam_rest=5.0)
    loss1, m1 = A.ftpo_loss(z, z0, chosen, rejected, lam_rest=5.0)
    assert m1["rest"] > 0 and float(loss1) > float(loss0)


def test_dobradica_so_cobra_acima_de_tau_e_metricas_de_vitoria():
    z0, chosen, rejected = _setup()
    z = z0.clone()
    z[rejected] -= 2.0  # dentro de tau = 3: sem custo
    _, m = A.ftpo_loss(z, z0, chosen, rejected, tau=3.0)
    assert m["cr"] == 0.0
    z[rejected] -= 10.0
    z[chosen] += 10.0
    _, m = A.ftpo_loss(z, z0, chosen, rejected, tau=3.0)
    assert m["cr"] > 0 and m["chosen_win"] == 1.0 and m["rejected_is_argmax"] == 0.0 and m["margin"] > 0


# ---------------------------------------------------------------- conjunto de prompts
def _load_prompts_module():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("make_antidoom_prompts", Path(__file__).resolve().parents[1] / "scripts" / "make_antidoom_prompts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_prompts_particoes_disjuntas_e_deterministicas():
    M = _load_prompts_module()
    old = [{"id": f"mcq:{i}", "kind": "mcq", "prompt": f"q{i}"} for i in range(30)] + [{"id": f"m:{i}", "kind": "math", "prompt": f"p{i}"} for i in range(20)]
    gold = [{"id": f"g:{i}", "kind": "mcq", "prompt": f"g{i}"} for i in range(40)] + [{"id": "m:dup", "kind": "math", "prompt": "p0"}]
    a = M.build(old, gold, mcq_mine=50, math_mine=10, mcq_heldout=8, math_heldout=5, seed=0)
    b = M.build(old, gold, mcq_mine=50, math_mine=10, mcq_heldout=8, math_heldout=5, seed=0)
    assert a == b
    mine = [r for r in a if r["split"] == "mine"]
    held = [r for r in a if r["split"] == "heldout"]
    assert sum(r["kind"] == "mcq" for r in mine) == 50 and sum(r["kind"] == "math" for r in mine) == 10
    assert sum(r["kind"] == "mcq" for r in held) == 8 and sum(r["kind"] == "math" for r in held) == 5
    assert not ({r["prompt"] for r in mine} & {r["prompt"] for r in held})
    assert len({r["prompt"] for r in a}) == len(a)  # sem prompt repetido
    # as 30 do conjunto antigo entram todas na mineração (vêm primeiro)
    assert sum(r["id"].startswith("mcq:") for r in mine) == 30
