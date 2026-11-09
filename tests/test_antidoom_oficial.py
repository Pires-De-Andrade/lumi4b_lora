"""Receita oficial do Antidoom portada (src/lumi_ft/train/antidoom_oficial.py).

Os testes de equivalência comparam com o código do repositório oficial quando ele está clonado ao lado do projeto
(../antidoom_oficial, commit bd6a126); sem o clone, eles são pulados e só os testes da conta própria rodam.
"""
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from lumi_ft.train import antidoom_oficial as O

OFICIAL = Path(__file__).resolve().parents[2] / "antidoom_oficial" / "src"


def _oficial(modname):
    if not OFICIAL.is_dir():
        pytest.skip("repositório oficial não clonado em ../antidoom_oficial")
    if str(OFICIAL) not in sys.path:
        sys.path.insert(0, str(OFICIAL))
    import importlib
    import types
    if "antidoom.client" not in sys.modules:  # o cliente HTTP do original (requests) não é usado aqui; só as funções puras
        fake = types.ModuleType("antidoom.client")
        fake.CompletionChunk = type("CompletionChunk", (), {})
        fake.VllmClient = type("VllmClient", (), {})
        sys.modules["antidoom.client"] = fake
    try:
        return importlib.import_module(modname)
    except Exception as e:  # dependência do original ausente no ambiente
        pytest.skip(f"não importa {modname}: {e}")


# ---------------------------------------------------------------- detector e rejeitado
def _texto_com_laco(seed):
    rng = random.Random(seed)
    palavras = ["Na", "verdade", ",", "o", "osso", "hioide", "se", "origina", "do", "arco", "faríngeo", ".", "Então", "a", "resposta"]
    pre = " ".join(rng.choice(palavras) for _ in range(rng.randint(20, 80)))
    bloco = " ".join(rng.choice(palavras) for _ in range(rng.randint(3, 15))) + ". "
    pos = " ".join(rng.choice(palavras) for _ in range(rng.randint(0, 30)))
    return pre + " " + bloco * rng.randint(2, 9) + pos


def test_detector_igual_ao_oficial():
    R = _oficial("antidoom.repetition")
    for seed in range(300):
        t = _texto_com_laco(seed)
        a = O.find_inner_repetition(t)
        b = R.find_inner_repetition(t)
        assert a[0] == b[0]
        if a[0]:
            assert (a[1].start, a[1].end, a[1].period, a[1].repeats) == (b[1].start, b[1].end, b[1].period, b[1].repeats)


def test_detector_pega_laco_no_meio_e_exige_60_caracteres_no_total():
    rng = random.Random(7)
    letras = "abcdefghijklmnopqrstuvwxyz ,.;"
    sem_rep = lambda n: "".join(rng.choice(letras) for _ in range(n))
    bloco = "Na verdade, o osso. "  # 20 caracteres
    t = sem_rep(150) + bloco * 10 + sem_rep(400)  # laço no meio, seguido de texto que não repete
    found, hit = O.find_inner_repetition(t)
    assert found and hit.period == 20 and hit.repeats >= 4
    # amostragem a cada 128 caracteres: um laço curto que não contém nenhum ponto amostrado não é visto (comportamento do original)
    assert not O.find_inner_repetition(sem_rep(20) + "abcdefghijklmno" * 4 + "fim da resposta")[0]
    # menos de 60 caracteres repetidos no total não conta, mesmo com muitas repetições
    assert not O.find_inner_repetition(sem_rep(125) + "ab" * 25 + sem_rep(200))[0]  # 2 x 25 = 50 < 60
    assert O.find_inner_repetition(sem_rep(125) + "ab" * 40 + sem_rep(200))[0]  # 2 x 40 = 80


def test_rejeitado_token_aware_igual_ao_oficial(monkeypatch):
    monkeypatch.setattr(O, "RESTART_WORDS", O._RESTART_WORDS)  # compara com o conjunto original (sem os equivalentes em português)
    G = _oficial("antidoom.generate")
    T = _oficial("antidoom.tokens")
    for seed in range(200):
        rng = random.Random(seed)
        vocab = ["Ġthe", "Ġosso", "Ġhioide", ".", "Ċ", "ĠNa", "Ġverdade", ",", "ĠEntão", "Ġresposta", "Ġé", "ĠB", "Ġwait", "ĠSo"]
        pre = [rng.choice(vocab) for _ in range(rng.randint(5, 40))]
        bloco = [rng.choice(vocab) for _ in range(rng.randint(2, 10))]
        toks = pre + bloco * rng.randint(4, 8) + [rng.choice(vocab) for _ in range(rng.randint(0, 6))]
        ids = [vocab.index(t) for t in toks]
        ev = O.find_loop_event(ids, toks)
        st = T.TokenState("")
        st.append(toks, {})
        found, hit = _oficial("antidoom.repetition").find_inner_repetition(st.text)
        if not found:
            assert ev is None
            continue
        ref = G.select_rejected_token_index_token_aware(st, hit.start, hit.repeat_start, hit.end, ids)
        if ref is None:
            assert ev is None
        else:
            # as palavras de recomeço em português são a única diferença permitida; este vocabulário de teste só tem as do original
            assert ev is not None and ev["reject_idx"] == ref


# ---------------------------------------------------------------- escolhidos
def test_escolhidos_iguais_ao_oficial():
    S = _oficial("antidoom.sampling")
    rng = random.Random(1)
    vocab = [f"Ġtok{i}" for i in range(30)] + [",", "Ċ", " "]
    for _ in range(200):
        ids = rng.sample(range(len(vocab)), 20)
        lps = sorted([rng.uniform(-8, 0) for _ in ids], reverse=True)
        items = [(i, vocab[i], lp) for i, lp in zip(ids, lps)]
        rej = rng.choice(ids)
        ours = O.select_chosen_tokens(items, rejected_id=rej, rejected_token=vocab[rej])
        raw, _dec = S.select_chosen_tokens([(vocab[i], lp) for i, _t, lp in items], rejected_token=vocab[rej], temperature=0.1, min_p=0.01,
                                           top_k=50, max_tokens=20, min_decoded_chars=1, require_alnum=False)
        assert [vocab[i] for i in ours] == raw


def test_escolhidos_corte_min_p_na_temperatura_0_1_fica_no_topo():
    # na temperatura 0,1 o corte min-p de 0,01 guarda só alternativas com probabilidade >= 0,01^0,1 ~ 63% da maior
    import math
    items = [(1, "ĠA", math.log(0.40)), (2, "ĠB", math.log(0.30)), (3, "ĠC", math.log(0.20)), (4, "ĠD", math.log(0.05))]
    assert O.select_chosen_tokens(items, rejected_id=1, rejected_token="ĠA") == [3, 2]  # 0,20/0,30 = 0,67 >= 0,63; 0,05 fica fora
    assert O.select_chosen_tokens(items, rejected_id=2, rejected_token="ĠB") == [1]  # 0,20/0,40 = 0,5 < 0,63


# ---------------------------------------------------------------- regularização de frequência
def test_sample_rows_igual_ao_oficial_e_inerte_abaixo_do_teto():
    D = _oficial("antidoom.ftpo_data")
    rng = random.Random(3)
    rows = [{"rejected_decoded": rng.choice([" Wait", " So", " the", " B", " osso", " Na"] + [f" t{i}" for i in range(10)]),
             "source_dataset": rng.choice(["mcq", "math"]), "multi_chosen_decoded": ["x"], "i": i} for i in range(400)]
    for r in rows:
        r["kind"] = r["source_dataset"]
    for cap in (None, 150):
        ours = O.sample_rows(rows, max_train_examples=cap, rejected_strength=0.3, seed=5)
        ref = D.sample_rows(rows, max_train_examples=cap, rejected_strength=0.3, min_chosen_tokens=1, seed=5)
        assert [r["i"] for r in ours] == [r["i"] for r in ref]
    assert len(O.sample_rows(rows, max_train_examples=12000, rejected_strength=0.3, seed=0)) == len(rows)


def test_regularizacao_dos_escolhidos_igual_ao_oficial():
    D = _oficial("antidoom.ftpo_data")
    rng = random.Random(4)
    keys = ["wait", "so", "then", "na", "osso"] + [f"k{i}" for i in range(40)]
    exs = []
    for i in range(600):
        n = rng.randint(1, 5)
        ks = [rng.choice(keys[:5]) if rng.random() < 0.6 else rng.choice(keys) for _ in range(n)]
        exs.append({"prompt_ids": [i], "chosen_ids": [keys.index(k) for k in ks], "chosen_keys": ks, "rejected_token_id": 999})
    ours = O.regularise_chosen([dict(e, chosen_ids=list(e["chosen_ids"]), chosen_keys=list(e["chosen_keys"])) for e in exs], 0.5)
    ref = D._regularise_chosen_examples([dict(e, chosen_ids=list(e["chosen_ids"]), chosen_keys=list(e["chosen_keys"])) for e in exs], 0.5,
                                        Counter())
    assert [(e["prompt_ids"], e["chosen_ids"]) for e in ours] == [(e["prompt_ids"], e["chosen_ids"]) for e in ref]


# ---------------------------------------------------------------- perda
def _perda_referencia(z, z0, chosen, rej, eps=2.0, lm=0.4, lt=0.05, tau=0.5):
    """A conta do compute_loss oficial reescrita para um lote de 1 (independente da implementação testada)."""
    zl, zr = z.unsqueeze(0), z0.unsqueeze(0)
    ch = torch.tensor([chosen])
    mask = torch.ones_like(ch, dtype=torch.bool)
    rejt = torch.tensor([rej])
    delta = zl[torch.arange(1).unsqueeze(1), ch] - zl.gather(-1, rejt.unsqueeze(-1))
    w = torch.clamp((eps - delta) / eps, 0.0, 1.0) * mask
    pref = ((F.softplus((eps - delta) / 1.0) * w).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    tether = torch.ones_like(zl, dtype=torch.bool)
    tether[0, ch[0]] = False
    tether.scatter_(1, rejt.unsqueeze(-1), False)
    diff = zl - zr
    mse_elem = (tether * diff.pow(2)).sum() / tether.sum()
    tgt = ~tether
    excess = torch.clamp((diff * tgt).abs() - tau, min=0.0)
    mse_tgt = excess.pow(2).sum() / tgt.sum()
    return pref + lm * mse_elem + lt * mse_tgt


def test_perda_igual_a_conta_oficial_e_gradiente():
    torch.manual_seed(0)
    for _ in range(20):
        z0 = torch.randn(300) * 3
        z = (z0 + torch.randn(300) * 0.7).requires_grad_(True)
        rej = int(z0.argmax())
        chosen = [i for i in torch.topk(z0, 8).indices.tolist() if i != rej][:5]
        loss, m = O.ftpo_loss_oficial(z, z0, chosen, rej)
        ref = _perda_referencia(z.detach().clone(), z0, chosen, rej)
        assert float(loss) == pytest.approx(float(ref), rel=1e-5, abs=1e-6)
        loss.backward()
        assert z.grad[rej] > 0 or m["active_weight"] == 0.0
    # a preferência desliga quando todos os escolhidos passam o rejeitado por eps
    z0 = torch.zeros(50)
    z = z0.clone()
    z[[1, 2]] = 5.0
    _, m = O.ftpo_loss_oficial(z, z0, [1, 2], 0)
    assert m["pref_loss"] == 0.0 and m["chosen_win"] == 1.0 and m["margin_win"] == 1.0
