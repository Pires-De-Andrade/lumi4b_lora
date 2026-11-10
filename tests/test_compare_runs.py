"""Bootstrap pareado da nota média (scripts/compare_runs.py): ponto, IC e propriedades básicas."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_runs import paired_bootstrap, paired_bootstrap_mean  # noqa: E402


def _provas(seed=0):
    rng = random.Random(seed)
    return [[int(rng.random() < p) for _ in range(n)] for p, n in ((0.8, 400), (0.5, 900), (0.3, 250))]


def test_ponto_e_a_media_das_diferencas_por_prova():
    a, b = _provas(1), _provas(2)
    diff, lo, hi = paired_bootstrap_mean(list(zip(a, b)))
    esperado = sum(sum(x) / len(x) - sum(y) / len(y) for x, y in zip(a, b)) / 3
    assert abs(diff - esperado) < 1e-12
    assert lo <= diff <= hi


def test_avaliacoes_iguais_tem_diferenca_zero_e_ic_em_zero():
    a = _provas(3)
    diff, lo, hi = paired_bootstrap_mean(list(zip(a, a)))
    assert diff == 0.0 and lo == 0.0 and hi == 0.0


def test_ganho_claro_deixa_o_ic_acima_de_zero_e_perda_clara_abaixo():
    ruim = [[0] * 300 for _ in range(3)]
    bom = [[1] * 300 for _ in range(3)]
    diff, lo, hi = paired_bootstrap_mean(list(zip(bom, ruim)))
    assert diff == 1.0 and lo == 1.0
    diff, lo, hi = paired_bootstrap_mean(list(zip(ruim, bom)))
    assert diff == -1.0 and hi == -1.0


def test_com_uma_prova_equivale_ao_bootstrap_pareado_simples_no_ponto_e_na_largura():
    rng = random.Random(5)
    a = [int(rng.random() < 0.6) for _ in range(600)]
    b = [int(rng.random() < 0.55) for _ in range(600)]
    d1, lo1, hi1 = paired_bootstrap(a, b)
    d2, lo2, hi2 = paired_bootstrap_mean([(a, b)])
    assert abs(d1 - d2) < 1e-12
    assert abs((hi1 - lo1) - (hi2 - lo2)) < 0.01  # mesmas distribuições, sorteios diferentes
