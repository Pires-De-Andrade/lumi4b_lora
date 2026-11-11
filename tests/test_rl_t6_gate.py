import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from rl_t6_gate import gate  # noqa: E402


def _s(acc, closed, pt, other=0):
    return {"mcq_val|thinking": {"n": 385, "strict_acc": acc, "closed_rate": closed, "trace_language": {"pt": pt, "en": other}}}


def test_passa_so_com_os_tres_requisitos():
    assert gate(_s(0.76, 0.95, 99, 1))["passa"]
    assert not gate(_s(0.7325, 0.9558, 99, 1))["passa"]  # o T6-Lumi sozinho não passa (precisa de +2,0)
    assert not gate(_s(0.76, 0.89, 99, 1))["passa"]  # fecha menos de 90%
    assert not gate(_s(0.76, 0.95, 80, 20))["passa"]  # português abaixo de 90%
    assert gate(_s(0.752, 0.90, 90, 10))["passa"]  # os limiares valem com >=
