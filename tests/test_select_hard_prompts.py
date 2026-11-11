import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from select_hard_prompts import pass_counts, select  # noqa: E402


def _grupo(pid, rewards, finished=None):
    finished = finished if finished is not None else [True] * len(rewards)
    return [{"prompt_id": pid, "reward": r, "finished": f} for r, f in zip(rewards, finished)]


def test_so_ficam_os_prompts_com_sinal(tmp_path):
    rows = (_grupo("a", [1.0] * 4) + _grupo("b", [0.0] * 4) + _grupo("c", [1.0, 0.0, 0.0, 0.0]) + _grupo("d", [1.0, 1.0, 1.0, 0.0], [True, True, True, False])
            + _grupo("e", [1.0, 0.0]))  # grupo incompleto: fica de fora
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    counts = pass_counts(f)
    assert counts["c"] == [4, 1, 0] and counts["d"] == [4, 3, 1]
    pool = [{"id": i, "kind": "mcq" if i in "abc" else "math"} for i in "abcdef"]  # "f" não foi gerado
    kept, resumo = select(pool, counts, group=4)
    assert [p["id"] for p in kept] == ["c", "d"]
    assert kept[0]["t6_correct_of_group"] == 1 and kept[1]["t6_truncated_of_group"] == 1
    assert resumo["mcq"]["sempre_certos"] == 1 and resumo["mcq"]["sempre_errados"] == 1 and resumo["mcq"]["com_sinal"] == 1
    assert resumo["math"]["prompts_gerados"] == 1  # só o "d" (o "e" tem grupo incompleto e o "f" não existe nos rollouts)
