"""hf/t6-lumi/publicar.py: só a lista permitida vai ao Hugging Face (nada de handoff, scripts ou estudos) e o envio é recusado com o card pendente."""
import importlib.util
from pathlib import Path

import pytest

CARD_DIR = Path(__file__).resolve().parents[1] / "hf" / "t6-lumi"
spec = importlib.util.spec_from_file_location("publicar", CARD_DIR / "publicar.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def test_plan_has_only_what_the_model_needs():
    plan = P.plan(CARD_DIR)
    assert set(plan) == P.ALLOWED
    for name in plan:
        assert "/" not in name and not name.endswith((".sh", ".md")) or name == "README.md"
    assert not any(k in n for n in plan for k in ("handoff", "estudos", "scripts", "data", "tar.gz", ".png"))
    assert {"model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja", "modeling_nemotron_h.py"} <= set(plan)


def test_every_figure_in_the_card_is_in_the_plan_and_exists():
    card = (CARD_DIR / "README.md").read_text(encoding="utf-8")
    import re

    shown = re.findall(r"\(\./(fig[^)]+\.svg)\)", card)
    assert shown and set(shown) <= set(P.FIGURES)
    assert all((CARD_DIR / f).is_file() for f in P.FIGURES)


def test_pending_markers_and_repo_id_substitution():
    text = "a\n<!-- DECISION: x -->\nurl: https://huggingface.co/<repo-id>\nTODO: y\n"
    assert [m.split(":")[0] for m in P.pending_markers(text)] == ["2", "4"]
    assert P.pending_markers("tudo certo") == []
    assert "<repo-id>" not in P.render_card(text, "ORG/t6-lumi-4b") and "ORG/t6-lumi-4b" in P.render_card(text, "ORG/t6-lumi-4b")


def test_plan_refuses_a_missing_figure(tmp_path):
    (tmp_path / "README.md").write_text("x")
    with pytest.raises(SystemExit):
        P.plan(tmp_path)
