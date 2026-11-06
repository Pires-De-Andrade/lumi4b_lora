"""Patch local do DoRA do PEFT para linhas de peso exatamente zero.

Problema: a in_proj do Lumi tem 213 linhas com pesos exatamente zero. O DoRA inicializa a magnitude com a norma
da linha (0) e divide magnitude / norma no forward e na fusão: 0/0 = NaN já nos logits do primeiro passo.

Correção: onde a norma é exatamente 0, o DIVISOR vale 1. A magnitude inicial continua sendo a norma verdadeira (0),
então essas linhas nascem inertes (fator = 0/1 = 0) e continuam produzindo 0, como na base. Linhas com norma != 0
não mudam. Durante update_layer (inicialização da magnitude) o patch fica desligado.

Limite: desfazer a fusão (unmerge) divide pelo fator e daria 0 nessas linhas; o projeto não usa unmerge
(os adaptadores são salvos à parte e o fundido vem de merge_and_unload).
"""
import torch
from peft.tuners.lora import dora

_state = {"applied": False, "raw": False}
_orig = {}


def _safe_get_weight_norm(self, *args, **kwargs):
    norm = _orig["get_weight_norm"](self, *args, **kwargs)
    if _state["raw"]:
        return norm
    return torch.where(norm == 0, torch.ones_like(norm), norm)


def _update_layer_raw(self, *args, **kwargs):
    _state["raw"] = True
    try:
        return _orig["update_layer"](self, *args, **kwargs)
    finally:
        _state["raw"] = False


def apply() -> None:
    if _state["applied"]:
        return
    cls = dora.DoraLinearLayer
    _orig["get_weight_norm"], _orig["update_layer"] = cls.get_weight_norm, cls.update_layer
    cls.get_weight_norm, cls.update_layer = _safe_get_weight_norm, _update_layer_raw
    _state["applied"] = True


def revert() -> None:
    if not _state["applied"]:
        return
    cls = dora.DoraLinearLayer
    cls.get_weight_norm, cls.update_layer = _orig["get_weight_norm"], _orig["update_layer"]
    _state["applied"] = False
