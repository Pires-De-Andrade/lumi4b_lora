"""Carga do Lumi-4B em BF16, com eos e pad conforme o PRD."""
import gc
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ceia-br/lumi-4b"
REVISION = "27b30f40b930fd05cee5e1c1f8790f70fa5b816d"  # v0.2
EOS_IDS = [2, 11]  # </s> e <|im_end|>; as configs do modelo só listam o 2
PAD_TOKEN = "<unk>"  # id 0, o pad_token_id do config.json


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tok.pad_token = PAD_TOKEN
    assert tok.pad_token_id == 0, tok.pad_token_id
    return tok


def load_model(impl: str = "remote", device: str = "cuda"):
    """impl='remote' usa o modeling_nemotron_h.py do repositório (auto_map);
    impl='native' usa a implementação nativa do Transformers."""
    assert impl in ("remote", "native")
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.bfloat16,
        trust_remote_code=(impl == "remote"),
        device_map={"": device},
    )


def is_fast_path(model):
    """Flag do módulo do código remoto. Retorna None na implementação nativa, que não tem a flag:
    ela escolhe mamba-ssm/causal-conv1d (se instalados) por função, com fallback em PyTorch."""
    mod = sys.modules[type(model).__module__]
    return bool(mod.is_fast_path_available) if hasattr(mod, "is_fast_path_available") else None


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
