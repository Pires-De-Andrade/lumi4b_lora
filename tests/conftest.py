import os

import pytest
import torch

from lumi_ft.modeling.load import free_gpu, load_model, load_tokenizer

IMPL = os.environ.get("LUMI_IMPL", "native")  # native | remote

PROMPTS = [
    "Quanto é 17 vezes 23? Explique.",
    "Escreva uma frase curta sobre o Rio de Janeiro.",
]


@pytest.fixture(scope="session")
def tok():
    return load_tokenizer()


@pytest.fixture(scope="session")
def batch(tok):
    """Batch fixo; enable_thinking declarado (regra 7 do PRD), um exemplo de cada modo."""
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=think)
        for p, think in zip(PROMPTS, (True, False))
    ]
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="left")
    return {k: v.to("cuda") for k, v in enc.items()}


@pytest.fixture
def fresh_model():
    model = load_model(IMPL)
    yield model
    del model
    free_gpu()
    torch.cuda.reset_peak_memory_stats()


BIG_PROMPTS = PROMPTS + [
    "Qual a capital da Argentina?", "Resuma em uma frase o que é fotossíntese.",
    "Se um trem anda 60 km em 45 minutos, qual a velocidade média em km/h?", "Liste três frutas tropicais.",
    "Explique o que é um número primo.", "Traduza para o inglês: bom dia, tudo bem?", "Quem escreveu Dom Casmurro?",
    "Calcule 15% de 240.", "Dê um conselho para estudar para o ENEM.", "O que é uma função em programação?",
    "Escreva um parágrafo curto sobre a importância da leitura para a formação de uma pessoa.",
    "Explique a diferença entre clima e tempo.", "Quais são os planetas do sistema solar?", "Como funciona um termostato?",
]


@pytest.fixture(scope="session")
def big_batch(tok):
    """Mais posições que o batch pequeno, para o top-1 do teste de fusão não depender de 1 token."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=bool(i % 2)) for i, p in enumerate(BIG_PROMPTS)]
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="right")
    return {k: v.to("cuda") for k, v in enc.items()}


def equivalence_stats(a, b, mask):
    """Compara dois conjuntos de logits nas posições reais (mask): diferença média, discordância do top-1 e KL médio."""
    a, b = a[mask].float(), b[mask].float()
    return {
        "n": int(mask.sum()),
        "mean_abs": (a - b).abs().mean().item(),
        "top1_disagree": (a.argmax(-1) != b.argmax(-1)).float().mean().item(),
        "kl": torch.nn.functional.kl_div(b.log_softmax(-1), a.log_softmax(-1), log_target=True,
                                         reduction="none").sum(-1).mean().item(),
        "scale": a.abs().max().item(),
    }


@pytest.fixture(scope="session")
def noise_floor(big_batch):
    """Ruído natural do modelo em BF16: o mesmo batch em batch (com padding) x cada sequência sozinha.
    É o critério de 'equivalente' dos testes de adaptador: nada pior que 2x esse piso."""
    model = load_model(IMPL).eval()
    mask = big_batch["attention_mask"].bool()
    with torch.no_grad():
        batched = model(**big_batch).logits.float()
        single = torch.zeros_like(batched)
        for i in range(batched.shape[0]):
            n = int(mask[i].sum())
            single[i, :n] = model(input_ids=big_batch["input_ids"][i:i + 1, :n]).logits.float()[0]
    stats = equivalence_stats(batched, single, mask)
    del model, batched, single
    free_gpu()
    print(f"\n[piso de ruído BF16] {stats}")
    return stats


def assert_within_noise(stats, floor, what):
    """Mesmo critério para todos os testes: no máximo 2x o piso de ruído (+ 1 posição de folga no top-1)."""
    slack = 1.0 / stats["n"]
    ok = (stats["mean_abs"] <= 2 * floor["mean_abs"] and stats["kl"] <= 2 * floor["kl"]
          and stats["top1_disagree"] <= 2 * floor["top1_disagree"] + slack)
    assert ok, f"{what}: {stats} acima de 2x o piso {floor}"
