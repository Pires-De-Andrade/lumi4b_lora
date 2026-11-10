"""Provas novas (checagem externa, plano de 10/11): OAB 2018 a 2025, ENEM 2024 e três fatias do MMLU-PT do CEIA.

Nenhuma delas entrou na seleção do candidato. Este módulo tem só funções puras (testáveis na CPU): transformar as linhas de cada fonte em itens
`{id, question, options, gold}` (o mesmo formato de `mcq.py`), descontaminar contra as provas de teste, a validação e os dados de treino pelo índice de
8-gramas de `lumi_ft.data.decontaminate`, e ler os arquivos gravados por `scripts/build_novas_provas.py`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from lumi_ft.data.decontaminate import NgramIndex, flatten_strings

LETTERS = "ABCDEFGHIJ"
PLACEHOLDER = "[[placeholder]]"
OUT_DIR = "data/novas_provas"
TASKS = ["oab_novas", "enem_2024", "mmlu_pt_exatas", "mmlu_pt_enade", "mmlu_pt_concursos"]
# fatias do MMLU-PT (Portuguese-MMLU/mmlu-pt-knowledge-annotated), pela coluna `exam`
MMLU_PT_SLICES = {
    "mmlu_pt_exatas": ("OBI", "POSCOMP", "IME", "AFA"),
    "mmlu_pt_enade": ("ENADE",),
    "mmlu_pt_concursos": ("BNDES", "BACEN", "CNU"),
}
FIGURE_WORDS = re.compile(r"\b(figura|imagem|gráfico|grafico|tabela|charge|mapa|ilustra[çc][ãa]o)\b", re.IGNORECASE)


def norm_text(t: str) -> str:
    return " ".join(re.findall(r"\w+", t.lower()))


# ------------------------------------------------------------------------------------------------ fontes
def parse_oab_novas(rows, excluded_exams: set[str]) -> tuple[list[dict], dict]:
    """`russ7/oab_exams_2011_2025_combined`: mesmo esquema do `eduagarcia/oab_exams`, mas `answerKey` é o índice (0 a 3).
    Fica fora toda edição (`exam_id`) que já está na suíte de teste."""
    items, drops = [], Counter()
    for r in rows:
        if r["exam_id"] in excluded_exams:
            drops["edicao_na_suite"] += 1
            continue
        if r.get("nullified"):
            drops["anulada"] += 1
            continue
        texts = [str(t).strip() for t in r["choices"]["text"]]
        labels = list(r["choices"]["label"])
        k = r["answerKey"]
        try:
            idx = labels.index(k) if isinstance(k, str) and k in labels else int(k)
        except (TypeError, ValueError):
            drops["gabarito_invalido"] += 1
            continue
        if not 0 <= idx < len(texts) or any(not t for t in texts) or not str(r["question"]).strip():
            drops["formato"] += 1
            continue
        items.append({"id": f"oab_novas:{r['id']}", "question": str(r["question"]).strip(), "options": texts, "gold": LETTERS[idx]})
    return items, dict(drops)


def parse_enem_2024(rows) -> tuple[list[dict], dict]:
    """`maritaca-ai/enem`, config 2024. As figuras vêm descritas em `description`; cada `[[placeholder]]` (no enunciado ou nas alternativas)
    é trocado, em ordem, pela descrição correspondente. A questão anulada (`label` fora das letras) e a que não tem uma descrição por
    placeholder ficam de fora."""
    items, drops = [], Counter()
    for r in rows:
        alts = [str(a).strip() for a in r["alternatives"]]
        if r["label"] not in LETTERS[: len(alts)]:
            drops["anulada"] += 1
            continue
        descs = [str(d).strip() for d in r["description"]]
        n_ph = str(r["question"]).count(PLACEHOLDER) + sum(a.count(PLACEHOLDER) for a in alts)
        if n_ph != len(descs):
            drops["figura_sem_descricao"] += 1
            continue
        it = iter(descs)

        def fill(s: str) -> str:
            return re.sub(re.escape(PLACEHOLDER), lambda _m: next(it), s)

        q = fill(str(r["question"])).strip()
        alts = [fill(a) for a in alts]
        if not q or any(not a for a in alts):
            drops["formato"] += 1
            continue
        items.append({"id": f"enem_2024:{r['id']}", "question": q, "options": alts, "gold": r["label"]})
    return items, dict(drops)


def parse_mmlu_pt(rows, exams: tuple[str, ...], tag: str) -> tuple[list[dict], dict]:
    """`Portuguese-MMLU/mmlu-pt-knowledge-annotated`, filtrado pela coluna `exam`; `answer` é o índice; 4 ou 5 alternativas."""
    items, drops = [], Counter()
    for r in rows:
        if r["exam"] not in exams:
            continue
        ch = [str(c).strip() for c in r["choices"]]
        a = int(r["answer"])
        if len(ch) not in (4, 5) or not 0 <= a < len(ch) or any(not c for c in ch) or not str(r["question"]).strip():
            drops["formato"] += 1
            continue
        items.append({"id": f"{tag}:{r['annotation_id'][:16]}", "question": str(r["question"]).strip(), "options": ch, "gold": LETTERS[a]})
    return items, dict(drops)


# ------------------------------------------------------------------------------------------------ descontaminação
def item_text(it: dict) -> str:
    return "\n".join(s for s in flatten_strings({"q": it["question"], "o": it["options"]}) if len(s) > 3)


def overlap(it: dict, idx: NgramIndex) -> tuple[int, int]:
    """(8-gramas do item que estão no índice, 8-gramas do item)."""
    from lumi_ft.data.decontaminate import ngram_hashes

    hs = list(ngram_hashes(item_text(it), idx.n))
    return sum(h in idx.hashes for h in hs), len(hs)


def decontaminate(items: list[dict], indexes: dict[str, NgramIndex], min_frac: float = 0.10, min_hits: int = 3) -> tuple[list[dict], dict, set[str]]:
    """Tira o item em que pelo menos `min_frac` dos 8-gramas (e pelo menos `min_hits`) estão em algum índice; conta no primeiro índice que bater.

    Por que não "qualquer 8-grama": frases de abertura de prova e trechos de lei repetem 8-gramas sem que a questão seja a mesma (no ENADE, a
    alternativa "As asserções I e II são proposições verdadeiras, e a II é uma justificativa correta da I" marca metade das questões). O critério
    estrito do plano (qualquer 8-grama) não é perdido: o terceiro valor devolvido é o conjunto de ids que não têm NENHUM 8-grama em comum com nenhum
    índice, e as análises de sensibilidade usam esse subconjunto.
    """
    kept, removed, strict = [], Counter(), set()
    for it in items:
        drop, any_hit = None, False
        for name, idx in indexes.items():
            h, n = overlap(it, idx)
            any_hit |= h > 0
            if drop is None and h >= min_hits and n and h / n >= min_frac:
                drop = name
        if drop:
            removed[drop] += 1
            continue
        kept.append(it)
        if not any_hit:
            strict.add(it["id"])
    return kept, dict(removed), strict


def drop_same_text(items: list[dict], texts: set[str]) -> tuple[list[dict], int]:
    """Tira o item cujo enunciado (normalizado) é igual ao de uma questão da suíte."""
    kept = [it for it in items if norm_text(it["question"]) not in texts]
    return kept, len(items) - len(kept)


# ------------------------------------------------------------------------------------------------ arquivos
def write_task(root: str | Path, name: str, items: list[dict]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.jsonl").write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n")
    (root / f"{name}_ids.json").write_text(json.dumps([it["id"] for it in items]))


def read_task(name: str, root: str | Path = OUT_DIR) -> list[dict]:
    p = Path(root) / f"{name}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe; rode scripts/build_novas_provas.py")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
