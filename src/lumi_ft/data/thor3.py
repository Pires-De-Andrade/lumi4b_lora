"""Questões de múltipla escolha com gabarito para o professor da Thor (pacote 3): MedMCQA e WebInstruct-verified.

Fontes fora da lista de treino do Lumi (card do ceia-br/lumi-4b) e com gabarito, para ampliar o domínio além da ciência do
Mixture-of-Thoughts (pacote 2):
- `openlifescienceai/medmcqa` (Apache-2.0): 182.822 questões de treino de provas médicas indianas, 4 alternativas, `cop` = índice
  da correta. Só `choice_type == "single"`; fora as que citam imagem e as com o defeito de OCR "aery" (artery sem o "rt").
- `TIGER-Lab/WebInstruct-verified` (Apache-2.0): 228.736 questões de várias áreas (física, química, negócios, finanças, economia,
  história, biologia, psicologia, filosofia, direito...); 27.764 têm `answer_type == "Multiple Choice"`, com as alternativas
  dentro do enunciado em formatos variados (`a.`, `(a)`, `A)`, às vezes fora de ordem). Só entram as que se separam com segurança.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

LETTERS = "ABCDEFGHIJ"
_EQ = re.compile(r"\{eq\}(.*?)\{/eq\}", re.S)
_MARK = re.compile(r"(?m)^[ \t]*(?:\(([a-jA-J])\)|([a-jA-J])[\.\)])(?=\s)")
_IMG = re.compile(r"(?i)\b(image|figure|picture|photo(graph)?|diagram|graph below|shown (below|above|here)|following (table|graph|chart))\b")
_OCR_MED = re.compile(r"(?i)\b(aery|aeries|aerial blood)\b")
_REF = re.compile(r"^\(?([a-jA-J])\)?\.?$")


def clean_eq(text: str) -> str:
    """`{eq} ... {/eq}` (marcação do WebInstruct) vira `$...$`, sem as quebras de linha internas."""
    return _EQ.sub(lambda m: "$" + " ".join(m.group(1).split()) + "$", text)


def parse_embedded_options(text: str) -> tuple[str, list[str], list[str]] | None:
    """(enunciado, alternativas em ordem de letra, letras) de um texto com as alternativas marcadas no começo da linha
    (`a.`, `a)`, `(a)`, maiúsculas ou minúsculas, todas no mesmo estilo de caixa). None se não houver ao menos 3 letras
    consecutivas a partir de A, sem repetição, cada uma com texto."""
    marks = list(_MARK.finditer(text))
    if len(marks) < 3:
        return None
    letters = [(m.group(1) or m.group(2)) for m in marks]
    if len({ch.isupper() for ch in letters}) != 1:
        return None
    up = [ch.upper() for ch in letters]
    if len(set(up)) != len(up) or sorted(up) != list(LETTERS[: len(up)]):
        return None
    stem = text[: marks[0].start()].strip()
    bounds = [m.end() for m in marks]
    ends = [m.start() for m in marks[1:]] + [len(text)]
    opts = {u: " ".join(text[b:e].split()) for u, b, e in zip(up, bounds, ends)}
    ordered = [opts[ch] for ch in LETTERS[: len(up)]]
    if not stem or not all(ordered) or len(set(ordered)) != len(ordered):
        return None
    return stem, ordered, list(LETTERS[: len(up)])


def webinstruct_item(row: dict) -> dict | None:
    if row.get("answer_type") != "Multiple Choice":
        return None
    m = _REF.match(str(row.get("answer", "")).strip())
    if not m:
        return None
    text = clean_eq(row["question"])
    if _IMG.search(text):
        return None
    sp = parse_embedded_options(text)
    if sp is None:
        return None
    stem, opts, letters = sp
    ref = m.group(1).upper()
    if ref not in letters or max(len(o) for o in opts) > 400 or len(stem) > 3000:
        return None
    return {"id": f"webinstruct:{row['id']}", "source": "webinstruct", "category": row.get("category"), "question": " ".join(stem.split()) if "\n" not in stem.strip() else stem.strip(),
            "options": opts, "ref": ref}


def medmcqa_item(row: dict) -> dict | None:
    if row.get("choice_type") != "single":
        return None
    opts = [str(row.get(k) or "").strip() for k in ("opa", "opb", "opc", "opd")]
    q = str(row.get("question") or "").strip()
    cop = row.get("cop")
    if not q or not all(opts) or len(set(o.lower() for o in opts)) != 4 or cop not in (0, 1, 2, 3):
        return None
    blob = q + "\n" + "\n".join(opts)
    if _IMG.search(blob) or _OCR_MED.search(blob):
        return None
    return {"id": f"medmcqa:{row['id']}", "source": "medmcqa", "category": row.get("subject_name"), "question": q, "options": opts,
            "ref": LETTERS[int(cop)]}


def balanced_sample(items: list[dict], n: int, cap_share: float, seed: int = 0) -> list[dict]:
    """Até `n` itens por rodízio entre as categorias: as pequenas entram inteiras, as grandes dividem o resto por igual, e
    nenhuma passa de `cap_share` de `n` (teto rígido: pode devolver menos de `n`). Determinístico em `seed`."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in sorted(items, key=lambda x: x["id"]):
        by[it["category"] or "?"].append(it)
    for v in by.values():
        rng.shuffle(v)
    cap = max(1, int(cap_share * n))
    out, taken = [], Counter()
    cats = sorted(by)
    while len(out) < n:
        moved = False
        for c in cats:
            if len(out) >= n:
                break
            if taken[c] < min(cap, len(by[c])):
                out.append(by[c][taken[c]])
                taken[c] += 1
                moved = True
        if not moved:
            break
    rng.shuffle(out)
    return out
