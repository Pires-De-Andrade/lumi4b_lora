"""Fase 0: compara logits do código do repositório (remote) com a implementação nativa do Transformers
num batch fixo, e testa a geração (parada em <|im_end|>) em cada uma. Carrega um modelo por vez."""
import json, sys, time
import torch
from lumi_ft.modeling.load import EOS_IDS, free_gpu, is_fast_path, load_model, load_tokenizer

PROMPTS = [
    "Quanto é 17 vezes 23? Explique.", "Escreva uma frase curta sobre o Rio de Janeiro.",
    "Qual a capital da Argentina?", "Resuma em uma frase o que é fotossíntese.",
    "Se um trem anda 60 km em 45 minutos, qual a velocidade média em km/h?",
    "Liste três frutas tropicais.", "Explique o que é um número primo.",
    "Traduza para o inglês: bom dia, tudo bem?", "Quem escreveu Dom Casmurro?",
    "Calcule 15% de 240.", "Dê um conselho para estudar para o ENEM.", "O que é uma função em programação?",
]
tok = load_tokenizer()
texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=bool(i % 2)) for i, p in enumerate(PROMPTS)]
batch = {k: v.to("cuda") for k, v in tok(texts, return_tensors="pt", padding=True, padding_side="left").items()}
res, logits = {}, {}
for impl in ("remote", "native"):
    print(f"=== {impl}", flush=True)
    model = load_model(impl).eval()
    r = {"class": f"{type(model).__module__}.{type(model).__name__}", "fast_path": is_fast_path(model)}
    with torch.no_grad():
        logits[impl] = model(**batch).logits.float().cpu()
    one = tok(tok.apply_chat_template([{"role": "user", "content": "Diga apenas: olá."}], tokenize=False,
              add_generation_prompt=True, enable_thinking=False), return_tensors="pt").to("cuda")
    n = one["input_ids"].shape[1]
    try:
        t = time.time()
        out = model.generate(**one, max_new_tokens=128, do_sample=False, eos_token_id=EOS_IDS, pad_token_id=0)[0, n:]
        r["gen_eos_2_11"] = {"tokens": len(out), "last_id": out[-1].item(), "text": tok.decode(out, skip_special_tokens=False)[:200], "s": round(time.time() - t, 2)}
        out2 = model.generate(**one, max_new_tokens=128, do_sample=False, eos_token_id=2, pad_token_id=0)[0, n:]
        r["gen_eos_2_only"] = {"tokens": len(out2), "last_id": out2[-1].item()}
    except Exception as e:
        r["gen_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    res[impl] = r
    del model; free_gpu()
a, b = logits["remote"], logits["native"]
mask = batch["attention_mask"].cpu().bool()
d = (a - b).abs()[mask]
res["compare"] = {"max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item(), "logit_scale": a[mask].abs().max().item(),
                  "top1_agreement": (a.argmax(-1) == b.argmax(-1))[mask].float().mean().item(), "positions": int(mask.sum())}
print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open("runs/phase0/compare_impls.json", "w"), indent=2, ensure_ascii=False)
