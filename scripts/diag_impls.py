"""Diagnóstico: a diferença remote x nativo vem do padding? Também mostra qual função de kernel a versão nativa usa."""
import json, sys
import torch
from lumi_ft.modeling.load import free_gpu, load_model, load_tokenizer

PROMPTS = ["Quanto é 17 vezes 23? Explique.", "Qual a capital da Argentina?",
           "Se um trem anda 60 km em 45 minutos, qual a velocidade média em km/h?", "Liste três frutas tropicais."]
tok = load_tokenizer()
texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False) for p in PROMPTS]
single = [tok(t, return_tensors="pt").to("cuda") for t in texts]
def padded(side):
    return {k: v.to("cuda") for k, v in tok(texts, return_tensors="pt", padding=True, padding_side=side).items()}
out, store = {}, {}
for impl in ("remote", "native"):
    model = load_model(impl).eval()
    if impl == "native":
        names = [n for n, _ in model.named_modules()]
        out["native_module_names_sample"] = [n for n in names if n.endswith("layers.1.mixer.up_proj") or n.endswith("layers.0.mixer.in_proj") or n.endswith("embeddings") or n.endswith("embed_tokens")][:4]
        mx = dict(model.named_modules())[next(n for n in names if n.endswith("layers.0.mixer"))]
        out["native_kernel_funcs"] = {n: f"{getattr(getattr(mx, n, None), '__module__', None)}" for n in
                                      ("causal_conv1d_fn", "causal_conv1d_update", "mamba_chunk_scan_combined", "selective_state_update", "mamba_split_conv1d_scan_combined")}
        import transformers.models.nemotron_h.modeling_nemotron_h as nm
        out["native_module_funcs"] = {n: f"{type(f).__name__} de {getattr(f, '__module__', '?')}" for n, f in vars(nm).items()
                                      if callable(f) and any(k in n for k in ("conv1d", "scan", "selective_state"))}
    with torch.no_grad():
        store[impl, "single"] = [model(**s).logits[0].float().cpu() for s in single]
        for side in ("left", "right"):
            b = padded(side); lg = model(**b).logits.float().cpu(); m = b["attention_mask"].cpu().bool()
            store[impl, side] = [lg[i][m[i]] for i in range(len(PROMPTS))]
    del model; free_gpu()
def stats(xs, ys):
    d = torch.cat([(x - y).abs().flatten() for x, y in zip(xs, ys)])
    t = torch.cat([(x.argmax(-1) == y.argmax(-1)) for x, y in zip(xs, ys)]).float().mean().item()
    return {"max": round(d.max().item(), 3), "mean": round(d.mean().item(), 4), "top1": round(t, 4)}
for impl in ("remote", "native"):
    out[f"{impl}: sem padding x batch com padding à ESQUERDA"] = stats(store[impl, "single"], store[impl, "left"])
    out[f"{impl}: sem padding x batch com padding à DIREITA"] = stats(store[impl, "single"], store[impl, "right"])
out["remote x nativo, sem padding (batch 1)"] = stats(store["remote", "single"], store["native", "single"])
print(json.dumps(out, indent=2, ensure_ascii=False))
json.dump(out, open("runs/phase0/diag_impls.json", "w"), indent=2, ensure_ascii=False)
