# Pacote 5: provas novas do plano de 10/11 na Thor (Lumi e Nemotron)

**Para que serve.** A 4090 está ocupada por outro usuário e o plano `plano_benchmarks_finais.md` tem prazo (experimentos até quarta, 11/11, meio-dia). A Thor roda **só o Lumi e o Nemotron** nas provas novas, com o mesmo protocolo da suíte (vLLM, gulosa, 4.096 tokens). O T6-Lumi (nosso modelo fundido, 7,5 GB) roda na 4090, quando a GPU liberar (~1 h). A comparação final é na 4090 sempre que ela terminar o seu lado; o que vier da Thor é comparação secundária, como a suíte de 09/11 (a nota do Lumi difere ≤ 0,83 ponto entre as máquinas em 9 de 10 combinações; exceção: BLUEX sem thinking, 1,52).

**Regras (as dos outros pacotes).** A Thor não altera scripts nem parâmetros, não publica nada no Hugging Face, roda em `tmux` e anota divergências em `out/NOTAS.md`. As provas **não entram em treino**.

## Conteúdo de `thor5_pacote.tar.gz` (extrair na raiz do projeto, por cima)

- `src/lumi_ft/eval/novas.py` (novo) e `src/lumi_ft/eval/mcq.py` (carregadores das provas novas e `subsample`);
- `scripts/eval_vllm.py` (opções de amostragem, que não mudam o padrão guloso);
- `data/novas_provas/`: as cinco provas já montadas (`<tarefa>.jsonl`), os ids e `report.json` (a Thor não precisa baixar nem montar nada).

Provas: `oab_novas` (1.093 itens), `enem_2024` (179), `mmlu_pt_exatas`, `mmlu_pt_enade`, `mmlu_pt_concursos` (1.000 cada).

## Comandos (a mesma linha de base da suíte de 09/11: `--mem 0.5 --max-seqs 256`)

```bash
source env.sh   # ou o equivalente da Thor
E="VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-eval/bin/python scripts/eval_vllm.py --mem 0.5 --max-seqs 256"
NOVAS=oab_novas,enem_2024,mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
# 0) fumaça (3 min): confere que as provas carregam e pontuam
$E --name smoke_novas --tasks $NOVAS --modes thinking --limit 20 --out runs/eval_smoke
# 1) Lumi, as provas novas, os dois modos (a ordem põe primeiro o que mais importa)
$E --name base_novas_thor --tasks oab_novas,enem_2024
$E --name base_novas_thor --tasks mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
# 2) Nemotron 3 Nano 4B, as provas novas, os dois modos
$E --name nemotron4b_novas_thor --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --tasks oab_novas,enem_2024
$E --name nemotron4b_novas_thor --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --tasks mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
# 3) Nemotron na suíte do PRD (fecha a Nota dele; o MMLU-ProX é o mais longo, deixar por último)
$E --name nemotron4b_suite_thor --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --tasks gsm8k_test,enem,bluex,oab
$E --name nemotron4b_suite_thor --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --tasks mmlu_prox_test
```

`eval_vllm.py` pula o que já está no `summary.json`, então pode ser interrompido e retomado. Os nomes `*_thor` são diferentes dos da 4090 de propósito: nada é sobrescrito na volta.

## Devolução

```bash
tar czf novas_thor_saida.tar.gz runs/eval/base_novas_thor runs/eval/nemotron4b_novas_thor runs/eval/nemotron4b_suite_thor runs/eval_smoke runs/jobs/*.log out/NOTAS.md
sha256sum novas_thor_saida.tar.gz
```

Estimativas pelos tempos da suíte (57 a 98 itens por minuto com thinking na 4090; a Thor foi ~3 vezes mais lenta no MMLU-ProX): passos 1 e 2, ~3 h na Thor; passo 3 sem o MMLU-ProX, ~1,5 h; o MMLU-ProX do Nemotron, ~8 h (só se sobrar tempo).
