# Pacote 6: T6-Lumi e Lumi na DGX Spark (provas novas)

A 4090 está ocupada por outro usuário e o plano `plano_benchmarks_finais.md` tem prazo (experimentos até quarta, 11/11, meio-dia). A Spark roda o **T6-Lumi** (nosso modelo fundido) **e o Lumi na mesma máquina** nas cinco provas novas (`oab_novas`, `enem_2024`, `mmlu_pt_exatas`, `mmlu_pt_enade`, `mmlu_pt_concursos`), no mesmo protocolo da suíte (vLLM, gulosa, 4.096 tokens). Previsão: ~5 h (tempos da 4090 × 3; a Spark não foi medida). Nada é alterado nos scripts, nada é publicado e as provas não entram em treino; elas têm licença não declarada e ficam só na pasta `data/`, para avaliação.

## O que fazer (quem opera a Spark)

1. **Subir um arquivo só** para uma pasta do Jupyter (a que ele mostra): `spark1_pacote.tar.gz` (~2 GB; dentro vai a diferença exata entre os pesos do T6-Lumi e os do Lumi, que o script soma ao Lumi baixado do Hugging Face).
2. **Abrir um terminal no Jupyter e rodar uma linha:**

```bash
tar xzf spark1_pacote.tar.gz && bash handoff/spark1/rodar.sh
```

O script acha (ou instala) o vLLM e um compilador C (o Triton, que o vLLM usa, precisa; sem root, via `zig cc` do pip, `compilador.sh`), baixa o Lumi (~8 GB, do Hugging Face), reconstrói os pesos do T6-Lumi e confere o sha256 (têm de ser idênticos byte a byte aos da 4090), confere o ambiente e lança a fila em segundo plano (pode fechar o navegador). Se falhar, ele diz o quê (e deixa tudo em `out/rodar.log`); rodar de novo é seguro.

3. **Acompanhar:** `bash handoff/spark1/status.sh`. A cada par pronto aparece `spark1_saida.tar.gz` (com `spark1_saida.sha256`) na pasta: **baixar pelo Jupyter (botão direito > Download)**, mesmo com a fila rodando. O fim é a linha `FIM da fila spark1`. Para parar: `touch runs/jobs/spark1.stop` (para depois do passo em andamento).

Precisa de ~40 GB livres (o Lumi baixado, os pesos reconstruídos do T6-Lumi, o vLLM). Variáveis opcionais, se o vLLM recusar por memória: `MEM=0.4 bash handoff/spark1/rodar.sh` (padrão 0.5 = metade da memória) ou `SEQS=128` (padrão 256).

## O que a fila faz (`run_spark1.sh`)

| Passo | O quê | Nomes em `runs/eval/` | ~Tempo |
| --- | --- | --- | --- |
| 0 | Fumaça: Lumi, 20 itens por prova (confere vLLM, Hub e provas; a fila para se falhar) | `runs/eval_smoke/smoke_novas` | 5 min |
| 1 | `oab_novas` e `enem_2024`: T6-Lumi (thinking), depois Lumi (os dois modos) | `thor_pt_antidoom_of_spark`, `base_spark` | 2,5 h |
| 2 | As três fatias do MMLU-PT: T6-Lumi (thinking), depois Lumi (os dois modos) | idem | 2,5 h |

## Devolução e leitura (na 4090)

```bash
python scripts/import_thor_eval.py --origem spark --tar spark1_saida.tar.gz --sha256 <hash do spark1_saida.sha256>
bash scripts/report_spark1.sh      # runs/jobs/spark1_comparacao.md
```

- **Comparação principal da Spark:** T6-Lumi contra o Lumi com thinking e sem thinking, ambos na Spark. As previsões e a regra de leitura são as da H74 (`estudos/07`), sem mudança.
- **Calibração entre máquinas:** o Lumi da Spark contra o Lumi da Thor (pacote 5) mostra quanto as máquinas diferem (equivalentes se ≤ 1,0 ponto por prova e modo).
- Tudo o que ganhar do Lumi vem de fechar o `</think>` dentro do limite; o relatório diz isso sempre (H73).
- O que fica para a 4090: limite de 8.192 tokens (H67), as outras execuções da receita (H69) e a amostragem (H68), que estão na `noite18`.
