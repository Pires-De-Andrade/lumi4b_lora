# scripts/

Índice dos 62 arquivos desta pasta, por papel. A pasta é **plana de propósito**: os registros (`estudos/06`, `discovery.md`, `runs/jobs/*.log`) citam estes caminhos nos comandos que rodaram, e o RL em andamento os carrega por caminho. Mover um arquivo quebraria a reprodução do que está registrado.

Convenções: rode sempre com `source env.sh`; tarefas longas com `scripts/detach.sh [--gpu] <nome> <comando>`; Python do treino com `PYTHONPATH=src python`, do vLLM com `.venv-vllm/bin/python` (e `VLLM_USE_FLASHINFER_SAMPLER=0`). **Não edite um `.sh` enquanto ele roda** (o bash lê o arquivo aos poucos). Marcas: **ativo** = parte do fluxo atual; **histórico** = guardado para reproduzir uma rodada já registrada.

## Infraestrutura

| Arquivo | Para quê | Estado |
| --- | --- | --- |
| `detach.sh` | Roda um comando desacoplado em tmux, com `runs/jobs/<nome>.log` e `.exit`; `--gpu` espera a GPU livre | ativo |
| `build_kernels.sh` | Compila `causal-conv1d` e `mamba-ssm` (nvcc do conda-forge, sem sudo); usa `patches/mamba-ssm-sm89-cxx20.patch` | ativo |
| `install_vllm.sh` | Cria o `.venv-vllm` | ativo |
| `chat.py` | Conversa interativa com o Lumi ou, com `--model`, com um modelo fundido (ex.: `runs/merged/thor_pt_antidoom_of`, o T6-Lumi); `--device cpu` roda sem GPU (float32, ~5 tokens/s com 8 threads) | ativo |

## Dados: construção, descontaminação e validação

| Arquivo | Para quê |
| --- | --- |
| `download_data.py` | Baixa o GigaVerbo-v2 e as provas |
| `build_dataset.py` | Fase 1: `data/train.parquet`, `valid.parquet` e `reports/data_manifest.json` |
| `build_validation.py`, `clean_validation.py` | Conjuntos de validação (A-10) e o subconjunto limpo do GSM8K (676 itens, `data/val_gsm8k_clean_ids.json`) |
| `build_teacher_dataset.py` | Parquet de SFT a partir de registros de um professor |
| `concat_parquets.py` | Junta parquets de SFT de mesmo esquema (confere esquema e `id` únicos) e grava o manifest |
| `prepare_adalberto.py` | Dataset `adalberto` (Qwen3-30B, GSM8K em português) |
| `prepare_mot_science.py`, `prepare_nemotron_pt.py` | Questões de ciência (Mixture-of-Thoughts e Nemotron-SFT-Multilingual) |
| `prepare_thor.py`, `prepare_thor_translated.py`, `filter_translated.py` | Entregas da Thor viram registros de treino; descontaminação por 8-grama |
| `subset_think_by_length.py` | Filtra um parquet de SFT pelo tamanho do raciocínio |
| `make_review_sample.py` | Amostra para revisão humana (D-10, `reports/review_d10.md`) |
| `gen_selfdistill.py` | Autodestilação do Lumi (E3; **histórico**, resultado negativo) |
| `translate_questions.py`, `translate_traces.py` | Tradução com o Lumi no vLLM (piloto E8; **histórico**: a tradução em escala foi feita na Thor) |
| `run_nemo_mix.sh` | Tradução do raciocínio de ciência da NVIDIA e mistura com o `adalberto` |

## SFT, fusão e avaliação

| Arquivo | Para quê | Estado |
| --- | --- | --- |
| `train_sft.py` | Treina o adaptador LoRA/DoRA; retomável | ativo |
| `run_experiment.sh` | Treina, funde e avalia, em sequência | ativo |
| `merge_adapter.py` | Funde um adaptador no Lumi em BF16 (`--scale` muda o efeito); com `--base` (ou a variável `GRPO_BASE`) funde sobre outro modelo fundido, como o T6-Lumi, sem a conferência contra o Lumi | ativo |
| `merge_adapter_layers.py` | Funde só parte do adaptador (triagem por camadas, E9) | histórico |
| `eval_vllm.py` | Avalia um modelo no vLLM por (prova, modo); retomável; `--tasks validation|suite`; gulosa por padrão, `--temperature`, `--top-p` e `--sample-seed` só para análises de sensibilidade (H68) | ativo |
| `eval_merged.sh` | Avalia vários fundidos na validação | ativo |
| `stop_eval_after_task.sh` | Encerra uma avaliação em curso depois de uma (prova, modo) | ativo |
| `run_suite.sh`, `run_suite_models.sh` | Suíte de teste do PRD (GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX, OAB) | ativo (só o Lumi tem resultado parcial) |
| `compare_runs.py`, `report_compare.py` | Bootstrap pareado: diferença e IC 95% entre rodadas, por prova e (`report_compare.py --mean`, `paired_bootstrap_mean`; `--subset-ids` restringe uma prova a uma lista de ids) na nota média das provas | ativo |
| `build_novas_provas.py` | Monta as provas novas (`oab_novas`, `enem_2024`, `mmlu_pt_*`) em `data/novas_provas/`: conversão, descontaminação por 8-gramas (≥ 10% do item) contra a suíte, a validação e os dados de treino, subconjunto estrito, amostra e `report.json`; lógica em `src/lumi_ft/eval/novas.py` | ativo |
| `interpolate_models.py` | Interpolação de pesos entre dois modelos com as mesmas chaves (WiSE-FT): `alpha` de A + (1 − `alpha`) de B; `--b lumi` usa o Lumi original (H71, adiada) | ativo |
| `decompose_paired.py` | Decomposição pareada da diferença de acerto contra o Lumi por quadrante de fechamento do `</think>` (onde o modelo ganha e onde perde) | ativo |
| `gen_estudos_resultados.py` | Regera `estudos/resultados_auto.md` de `runs/eval/*` | ativo |
| `run_gsm8k.py`, `run_gsm8k_vllm.py` | GSM8K-PT do piloto (HF e vLLM) | histórico |
| `run_capacity.sh` + `capacity_plan.txt` | Fila do estudo de capacidade (E11; encerrado com 1 de 15 rodadas) | histórico |

## RL (GRPO e variantes)

Fluxo de um ciclo (`run_grpo.sh`): fundir o adaptador → `grpo_rollout.py` (vLLM, 256 prompts × 8 respostas, recompensa e vantagem) → `grpo_guard.py` → `grpo_train.py` (atualiza o adaptador) → avaliação a cada `--eval-every`.

| Arquivo | Para quê | Estado |
| --- | --- | --- |
| `run_grpo.sh` | Laço do RL, retomável; é o que as rodadas atuais usam | ativo |
| `grpo_rollout.py`, `grpo_train.py`, `grpo_guard.py` | Geração e recompensa, atualização (com `GRPO_BASE` treina sobre um modelo fundido em vez do Lumi), guarda de truncadas | ativo |
| `grpo_status.py` | Resumo por ciclo (recompensa, truncadas, entropia, corte, avaliações) | ativo |
| `make_grpo_prompts.py` | Prompts antigos (letra majoritária do R1) | histórico (`gspo1`, `gspo_thor`) |
| `make_questions_rl_prompts.py`, `audit_gold_questions.py` | Prompts com gabarito oficial (`data/grpo_prompts_gold.jsonl`) e auditoria estrutural | ativo |
| `make_rl_pool_t6.py` | Pool de RL a partir do T6-Lumi (`data/grpo_prompts_t6.jsonl`): o pool com gabarito sem os itens que o SFT do T6-Lumi já viu (8-gramas) e conferido contra as provas novas | ativo |
| `make_blank_adapter.py` | Adaptador LoRA em branco (B = 0, r 16, alpha 32, os 71 módulos) sobre um modelo fundido; ponto de partida do RL a partir do T6-Lumi (a fusão devolve a base bit a bit) | ativo |
| `select_hard_prompts.py` | Dos rollouts do T6-Lumi, os prompts que dão sinal (1 a 7 acertos de 8): `data/grpo_prompts_t6_hard.jsonl` | ativo |
| `rl_t6_gate.py` | Portão da H75: a validação do `rl_t6` passa para a suíte de teste (múltipla escolha ≥ 75,2%, fecha ≥ 90%, português ≥ 90%) | ativo |
| `run_rl_t6_chain.sh` | Cadeia do RL a partir do T6-Lumi (H75): espera a `noite18`, seleciona prompts, RL de 4 iterações, validação, portão e, se passar, suíte e provas novas | ativo |
| `reordenar_h75_antes_h69.sh` | Vigia que encerra a fila `noite18` quando ela começa a H69, para a cadeia do RL (H75) rodar antes (sem editar o script da fila) | ativo |
| `reordenar_h75_antes_h68.sh` | Vigia que encerra a `fila_final2` quando ela começa a H68 (primeira avaliação, `base_t06`), escreve a tabela da H69 (`runs/jobs/h69_comparacao.md`), roda a cadeia do RL (H75) e retoma o `run_final_chain.sh` (que pula o que está completo); uma sessão só, sem espera entre sessões | ativo |
| `run_noite18_resto.sh` | Retomada da fila depois do RL (**travou**: o `tmux has-session -t noite18` casa por prefixo com a própria `noite18b`; substituída pelo `run_final_chain.sh`) | histórico |
| `run_final_chain.sh` | Fila final da 4090 (11/11): H69 e H68 direto, tabelas em segundo plano (copia o cabeçalho de `run_noite18_chain.sh`) | ativo |
| `run_grpo_kl.sh`, `grpo_train_kl.py` | Variante com KL ao SFT (`grpo4`) | histórico |
| `run_grpo_entropy.sh` | Entropia por ciclo de um RL já rodado | histórico |
| `bench_rollout.py`, `run_bench_rollout.sh` | Benchmark de velocidade da geração (resultado: nada adotado, estudos/08) | histórico |
| `run_gold_chain.sh` | `gspo_gold4k` (limite 4.096; parou pela regra no ciclo 4) | histórico |
| `run_thor4k_chain.sh` | RL a partir do `thor_pt` com 4.096 (**cancelada** antes de rodar) | histórico |
| `run_gold2048_resume.sh` | Retoma o `gspo_gold` (2.048) até o ciclo 4 e avalia | **em execução** |
| `run_noite4_chain.sh` | Fila da noite (v4): RLs de teste da receita do `gspo1` (T nos logits, semente 1, corte 3×, log-probabilidades antigas na composição do treino, limite 4.096), suíte do Lumi como preenchimento até o pacote 4 chegar, trabalho com o pacote 4 e suíte até o fim | congelada em 09/11 01:22 (substituída pela `run_noite5_chain.sh`) |
| `run_noite5_chain.sh` | Fila da noite (v5): espera o `gspo1_s1`, teste parcial `qwen_mix_p` com a entrega parcial do pacote 4, diagnósticos de RL (`gspo1_eps3x`, `gspo1_same`, `gspo1_4k`), pacote 4 final e suíte do Lumi por último | encerrada em 09/11 13:54 (queda da máquina), substituída pela `run_noite6_chain.sh` |
| `run_noite6_chain.sh` | Fila (v6): avaliação do SFT parcial `qwen_mix_p` sem RL, pacote 4 completo (`qwen_mix` e `gspo_qwen_mix`), diagnósticos de RL (`gspo1_eps3x`, `gspo1_same`, `gspo1_4k`) e suíte do Lumi por último | congelada em 09/11 15:46 (substituída pela `run_noite7_chain.sh`) |
| `run_noite7_chain.sh` | Fila (v7): espera o RL `gspo_qwen_mix`, SFTs de teste `qwen_mix_short` (H53) e `selfdist_min` (H54) e suíte do Lumi por último | encerrada em 09/11 16:58 (substituída pela `run_noite8_chain.sh`) |
| `run_noite8_chain.sh` | Fila (v8): sopas de pesos, `base_ptprefill`, SFTs `qwen_mix_short`, `selfdist_min` e `selfdist_300`, suíte do Lumi por último | congelada em 09/11 17:13 (substituída pela `run_noite9_chain.sh`) |
| `run_noite9_chain.sh` | Fila (v9): o que a v8 tinha (sopas, `base_ptprefill`, SFTs de teste, suíte por último) mais a fase anti-laço (H57) sobre o `gspo1` de 8 ciclos | congelada em 09/11 17:22 (substituída pela `run_noite10_chain.sh`) |
| `run_noite10_chain.sh` | Fila (v10): igual à v9, mas termina com a suíte de teste do `gspo1` (a do Lumi roda na Jetson Thor); espera só os processos vivos da v9 | congelada em 09/11 18:28; encerrada às 18:52, ao fim da mineração do `gspo1` |
| `run_noite11_chain.sh` | Fila (v11): teste de fumaça do treino anti-laço, fase anti-laço no `gspo_thor` (H58), SFTs de teste, fase anti-laço no `gspo1` (H57) e suíte do `gspo1` por último | encerrada em 09/11 18:38 antes de começar (substituída pela `run_noite12_chain.sh`) |
| `run_noite12_chain.sh` | Fila (v12): SFTs de teste até o código da receita oficial ficar pronto; fase anti-laço na receita oficial no `thor_pt` (H59), no `gspo_thor` (H58), braço com o treinador oficial, SFTs restantes, `gspo1` (H57) e suíte de um candidato escolhido por regra | congelada em 09/11 19:40 e encerrada (substituída pela `run_noite13_chain.sh`) |
| `run_noite13_chain.sh` | Fila (v13): RL `gspo_qwen_mix_short` (H60) primeiro, depois a fase anti-laço na receita oficial (`thor_pt`, `gspo_thor`, braço do treinador oficial), SFTs `selfdist_min` e `selfdist_300`, `gspo1` e a suíte de um candidato escolhido por regra | executada até a suíte de teste do `thor_pt_antidoom_of` (em andamento desde 01:40 de 10/11) |
| `run_noite14_chain.sh` | Fila (v14): espera a `noite13` acabar e faz a sonda da segunda rodada do Antidoom (H64) e duas sementes do treino da primeira rodada (H65); não roda suíte de teste | **terminou em 10/11 05:53** |
| `run_noite15_chain.sh` | Fila (v15): espera a `noite14` acabar e roda o Lumi original na 4090 em MMLU-ProX PT (sem thinking), ENEM, BLUEX e OAB (os dois modos) (H66); ao fim, duas tabelas de comparação com a nota média (contra o Lumi com e sem thinking) se a suíte do `thor_pt_antidoom_of` estiver completa; não avalia modelo nosso | **terminou em 10/11 07:29** |
| `run_noite18_chain.sh` | Fila (v18): o plano de provas novas do usuário dividido com a Thor (fumaça, T6-Lumi nas provas novas, Lumi na 4090, 8.192 tokens (H67), outras execuções da receita (H69), amostragem T 0,6 (H68) e o Nemotron que a Thor não devolver); não troca o candidato | **na fila** |
| `make_card_figures.py` | Gráficos do card do T6-Lumi no Hugging Face (`hf/t6-lumi/fig*.svg`): barras por prova, diferença pareada com IC, fechamento do `</think>`, decomposição por quadrante e efeito de cada etapa. Os números vêm dos registros (cada bloco cita a seção); `tests/test_card_figures.py` confere a consistência |
| `make_card_figures_novas.py` | Gráficos 6 a 9 do card do T6-Lumi (provas novas): calcula acerto, diferença pareada com IC, fechamento e decomposição a partir de `runs/eval` (nada digitado à mão) | ativo |
| `antidoom_oficial_train.py` | Braço de conferência: treina com o `FTPOTrainer` do repositório oficial (clonado em `../antidoom_oficial`) sobre os nossos eventos | pronto (testado na CPU com modelo minúsculo) |
| `merge_on_base.py` | Funde um LoRA treinado sobre um modelo já fundido (qualquer alvo, inclusive `lm_head`) e confere chaves, formas e tensores alterados | pronto |
| `pick_suite_candidate.py` | Escolhe o único modelo da suíte de teste pela regra escrita antes dos resultados (padrão: `gspo1`) | pronto |
| `make_antidoom_prompts.py`, `antidoom_mine.py`, `antidoom_train.py` | Fase anti-laço: prompts de mineração e de medida (`data/antidoom_prompts.jsonl`); mineração com o vLLM e dois detectores (`--detector exato` ou `oficial`; `stats.json` traz as taxas dos dois); treino na receita oficial do Antidoom (`--recipe oficial`, padrão: LoRA novo r=256 com `lm_head` sobre o modelo fundido) ou na primeira versão própria (`--recipe propria`) | pronto; receita oficial testada na CPU com modelo minúsculo |
| `prepare_soups.sh`, `average_adapters.py` | Média de pesos de adaptadores LoRA treinados do mesmo ponto de partida (confere esquema e configuração, relata a distância à média e o erro da aproximação B·A) e fusão em `runs/merged/gspo1_soup2` e `gspo1_soup3` | pronto |
| `prepare_sft_curtos.sh` | Dados dos dois SFTs de teste (`data/qwen_mix_short.parquet`, `data/selfdist_min.parquet`), só CPU, idempotente | pronto |
| `run_qwen_mix.sh` | Trabalho com o pacote 4 da Thor: extrai, confere, converte (`prepare_thor4.py`), filtra contaminação, monta `data/<NOME>.parquet` (os dados do `adalberto` + a múltipla escolha do pacote, junto por `concat_parquets.py`), SFT `qwen_mix` e RL `gspo_qwen_mix`; chamado pela `run_noite4_chain.sh` quando `data/thor4/thor4_out.tar.gz` chega | pronto, ensaiado só nas partes de CPU |
| `prepare_thor4.py` | Converte o `kept.jsonl` do pacote 4 em registros de treino (`think.jsonl`) | ativo |
| `run_noite3_chain.sh` | Fila da noite (v3, substituída pela v4): RL do `thor_pt_e1`, diagnóstico, SFT com perda DFT | histórico |
| `diag_logprob.py` | Diagnóstico de log-probabilidades: vLLM x treinador, ruído do corte do GSPO, cabeça FP32, adaptador desligado = Lumi (dois estágios, `vllm` e `hf`); rodado em 08/11, relatório em `runs/diag/lp_report.json` | ativo |
| `run_noite_chain.sh` | Fila da noite v2 (substituída pela v3): SFT `thor_pt_e1`, RL `gspo_thor_e1`, RL do ramo A/B/C (hoje B) e a suíte do Lumi (`--decide NOME` só imprime o ramo) | histórico |

## Pacotes da Thor (geração de dados)

| Arquivo | Pacote |
| --- | --- |
| `make_thor_prompts.py` | 1: matemática, Qwen3-30B |
| `make_thor2_prompts.py` | 2: Nemotron 30B (matemática em português e ciência em inglês) |
| `make_thor3_prompts.py` | 3: múltipla escolha de outros domínios, mais a tradução |
| `make_thor4_prompts.py` | 4: múltipla escolha em português com o Qwen3-30B-A3B-Thinking (pacote **não versionado**) |

## Fase 0: medições e equivalência (históricos)

`measure_train.py` e `measure_generate.py` (memória e velocidade, A-14), `compare_impls.py` e `diag_impls.py` (código remoto contra a implementação nativa), `check_merged_keys.py` (formato de chaves do fundido). Detalhes em `discovery.md` §1 a §12.

## Testes

Cada script importante tem teste em `tests/` (CPU): `python -m pytest tests --ignore tests/test_phase0_model.py`. Não rode testes de GPU com uma fila na 4090.
