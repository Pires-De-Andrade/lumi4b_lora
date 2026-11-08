# scripts/

Índice dos 62 arquivos desta pasta, por papel. A pasta é **plana de propósito**: os registros (`estudos/06`, `discovery.md`, `runs/jobs/*.log`) citam estes caminhos nos comandos que rodaram, e o RL em andamento os carrega por caminho. Mover um arquivo quebraria a reprodução do que está registrado.

Convenções: rode sempre com `source env.sh`; tarefas longas com `scripts/detach.sh [--gpu] <nome> <comando>`; Python do treino com `PYTHONPATH=src python`, do vLLM com `.venv-vllm/bin/python` (e `VLLM_USE_FLASHINFER_SAMPLER=0`). **Não edite um `.sh` enquanto ele roda** (o bash lê o arquivo aos poucos). Marcas: **ativo** = parte do fluxo atual; **histórico** = guardado para reproduzir uma rodada já registrada.

## Infraestrutura

| Arquivo | Para quê | Estado |
| --- | --- | --- |
| `detach.sh` | Roda um comando desacoplado em tmux, com `runs/jobs/<nome>.log` e `.exit`; `--gpu` espera a GPU livre | ativo |
| `build_kernels.sh` | Compila `causal-conv1d` e `mamba-ssm` (nvcc do conda-forge, sem sudo); usa `patches/mamba-ssm-sm89-cxx20.patch` | ativo |
| `install_vllm.sh` | Cria o `.venv-vllm` | ativo |
| `chat.py` | Conversa interativa com o Lumi | ativo |

## Dados: construção, descontaminação e validação

| Arquivo | Para quê |
| --- | --- |
| `download_data.py` | Baixa o GigaVerbo-v2 e as provas |
| `build_dataset.py` | Fase 1: `data/train.parquet`, `valid.parquet` e `reports/data_manifest.json` |
| `build_validation.py`, `clean_validation.py` | Conjuntos de validação (A-10) e o subconjunto limpo do GSM8K (676 itens, `data/val_gsm8k_clean_ids.json`) |
| `build_teacher_dataset.py` | Parquet de SFT a partir de registros de um professor |
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
| `merge_adapter.py` | Funde um adaptador no Lumi em BF16 (`--scale` muda o efeito) | ativo |
| `merge_adapter_layers.py` | Funde só parte do adaptador (triagem por camadas, E9) | histórico |
| `eval_vllm.py` | Avalia um modelo no vLLM por (prova, modo); retomável; `--tasks validation|suite` | ativo |
| `eval_merged.sh` | Avalia vários fundidos na validação | ativo |
| `stop_eval_after_task.sh` | Encerra uma avaliação em curso depois de uma (prova, modo) | ativo |
| `run_suite.sh`, `run_suite_models.sh` | Suíte de teste do PRD (GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX, OAB) | ativo (só o Lumi tem resultado parcial) |
| `compare_runs.py`, `report_compare.py` | Bootstrap pareado: diferença e IC 95% entre rodadas | ativo |
| `gen_estudos_resultados.py` | Regera `estudos/resultados_auto.md` de `runs/eval/*` | ativo |
| `run_gsm8k.py`, `run_gsm8k_vllm.py` | GSM8K-PT do piloto (HF e vLLM) | histórico |
| `run_capacity.sh` + `capacity_plan.txt` | Fila do estudo de capacidade (E11; encerrado com 1 de 15 rodadas) | histórico |

## RL (GRPO e variantes)

Fluxo de um ciclo (`run_grpo.sh`): fundir o adaptador → `grpo_rollout.py` (vLLM, 256 prompts × 8 respostas, recompensa e vantagem) → `grpo_guard.py` → `grpo_train.py` (atualiza o adaptador) → avaliação a cada `--eval-every`.

| Arquivo | Para quê | Estado |
| --- | --- | --- |
| `run_grpo.sh` | Laço do RL, retomável; é o que as rodadas atuais usam | ativo |
| `grpo_rollout.py`, `grpo_train.py`, `grpo_guard.py` | Geração e recompensa, atualização, guarda de truncadas | ativo |
| `grpo_status.py` | Resumo por ciclo (recompensa, truncadas, entropia, corte, avaliações) | ativo |
| `make_grpo_prompts.py` | Prompts antigos (letra majoritária do R1) | histórico (`gspo1`, `gspo_thor`) |
| `make_questions_rl_prompts.py`, `audit_gold_questions.py` | Prompts com gabarito oficial (`data/grpo_prompts_gold.jsonl`) e auditoria estrutural | ativo |
| `run_grpo_kl.sh`, `grpo_train_kl.py` | Variante com KL ao SFT (`grpo4`) | histórico |
| `run_grpo_entropy.sh` | Entropia por ciclo de um RL já rodado | histórico |
| `bench_rollout.py`, `run_bench_rollout.sh` | Benchmark de velocidade da geração (resultado: nada adotado, estudos/08) | histórico |
| `run_gold_chain.sh` | `gspo_gold4k` (limite 4.096; parou pela regra no ciclo 4) | histórico |
| `run_thor4k_chain.sh` | RL a partir do `thor_pt` com 4.096 (**cancelada** antes de rodar) | histórico |
| `run_gold2048_resume.sh` | Retoma o `gspo_gold` (2.048) até o ciclo 4 e avalia | **em execução** |

## Pacotes da Thor (geração de dados)

| Arquivo | Pacote |
| --- | --- |
| `make_thor_prompts.py` | 1: matemática, Qwen3-30B |
| `make_thor2_prompts.py` | 2: Nemotron 30B (matemática em português e ciência em inglês) |
| `make_thor3_prompts.py` | 3: múltipla escolha de outros domínios, mais a tradução |
| `make_thor4_prompts.py` | 4: múltipla escolha em português com o Qwen3-30B-A3B-Thinking (pacote **não versionado**, ver [handoff/README.md](../handoff/README.md)) |

## Fase 0: medições e equivalência (históricos)

`measure_train.py` e `measure_generate.py` (memória e velocidade, A-14), `compare_impls.py` e `diag_impls.py` (código remoto contra a implementação nativa), `check_merged_keys.py` (formato de chaves do fundido). Detalhes em `discovery.md` §1 a §12.

## Testes

Cada script importante tem teste em `tests/` (CPU): `python -m pytest tests --ignore tests/test_phase0_model.py`. Não rode testes de GPU com uma fila na 4090.
