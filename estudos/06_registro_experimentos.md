# 06. Registro de experimentos

Uma linha por rodada: o que foi, de onde partiu, com que dados e configuração, quanto demorou e onde estão os arquivos. Notas e comparações: [resultados_auto.md](resultados_auto.md). Estudo de cada rodada: [02](02_sft_dados_e_professor.md) a [05](05_diagnosticos_e_referencias.md). Tempos medidos nos registros (`runs/train/*/train_log.jsonl`, `runs/grpo/*/itNNN`, `runs/eval/*/summary.json`) na RTX 4090.

## SFT (LoRA r=16, alpha 32, lr 1e-4, 1 época, batch efetivo 32, salvo indicação)

| ID | Estudo | Dados | Config. | Passos | **Treino** | Tokens/s | Pico (GiB) | **Avaliação** | Pasta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke`, `smoke_dora` | medição | `train.parquet` | 3 a 5 passos | 5 e 3 | 20 s e 50 s | 5.642 e 2.124 | 18,8 e 15,9 | – | `runs/train/smoke*` |
| `viab` | E1 | GigaVerbo, 15.544 ex. (6,35 M tok) | | 486 | **21,7 min** | 5.592 | 19,0 | 6,7 min | `runs/train/viab` |
| `adalberto` | E2 | Qwen3-30B, 14.386 ex. (7,72 M tok) | limite 1.024 tok | 450 | **25,7 min** | 5.494 | 19,0 | 12,5 min | `runs/train/adalberto` |
| `adalberto_L2048` | E4 | idem, limite 2.048 (8,79 M tok) | | 450 | **28,6 min** | 5.755 | 19,0 | 13,6 min | `runs/train/adalberto_L2048` |
| `adalberto_lr5e5` | E5 | idem `adalberto` | lr 5e-5 | 450 | **25,7 min** | 5.494 | 19,0 | 12,4 min | `runs/train/adalberto_lr5e5` |
| `adalberto_dora` | E6 | idem `adalberto` | **DoRA** r=16 | 450 | **63,9 min** | 2.062 | 15,9 | 12,4 min | `runs/train/adalberto_dora` |
| `selfdist` | E3 | Lumi com prefill, 14.113 ex. (5,59 M tok) | | 442 | **19,4 min** | 4.381 | 19,0 | 9,5 min | `runs/train/selfdist` |
| `adalberto_mlp` | E10 | idem `adalberto` | **só MLP** (34 módulos, 8,5 M param.) | 450 | **23,9 min** | 6.087 | 18,3 | 12 min | `runs/train/adalberto_mlp` |

### SFT com os dados da Thor (E12) e estudo de capacidade (E11): fila `cap-queue`

Planejadas, na ordem (`scripts/capacity_plan.txt`); tempos entram aqui quando cada rodada terminar. Cada uma ~25 min de treino mais ~12 de avaliação; `thor_*` treinam menos passos (328 contra 450).

| ID | Estudo | Dados e mudança | Estado |
| --- | --- | --- | --- |
| `cap_r04_std` | E11 | `adalberto`, r=4, alpha 32 (4,3 M param.) | **feito**: treino 25,6 min, avaliação 12 min; MCQ 62,3% ▼ (igual ao r=16) |
| `thor_pt` | E12 | Nemotron 30B traduzido para o português (4.927 problemas) | **feito**: treino 17,6 min (328 passos), avaliação 12 min; MCQ 54,3% ▼, trunca 34% (laços) |
| `thor_en` | E12 | os mesmos 4.927 problemas, raciocínio original em inglês | **feito**: treino 16 min, avaliação 10 min; MCQ 60,3% ▼, GSM8K 83,1% (com thinking) |
| `thor_pt_short` | E12 | `thor_pt` só com raciocínio de até 600 tokens (3.738 de 4.927 problemas; ciência 1.255 de 2.174), mesmos 5.544 sem thinking | **na fila** (08/11): começa quando o `gspo_thor` terminar (~16:00); ~290 passos, ~15 min de treino e ~12 de avaliação |
| `thor_pt_math` | E12 | só a matemática, em português | **descartado** (sem ganho a explicar; só matemática) |
| `nt_both`, `nt_thor`, `nt_qwen` | E13 | SFT só com exemplos sem thinking | **adiados** (dados prontos) |
| `cap_r64_rs`, `cap_r64_std`, `cap_r04_rs` | E11 | rank e lei de escala | **cancelada** (08/11) |
| `type_nomlp`, `type_attn`, `type_mamba`, `type_mlp_attn`, `type_mlp_mamba` | E11 | tipo de bloco | **cancelada** (08/11) |
| `pos_L00_20`, `pos_L21_41` | E11 | posição | **cancelada** (08/11) |
| `sc_a08`, `sc_a16`, `sc_a64` | E11 | r=16 fixo, alpha 8, 16 e 64 (escala 0,5; 1; 4) | **cancelada** (08/11) |
| `cap_r32_a16` | E11 | r=32, alpha 16 (escala 0,5) | **cancelada** (08/11) |

## Fusões sem treino (CPU) e avaliações associadas

| ID | Estudo | O que é | Avaliação | Pasta |
| --- | --- | --- | --- | --- |
| `adalberto_s050`, `adalberto_s075` | E7 | `adalberto` com o efeito do adaptador x0,5 e x0,75 | 12,6 e 12,7 min | `runs/merged/adalberto_s0*` |
| `adalberto_L00_20`, `_L21_41`, `_noMLP`, `_MLPonly` | E9 | `adalberto` fundido só em parte (36, 35, 37 e 34 módulos) | 8,8; 10,0; 9,2 e 7,6 min (só thinking) | `runs/merged/adalberto_*` |

## RL (partem de `runs/train/adalberto/adapter`; 256 questões × 8 respostas por ciclo, lr 2e-5, até 2.048 tokens)

"Total" = geração + treino + avaliação final; as avaliações intermediárias ficam fora.

| ID | Estudo | Configuração | Ciclos | **Geração** | **Treino** | **Avaliação final** | **Total** | Pasta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `grpo1` | RL | T 0,8, top-p 1,0, padrão | 8 | 82 min | 58 min | 11 min | **2,5 h** | `runs/grpo/grpo1` |
| `grpo2` | RL | T 0,6, top-p 0,95, truncadas fora da perda, guarda 5% | 4 | 42 min | 17 min | 12 min (+8 min do ciclo 2, fora do total) | **1,2 h** | `runs/grpo/grpo2` |
| `grpo3` | RL | T 0,6, top-p 0,95, Dr. GRPO, 70% de múlt. escolha | 4 | 45 min | 31 min | 11 min | **1,5 h** | `runs/grpo/grpo3` |
| `grpo4` | RL | T 0,6, top-p 0,95, KL 0,04 ao SFT | 4 | 42 min | 31 min | 11 min | **1,4 h** | `runs/grpo/grpo4` |
| `raft1` | RL | T 0,6, top-p 0,95, só respostas certas | 4 | 42 min | 12 min | 13 min | **1,1 h** | `runs/grpo/raft1` |
| `gspo1` | RL | T 0,6, top-p 0,95, razão por sequência (corte 3e-4 e 4e-4) | 4 e depois 8 | 82 min | 47 min | 12 min (+12 min da avaliação de 4 ciclos, fora do total) | **2,4 h** | `runs/grpo/gspo1` |

Avaliações intermediárias de RL (só thinking, ~8 min cada): `grpo1_it3`, `grpo1_u4` (`grpo1` com 4 atualizações), `grpo1_it6`, `grpo2_it2`. A avaliação de 4 ciclos do `gspo1` ficou em `runs/eval/gspo1_c4`. O treino do ciclo 0 do `grpo1` estourou a memória na primeira tentativa e foi refeito com gradient checkpointing (pico de 12,3 GiB); a geração do ciclo 0 foi aproveitada.

RL a partir do SFT da Thor: `gspo_thor` (8 ciclos, receita do `gspo1`, `--init runs/train/thor_pt/adapter`, `--max-noclose 0.85`), sessão `rl-thor`, começa quando a `cap-queue` parar (depois do `thor_en`, ~13:17); pasta `runs/grpo/gspo_thor`. Tempos entram aqui quando terminar.

## Referências, diagnósticos e medições

| ID | Estudo | O que é | **Tempo** | Pasta |
| --- | --- | --- | --- | --- |
| `base` | D1 e D4 | Lumi original: validação nos dois modos, GSM8K-PT de teste (duas execuções) | 47 min | `runs/eval/base` |
| `nemotron4b` | D1 | Nemotron 3 Nano 4B, validação | 5,3 min | `runs/eval/nemotron4b` |
| `base_ptprefill`, `base_ptinstr` | E0 | Lumi com prefill e com instrução em português | 6,3 e 6,0 min | `runs/eval/base_pt*` |
| `mmlu_prox_diag_pt` e `_en` | D2 | Lumi original, 1.200 questões nos dois idiomas, com e sem thinking | 32 min | `runs/eval/base` |
| Entropia dos ciclos do `grpo1` | RL | 8 medições com 512 respostas cada | ~7 min | `runs/grpo/grpo1/entropy` |
| Geração da autodestilação | E3 | 17.000 problemas, vLLM | ~60 min | `data/selfdist` |
| Piloto de tradução | E8 | 200 registros, Lumi como tradutor | 88 s | `data/mot_pilot` |
| Teste de fumaça do RL | RL | 1 iteração, 16 × 4 respostas, 512 tokens | ~2 min | `runs/grpo/smoke` |
| Teste do KL | RL | duas passadas curtas (KL ≈ 0 e KL > 0) | ~3 min | `runs/grpo/klsmoke` |

## Orçamento de GPU (rodadas acima, sem a fase 0 de medições)

| Parte | Tempo aproximado |
| --- | --- |
| Treino SFT (8 rodadas) | 3,9 h |
| RL (6 rodadas e avaliações intermediárias) | 10,8 h |
| Avaliações (SFT, fusões, referências, diagnóstico) | 3,9 h |
| **Total** | **≈ 18,6 h** em ~1,5 dia de calendário |

A fila é sequencial (uma GPU de 24 GiB): os ajustes de um experimento esperam o anterior. A Thor gera dados em paralelo, sem custo na 4090.

## Dados gerados fora da 4090

Pacote 1 (Qwen3-30B), pacote 2 (Nemotron 30B) e pacote 3 (tradução e questões novas): descrição e estado em [05](05_diagnosticos_e_referencias.md), D7. Nenhum devolveu dados ainda.
