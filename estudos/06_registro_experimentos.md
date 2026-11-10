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
| `thor_pt_short` | E12 | `thor_pt` só com raciocínio de até 600 tokens (3.738 de 4.927 problemas; ciência 1.255 de 2.174), mesmos 5.544 sem thinking | **feito**: treino 13 min (291 passos), avaliação 10 min; MCQ com thinking 57,4% ▼ (+3,1 sobre o `thor_pt`, IC −1,0 a +7,5), trunca 26,2% |
| `thor_pt_e1` | E12 | entrega 1 do pacote 3: 9.850 exemplos com raciocínio (5.426 de múltipla escolha), limite 2.048 tokens, mais as mesmas 5.544 linhas sem raciocínio; receita do `thor_pt` | **feito** (08/11): treino de 481 passos em ~31 min (a primeira avaliação caiu com segfault do vLLM ao carregar o tokenizador e foi refeita sem mudança, 12 min). Múlt. escolha com thinking **51,4%** ▼ (−18,2 contra o Lumi; −2,9 contra o `thor_pt`, IC −7,0 a +1,3), GSM8K limpo 76,9% ▼, fecha 63,4%, trunca 36,6% (mediana 802 tokens), sem thinking 61,8% |
| `adalberto_dft` | E2 | os mesmos dados do `adalberto` (limite 1.024 tokens), `--loss dft` | **feito** (08/11 22:44 a 23:22; 450 passos, ~25 min de treino + ~12 de avaliação): múlt. escolha com thinking **61,3%** ▼ (−1,0 contra o `adalberto`, IC −6,0 a +4,2; sem ganho), GSM8K limpo 83,3%, fecha 77,7%, **traço em português só 0,7%** na múltipla escolha (15,6% no GSM8K): a perda DFT não ensinou o raciocínio em português. Sem thinking: múlt. escolha 63,1% (o `adalberto` tem 58,2%) | `runs/train/adalberto_dft` |
| `qwen_mix_p` | E12 | os dados do `adalberto` como estão (limite 1.024 tokens) + 1.810 traços de múltipla escolha da entrega parcial do pacote 4 (1.141 MedMCQA, 669 WebInstruct; até 2.048 tokens); receita do `adalberto` | **feito** (09/11 02:13 a 02:45, 507 passos, 1.903 s; a avaliação foi refeita em 14:28 a 14:41 depois da queda da máquina): múlt. escolha com thinking **61,8%** (−0,5 contra o `adalberto`, IC −4,7 a +3,6; −7,8 contra o Lumi), fecha 83,1% (86,8% no `adalberto`), trunca 17,7%, mediana 1.044 tokens, traço em português 100%; GSM8K limpo 84,5% (+1,2, IC −0,7 a +3,1); múlt. escolha sem thinking 61,6% (58,2% no `adalberto`). Sem ganho em thinking; sem RL (cancelado quando o pacote completo chegou) | `runs/train/qwen_mix_p` |
| `qwen_mix` | E12 | os dados do `adalberto` como estão + 5.055 traços de múltipla escolha do pacote 4 completo (2.752 MedMCQA, 2.303 WebInstruct; até 2.048 tokens), 19.441 linhas; receita do `adalberto` | **feito** (09/11 14:42 a 15:39, 608 passos; treino ~35 min + avaliação ~13 min): múlt. escolha com thinking **62,1%** (−0,3 contra o `adalberto`, IC −4,7 a +4,2; −7,5 contra o Lumi), fecha 83,9% (86,8% no `adalberto`), trunca 16,9%, mediana 1.021 tokens, traço em português 99,4%; GSM8K limpo 83,4% (+0,1, IC −1,6 a +1,9); múlt. escolha sem thinking 62,9% (58,2% no `adalberto`). Sem ganho em thinking, igual ao `qwen_mix_p` com 2,8 vezes menos traços | `runs/train/qwen_mix` |
| `qwen_mix_short` | E12 | os dados do `adalberto` como estão + 2.221 traços de múltipla escolha do pacote 4 de até 1.024 tokens (1.517 MedMCQA, 704 WebInstruct), 16.607 linhas; receita do `adalberto` (H53) | **feito em 09/11 (`noite12`, 44 min de treino, 519 passos, avaliação de 13 min)**: com thinking, múltipla escolha **66,5%** (−3,1 contra o Lumi, IC −7,5 a +1,3: empate estatístico; contra o `adalberto`, +4,2, IC −0,5 a +8,8), fecha **89,1%**, traço em português 99,1%, mediana de 884 tokens; GSM8K limpo 83,7% (+2,1, IC −0,4 a +4,6), fecha 97,0%. Sem thinking (o adaptador não é usado): múltipla escolha 62,3%. Decomposição pareada: nos 307 itens em que o Lumi e ele fecham, 77,5% contra 82,4% (−4,9 pontos, contra −7,5 do `gspo1`); só o Lumi fecha em 25 itens (−3,9); só ele fecha em 36 (+4,7). **H53 apoiada** (fecha ≥ 87% e múltipla escolha ≥ 62%): cortar os traços de múltipla escolha em 1.024 tokens rende o melhor SFT até agora, no nível do `gspo1`, sem RL. Das 42 respostas que não fecham, 17 têm laço exato e 11 laço pelo detector oficial: o resto é raciocínio longo. Próximo: RL (H60) | `runs/train/qwen_mix_short` |
| `gspo_qwen_mix_short` | RL (receita do `gspo1`) | – | **feito em 09/11 (`noite13`, 4 ciclos, 75 min, avaliação de 13 min)**: com thinking, múltipla escolha **67,0%** (−2,6 contra o Lumi, IC −7,0 a +1,8: empate estatístico), fecha **90,1%**, traço em português 98,6%, mediana de 905 tokens; GSM8K limpo 83,1% (+1,5, IC −1,0 a +4,1), fecha 97,2%. Contra o próprio SFT `qwen_mix_short` (66,5%; fecha 89,1%; GSM8K limpo 83,7%): +0,5, +1,0 e −0,6 pontos, ou seja, o RL quase não acrescentou. Igual ao `gspo1_c4` (67,8%; fecha 90,9%; 84,3%), que usou os mesmos 4 ciclos. Decomposição pareada contra o Lumi: os dois fecham, 307 itens: 78,2% contra 82,1% (−3,9 por item; o `gspo1` perde 7,5); só o Lumi fecha, 25 itens: −4,2 pontos; só ele fecha, 40 itens (45% de acerto): +4,7. Das 38 respostas que não fecham, 19 têm laço (qualquer dos dois detectores). **H60 sem apoio pela regra** (múltipla escolha ≥ 69,5%): cai na faixa de um segundo `gspo1`. Recompensa média de múltipla escolha por ciclo: 0,59, 0,65, 0,62, 0,63; truncadas na amostragem de treino (de 1.024): 186, 131, 133, 157 (o truncamento deixou de cair depois do primeiro ciclo). Comando:  `scripts/run_grpo.sh gspo_qwen_mix_short --init runs/train/qwen_mix_short/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq` | `runs/grpo/gspo_qwen_mix_short` |
| `selfdist_min` | E12 | 3.000 traços de matemática do próprio Lumi, em português com o começo forçado (os aceitos do `selfdist`, um por problema, até 1.024 tokens), sem linhas sem raciocínio; ~94 passos; receita do `adalberto` (H54) | **na fila** (`noite7`, teste 1, depois do `qwen_mix_short`): ~6 min de treino + ~13 de avaliação | `runs/train/selfdist_min` |
| `selfdist_300` | E12 | 300 traços do mesmo conjunto do `selfdist_min` (~10 passos); receita do `adalberto` (H54b; o XReasoning, arXiv 2505.22888, relata que 100 exemplos bastam para o idioma) | **na fila** (`noite8`, depois do `selfdist_min`): ~2 min de treino + ~13 de avaliação | `runs/train/selfdist_300` |
| `base_ptprefill` | E0 | Lumi sem adaptador, com o começo do raciocínio forçado em português (`--prefill "Vou raciocinar em português."`, só no modo thinking) (H56) | **feito em 07/11** (E0): múlt. escolha 61,3% (−8,3 contra o Lumi), fecha 79,7%, trunca 18,4%; GSM8K 48,1% (formato quebrado). Só com thinking; a reavaliação completa de 09/11 reproduziu a medida (61,3%; GSM8K limpo 47,5%) | `runs/eval/base_ptprefill` |
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

RL a partir do SFT da Thor: `gspo_thor` (8 ciclos, receita do `gspo1`, `--init runs/train/thor_pt/adapter`, `--max-noclose 0.85`), sessão `rl-thor`, 13:14 a 14:50; pasta `runs/grpo/gspo_thor`. Geração 47 min (9,1 M tokens), treino 29 min, avaliação final 7 min (+6 min da avaliação do ciclo 4, em `runs/eval/gspo_thor_it4`, fora do total): **1,4 h**. Resultado em [04](04_rl_grpo.md): múltipla escolha com thinking 67,0%, empata com o `gspo1`.

RL com gabarito oficial (`--prompts data/grpo_prompts_gold.jsonl`, receita do `gspo1`: `--init runs/train/adalberto/adapter`, T 0,6, top-p 0,95, `--ratio seq`, `--max-noclose 0.05`; ciclos de ~18 min com 256 questões × 8 respostas). Mesmas 256 questões por ciclo nas três rodadas (a seleção é determinística):

| ID | Limite de geração | Ciclos | Estado e resultado (validação limpa, com thinking) | Pasta |
| --- | --- | --- | --- | --- |
| `gspo_gold` | 2.048 | 4 (retomada) | ciclos 0 e 1 em 17:20–17:59; interrompida no ciclo 2 às 18:02 para testar 4.096; retomada às 20:02 (`scripts/run_gold2048_resume.sh`), ciclos 2 e 3 e avaliação do ciclo 4 até 20:50. Avaliação em `runs/eval/gspo_gold_it4`: múlt. escolha **63,9%** (−5,7 contra o Lumi, IC −10,1 a −1,3 ▼; −3,9 contra o `gspo1_c4`, IC −7,8 a +0,0), GSM8K 85,2% ▲, fecha 89,4%, trunca 11,4% (mediana 1.167 tokens). Não continua até o ciclo 8 (ramo B de [08](08_registro_de_decisoes.md)) | `runs/grpo/gspo_gold` |
| `gspo_gold4k` | 4.096 | 4 de 8 (parou pela regra de [08](08_registro_de_decisoes.md) às 19:41) | ciclo 4 avaliado em `runs/eval/gspo_gold4k_it4`: múlt. escolha **63,1%**, GSM8K 84,6%, fecha 83,9%, trunca 16,4% (mediana 1.250 tokens) | `runs/grpo/gspo_gold4k` |
| `gspo_gold4k_thor` | 4.096, a partir do `thor_pt` | – | **cancelada** antes de rodar (19:45): repetia a configuração suspeita | – |
| `gspo1_4k` | 4.096, questões antigas | 4 | **na fila** (sessão `noite`, ramo B, 3ª etapa): receita do `gspo1` (a partir do `adalberto`, T 0,6, top-p 0,95, `--ratio seq`) só com o limite trocado; um início às 20:51 foi interrompido na fusão e refeito | `runs/grpo/gspo1_4k` |
| `gspo_thor_e1` | 2.048, questões antigas | 4 | **feito** (21:43 a 22:42, 4 ciclos, a partir do `thor_pt_e1`; as sessões `noite` e `noite3`): avaliação em `runs/eval/gspo_thor_e1`: múlt. escolha com thinking **62,1%** ▼ (−7,5 contra o Lumi, IC −11,9 a −3,1; −3,6 contra o `gspo_thor` no ciclo 4, IC −7,5 a +0,0; +10,7 sobre o próprio ponto de partida), GSM8K limpo 84,3% ▲, fecha 80,8%, trunca 19,2% (36,6% no SFT), mediana 612 tokens. Sem decisão de continuar até o ciclo 8 | `runs/grpo/gspo_thor_e1` |
| `gspo1_T06` | 2.048, questões antigas | 4 | **feito** (08/11 23:24 a 09/11 00:43, 4 ciclos): avaliação em `runs/eval/gspo1_T06`: múlt. escolha com thinking **63,6%** ▼ (−6,0 contra o Lumi, IC −10,4 a −1,6; **−4,2 contra o `gspo1_c4`**, IC −8,3 a +0,0), GSM8K limpo 84,0% (−0,3 contra o `gspo1_c4`), fecha 87,3% (90,9% no `gspo1_c4`), trunca 12,7%, mediana 1.027 tokens. Sem ganho; a leitura depende do ruído entre sementes (`gspo1_s1`) | `runs/grpo/gspo1_T06` |
| `gspo1_s1` | 2.048, questões antigas | 4 | **feito** (09/11 00:45 a 02:11, 4 ciclos; receita do `gspo1` com a semente 1, `GRPO_SEED=1`): avaliação em `runs/eval/gspo1_s1`: múlt. escolha com thinking **64,4%** (**−3,4 contra o `gspo1_c4`**, IC −7,5 a +0,8; −5,2 contra o Lumi, IC −9,9 a −0,8 ▼), GSM8K limpo 82,5% (−1,8, IC −3,7 a +0,1; +0,9 contra o Lumi), fecha 87,8% (90,9% no `gspo1_c4`), trunca 13,0%, mediana 1.222 tokens, traço em português 97,9%, múlt. escolha sem thinking 58,7%. É o tamanho do ruído entre sementes da receita | `runs/grpo/gspo1_s1` |
| `gspo1_soup2` | média de pesos | – | média dos adaptadores do `gspo1` (ciclo 4) e do `gspo1_s1`, todos a partir do `adalberto`; distância relativa à média ~1,7%, erro da aproximação B·A ≤ 0,14% (`runs/train/gspo1_soup2/adapter/soup_report.json`). **Validação com thinking (09/11):** múltipla escolha 63,6% (−6,0 contra o Lumi; IC −10,6 a −1,3), fecha 88,1%, traço em português 98,2%; GSM8K limpo 84,6% (+3,0; IC +0,4 a +5,5). Abaixo das duas rodadas de origem (67,8% e 64,4%) | `runs/merged/gspo1_soup2` |
| `gspo1_soup3` | média de pesos | – | a anterior mais o `gspo1_T06`; distância relativa ~2,0 a 2,2%, erro B·A ≤ 0,17%. **Validação com thinking (09/11):** múltipla escolha 64,7% (−4,9; IC −9,6 a −0,3), fecha 88,8%, traço em português 97,7%; GSM8K limpo 84,9% (+3,3; IC +0,9 a +5,6) | `runs/merged/gspo1_soup3` |
| `thor_pt_antidoom` | fase anti-laço (receita oficial) | – | **na fila** (`noite12`, primeira da fase; H59): o `thor_pt` fundido + LoRA novo r=256/alpha 128 (alvos do PEFT no NemotronH, sem o `out_proj` do Mamba, mais o `lm_head`); mineração em 10.000 prompts (temperatura 0,1, top-k 50, min-p 0,01, 4.000 tokens, detector oficial); perda FTPO oficial; parada no `chosen_win` de treino ≥ 0,4 | `runs/merged/thor_pt_antidoom` |
| `gspo_thor_antidoom` | fase anti-laço (receita oficial) | – | **na fila** (`noite12`, H58): o mesmo sobre o `gspo_thor` (`runs/grpo/gspo_thor/merged`), mineração em 8.000 prompts de múltipla escolha | `runs/merged/gspo_thor_antidoom` |
| `thor_pt_antidoom_of` | fase anti-laço (treinador oficial) | – | **na fila** (`noite12`, conferência): os mesmos eventos do `thor_pt`, treinados com o `FTPOTrainer` do repositório oficial (`scripts/antidoom_oficial_train.py`) | `runs/merged/thor_pt_antidoom_of` |
| `gspo1_antidoom` | fase anti-laço (receita oficial) | – | **na fila** (`noite12`, H57, última da fase; teto de ~+0,6 ponto): eventos da mineração gulosa do `gspo1` (136 em 10.000 prompts) | `runs/merged/gspo1_antidoom` |
| `gspo1_eps3x` | 2.048, questões antigas | 4 | **na fila** (sessão `noite4`, 3ª rodada de RL): receita do `gspo1` com o corte do GSPO 3x maior (`GRPO_EPS_LOW=1e-3`, `GRPO_EPS_HIGH=1.3e-3`), semente 0 | `runs/grpo/gspo1_eps3x` |
| `gspo1_same` | 2.048, questões antigas | 4 | **na fila** (sessão `noite4`, 4ª rodada de RL): receita do `gspo1` com as log-probabilidades antigas nos mesmos micro-batches do treino (`GRPO_OLD_SAME_BATCHES=1`), semente 0 | `runs/grpo/gspo1_same` |
| `gspo_qwen_mix` | 2.048, questões antigas | 4 | **feito** (09/11 15:39 a 16:59, 4 ciclos): avaliação em `runs/eval/gspo_qwen_mix`: múlt. escolha com thinking **64,4%** (−3,4 contra o `gspo1_c4`, IC −7,5 a +0,8; −5,2 contra o Lumi, IC −9,9 a −0,8 ▼), igual ao `gspo1_s1`; fecha 87,8%, trunca 12,5%, mediana 982 tokens, traço em português 98,8%; GSM8K limpo 84,0% (−0,3 contra o `gspo1_c4`; +2,4 contra o Lumi, IC −0,1 a +5,0); múlt. escolha sem thinking 65,7%. Dentro da faixa de 64,4 a 67,8% do `gspo1`: os traços de MCQ do Qwen não acrescentam nada depois do RL | `runs/grpo/gspo_qwen_mix` |

Benchmark de velocidade da geração (08/11, 5 configurações do vLLM, `runs/bench_rollout/`): ganho máximo +9,3% contra ruído de 5,4%; o critério combinado (≥15%) não foi atingido e nada foi adotado ([discovery.md](../discovery.md) §20.35 e §20.41).

## Referências, diagnósticos e medições

| ID | Estudo | O que é | **Tempo** | Pasta |
| --- | --- | --- | --- | --- |
| `base` | D1 e D4 | Lumi original: validação nos dois modos, GSM8K-PT de teste (duas execuções) | 47 min | `runs/eval/base` |
| `nemotron4b` | D1 | Nemotron 3 Nano 4B, validação | 5,3 min | `runs/eval/nemotron4b` |
| `base_ptprefill`, `base_ptinstr` | E0 | Lumi com prefill e com instrução em português | 6,3 e 6,0 min | `runs/eval/base_pt*` |
| `mmlu_prox_diag_pt` e `_en` | D2 | Lumi original, 1.200 questões nos dois idiomas, com e sem thinking | 32 min | `runs/eval/base` |
| Suíte de teste do Lumi (`base`; o `gspo1` e o `gspo_thor` ficaram de fora) | PRD | GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX e OAB (17.417 itens), nos dois modos, `eval_vllm.py --tasks suite` | **parcial** (08/11): GSM8K-PT de teste nos dois modos (88,73% e 85,33%) e MMLU-ProX PT com thinking (53,57%); parada de propósito (`stop_eval_after_task.sh`). Faltam MMLU-ProX sem thinking, ENEM, BLUEX e OAB do Lumi, e a suíte de qualquer candidato | `runs/eval/base` |
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

Pacote 1 (Qwen3-30B), pacote 2 (Nemotron 30B) e pacote 3 (tradução e questões novas): descrição e estado em [05](05_diagnosticos_e_referencias.md), D7. Devolvido até agora: do pacote 2 (traduzido), 5.278 registros com raciocínio (08/11, 11:47); do pacote 3, as **12.000 questões traduzidas** (`questions_pt.tar.gz`, 08/11 15:26, sha256 `8c4e3339…ca941`; 11.455 aceitas, 95,5%; 4.717 MedMCQA e 6.738 WebInstruct-verified), convertidas em `data/grpo_prompts_gold.jsonl` (10.981 questões de múltipla escolha com gabarito oficial após o filtro de 8-gramas, mais as 5.612 de matemática).
