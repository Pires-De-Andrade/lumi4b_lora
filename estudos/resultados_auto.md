# Resultados de todas as rodadas (gerado por `scripts/gen_estudos_resultados.py`)

Validação limpa, nota **estrita**, decodificação gulosa, 4.096 tokens. GSM8K = 676 itens (sem duplicatas por tradução); múltipla escolha = 385 itens. Entre parênteses, a diferença em pontos contra o Lumi no mesmo modo, com a marca ▲ (IC 95% acima de 0) ou ▼ (IC abaixo de 0); sem marca, empate estatístico. O ruído de 385 itens é de ±4 a 5 pontos.

## Referências (sem treino nosso)

| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |
| --- | --- | --- | --- | --- | --- | --- |
| Lumi original (`base`) | 81,7 | 77,2 | 69,6 | 63,9 | 86,2 / 0,0 / 13,8 / 490 | 47 |
| Nemotron 3 Nano 4B (pai do Lumi) (`nemotron4b`) | 85,2 (+3,6) ▲ | 63,5 (-13,8) ▼ | 68,3 (-1,3) | 58,7 (-5,2) ▼ | 89,6 / 0,3 / 10,4 / 503 | 5 |

## SFT: dados e professor

| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |
| --- | --- | --- | --- | --- | --- | --- |
| viab: 10 mil traços do GigaVerbo, 1 época · 17,1 M param. (`viab`) | 67,2 (-14,5) ▼ | 57,7 (-19,5) ▼ | 43,4 (-26,2) ▼ | 43,1 (-20,8) ▼ | 99,2 / 100,0 / 0,8 / 566 | 7 |
| adalberto: traços do Qwen3-30B (GSM8K), lr 1e-4 · 17,1 M param. (`adalberto`) | 83,3 (+1,6) | 79,3 (+2,1) | 62,3 (-7,3) ▼ | 58,2 (-5,7) ▼ | 86,8 / 99,4 / 13,8 / 1176 | 13 |
| adalberto com lr 5e-5 · 17,1 M param. (`adalberto_lr5e5`) | 83,1 (+1,5) | 80,2 (+3,0) ▲ | 63,4 (-6,2) ▼ | 57,9 (-6,0) ▼ | 86,5 / 97,9 / 14,5 / 1149 | 12 |
| adalberto com limite de 2.048 tokens · 17,1 M param. (`adalberto_L2048`) | 82,2 (+0,6) | 81,7 (+4,4) ▲ | 64,2 (-5,5) ▼ | 55,3 (-8,6) ▼ | 85,5 / 99,4 / 15,1 / 1292 | 14 |
| selfdist: traços do próprio Lumi com prefill em português · 17,1 M param. (`selfdist`) | 78,1 (-3,6) ▼ | 77,5 (+0,3) | 57,9 (-11,7) ▼ | 63,4 (-0,5) | 79,2 / 93,8 / 19,7 / 475 | 9 |
| adalberto com DoRA · 17,7 M param. (`adalberto_dora`) | 84,0 (+2,4) | 79,1 (+1,9) | 62,6 (-7,0) ▼ | 57,7 (-6,2) ▼ | 87,5 / 99,4 / 12,7 / 1146 | 12 |

## SFT: dados do Nemotron 30B da Thor, traduzidos (entrega parcial: 4.927 problemas com raciocínio + 5.544 linhas sem raciocínio do GigaVerbo)

| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |
| --- | --- | --- | --- | --- | --- | --- |
| **adalberto: traços do Qwen3-30B (referência)** · 17,1 M param. (`adalberto`) | 83,3 (+1,6) | 79,3 (+2,1) | 62,3 (-7,3) ▼ | 58,2 (-5,7) ▼ | 86,8 / 99,4 / 13,8 / 1176 | 13 |
| thor_pt: raciocínio do Nemotron 30B traduzido para o português · 17,1 M param. (`thor_pt`) | 77,8 (-3,8) ▼ | 74,6 (-2,7) | 54,3 (-15,3) ▼ | 62,9 (-1,0) | 66,0 / 99,6 / 34,0 / 703 | 12 |
| thor_en: os mesmos problemas, raciocínio original em inglês · 17,1 M param. (`thor_en`) | 83,1 (+1,5) | 65,7 (-11,5) ▼ | 60,3 (-9,4) ▼ | 56,1 (-7,8) ▼ | 75,3 / 0,0 / 24,7 / 655 | 10 |
| thor_pt_short: o thor_pt só com raciocínio de até 600 tokens (3.738 dos 4.927 problemas) (`thor_pt_short`) | – | – | – | – | – | – |
| thor_pt_math: só a matemática do mesmo professor (sem múltipla escolha), em português (`thor_pt_math`) | – | – | – | – | – | – |

## SFT: tamanho do efeito do adaptador

| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |
| --- | --- | --- | --- | --- | --- | --- |
| adalberto, efeito do adaptador x0,5 (`adalberto_s050`) | 84,0 (+2,4) | 79,6 (+2,4) | 63,4 (-6,2) ▼ | 64,9 (+1,0) | 81,3 / 97,8 / 18,2 / 1121 | 13 |
| adalberto, efeito do adaptador x0,75 (`adalberto_s075`) | 84,0 (+2,4) | 79,1 (+1,9) | 64,4 (-5,2) ▼ | 58,7 (-5,2) ▼ | 87,0 / 98,2 / 13,5 / 1151 | 13 |

## SFT: só os MLPs

| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |
| --- | --- | --- | --- | --- | --- | --- |
| adalberto_mlp: LoRA só em up_proj e down_proj (34 módulos), treinado do zero · 8,5 M param. (`adalberto_mlp`) | 83,4 (+1,8) | 80,2 (+3,0) ▲ | 62,3 (-7,3) ▼ | 54,3 (-9,6) ▼ | 84,7 / 98,5 / 15,8 / 1147 | 12 |

## Capacidade e tipo de bloco (treinado do zero, 71 módulos salvo indicação; E11 encerrado em 1 de 15 rodadas)

| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |
| --- | --- | --- | --- | --- | --- | --- |
| **r=16, alpha 32, 71 módulos (base do `adalberto`)** · 17,1 M param. (`adalberto`) | 83,3 (+1,6) | 79,3 (+2,1) | 62,3 (-7,3) ▼ | 58,2 (-5,7) ▼ | 86,8 / 99,4 / 13,8 / 1176 | 13 |
| r=4, alpha 32 (escala 8), 71 módulos · 4,3 M param. (`cap_r04_std`) | 82,5 (+0,9) | 81,2 (+4,0) ▲ | 62,3 (-7,3) ▼ | 60,0 (-3,9) | 86,8 / 98,8 / 14,0 / 1139 | 12 |
| r=16, só MLP (34 módulos) · 8,5 M param. (`adalberto_mlp`) | 83,4 (+1,8) | 80,2 (+3,0) ▲ | 62,3 (-7,3) ▼ | 54,3 (-9,6) ▼ | 84,7 / 98,5 / 15,8 / 1147 | 12 |

## RL (a partir do adalberto; `gspo_thor` a partir do `thor_pt`)

| Rodada | GSM8K com thinking | GSM8K sem | Múlt. escolha com thinking | Múlt. escolha sem | Forma da múlt. escolha: fecha / PT / trunca / mediana tok | Avaliação (min) |
| --- | --- | --- | --- | --- | --- | --- |
| grpo1: 8 ciclos, T 0,8 (`grpo1`) | 82,8 (+1,2) | 77,5 (+0,3) | 66,8 (-2,9) | 61,6 (-2,3) | 91,7 / 99,2 / 8,6 / 904 | 11 |
| grpo2: 4 ciclos, T 0,6, top-p 0,95, truncadas fora da perda (`grpo2`) | 82,7 (+1,0) | 79,6 (+2,4) | 66,0 (-3,6) | 59,5 (-4,4) | 84,2 / 98,1 / 16,1 / 1143 | 12 |
| grpo3: Dr. GRPO + 70% múltipla escolha (`grpo3`) | 84,6 (+3,0) ▲ | 79,1 (+1,9) | 66,5 (-3,1) | 43,1 (-20,8) ▼ | 91,7 / 96,6 / 8,1 / 999 | 11 |
| grpo4: KL 0,04 (`grpo4`) | 84,3 (+2,7) ▲ | 79,7 (+2,5) | 62,9 (-6,8) ▼ | 43,1 (-20,8) ▼ | 90,6 / 99,1 / 8,6 / 1010 | 11 |
| raft1: só respostas certas (`raft1`) | 82,8 (+1,2) | 78,4 (+1,2) | 61,6 (-8,1) ▼ | 59,2 (-4,7) | 84,2 / 96,6 / 15,1 / 1155 | 13 |
| gspo1: razão por sequência, 4 ciclos (`gspo1_c4`) | 84,3 (+2,7) | 79,9 (+2,7) | 67,8 (-1,8) | 59,5 (-4,4) | 90,9 / 95,4 / 10,6 / 1086 | 12 |
| gspo1: 8 ciclos (`gspo1`) | 84,3 (+2,7) ▲ | 78,6 (+1,3) | 67,5 (-2,1) | 57,7 (-6,2) ▼ | 93,2 / 96,1 / 8,1 / 1035 | 12 |
| gspo_thor: 4 ciclos a partir do `thor_pt` (só thinking) (`gspo_thor_it4`) | 83,7 (+2,1) | – | 65,7 (-3,9) | – | 82,9 / 98,4 / 17,1 / 520 | 6 |
| gspo_thor: 8 ciclos a partir do `thor_pt` (Nemotron 30B) (`gspo_thor`) | – | – | – | – | – | – |

## Triagem por camadas (adaptador `adalberto` fundido só em parte; só com thinking)

| Parte mantida | GSM8K | Múlt. escolha | Forma da múlt. escolha: fecha / PT / trunca / mediana tok |
| --- | --- | --- | --- |
| adaptador completo (71 módulos) | 83,3 (+1,6) | 62,3 (-7,3) ▼ | 86,8 / 99,4 / 13,8 / 1176 |
| só camadas 0 a 20 | 83,6 (+1,9) | 62,6 (-7,0) ▼ | 81,3 / 0,0 / 16,9 / 681 |
| só camadas 21 a 41 | 76,6 (-5,0) ▼ | 18,4 (-51,2) ▼ | 22,9 / 98,9 / 23,1 / 688 |
| sem MLP (Mamba e atenção) | 81,7 (+0,0) | 60,0 (-9,6) ▼ | 81,3 / 98,7 / 19,0 / 1192 |
| só MLP (do adaptador completo, sem retreinar) | 83,9 (+2,2) | 69,6 (+0,0) | 87,8 / 2,4 / 11,9 / 540 |

'PT' é a fração dos traços classificados em português pela heurística de palavras funcionais (`lumi_ft/eval/shape.py`); 'fecha' é a fração que fecha um `</think>` e escreve a resposta depois.
