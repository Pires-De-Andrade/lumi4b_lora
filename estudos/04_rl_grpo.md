# 04. Estudos de RL: GRPO e variantes

Pergunta: partindo do adaptador do SFT (`adalberto`), o RL com recompensa verificável **recupera a múltipla escolha** sem depender de traços de professor, e o que o mantém estável? Protocolo e convenções: [01_protocolo.md](01_protocolo.md). Todas as rodadas partem do adaptador `adalberto` (LoRA r=16, 71 módulos), exceto onde dito.

## Desenho do RL

Cada **ciclo** roda em processos separados (a 4090 não comporta o vLLM e o treino juntos): (1) funde o adaptador no Lumi; (2) gera **8 respostas por questão** para 256 questões (metade matemática, metade múltipla escolha) com o vLLM, até 2.048 tokens, modo thinking, sem começo forçado; (3) calcula a recompensa e a vantagem; (4) treina o adaptador por uma passada pelas respostas com vantagem diferente de zero (64 respostas por passo do otimizador, lr 2e-5, AdamW com o estado guardado entre ciclos). Um ciclo leva uns **13 a 20 min**.

| Componente | Definição |
| --- | --- |
| **Recompensa** | 1 só se a resposta fecha um único `</think>`, termina sozinha (não trunca), o traço está em português e a resposta final bate com o gabarito no formato estrito (`#### N` ou `Resposta: <letra>`); senão 0. Truncada vale 0 |
| **Vantagem** | recompensa menos a média do grupo de 8, dividida pelo desvio-padrão (`--adv std`); grupo com recompensas iguais vale 0 e não treina |
| **Perda (padrão)** | PPO por token com corte assimétrico (0,2 e 0,28), sem KL, normalizada pelo total de tokens de resposta do minibatch; as log-probabilidades "antigas" vêm de uma passada sem gradiente antes de qualquer atualização |
| **Prompts** | 5.612 de matemática (GSM8K-PT `train`, gabarito oficial) e 3.715 de múltipla escolha (questões de ciência do Mixture-of-Thoughts com **só o enunciado traduzido**, descontaminadas). Cada ciclo usa 128 de cada, sem repetir |
| **Ponto fraco** | o gabarito da múltipla escolha é a **letra majoritária dos traços do R1** (≥ 75% de pelo menos 3 traços), não um gabarito oficial |

**Ferramentas criadas durante o estudo** (todas testadas na CPU, 14 testes em `tests/test_grpo.py`): entropia do modelo nas próprias respostas por ciclo; **guarda** que para a rodada se mais de X% das respostas de um tipo terminarem sem um único `</think>`; opções `--top-p`, `--mask-truncated`, `--ratio seq` (GSPO), `--positives-only` (RAFT), `--kl-coef`; `scripts/grpo_status.py` (resumo por ciclo); `scripts/run_grpo_entropy.sh` (entropia de rodadas já feitas).

## Resultados (validação limpa, com e sem thinking; ▲ ▼ = IC contra o Lumi)

| Rodada | Ciclos | GSM8K | Múlt. escolha | Fecha `</think>` (MCQ) | Trunca (MCQ) | Tempo total |
| --- | --- | --- | --- | --- | --- | --- |
| Lumi original | – | 81,7 / 77,2 | 69,6 / 63,9 | 86,2% | 13,8% | – |
| `adalberto` (ponto de partida) | 0 | 83,3 / 79,3 | 62,3 ▼ / 58,2 ▼ | 86,8% | 13,8% | – |
| `grpo1` (T 0,8, padrão) | 8 | 82,8 / 77,5 | 66,8 / 61,6 | 91,7% | 8,6% | 2,5 h |
| `grpo2` (T 0,6, top-p 0,95, truncadas fora da perda) | 4 | 82,7 / 79,6 | 66,0 / 59,5 | 84,2% | **16,1%** | 1,2 h |
| `grpo3` (Dr. GRPO, 70% de múlt. escolha) | 4 | **84,6 ▲** / 79,1 | 66,5 / **43,1 ▼** | 91,7% | 8,1% | 1,5 h |
| `grpo4` (KL 0,04) | 4 | 84,3 ▲ / 79,7 | 62,9 ▼ / **43,1 ▼** | 90,6% | 8,6% | 1,4 h |
| `raft1` (só respostas certas) | 4 | 82,8 / 78,4 | 61,6 ▼ / 59,2 | 84,2% | 15,1% | 1,1 h |
| `gspo1` (razão por sequência) | 4 | 84,3 / 79,9 | 67,8 / 59,5 | 90,9% | 10,6% | 1,3 h |
| **`gspo1`** | **8** | **84,3 ▲** / 78,6 | 67,5 / 57,7 ▼ | **93,2%** | 8,1% | 2,4 h |

`grpo2` a `gspo1` usam T 0,6 e top-p 0,95 (salvo a coluna). Tempo total = geração + treino + avaliação final (com e sem thinking). `grpo3` e `grpo4` e `gspo1` não usam o filtro de truncadas. A tabela completa, com diferenças e IC, está em [resultados_auto.md](resultados_auto.md).

## O estudo da entropia

O `grpo1` mostrou degeneração: respostas sem `</think>` de 11 para 223 (de 1.024 na múltipla escolha) entre os ciclos 0 e 6, lixo no fim das respostas, recompensa de treino na múltipla escolha caindo de 0,59 para 0,44. Para medir a causa, calculei a **entropia do modelo nas próprias respostas** (nats por token, temperatura 1) em cada ciclo:

| Ciclo | `grpo1` (T 0,8) | `grpo2` | `grpo4` (KL) | `grpo3` | `raft1` | `gspo1` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0,42 | 0,42 | 0,42 | 0,45 | 0,40 | 0,41 |
| 1 | 0,49 | 0,34 | 0,52 | 0,55 | 0,29 | 0,42 |
| 2 | 0,55 | 0,27 | 0,63 | 0,64 | 0,22 | 0,44 |
| 3 | 0,71 | 0,23 | 0,75 | **0,89** | 0,18 | 0,45 |
| 4 | 0,97 | | | | | 0,44 |
| 5 | 1,56 | | | | | 0,45 |
| 6 | **2,14** | | | | | 0,44 |
| 7 | 1,98 | | | | | 0,41 |

(As rodadas de T 0,6 geram com distribuição mais fria, o que puxa a entropia medida para baixo; comparar a tendência dentro de cada rodada, não os valores entre rodadas de temperaturas diferentes. O KL do `grpo4` ao SFT foi 0,002, 0,005, 0,014 e 0,040 por token nos ciclos 0 a 3.)

## Hipóteses e vereditos

| ID | Hipótese | Evidência | Veredito |
| --- | --- | --- | --- |
| H11 | O RL a partir do SFT recupera a múltipla escolha com thinking | 62,3% → 66 a 68% (empate com o Lumi, IC inclui 0) em `grpo1`, `grpo2`, `grpo3`, `gspo1` | **Apoiada em parte** (não passa o Lumi) |
| H12 | A explosão da entropia vem de gradientes negativos longos sem KL e T alta | T 0,6 + filtro de truncadas (`grpo2`) evita a explosão; KL 0,04 sem filtro (`grpo4`) não segura (0,42 → 0,75); Dr. GRPO + 70% MCQ (`grpo3`) também sobe (→ 0,89) | **Aberta**: o KL de 0,04 é fraco (KL medido 0,04 por token no ciclo 3) e o mecanismo não foi isolado |
| H13 | O filtro de truncadas tira a pressão de parar | `grpo2`: trunca 16,1% e fecha 84,2%, contra 9,6% e 90,9% do `grpo1` com o mesmo número de atualizações | **Apoiada** |
| H14 | Treinar só com respostas certas (RAFT) é estável, mas estaciona mais baixo | `raft1`: entropia cai (0,40 → 0,18), múltipla escolha 61,6% ▼, trunca 15,1% | **Apoiada** (4 ciclos) |
| H15 | O RL treinado só com thinking prejudica o modo sem thinking | `grpo3` e `grpo4`: múltipla escolha sem thinking 43,1% (−20,8); `gspo1` com 8 ciclos: 57,7% ▼ (−6,2), contra 59,5% com 4 | **Apoiada** |
| H16 | A razão por sequência com corte minúsculo (GSPO) estabiliza o treino | entropia 0,41 → 0,41 em 8 ciclos; porém 40 a 50% dos tokens ficam cortados, ou seja, o modelo se mexe pouco por ciclo | **Apoiada**, mecanismo ambíguo |
| H17 | Mais ciclos melhoram a nota | `gspo1` com 4 → 8 ciclos: múltipla escolha 67,8 → 67,5, sem thinking 59,5 → 57,7; forma melhora (fecha 90,9 → 93,2%) | **Refutada** para a nota; vale para a forma |

## Custos

Geração: ~10 a 11 min por ciclo (256 × 8 respostas, ~2,2 milhões de tokens a ~3,3 mil tokens/s). Treino: 4 a 8 min por ciclo (com gradient checkpointing; pico de 12,5 GiB). Avaliação completa: 11 a 13 min. Rodadas de 4 ciclos: 1,1 a 1,5 h; de 8 ciclos: 2,4 a 2,5 h.

Incidente de processo: o treino do primeiro ciclo estourou a memória (22 GiB, uma só resposta de ~2.350 tokens, sem checkpointing); corrigido com `--ckpt full` em `grpo_train.py` (pico de 12,3 GiB). Desde então: não editar um `.sh` em execução (o `grpo1` saiu com código 1 por isso, com os resultados intactos).

## Ameaças à validade

- Uma execução por rodada; **sem repetição por semente**. Diferenças de 1 a 2 pontos na múltipla escolha (385 itens) são ruído.
- Os prompts de múltipla escolha vêm de **ciência** (pós-graduação, sobretudo química) e o gabarito é fraco; a validação mistura outros domínios, o que mede transferência.
- O limite de 2.048 tokens na geração é mais curto que o de 4.096 da avaliação: respostas corretas e longas valem 0 no treino.
- `grpo2`, `grpo3`, `grpo4` e `gspo1` mudam **mais de uma coisa** em relação ao `grpo1` (temperatura, top-p, filtro, KL, vantagem, proporção de questões). Por isso a causa de cada diferença não está isolada.
- A heurística de idioma e a recompensa de "traço em português" podem ser enganadas por traços mistos.

## O que o RL ainda não responde

Nenhuma rodada passa o critério do PRD (múltipla escolha com IC acima de 0 contra o Lumi, queda sem thinking de no máximo 1 ponto, 98% de fechamento). O RL melhora a forma e quase fecha a distância com thinking, mas não protege o modo sem thinking. Caminhos em [07_hipoteses_e_proximos_passos.md](07_hipoteses_e_proximos_passos.md): proteger o modo sem thinking no RL (perda de SFT sem thinking misturada), partir de um SFT melhor (traços do Nemotron 30B, adaptador só com MLP), e rodar os finalistas na suíte de teste.
