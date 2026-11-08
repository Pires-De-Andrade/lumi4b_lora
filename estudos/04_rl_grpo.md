# 04. Estudos de RL: GRPO e variantes

Pergunta: partindo do adaptador do SFT (`adalberto`), o RL com recompensa verificável **recupera a múltipla escolha** sem depender de traços de professor, e o que o mantém estável? Protocolo e convenções: [01_protocolo.md](01_protocolo.md). Todas as rodadas partem do adaptador `adalberto` (LoRA r=16, 71 módulos), exceto onde dito.

## Desenho do RL

Cada **ciclo** roda em processos separados:
 (1) funde o adaptador no Lumi; 
(2) gera **8 respostas por questão** para 256 questões (metade matemática, metade múltipla escolha) com o vLLM, até 2.048 tokens, modo thinking, sem começo forçado; 
(3) calcula a recompensa e a vantagem; 
(4) treina o adaptador por uma passada pelas respostas com vantagem diferente de zero (64 respostas por passo do otimizador, lr 2e-5, AdamW com o estado guardado entre ciclos). Um ciclo leva uns **13 a 20 min**.

| Componente | Definição |
| --- | --- |
| **Recompensa** | 1 só se a resposta fecha um único `</think>`, termina sozinha (não trunca), o traço está em português e a resposta final bate com o gabarito no formato estrito (`#### N` ou `Resposta: <letra>`); senão 0. Truncada vale 0 |
| **Vantagem** | recompensa menos a média do grupo de 8, dividida pelo desvio-padrão (`--adv std`); grupo com recompensas iguais vale 0 e não treina |
| **Perda (padrão)** | PPO por token com corte assimétrico (0,2 e 0,28), sem KL, normalizada pelo total de tokens de resposta do minibatch (na variante GSPO, `gspo_loss_sum`, o termo de cada sequência é multiplicado pelo número de tokens e dividido por esse total, então **sequências longas pesam mais** que na média uniforme por sequência do paper); as log-probabilidades "antigas" vêm de uma passada sem gradiente antes de qualquer atualização |
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
| H16 | A razão por sequência com corte minúsculo (GSPO) estabiliza o treino | entropia 0,41 → 0,41 em 8 ciclos; o `clip_fraction` registrado (40 a 50%) é **por token**, contra os limites da sequência (3e-4 e 4e-4), e **não mede o corte da perda** (que é da razão da sequência); o corte real é desconhecido, então não se conclui que "o modelo se mexe pouco" (revisão de 08/11, abaixo) | **Apoiada**, mecanismo ambíguo |
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

## RL a partir do SFT da Thor (`gspo_thor`, **feito**: empata com o `gspo1`, não passa o Lumi)

**Decisão (08/11, 13:05):** o foco é **melhorar o modo com thinking**. O melhor candidato até agora (`gspo1`, 8 ciclos a partir do `adalberto`) empata com o Lumi na múltipla escolha (67,5%) e passa na matemática (84,3% ▲). O `thor_pt` ([02](02_sft_dados_e_professor.md), E12) tem o raciocínio tão bom quanto o do Lumi **quando termina**, mas entra em laço de repetição em 33% das respostas de múltipla escolha (fecha `</think>` em 66,0%), e o RL com recompensa zero para truncada é a ferramenta que já mostrou reduzir isso (`gspo1`: fechamento 86,8% → 93,2%).

**Hipótese H33:** o GSPO a partir do `thor_pt` elimina os laços sem perder a qualidade do raciocínio e leva a múltipla escolha com thinking acima do Lumi.

**Desenho:** a receita do `gspo1` (a única estável em 8 ciclos), só trocando o ponto de partida: `scripts/run_grpo.sh gspo_thor --init runs/train/thor_pt/adapter --iters 8 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 --max-new 2048 --temperature 0.6 --top-p 0.95 --ratio seq --max-noclose 0.85`. Truncadas valem 0 e **não** são filtradas (a pressão para parar é o ponto). O guarda de "sem `</think>`" sobe de 5% para 85%, porque o `thor_pt` já começa com ~34% (a 4.096 tokens) e a 5% a rodada pararia no primeiro ciclo; 85% só pega degeneração catastrófica. Conferido: as 3.715 questões de múltipla escolha do RL não repetem as 2.174 de ciência do SFT (nenhum id em comum); as de matemática do RL (GSM8K `train`) repetem em parte as 1.719 do `gsm8k_train` do SFT. Avaliação do modelo que entra no ciclo 4 (só thinking, ~8 min) e a completa no fim (12 min). ~2,5 h no total.

**Previsões** (escritas antes dos resultados; vale a regra de leitura do E11, sinal = ±4 pontos):

| Resultado do `gspo_thor` (8 ciclos) | Leitura |
| --- | --- |
| Fecha ≥ 90% (múlt. escolha) e múlt. escolha com thinking ≥ 69,6% (Lumi) | H33 apoiada: os laços eram o único problema do `thor_pt`; passa a ser o melhor candidato |
| Fecha ≥ 90% mas múlt. escolha < 66% | Os laços saem, mas o RL não recupera a nota: o SFT Nemotron não é melhor que o do Qwen para esta métrica |
| Fecha < 80% | Os laços resistem ao RL a partir deste SFT: o problema é do SFT (dados ou estilo) |
| Guarda de `</think>` dispara | O RL degenera a partir de um SFT que já entra em laço; trocar o ponto de partida |

**Resultado parcial (08/11, 14:07, modelo que entra no ciclo 4, só thinking, validação limpa):** múltipla escolha **65,7%** (−3,9 contra o Lumi, sem ▼), fecha 82,9%, trunca 17,1% (o `thor_pt` tinha 54,3% e trunca 34%), mediana de 520 tokens; GSM8K 83,7% (+2,1). Durante o treino, a truncação nas respostas amostradas (T 0,6, 2.048 tokens) caiu 25% → 16% → 8% → 11% (ciclos 0 a 3) e a recompensa de múltipla escolha subiu 0,62 → 0,67 → 0,70 → 0,72; a entropia subiu de 0,64 para 0,74 e desacelera. Contra o `gspo1` no mesmo ponto (`gspo1_c4`: 67,8%, trunca 10,6%) está 2 pontos abaixo (dentro do ruído) e com mais truncação. Falta metade da rodada: a leitura final é a do ciclo 8.

**Resultado final (08/11, 14:50; 8 ciclos, 1,4 h; validação limpa, nota estrita, IC 95% pareado):**

| | GSM8K com thinking | Múlt. escolha com thinking | Fecha / trunca (múlt. escolha) | Mediana de tokens (múlt. escolha) | GSM8K sem | Múlt. escolha sem |
| --- | --- | --- | --- | --- | --- | --- |
| Lumi | 81,7 | 69,6 | 86,2% / 13,8% | 490 | 77,2 | 63,9 |
| `thor_pt` (ponto de partida) | 77,8 ▼ | 54,3 ▼ | 66% / 34% | – | 74,6 | 62,9 |
| `gspo1` (a partir do `adalberto`) | 84,3 ▲ | 67,5 | 93,2% / 8,1% | 1.035 | 78,6 | 57,7 ▼ |
| **`gspo_thor`** | **82,5** (+0,9; IC −1,6 a +3,6) | **67,0** (−2,6; IC −7,0 a +1,6) | **87,8% / 12,2%** | **534** | 78,7 (+1,5) | 60,0 (−3,9; IC −8,3 a +0,5) |

- **Contra o ponto de partida:** o RL tirou os laços e recuperou a nota. Múltipla escolha +12,7 (IC +8,6 a +16,9), GSM8K +4,7 (IC +2,5 a +7,1), truncação 34% → 12,2%. Isso apoia a parte fraca da H33 ("o RL desfaz os laços do `thor_pt`").
- **Contra o Lumi:** não passa. A múltipla escolha com thinking fica 2,6 pontos abaixo (IC inclui 0), fecha `</think>` em 87,8% (o critério pede 98%) e o sem thinking perde 3,9 na múltipla escolha (o critério pede perda de no máximo 1). A parte forte da H33 ("passa o Lumi") **não foi apoiada**.
- **Contra o `gspo1`:** empate. Diferença de −0,5 na múltipla escolha (IC −4,7 a +3,6) e −1,8 no GSM8K (IC −4,1 a +0,6). Os traços do Nemotron não são melhores que os do Qwen como ponto de partida do RL para esta métrica.
- **O que o `gspo_thor` tem de diferente:** o raciocínio fica curto (mediana de 534 tokens na múltipla escolha, contra 1.035 do `gspo1` e 490 do Lumi). Mesma nota com metade dos tokens. Cuidado: é a mediana de todas as respostas, não controlada pelas que terminam; não conclua mais que "mais barato por resposta".
- **Leitura da tabela de previsões:** o resultado (fecha 87,8%, múltipla escolha 67,0%) cai entre as linhas 1 e 2, mais perto da 2 no espírito: os laços saem em boa parte, mas o RL não leva acima do Lumi, e o SFT do Nemotron não é melhor que o do Qwen para esta métrica. O guarda de `</think>` nunca disparou (0% em todos os ciclos).
- **Durante o treino:** a truncação das respostas amostradas (T 0,6, 2.048 tokens) caiu 25% → 4% e a recompensa de múltipla escolha subiu 0,62 → 0,76 nos 8 ciclos. A avaliação gulosa foi de 65,7% (ciclo 4) para 67,0% (ciclo 8): os ciclos 5 a 8 renderam pouco na validação, apesar da recompensa de treino subir.
- **Tempos:** geração 47 min (9,1 M tokens), treino 29 min, avaliação final 7 min; avaliação do ciclo 4 de 6 min.

**O que ainda faltará mesmo se passar:** o modo sem thinking (adiado por decisão de 08/11), a suíte de teste dos finalistas, e uma segunda semente.


## Revisão de 08/11 (leitura externa do código): três pontos conferidos

Conferidos no código (`scripts/grpo_train.py`, `src/lumi_ft/train/grpo.py`, `src/lumi_ft/eval/shape.py`) e nos dados dos ciclos.

1. **`clip_fraction` não mede o corte do GSPO (procede).** A perda corta a razão da **sequência**, `s = exp(média_t(logp_novo - logp_antigo))` (`gspo_loss_sum`), mas o registro (`grpo_train.py`, bloco `no_grad` do laço) calcula `exp(logp_novo - logp_antigo)` **por token** e conta os tokens fora de 1-3e-4 e 1+4e-4. Com razões por token ruidosas (bf16), esse número não diz nada sobre a perda. A perda e o gradiente estão certos; o diagnóstico "40 a 50% dos tokens cortados, logo o modelo se mexe pouco" (H16, `gspo1`) **não é sustentado**. O corte por sequência dos `gspo1`, `gspo_thor` é desconhecido (não há como recalcular sem as log-probabilidades antigas). Correção prevista: registrar a fração de **sequências** cortadas e a média de |log s|.
2. **A perda GSPO implementada é ponderada pelo comprimento (procede, e já estava no docstring, mas não nos estudos).** O gradiente de cada token é `-A·s/N` (N = tokens do minibatch): é a agregação por token, no estilo DAPO/Dr. GRPO, e não a média uniforme por sequência do paper. Sequências longas pesam mais; como as longas degeneradas têm recompensa 0 (vantagem negativa), isso empurra com mais força contra os laços, o que combina com a queda da truncação, mas também pode favorecer respostas curtas (o `gspo_thor` tem mediana de 534 tokens contra 1.035 do `gspo1`; a diferença vem em boa parte do SFT de partida, não separável). Todas as rodadas GSPO usam a mesma variante, então as comparações entre elas não mudam. Ablação válida: média uniforme por sequência, o resto fixo (~2 h de GPU); prioridade baixa, depois do RL com gabarito oficial.
3. **A recompensa mistura objetivos e esconde qual falhou (procede em parte).** A recompensa é 1 só se terminou, fechou uma vez, tem traço e resposta, acertou no formato estrito e o traço é português. Um acerto em inglês e um erro em português valem 0. Já registramos `whys` por tipo, mas é **um motivo por amostra em ordem de prioridade** (truncou > sem um `</think>` > vazio > errado > não português): "não português" só aparece em respostas certas, e as respostas erradas em inglês entram como "errada". Os rollouts guardam só `reward`, `advantage` e `finished` por amostra. Mantenho a recompensa (comparabilidade com `gspo1`) e registro separadamente acerto, idioma, fechamento e término, também nos grupos sem gradiente.

**Grupos sem gradiente, medidos nos rollouts (`finished` é o único motivo guardado por amostra):**

| Rodada, ciclo | Tipo | Todos certos | Todos zero | Mistos (com gradiente) | Truncadas nos grupos todos-zero |
| --- | --- | --- | --- | --- | --- |
| `gspo_thor` 0 | múlt. escolha | 34% | 16% | 49% | 62% |
| `gspo_thor` 7 | múlt. escolha | 53% | 5% | 41% | 12% |
| `gspo1` 0 | múlt. escolha | 26% | 12% | 62% | 44% |
| `gspo1` 7 | múlt. escolha | 45% | 9% | 46% | 34% |
| `gspo_thor` 0 e 7 | matemática | 72% e 78% | 5% e 4% | 23% e 18% | 15% e 18% |
| `gspo1` 0 e 7 | matemática | 75% e 78% | 5% e 3% | 20% e 19% | 7% e 38% |

- **A maior parte da falta de gradiente é "fácil demais", não "difícil demais":** em múltipla escolha, 26 a 53% dos grupos são todos certos (a fração sobe ao longo do RL) e só 5 a 16% são todos zero. Em matemática, 72 a 78% dos grupos são todos certos: só ~20% dos prompts de matemática ensinam algo, e eles ocupam metade de cada ciclo (`--frac-mcq 0.5`). Isso sugere testar mais múltipla escolha por ciclo (a `grpo3` usou 70%, mas junto com outra mudança).
- **No começo do `gspo_thor`, os grupos todos-zero de múltipla escolha são em 62% truncadas (laços):** onde as 8 respostas entram em laço, não há exemplo positivo e a vantagem é zero; o GSPO só desfaz laços nos grupos mistos. Um crédito parcial por terminar seria uma opção futura (muda a recompensa, então só como experimento à parte).
- **Idioma:** a heurística de palavras funcionais (`shape.trace_language`, a mesma na recompensa e na avaliação; o próprio docstring diz que não é um detector de idioma) classifica 97,9% (`gspo_thor`) e 96,1% (`gspo1`) dos traços de múltipla escolha como português. Olhei os traços limítrofes (razão pt/en entre 2 e 4: 2,1% e 1,4% dos 'pt'): são português com trechos do enunciado em inglês; a fração média de palavras funcionais inglesas nos traços 'pt' é de 1,6%. Não vejo inflação, mas a leitura foi minha: a conformidade de 90% do PRD precisa de uma amostra revisada por pessoa (ex.: 50 traços) para o candidato final. Em matemática a classe 'unk' quase não aparece (1 resposta de 1.024 foi recusada por idioma).

**O que foi feito depois da revisão (08/11, 16:50):**

- **Registro do treino (só registro: a perda, a recompensa e os dados não mudam).** `grpo_train.py` passa a gravar, no modo GSPO, a fração de **sequências** com corte ativo (A > 0 e s > 1+eps_alto, ou A < 0 e s < 1-eps_baixo; é exatamente "gradiente da sequência zerado", conferido contra o gradiente de `gspo_loss_sum` em teste), a fração fora do intervalo, a fração de tokens dessas sequências, a média de |log s| (por passo e no resumo) e `first_step_abs_log_s` (no primeiro passo a razão deveria ser 1: mede o ruído numérico entre as duas passadas). No modo por token, a fração de tokens com corte ativo. O `clip_fraction` antigo continua, marcado como legado (`*` em `grpo_status.py`).
- **Rollouts.** Cada amostra guarda `parts` (terminou, fechou um único `</think>`, não vazio, acertou, português: independentes, a recompensa é 1 se e só se os cinco valem; para respostas sem `</think>` o idioma é o do texto até ali) e `source`; `rollouts.stats.json` ganha `summary` com a taxa de cada requisito, grupos (todos certos, todos zero e a causa, mistos) e recompensa e truncamento por fonte e por faixa de comprimento. `grpo_status.py` mostra o corte e os grupos mistos.
- **Teste rápido antes do RL.** A cadeia `gold-chain` roda um ciclo pequeno (`smoke_diag`, 16 prompts x 4 respostas, 512 tokens, ~3 min) e **para** se o registro novo falhar (o laço de treino só roda na GPU, ocupada pela suíte quando o código foi escrito).
- **Filtro de qualidade das questões** (`src/lumi_ft/data/question_quality.py`, no conversor): a auditoria achou traduções que **perderam a negação** do inglês (ex.: "Which of these is **not** a feature…" virou "Qual dessas é uma característica…", o que inverte o gabarito), 219 enunciados deixados em inglês pela tradução (1,9% das aceitas; passaram nas conferências da Thor) e alternativas repetidas. Saem do RL: 59 por negação perdida, 210 por enunciado idêntico ao inglês, 40 por enunciado em inglês, 34 por alternativas repetidas (343 no total). O pool passa de 11.324 para **10.981** questões de múltipla escolha; os 1.024 usados em 8 ciclos continuam 618 WebInstruct e 406 MedMCQA em proporção parecida. Os marcadores são heurísticas de palavras (há falsos positivos, por isso só os de risco de gabarito invertido ou de enunciado em inglês excluem).
- **Auditoria estrutural** (`scripts/audit_gold_questions.py`, saída em `data/audit_gold/`): lista dos IDs usados, distribuição do usado contra o pool (letras, fontes, nº de alternativas) e amostra estratificada de 60 com inglês e português lado a lado (`sample.md`). Nos 1.024 usados, depois do filtro, restam 59 com algum marcador leve (46 "alternativa remete a outras", 8 sem correspondente aparente no português, 4 enunciados muito curtos, 1 enunciado repetido). **Não foi feita revisão do conteúdo médico**, que ninguém do projeto vai julgar; o rótulo do MedMCQA é de provas e o do WebInstruct-verified é resposta da web verificada por modelo. Li 5 pares de WebInstruct da amostra (fiéis; atenção ao ler: o enunciado tem várias linhas).
