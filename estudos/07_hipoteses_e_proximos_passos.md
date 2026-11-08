# 07. Hipóteses, ameaças à validade e próximos experimentos

Registro único das hipóteses do projeto, com o veredito atual e onde está a evidência. Convenção: **apoiada**, **refutada**, **parcial** (evidência a favor e contra) e **aberta** (não testada ou inconclusiva).

## 1. Registro de hipóteses

| ID | Hipótese | Veredito | Evidência | Estudo |
| --- | --- | --- | --- | --- |
| H0 | O Lumi raciocina em português sem treino, só com instrução ou prefill | **Refutada** | A instrução não muda o idioma; o prefill muda o idioma e quebra a estrutura (GSM8K 48,1%) | [02](02_sft_dados_e_professor.md) E0 |
| H1 | 10 mil traços do GigaVerbo bastam para o comportamento desejado | **Refutada** | Forma ok, nota −14 (GSM8K) e −26 (múlt. escolha) | 02, E1 |
| H2 | Traços de um professor forte (Qwen3-30B) ensinam português e mantêm ou melhoram a nota | **Parcial** | Matemática +1,6 (IC inclui 0); múltipla escolha −7,3 ▼ | 02, E2 |
| H3 | A autodestilação (o Lumi como professor) funciona | **Refutada** | GSM8K −3,6 ▼, múlt. escolha −11,7 ▼ | 02, E3 |
| H4 | O limite de 1.024 tokens causa a perda na múltipla escolha | **Refutada** | Com 2.048: 64,2% (+1,8 sem significância), sem thinking pior | 02, E4 |
| H5 | Taxa de aprendizado menor preserva a múltipla escolha | **Refutada** | 63,4% contra 62,3% | 02, E5 |
| H6 | O DoRA preserva melhor que o LoRA | **Refutada** | Diferença de 0,3 a 0,7 ponto; custa 2,5× o tempo | 02, E6 |
| H7 | Reduzir o efeito do adaptador (x0,5, x0,75) recupera a múltipla escolha | **Refutada** com thinking | 63,4% e 64,4% | 02, E7 |
| H8 | O Lumi traduz bem os traços de professor (inglês → português) | **Refutada em escala** | Piloto 86% aceito; em lote maior, 35% dos raciocínios (o tradutor respondia a questão) | 02, E8 |
| H9a | A perda na múltipla escolha vem de algumas partes do adaptador | **Refutada** por treino (a triagem a apoiava) | Triagem: só MLP 69,6%. Treinado do zero, só MLP: 62,3% ▼, igual ao completo | [03](03_ablacao_camadas_e_mlp.md) E9 e E10 |
| H9b | O idioma do raciocínio é decidido pelas camadas finais | **Aberta** | Triagem: só 21 a 41 dão 98,9% em português, mas o adaptador parcial quebra o modelo; treino só com MLP (todas as camadas) dá 98,5%; a posição não foi testada com treino | 03, E9 e E10 |
| H9c | Os MLPs concentram ganhos de capacidade; o Mamba e a atenção, o dano | **Refutada** | `adalberto_mlp`: português 98,5%, GSM8K +1,8 e múlt. escolha 62,3% ▼ (−7,3), sem thinking 54,3% ▼ | 03, E10 |
| H10 | O português do raciocínio, por si, causa a perda na múltipla escolha | **Aberta** | O argumento "contra" (camadas 0 a 20 com traço em inglês) vem da triagem, que enganou no E10; o par em inglês e português (H21) decide | 03 |
| H11 | O RL a partir do SFT recupera a múltipla escolha com thinking | **Parcial** | 62,3% → 66 a 68%, empate com o Lumi (IC inclui 0) | [04](04_rl_grpo.md) |
| H12 | A explosão de entropia vem de gradientes negativos longos sem KL e T alta | **Aberta** | T 0,6 + filtro evita; KL 0,04 não segura; mecanismo não isolado | 04 |
| H13 | O filtro de truncadas tira a pressão de parar | **Apoiada** | `grpo2`: trunca 16,1%, fecha 84,2% | 04 |
| H14 | Treinar só com respostas certas (RAFT) é estável e estaciona mais baixo | **Apoiada** (4 ciclos) | `raft1`: entropia 0,40 → 0,18, múlt. escolha 61,6% ▼ | 04 |
| H15 | O RL só com thinking prejudica o modo sem thinking | **Apoiada** | 43,1% em duas rodadas; `gspo1` 8 ciclos 57,7% ▼ | 04 |
| H16 | A razão por sequência com corte minúsculo (GSPO) estabiliza | **Apoiada**, mecanismo ambíguo | Entropia estável em 8 ciclos; o `clip_fraction` de 40 a 50% era por token e **não mede o corte da perda** (corte real desconhecido; [04](04_rl_grpo.md), revisão de 08/11) | 04 |
| H17 | Mais ciclos de RL melhoram a nota | **Refutada** (nota), apoiada (forma) | `gspo1` 4 → 8: 67,8 → 67,5 | 04 |
| H18 | A pergunta em português reduz o acerto do Lumi original | **Refutada** | PT − EN: −1,3 a +0,4 (flexível), sem significância | [05](05_diagnosticos_e_referencias.md) D2 |
| H19 | O estilo do professor (longo e falado contra curto em forma de plano) custa na múltipla escolha | **Parcial**, em teste | Qwen: erra mais (72,3% quando termina). Nemotron 30B (`thor_pt`): quando termina iguala o Lumi (84,0% contra 84,5% nos mesmos itens), mas entra em laço em 33% (Lumi 13%). `thor_pt_math` decide o papel do formato | [02](02_sft_dados_e_professor.md) E12, 05 D5 |

## 2. Hipóteses em teste (E11, `cap-queue`) e propostas

| ID | Hipótese | Teste | Estado |
| --- | --- | --- | --- |
| H26 | O dano na múltipla escolha cresce com a capacidade do adaptador | r=4 e r=64 ([03](03_ablacao_camadas_e_mlp.md) E11) | **Refutada** no lado de baixa capacidade (r=4: 62,3%, igual ao r=16); o r=64 não foi testado |
| H27 | A RSLoRA aprende mais que a escala padrão em rank alto | `cap_r64_rs` contra `cap_r64_std` | **Não testada** (cancelada, 08/11) |
| H28 | O dano não depende do tipo de bloco | `type_nomlp`, `type_attn`, `type_mamba` | **Parcial**: completo e só MLP iguais (E10); `type_nomlp`, `type_attn`, `type_mamba` cancelados |
| H29 | O adaptador só na atenção (16 módulos) já ensina português | `type_attn` | **Não testada** (cancelada) |
| H30 | O idioma é decidido nas camadas finais (com treino) | `pos_L00_20`, `pos_L21_41` | **Não testada** (cancelada) |
| H32 | O preço na múltipla escolha depende da escala alpha/r, não do rank | `sc_a08`, `sc_a16`, `sc_a64` (r=16, só o alpha muda) | **Não testada** (cancelada) |

Hipóteses propostas, ainda sem teste:

| ID | Hipótese | Teste | Custo |
| --- | --- | --- | --- |
| H20 | Traços de um professor da mesma família (Nemotron 30B), traduzidos na Thor, evitam a perda na múltipla escolha | **Refutada** na nota (54,3% ▼ com thinking); **apoiada** no raciocínio (sem erro a mais quando termina); o dano virou laço de repetição | `thor_pt` | 02, E12 |
| H21 | O raciocínio em inglês **não** perde na múltipla escolha, o português perde | **Apoiada**: `thor_en` − `thor_pt` = +6,0 (múlt. escolha, IC +1,6 a +10,4) e +5,3 (GSM8K) com thinking. Mas o inglês também perde para o Lumi (60,3%, ▼) | [02](02_sft_dados_e_professor.md) E12 |
| H33 | O RL (GSPO) a partir do `thor_pt` elimina os laços e leva a múltipla escolha acima do Lumi | `gspo_thor`: 8 ciclos com `--init runs/train/thor_pt/adapter`; recompensa zero para truncada | **Parcial** (08/11, 14:50): os laços saem (truncação 34% → 12,2%; múlt. escolha +12,7 sobre o `thor_pt`, IC +8,6 a +16,9), mas **não passa o Lumi** (67,0% contra 69,6%, IC −7,0 a +1,6; fecha 87,8%) e empata com o `gspo1` (−0,5) | [04](04_rl_grpo.md) |
| H38 | A cauda longa dos traços do Nemotron (P90 de 1.027 tokens contra 493 do Qwen; 10,8% acima de 1.000) é o que ensina o modelo a não fechar o raciocínio (laços); sem ela a múltipla escolha melhora e a truncação cai | `thor_pt_short`: o `thor_pt` só com raciocínio de até 600 tokens (3.738 de 4.927 problemas; ciência 1.255 de 2.174), os mesmos 5.544 sem thinking. Se melhorar, o controle de "menos dados" é desnecessário; se não, o SFT em si é o problema (mesmo o `selfdist` deu 57,9%) | **Sem sinal** (08/11, 15:14): a direção é a prevista, mas pequena. Múlt. escolha com thinking 57,4% (+3,1 sobre o `thor_pt`, IC −1,0 a +7,5; −12,2 ▼ contra o Lumi), truncação 34% → 26,2%, fecha 73,8%; sem thinking 63,1% (+0,3). Abaixo da regra de sinal (+4 com thinking e mesma direção sem). A cauda longa explica no máximo parte dos laços. Confusão não separada: o corte também tirou 42% dos exemplos de ciência |
| H34 a H37 | A perda estrita sem thinking é de formato (H34); o estilo passo a passo do Qwen sobe o GSM8K sem thinking (H35); treinar só o sem thinking deixa o thinking como está (H36); só formato não passa o Lumi, estaciona em ~64% (H37) | `nt_both`, `nt_thor`, `nt_qwen` ([02](02_sft_dados_e_professor.md) E13) | **Em fila** |
| H31 | O estilo do professor, e não o formato da prova, é o que recupera a múltipla escolha | **Descartada** (08/11): o `thor_pt` não teve ganho a explicar (54,3%), e o controle `thor_pt_math` só tem matemática | 02, E12 |
| H22 | Misturar uma perda de SFT no modo sem thinking ao RL protege esse modo | Nova opção de `grpo_train`, 4 ciclos | Implementar ~1 h; rodada 1,3 h |
| H23 | Um KL mais forte (coeficiente 0,2 ou mais) segura a entropia sem o filtro de truncadas | `grpo4` com coeficiente maior | 1,4 h |
| H24 | O RL a partir do SFT só com MLP ou dos dados da Thor passa o critério | Rodar o GSPO a partir do melhor SFT | 1,3 a 2,4 h |
| H25 | O ranking entre as rodadas se repete com outras sementes | Duas sementes do `gspo1` e do `grpo1` | 2 × 1,3 h cada |
| H39 | Subir o limite de geração do RL de 2.048 para 4.096 melhora a múltipla escolha (a truncação some e respostas certas e longas deixam de valer 0) | `gspo_gold4k` contra `gspo_gold` (mesmas questões) | **Aberta, a evidência até agora é contra**: ciclo 4 com 4.096 deu 63,1% e fecha 83,9% (mais laços), contra 67,8% e 90,9% do `gspo1` com 2.048; o `gspo_gold` com 2.048 no ciclo 4 separa limite e questões (~21:00; leitura em [08](08_registro_de_decisoes.md)) |
| H40 | As questões com gabarito oficial (MedMCQA e WebInstruct-verified) ajudam o RL mais que as de letra majoritária do R1 | `gspo_gold` contra `gspo1` (mesmo limite de 2.048) | **Aberta** (mesmo teste da H39) |
| H41 | O SFT só de matemática (`adalberto`) prejudica a **qualidade** do raciocínio em múltipla escolha, além do fechamento | Acerto condicionado aos itens em que todos os modelos fecham o `</think>` (189 itens) | **Parcial, indício**: `adalberto` 82,5% contra 87,8% do Lumi; `thor_pt` 87,3%; condicionar no fechamento escolhe itens mais fáceis ([discovery.md](../discovery.md) §20.39) |
| H42 | Traços de múltipla escolha em português de um professor que fecha limpo (Qwen, pacote 4), misturados aos dados do `thor_pt`, fecham o `</think>` e preservam a qualidade | SFT misto, depois triagem de RL de 4 ciclos | **Proposta**; depende da entrega do pacote 4 (rendimento do Qwen em cálculos longos da WebInstruct é desconhecido) |

## 3. Próximos experimentos

**Foco (decisão de 08/11, 13:05): melhorar o modo COM thinking.** O adaptador só liga com thinking; sem thinking roda o Lumi original (decisão de ~18:25, [08](08_registro_de_decisoes.md)). Por isso a queda sem thinking e a H22 saem do caminho crítico; um SFT que penalize a queda sem thinking fica como ideia futura. Regra de projeto: a suíte de teste é só checagem, nunca serve para escolher candidato.

### Feito
- **`gspo_thor`** ([04](04_rl_grpo.md)): GSPO de 8 ciclos a partir do `thor_pt`; H33 parcial. 67,0% contra 67,5% do `gspo1` e 69,6% do Lumi, com raciocínio mais curto (mediana de 534 tokens contra 1.035).
- **`thor_pt_short`** ([02](02_sft_dados_e_professor.md) E12): H38 sem sinal (57,4% contra 54,3%, dentro do ruído; trunca 26,2%). Não rodar RL a partir dele, não pedir traços mais curtos.
- **`adalberto_mlp`**: português 98,5%, múlt. escolha 62,3% ▼; a perda não é do Mamba nem da atenção. Não vira base do RL.
- **Estudo de capacidade e posição (E11)**: encerrado com 1 de 15 rodadas; as outras 14 foram canceladas por não atacarem o problema ([03](03_ablacao_camadas_e_mlp.md)).
- **RL com 4.096 e gabarito oficial (`gspo_gold4k`)**: parada pela regra no ciclo 4 (63,1%, fecha 83,9%); ver H39.
- **Benchmark de velocidade da geração**: nada adotado (ganho +9,3% contra critério de ≥15%).

### Em andamento
- **`gspo_gold` com 2.048 até o ciclo 4** (sessão `gold2048`, avaliação ~21:00). Separa o efeito do limite do efeito das questões (H39, H40). A leitura está **fixada antes do número** em [08](08_registro_de_decisoes.md): `X ≥ 67%` e fecha ≥ 90% → o 4.096 é o culpado; `X ≤ 64%` ou fecha < 87% → as questões são o problema; entre os dois → ambíguo (mais ciclos e outra semente).

### Próximos, na ordem
1. **Conforme a regra acima:** retomar o `gspo_gold` até o ciclo 8 e rodar a triagem do `thor_pt` a 2.048 com as questões do gold; ou `gspo1` com 4.096 e as questões antigas (1,5 h), subindo a prioridade do SFT misto.
2. **SFT misto com o Qwen** (H42, H41): depende do **pacote 4** na Thor ([handoff/README.md](../handoff/README.md)). Mistura prevista: dados do `thor_pt` + resto do pacote 2 + pacote 3 com até ~2.048 tokens + traços do Qwen; o limite de tamanho por exemplo é escolhido na 4090, olhando a distribuição devolvida. Depois: avaliar o ponto de partida (qualidade e fechamento, ~40 min de SFT + ~13 de avaliação) e uma triagem de RL de 4 ciclos.
3. **Suíte de teste (com thinking) do melhor candidato**, só como checagem, e as provas que faltam do Lumi para comparação (MMLU-ProX sem thinking, ENEM, BLUEX e OAB, nos dois modos; retomável com `scripts/run_suite_models.sh "base|-"`). O `gspo1` e o `gspo_thor` ficaram de fora. Estimativa do Lumi inteiro: ~3 a 3,5 h (a de 1,3 h por modelo estava errada); modelos com RL pensam menos e levam menos.
4. **Segunda semente** do melhor (H25): sem ela, diferenças de 1 a 2 pontos entre rodadas não são interpretáveis.

### Backlog (sem data)
- Ajustes do GSPO: 128 sequências por minibatch no treino, logits divididos pela temperatura da amostragem (hoje não são), média uniforme por sequência em vez da ponderada pelo comprimento.
- Outro ponto de partida de SFT (`adalberto_L2048`: o limite de 1.024 tokens descartou ~34% dos candidatos; agora que o modo sem thinking não pesa, vale reavaliar).
- Diagnósticos: logprob do vLLM contra o do HF; conferir que o adaptador desligado dá exatamente o Lumi.
- Revisão humana de ~50 traços em português; checagem de fidelidade da tradução das questões (Thor).
- KL mais forte (H23); proteger o sem thinking no RL (H22): baixa prioridade com o adaptador só no thinking.

**Detalhe das questões do gold (para quem for reproduzir):** `data/grpo_prompts_gold.jsonl` tem 10.981 de múltipla escolha e 5.612 de matemática; em 8 ciclos o RL usa só 1.024 das questões (618 WebInstruct e 406 MedMCQA). O rótulo é do dataset: MedMCQA vem de provas; WebInstruct-verified é resposta extraída da web e verificada por modelo (revisão `3e8a350b`, arquivo `train` corrigido, não o `train_legacy`). Logo a comparação com o `gspo1` troca questões, domínios, fontes e rótulos juntos e **não isola o efeito do rótulo**. Ponto de partida do RL: o `adalberto` (62,3%, fecha 86,8%), não o `thor_pt_short` (57,4%, fecha 73,8%).

## 4. Ameaças à validade (do conjunto)

| Ameaça | Efeito | Mitigação |
| --- | --- | --- |
| **Uma execução por configuração** | Não sabemos a variância entre sementes; diferenças pequenas podem ser sorte | H25; interpretar só diferenças grandes |
| **385 itens de múltipla escolha** | IC de ±4 a 5 pontos; a "maldição do vencedor" ao escolher o melhor entre muitos | Triagem pela estabilidade; só os finalistas vão à suíte de teste |
| **Nenhum candidato na suíte de teste** | Todos os resultados são de validação | Item 2 da lista acima |
| **Gabarito fraco no RL** (letra majoritária do R1) | O modelo pode ser premiado por concordar com o R1 | A validação usa gabaritos reais |
| **Várias mudanças por rodada de RL** | A causa de cada diferença não está isolada | Variar uma coisa por vez a partir de uma base fixa |
| **Triagem por camadas é pós-treino** | Zerar módulos depois do treino conjunto não equivale a treinar só neles (**confirmado** no E10: a triagem dava 69,6%, o treino 62,3%) | Não usar triagem pós-treino como evidência causal; treinar |
| **Heurística de idioma** | O "traço em português" é uma contagem de palavras funcionais | Revisão por amostra (feita em ~5 saídas por rodada, não em 50) |
| **Dados do adalberto sem licença** | Só para experimento | Não publicar adaptador treinado neles |
| **vLLM não reprodutível** | Ruído de ±0,15 a 1,2 ponto por execução | Fora do IC; mesma ferramenta para todas as rodadas |
| **Revisão humana dos traços (D-10) não feita** | Qualidade dos dados de SFT não auditada por pessoa | `reports/review_d10.md` aguarda revisão |

## 5. O que mudaria as conclusões

- Uma segunda semente do `gspo1` que **não** repita a múltipla escolha de ~67%: o ganho do RL seria ruído.
- A suíte de teste mostrando o `gspo1` abaixo do Lumi nas provas brasileiras (ENEM, BLUEX, OAB): a validação (Global-MMLU, MMLU-ProX, INCLUDE) não representaria as provas do PRD.
- ~~O `adalberto_mlp` aprendendo português com a nota do Lumi~~ (não aconteceu: 62,3%). Fica o contrário: se o par em inglês e português (H21) também perder 7 pontos no inglês, o dano vem do SFT com esses dados, não do idioma.
- Traços do Nemotron 30B que reduzam a perda na múltipla escolha: o estilo do professor era a causa principal.
