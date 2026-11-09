# Literatura, datasets e táticas (09/11/2026)

Levantamento feito para decidir o que testar depois do `qwen_mix` e do `gspo_qwen_mix` (nenhum dos dois passou da faixa de 64,4 a 67,8% das duas sementes do `gspo1`). As cifras das fontes vêm de páginas lidas por ferramenta de busca e de resumo automático; **conferir nos artigos antes de citar em relatório**. Onde a fonte é blog de fabricante, está dito.

## 1. O custo de pensar em português é conhecido

| Fonte | O que diz | O que significa para o projeto |
| --- | --- | --- |
| [XReasoning, arXiv 2505.22888](https://arxiv.org/abs/2505.22888) | Modelos de raciocínio frequentemente voltam ao inglês; forçar o idioma do usuário por prompt melhora a leitura mas **reduz o acerto**; pós-treino com **100 exemplos** corrige o idioma, "com alguma perda de acerto" | O custo que medimos (acerto entre os que fecham de 72 a 76% contra 80,7% do Lumi) é o esperado; 100 a 300 exemplos bastam para trocar o idioma (teste `selfdist_300`) |
| [Think Portuguese with Bode Reasoning, PROPOR 2026](https://aclanthology.org/2026.propor-1.95/) | SFT de Qwen3-4B e Qwen3-4B-Thinking em 13.961 exemplos em português: raciocínio em português em 86 a 98,7% das saídas, menos verbosidade, "degradação moderada de acerto" em múltipla escolha | Mesmo o SFT publicado em português paga o custo; sem números antes e depois na página lida |
| [M-Thinker, arXiv 2510.07300](https://arxiv.org/abs/2510.07300) | RL com recompensa de consistência de idioma (−1 se o raciocínio ou a resposta saem do idioma da pergunta) e de alinhamento com o raciocínio em inglês do próprio modelo (juiz LLM, [0, 1]); recompensa total Racc × (1 + Rcta). Treino: SFT de partida de ~20 mil exemplos em 5 idiomas (inclui português) e 3 mil por idioma por iteração de RL. No 4B (Qwen3-4B-Thinking): 72,49 de acerto com consistência, contra 77,96 de acerto base | O método publicado mais forte deixa ~5,5 pontos abaixo do inglês no 4B (matemática, outra prova): a diferença que temos (5 a 7) é da mesma ordem. O RL só com acerto volta ao inglês (consistência 0,00 no 1,5B) |
| [Language-Mixed CoT, arXiv 2510.04230](https://arxiv.org/abs/2510.04230) | Raciocínio que alterna inglês (âncora) e o idioma-alvo, em coreano; ganhos médios de 18,6 pontos em modelos pequenos e médios | Alternativa de desenho; **conflita com o critério de 90% de traços em português** se o traço misto não contar como português. Decisão do PRD, não técnica |

## 2. Laços de repetição são a falha dominante

- [Wait, Wait, Wait… Why Do Reasoning Models Loop? (arXiv 2512.12895)](https://arxiv.org/abs/2512.12895): modelos menores entram mais em laço; **estudantes destilados entram em laço muito mais que seus professores**; a temperatura alta reduz o laço só na superfície; os autores apontam o remédio no treino (tornar os exemplos mais fáceis de aprender para o modelo pequeno, isto é, traços mais curtos e simples). Sustenta a H53.
- [Antidoom, Liquid AI](https://liquid.ai/blog/antidoom) (**blog de fabricante, não revisado**): otimização de preferência no **token que inicia o laço** (FTPO). Rejeitado: o token que começa a primeira repetição; escolhidos: até 20 alternativas do top-k do modelo de referência naquela posição. Laço = trecho de ≥ 60 caracteres repetido ≥ 4 vezes. LoRA de rank 128 a 256 em atenção, MLP e `lm_head`, lr 4e-6 a 2e-5, 1 época, parada antecipada quando a fração de escolhidos que vencem chega a 0,35, até 20 mil pares. Relata laço de 22,9% para 1% no Qwen3.5-4B (amostragem gulosa) e de 10,2% para 1,4% em um 2,6B; só dois modelos, um deles um ponto de controle interno.
- [LZ Penalty (arXiv 2504.20131)](https://ar5iv.labs.arxiv.org/html/2504.20131): penalidade na decodificação que permite decodificação gulosa sem perda. **Não se aplica**: a decodificação da avaliação é fixa (gulosa, `repetition_penalty` 1,0).
- **Medição nossa** (validação, múltipla escolha com thinking, saídas já avaliadas; laço = bloco de ≥ 60 caracteres repetido em sequência ≥ 4 vezes, ou mais de 50% de 8-gramas repetidos nas últimas 1.500 palavras):

| Modelo | Não fecham | Com repetição forte no fim |
| --- | --- | --- |
| Lumi | 53 de 385 (13,8%) | 50 (94% dos que não fecham) |
| `adalberto` | 51 (13,2%) | 44 (86%) |
| `qwen_mix` | 62 (16,1%) | 50 (81%) |
| `gspo1` (ciclo 4) | 35 (9,1%) | 21 (60%) |
| `gspo1_s1` | 47 (12,2%) | 33 (70%) |
| `gspo_thor` | 47 (12,2%) | 46 (98%) |
| `gspo_qwen_mix` | 47 (12,2%) | 25 (53%) |

  Os laços explicam 53 a 98% das respostas que não fecham, inclusive no Lumi. Os itens que o Lumi acerta e o `qwen_mix` deixa sem fechar são 34 de 385 (8,8%). Conta aproximada: fechar 95% das respostas com 75% de acerto entre elas daria ~71%, acima dos 69,6% do Lumi; os itens que passam a fechar são os mais difíceis, então o acerto entre eles tende a ser menor e o ganho real, menor que esse.

## 3. Outras táticas

- **Média de pesos** (sopas): a busca encontrou o método de média ponderada por métricas para adaptadores LoRA, com ganhos de até 2,62% sobre o último ponto em matemática ([arXiv 2504.18580](https://arxiv.org/abs/2504.18580), resumo da busca; artigo não lido). Ressalva da fonte: a média de A e B separadas não é a média dos produtos B·A. **Medido aqui:** os adaptadores de RL ficam a ~2% de distância relativa da média e o erro da aproximação B·A é de no máximo 0,17% (mediana 0,07%), então a média direta vale. O limite é de medição, não de método: com 385 itens e ruído de 3 a 4 pontos entre sementes, uma sopa não se distingue de uma rodada.
- **Destilação on-policy** ([verl, OPD](https://verl.readthedocs.io/en/latest/_sources/algo/opd.md.txt); [glossário da Prime Intellect](https://primeintellect.ai/rl-glossary/on-policy-distillation)): o estudante gera, o professor dá as log-probabilidades por token (KL reversa). Sinal denso, mais barato que RL em alguns relatos; o sinal do professor degrada em cadeias longas; OPD seguida de RL superou cada um isolado em um relatório. O professor natural aqui seria o Lumi com o começo em português forçado (destilação de contexto), que **não pode passar do próprio Lumi com o começo forçado** (diagnóstico `base_ptprefill`); exige código novo.
- **Ceia/Lumi** ([cartão](https://huggingface.co/ceia-br/lumi-4b)): base Nemotron 3 Nano 4B, professor de destilação Nemotron 3.5 Lightning; o cartão diz que o modo thinking "deve permanecer desligado nesta versão". Treinos listados (não usar, pela regra do projeto): CPT com FinePDFs-Edu, FineWeb2, GigaVerbo-v2, Nemotron-CC-v2.1, Nemotron-Pretraining-Specialized-v1.1, Nemotron-Pretraining-SFT-v1 e três conjuntos CEIA; SFT com Nemotron-SFT-Multilingual-v2, Nemotron-SFT-Instruction-Following-Chat-v2 e v3, Nemotron-Science-v1, Nemotron-SFT-Safety-v2 e três conjuntos CEIA.

## 4. Datasets

| Dataset | O que é | Decisão |
| --- | --- | --- |
| [Bode-Reasoning (recogna-nlp)](https://huggingface.co/datasets/recogna-nlp/Bode-Reasoning) | 11.715 exemplos (10.287 de treino) com raciocínio em português gerado pelo Gemini-2.5 Pro; **52% vêm de provas: ENEM 2010–2018 e 2022 (1.398), BLUEX 2018–2024 sem 2023 (692), OAB 2010–2018 sem 2018 (2.081)**; mais 5.227 de uma tradução do reasoning-v1-20m (raciocínio de até 500 palavras), 400 de matemática sintética e EduBench (proprietário) | **Não usar as partes de prova**: são as provas do nosso teste (ENEM, BLUEX, OAB). As partes não-prova (reasoning-v1-20m traduzido, 5.227) só com checagem de 8-gramas e de licença; o cartão não declara licença |
| [`cnmoro/reasoning-v1-20m-portuguese`](https://huggingface.co/cnmoro) | tradução (GPT-4o-mini) do `glaiveai/reasoning-v1-20m` (traços do DeepSeek-R1-Distill-Llama-70B; ~19 milhões de linhas segundo página em cache; a página do dataset não foi lida) | Candidato para raciocínio **curto** em português de domínio geral (H53); verificar licença e qualidade da tradução antes |
| [GigaVerbo-v2 SFT e Preferências, Tucano 2](https://huggingface.co/blog/Polygl0t/tucano2) | ~4 milhões de exemplos em 12 tipos de tarefa, com raciocínio e matemática; ~28 mil pares de preferência (APO) | **Fora**: GigaVerbo-v2 consta na lista de pré-treino contínuo do Lumi |
| ARC, OpenBookQA, SciQ em português | a busca só achou ARC-Challenge em português europeu com treino e uma amostra sem treino; nada em pt-BR com partição de treino | Teria de ser traduzido na Thor (tradução com o Nemotron 30B, 93,6% de aproveitamento). **Baixa prioridade**: três doses de MCQ do Qwen não mudaram nada |

## 5. O que foi para a fila e o que fica proposto

| Item | Estado |
| --- | --- |
| H55: sopas de pesos `gspo1_soup2` e `gspo1_soup3` | na fila (`noite8`) |
| H56: Lumi sem adaptador com o começo forçado em português (`base_ptprefill`) | **já medido em 07/11** (E0): 61,3% em múlt. escolha contra 69,6%; os SFTs ficam no mesmo nível |
| H54b: `selfdist_300` (300 traços, ~10 passos) | na fila (`noite8`) |
| H57: fase anti-laço (preferência no token que inicia o laço) sobre o `gspo1` de 8 ciclos | **implementada e na fila** (`noite10`): `scripts/make_antidoom_prompts.py`, `antidoom_mine.py`, `antidoom_train.py`, `src/lumi_ft/train/antidoom.py` (11 testes) |
