# PRD: SFT do Lumi-4B com LoRA ou DoRA para adicionar raciocínio

Nov 6, 2026 · @gustavobarrosss

Atualizado em Nov 7, 2026: escopo ampliado com traços de professor, GRPO e comparação em quatro colunas (seção [Atualização de escopo](#atualização-de-escopo-2026-11-07)).

## Resumo e objetivo

Treinar por SFT um adaptador LoRA ou DoRA sobre o `ceia-br/lumi-4b` v0.2 para que `enable_thinking=True` produza raciocínio em português que melhore as respostas, sem piorar o modo sem raciocínio.

Hoje o cartão manda manter esse modo desligado, porque ele não foi treinado nem validado nesta versão. A entrega final é um adaptador vencedor, sua versão fundida em BF16 e um relatório reproduzível.

A **Nota de raciocínio** é a média simples de GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX e OAB. Todas as referências abaixo são medidas na fase 0, no mesmo harness.

| Métrica | Referência | Meta mínima | Meta alvo |
| --- | --- | --- | --- |
| Nota de raciocínio, com raciocínio | Lumi medido nos dois modos | IC 95% da diferença acima de 0 contra os dois | Lumi sem raciocínio + 5,0 |
| GSM8K-PT, com raciocínio | Lumi sem raciocínio | + 3,0, com IC 95% acima de 0 | + 10,0 |
| Respostas que fecham `</think>` e trazem texto depois | Lumi com raciocínio | ≥ 98% | ≥ 99,5% |
| Traços em português | Lumi com raciocínio | ≥ 90% | ≥ 98% |
| Overall sem raciocínio | Lumi medido (52,99 no cartão) | Queda de até 1 ponto, sem queda significativa em nenhuma nota | Sem queda |

As metas ficam entre dois precedentes. Nas 4 provas em comum com a Nota (sem MMLU-ProX PT), o Qwen3-4B ganha em média 15,73 pontos ao ligar o raciocínio e o Tucano 3.7B Think fica em -0,15 contra o Instruct. A meta alvo de 5 pontos é cerca de um terço do ganho do Qwen3-4B. No GSM8K-PT, a meta mínima repete o ganho do Tucano (2,89) e a alvo é cerca de um quarto do ganho do Qwen3-4B (37,14).

## Atualização de escopo (2026-11-07)

O gate da fase 2 falhou na nota: os dados do GigaVerbo `reasoning`, sozinhos, ensinam o formato e o português, mas o modelo acerta menos que o Lumi (ver fase 2). Com o prazo de 2 dias para o projeto, o usuário decidiu ampliar o escopo em vez de encerrar. **As metas, o critério de vitória e os limiares não mudam** (regra 4); muda o que se pode fazer para alcançá-los.

| O que entra | Onde |
| --- | --- |
| **Traços de professor**, aceitos só se fecham `</think>`, estão em português e acertam o gabarito: o dataset `adalberto-temp/gsm8k-sft-thinking-pt` (Qwen3-30B), o próprio Lumi (autodestilação), o Qwen3-30B-A3B e o Nemotron 3 Nano 30B-A3B rodando na segunda máquina, e as questões de ciência da NVIDIA em português | D-12 a D-16 e [Fontes de professor](#fontes-de-professor-novo-escopo) |
| **Tradução do raciocínio** para português, feita pelo próprio Lumi, quando o professor raciocina em inglês | D-15 |
| **GRPO** (RL com recompensa verificável) a partir do adaptador do SFT, como plano B para a múltipla escolha | [GRPO](#grpo-rl-com-recompensa-verificável) |
| **Segunda máquina**, uma Jetson AGX Thor (122 GiB, aarch64), que só **gera dados**; treino, tradução e avaliação ficam na RTX 4090 | Fase 2b |
| **Validação limpa**: cerca de um terço da validação do GSM8K estava duplicada em outra tradução nos dados de treino; as comparações usam os 676 itens limpos | A-15 |
| **Comparação em quatro colunas**: Nemotron 3 Nano 4B com raciocínio, Lumi com raciocínio, nosso Lumi com raciocínio em inglês e nosso Lumi com raciocínio em português | A-16, A-17 e [comparacao.md](comparacao.md) |

O que continua fora: APO, QLoRA, publicação, tool calling e contexto longo. **Continuar o treino com raciocínio em inglês** é um teste posterior, que o usuário conduz depois de esgotado o ganho em português; aqui ele só existe como uma coluna da comparação, para separar o ganho do treino do custo do idioma.

## Contexto e evidências

O Lumi-4B já tem a infraestrutura do modo de raciocínio, mas ninguém mediu o que ele faz com ela. A hipótese central é que um adaptador treinado com traços em português consegue ativar esse modo de forma útil.

| Item | Valor |
| --- | --- |
| Versão | v0.2, lançada em 6 de novembro de 2026, no commit `27b30f40` |
| Classe | `NemotronHForCausalLM`, BF16, 7,95 GB |
| Camadas | 42: 21 Mamba, 17 MLP e 4 de atenção (índices 12, 17, 24 e 32) |
| Vocabulário e cabeça | 131.072 tokens, sem pesos amarrados: embeddings e `lm_head` somam cerca de 20% dos parâmetros |
| Tokens de raciocínio | `<think>` (12) e `</think>` (13) já existem no vocabulário e têm `special=False`, então `skip_special_tokens=True` não os remove |
| Contexto máximo | 262.144 tokens |
| Template | `enable_thinking` ligado por padrão, tool calls em XML |
| Prompt de geração | Com raciocínio termina em `<think>\n`; sem raciocínio termina em `<think></think>` |
| Histórico | `truncate_history_thinking` remove o raciocínio dos turnos anteriores ao último pedido do usuário |
| Parada | `config.json` e `generation_config.json` usam `eos_token_id=2` (`</s>`), mas o template fecha o turno com `<\|im_end\|>` (11) |
| Posição do cartão | "Thinking must remain disabled for this release"; o suporte está previsto para uma versão futura |
| Treino | Pré-treino contínuo de 50B tokens, com destilação da Nemotron 3.5 Lightning, seguido de SFT, sem RL nem RLVR |
| Modelo-base | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, com raciocínio ligado por padrão e uso declarado em inglês e código |
| Licença | O Lumi tem `license: TODO` no cartão; a base segue a NVIDIA Nemotron Open Model License |

O modo sem raciocínio é a referência publicada. O cartão atual traz as notas só em gráfico, com 6 provas, e a tabela abaixo usa os valores das barras de `lumi_versoes.svg`. O cartão não publica nota com raciocínio. O MMLU-ProX PT saiu na v0.2; a v0.1 tinha 36,28 nessa prova, com harness não documentado.

| Prova | Nemotron 3 Nano 4B | Lumi v0.1 | Lumi v0.2 |
| --- | --- | --- | --- |
| MMLU PT, ensino médio | 42,67 | 45,72 | 47,63 |
| MMLU PT, graduação | 44,02 | 48,80 | 50,75 |
| Global MMLU PT | 55,46 | 59,36 | 61,73 |
| MMMLU PT | 54,87 | 59,31 | 61,50 |
| INCLUDE PT | 54,08 | 58,44 | 60,98 |
| IFEval PT estrito | 40,93 | 33,83 | 35,33 |
| Overall | 48,67 | 50,91 | 52,99 |

O artigo do Tucano 2 dá as duas referências mais próximas, com resultados bem diferentes. No Qwen3-4B, o mesmo modelo muda de modo. No Tucano, são dois modelos pós-treinados a partir da mesma base. O Think usou 100% do subset `reasoning`, com SFT e APO.

| Prova do artigo | Qwen3-4B sem | Qwen3-4B com | Diferença | Tucano Instruct | Tucano Think | Diferença |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K-PT | 39,88 | 77,02 | +37,14 | 53,81 | 56,70 | +2,89 |
| BLUEX | 63,28 | 78,30 | +15,02 | 64,53 | 63,00 | -1,53 |
| ENEM | 72,15 | 85,23 | +13,08 | 72,92 | 71,52 | -1,40 |
| IFEval-PT | 79,33 | 84,00 | +4,67 | 41,67 | 31,67 | -10,00 |
| MMLU | 26,93 | 31,53 | +4,60 | 64,64 | 61,18 | -3,46 |
| OAB | 50,30 | 47,97 | -2,33 | 54,31 | 53,76 | -0,55 |
| ARC | 43,08 | 39,66 | -3,42 | 60,34 | 55,38 | -4,96 |
| BELEBELE | 83,67 | 23,00 | -60,67 | 85,22 | 83,56 | -1,66 |

O IFEval-PT do artigo tem 300 amostras e não é o Portuguese-IFEval de 535 prompts usado no cartão. Os autores atribuem o resultado do Think à falta de dados de raciocínio em português, aos traços curtos e ao orçamento de 4.096 tokens. O artigo não explica a queda do Qwen3-4B no BELEBELE, que parece problema de extração ou de limite de tokens, e não perda real; A-05 trata esse risco. No mesmo artigo, o HumanEval cai em média 29 pontos ao ligar o raciocínio.

Três hipóteses guiam o projeto:

- **H1:** o Lumi herdou parte da capacidade de raciocínio do Nemotron 3 Nano 4B, talvez em inglês. A base foi treinada com traços sintéticos e tem AIME25 78,5 e MATH500 95,4 com raciocínio. A fase 0 mede se essa capacidade sobreviveu.
- **H2:** com essa herança, o Lumi pode ganhar mais que o Tucano com os mesmos dados. É uma inferência: o Tucano 3.7B vem do `Qwen3-4B-Base`, sem pós-treino, enquanto o Lumi vem de um modelo de chat que já raciocina. O pré-treino contínuo e o SFT do Lumi podem ter apagado essa vantagem, e a fase 2 testa isso antes do treino completo.
- **H3:** o ganho se concentra em matemática, que soma 82% dos tokens de `reasoning`. Por isso, o relatório separa o GSM8K-PT das provas de conhecimento.

## Escopo

Este projeto cobre um SFT por adaptador sobre os pesos do Lumi-4B, comparando LoRA e DoRA no modo com raciocínio e na preservação do modo sem raciocínio, e, a partir de 2026-11-07, o uso de traços de professor, a tradução desses traços e um GRPO sobre o adaptador.

**Dentro do escopo**

- Preparar o subset `reasoning` do GigaVerbo-v2 e uma parcela de exemplos sem raciocínio.
- Gerar e filtrar traços de professor (Qwen3-30B-A3B e Nemotron 3 Nano 30B-A3B na Thor, o próprio Lumi) e usar os datasets de raciocínio da NVIDIA e do adalberto, com os filtros D-12 a D-16.
- Traduzir para português os traços em inglês.
- Treinar e comparar LoRA e DoRA, incluindo a escala do adaptador na fusão (só LoRA).
- Aplicar GRPO sobre o melhor adaptador do SFT.
- Medir o Lumi, o Nemotron 3 Nano 4B de referência e os candidatos nos dois modos, no mesmo harness.
- Entregar o adaptador vencedor, o modelo fundido em BF16 e o relatório.

**Fora do escopo**

- QLoRA e QDoRA: os pesos em BF16 cabem na RTX 4090, e a quantização não promete ganho de qualidade. Se a memória apertar, a saída é reduzir o batch físico e acumular gradientes.
- APO. Pede outro PRD.
- Continuar o treino com raciocínio em inglês: teste posterior do usuário; aqui é só uma coluna da comparação.
- Melhorar o IFEval: aqui ele é só uma trava contra regressão.
- Raciocínio em conversa com vários turnos: todos os exemplos de treino têm um só turno. O model card registra essa limitação.
- Pré-treino continuado dos pesos completos (CPT): avaliado e adiado. O Lumi já raciocina em português com forma correta; a queda está na nota da múltipla escolha, que o CPT não corrige de forma previsível, e a 4090 treina ~0,5 bilhão de tokens por dia. Um diagnóstico português x inglês no MMLU-ProX (`mmlu_prox_diag_*`) decide se vale reabrir a ideia.
- Tool calling, contexto longo, exportação GGUF e publicação no Hugging Face.

## Requisitos de dados

O treino usa o subset `reasoning` e uma parcela de exemplos sem raciocínio, todos do GigaVerbo-v2 SFT (Apache-2.0). Os números abaixo foram calculados sobre os arquivos Parquet completos.

| Fonte | Linhas | Tokens | Decisão |
| --- | --- | --- | --- |
| `reasoning`, tipo `math_reasoning` | 66.067 | 28,7 M | Usar com raciocínio |
| `reasoning`, tipo `reasoning` | 12.182 | 6,1 M | Usar com raciocínio |
| `general`, `structured`, `system_prompts`, `rewriting`, `summarization`, `math` e `retrieval` | Amostra | Até 25% dos tokens | Usar sem raciocínio, com `instruct_score` ≥ 4,0 |
| `math_cot` | 63.413 | 27,1 M | Fora: duplica `reasoning` |
| `chosen` de `gigaverbo-v2-preferences`, subsets `*-reasoning` | 13.649 | 7,6 M | Fora: só 1.502 de 13.627 prompts são novos |
| `function_call`, `code` e `translation` | | | Fora: não servem a este objetivo |

O que os dados permitem e o que limitam:

- **Formato limpo:** todas as linhas têm um único `<think>` fechado, no início da resposta, e `instruct_score` ≥ 4,5. Nenhuma passa de 1.000 tokens, e o p99 é 935.
- **Traços curtos:** a mediana é de 174 palavras em matemática e 157 nos demais, sem revisão de hipóteses nem checagem da resposta nas amostras lidas.
- **Pouca variedade:** dos 12.182 exemplos fora de matemática, 4.007 usam os prompts de `harmfull-reasoning`, e as amostras lidas desse grupo são recusas. Sobram cerca de 8.200 exemplos gerais.
- **`math_cot` não é fonte independente:** 99,4% dos seus prompts únicos estão em `reasoning`. Em 95% das linhas em comum, o traço é idêntico à resposta do `math_cot`, ou seja, é a solução passo a passo dentro das tags.
- **Traços escritos depois da resposta:** nos dados com pipeline de duas etapas, um modelo gera a resposta e outro escreve o traço. O texto pode justificar a resposta em vez de levar até ela.
- **Sem gabarito:** os dados são sintéticos, e nada garante que os traços acertem.
- **Resíduos do pipeline:** cerca de 11% dos traços (8.951) têm expressões como "contexto fornecido" ou "resposta fornecida". Nas amostras lidas, o prompt não tinha contexto nenhum. A conta usa regex e ainda não foi revisada.

A mistura inicial abaixo é uma hipótese, testada contra 100% de raciocínio na fase 3. A Unsloth recomenda ao menos 75% de exemplos com raciocínio para manter essa capacidade, mas o texto é genérico para Nemotron, e o único notebook de SFT LoRA citado é o do modelo de 30B. Aqui os 25% restantes também existem para proteger o modo sem raciocínio.

| Grupo | Fonte | Participação nos tokens |
| --- | --- | --- |
| Com raciocínio | `reasoning` filtrado, inteiro | 75% |
| Sem raciocínio | `general` | 12% |
| Sem raciocínio | `structured`, `system_prompts`, `rewriting` e `summarization` | 8% |
| Sem raciocínio | `math` e `retrieval` | 5% |

### Requisitos

| ID | Requisito |
| --- | --- |
| D-01 | Renderizar com `tokenizer.apply_chat_template` do Lumi, sem formatação manual. Declarar `enable_thinking` em todo exemplo; nos trainers do TRL, usar a coluna `chat_template_kwargs`. |
| D-02 | Separar traço e resposta em `reasoning_content` e `content`, para o template gerar sempre `<think>\n...\n</think>\n`. O subset traz `</think>\n\n`, diferente do que o template gera com esse campo. |
| D-03 | O prompt deve ser prefixo exato do exemplo renderizado e terminar em `<think>\n` com raciocínio ou em `<think></think>` sem raciocínio. A loss cobre traço, `</think>`, resposta e `<\|im_end\|>`; todo o resto recebe `labels = -100`. |
| D-04 | Exemplos sem raciocínio não podem ter `<think>` com conteúdo. Descartar os que tiverem. |
| D-05 | Remover sobreposição com todas as provas de teste e de validação por k-gramas de 8 a 32 tokens. Na checagem com palavras normalizadas, o GSM8K-PT `test` divide 12-gramas com `reasoning` em 2 de 1.295 itens e 8-gramas em 66. |
| D-06 | Remover os prompts duplicados (3.112 linhas) e os traços com resíduo do pipeline, depois de revisar uma amostra do filtro. |
| D-07 | Em `math_reasoning`, medir e descartar os exemplos em que o último número do traço não aparece na resposta. |
| D-08 | Nunca truncar um exemplo: descartá-lo se passar do limite, porque cortar `</think>` ensina o modelo a não fechar o traço. |
| D-09 | Descartar traços fora do português, com detector de idioma fixado na fase 1. |
| D-10 | Revisão humana de 200 traços, 100 de cada tipo, antes do treino completo, para avaliar coerência e utilidade. |
| D-11 | Separar 2% para validação interna, estratificado por `task_type`, e registrar hashes em `data_manifest.json`. |
| D-12 | Além do 8-grama (D-05), tirar do treino os problemas que repetem um item de validação em outra tradução: mesma assinatura de números do enunciado (normalizando `1.000` e `1000`, `0,5` e `0.5`) e sobreposição de palavras (Jaccard) ≥ 0,3 (`teacher.is_val_duplicate`). O 8-grama sozinho deixou passar cerca de 325 dos 1.000 itens da validação do GSM8K. |
| D-13 | Um traço de professor só entra se fechar um único `</think>`, terminar sozinho, tiver traço em português (ou for traduzido, D-15) e a resposta final bater com a referência no formato estrito do prompt (`#### N` ou `Resposta: <letra>`). A letra da múltipla escolha é a majoritária dos traços de um problema (≥ 75% de ao menos 3), um indicador e não verdade. |
| D-14 | O prompt de treino de professor é o da avaliação (`gsm8k.build_prompt` ou `mcq.build_prompt`), sem system, para o modelo aprender a seguir a instrução de formato. |
| D-15 | Tradução do raciocínio pelo Lumi (temperatura 0, em blocos de parágrafos de até 1.800 caracteres). Um bloco só vale se terminou, não traz `<think>`, tem entre 0,8 e 1,7 vezes o tamanho do original e o raciocínio final está em português; as alternativas da questão precisam voltar na mesma ordem. Depois, 8-grama contra as provas e a validação. |
| D-16 | Registrar a licença de cada fonte de professor. O dataset do adalberto não declara licença e só serve a experimentos; a NVIDIA (CC-BY-4.0, com partes CC-BY-SA em matemática derivada do StackOverflow) e o Qwen (Apache-2.0) permitem uso. Conferir os termos de qualquer dataset gerado por modelo comercial antes de treinar com ele. |

### Fontes de professor (novo escopo)

| Fonte | Conteúdo | Estado |
| --- | --- | --- |
| [`adalberto-temp/gsm8k-sft-thinking-pt`](https://huggingface.co/datasets/adalberto-temp/gsm8k-sft-thinking-pt) | Traços do Qwen3-30B-A3B-Thinking para problemas do GSM8K `train` em português; sem licença | Usado: matemática boa, múltipla escolha ruim |
| Autodestilação do Lumi | Raciocínio do Lumi forçado em português (prefill) e filtrado | Testado: pior que o Lumi |
| Pacote 1 da Thor (`handoff/thor/`) | Qwen3-30B-A3B-Thinking, prefill em português, 50.908 problemas | Gerando na Thor |
| Pacote 2 da Thor (`handoff/thor2/`) | Nemotron 3 Nano 30B-A3B, raciocínio em inglês, 9.612 de matemática e 12.000 de múltipla escolha | Gerando na Thor; o raciocínio será traduzido aqui |
| [`nvidia/Nemotron-SFT-Multilingual-v2`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) (`stem_pt`) | Questões de ciência em português do Brasil (enunciado e resposta final traduzidos pelo DeepSeek-V3) com raciocínio do GPT-OSS-120B em inglês, o estilo do Nemotron 3 e do Lumi | Em tradução do raciocínio (`nemo_mix`): 8.000 de 26.671 questões de múltipla escolha com traço de até 1.000 palavras |
| Questões de ciência do `open-r1/Mixture-of-Thoughts` | Múltipla escolha com traços do R1 | Só os enunciados, traduzidos, como prompts do GRPO (3.715 após o 8-grama) |
| `recogna-nlp/Bode-reasoning` | Exames e raciocínio traduzido | **Fora:** os 3.869 itens de ENEM, BLUEX e OAB têm 100% de 8-grama com as nossas provas |

O estilo do raciocínio importa: o do Lumi é curto e em forma de plano ("We need to…", mediana de 96 palavras no GSM8K); o do Qwen3-30B é mais longo e falado (177 palavras, "Certo, vamos…"), e o do R1 ainda mais (610). Treinar com um estilo que o Lumi não tem é a hipótese para a perda do `adalberto` na múltipla escolha; o professor da mesma família (Nemotron) e os traços do GPT-OSS testam essa hipótese.

## Requisitos de treino

Os dois métodos usam os mesmos módulos-alvo e diferem apenas no `use_dora`, com um patch local no DoRA (abaixo). O LoRA é a referência; o DoRA testa se há ganho em rank baixo, com mais custo de tempo.

| Braço | Pesos-base | `use_dora` | Kernel Mamba | Papel |
| --- | --- | --- | --- | --- |
| LoRA | BF16 | `False` | Kernels permitidos | Referência |
| DoRA | BF16 | `True` | Kernels permitidos | Testar ganho em rank baixo |

Os kernels são os pacotes `mamba-ssm` e `causal-conv1d`. O código do Lumi só ativa o caminho rápido com os dois instalados e, sem eles, cai no caminho PyTorch apenas com um aviso.

### DoRA e as linhas zeradas da `in_proj`

A `in_proj` do Lumi tem 213 linhas com pesos exatamente zero (média de cerca de 10 por camada). O DoRA do PEFT inicializa a magnitude com a norma da linha (0) e depois divide magnitude por norma, o que dá 0/0 = NaN nos logits e nos gradientes desde o primeiro passo. O LoRA não é afetado.

O projeto usa o patch `src/lumi_ft/modeling/peft_patch.py`: onde a norma é exatamente 0, o divisor vale 1, e a magnitude inicial continua sendo a norma verdadeira (0). Essas linhas nascem inertes e continuam produzindo zero, como na base; as outras linhas não mudam (saída bit a bit igual, testado). Consequência para a comparação: nessas 213 linhas o DoRA não se comporta como o LoRA, que pode ativá-las. O unmerge não é suportado, e o projeto não o usa. Evidências em [discovery.md](discovery.md).

### Módulos-alvo

```python
TARGET_REGEX = r".*\.mixer\.(in_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)"

LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_REGEX,
    use_dora=False,  # True no braço DoRA
)
```

- **`out_proj` do Mamba fica de fora.** Com `mamba-ssm`, o treino passa `self.out_proj.weight` direto ao kernel e ignora o `forward` do módulo. A [receita LoRA da NVIDIA](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_nano_4b_squad_peft.yaml) exclui `*.out_proj` por esse motivo.
- **`conv1d` fica de fora.** O peso também é lido direto, e o PEFT não funde convoluções com `groups > 1`.
- **`embeddings` e `lm_head` ficam de fora.** `<think>` e `</think>` já estão no vocabulário, então o adaptador não precisa tocar neles.

Os nomes foram confirmados na fase 0 com `named_modules()`: o regex casa exatamente 71 módulos nas duas implementações (21 `in_proj`, 17 `up_proj`, 17 `down_proj` e 4 de cada projeção de atenção). Os 21 `out_proj`, o `lm_head`, os `conv1d` e os embeddings ficam de fora. O código do repositório nomeia as camadas `backbone.layers.N`, e a implementação nativa `model.layers.N`; o regex vale para os dois.

### Implementação do modelo

**Decisão da fase 0: implementação nativa do Transformers** (`AutoModelForCausalLM` sem `trust_remote_code`). O código do repositório (`modeling_nemotron_h.py`) fica só como referência. Sem padding, as duas implementações calculam o mesmo. O código do repositório erra com padding à esquerda, que o `generate` em batch usa, e seu `generate` quebra no Transformers 5.22.2. A nativa usa os mesmos kernels e para em `<|im_end|>`. Medições, números e limites estão em [discovery.md](discovery.md).

**Decisão de 2026-11-07 (prazo de 2 dias): a avaliação gera com o vLLM 0.33.0**, não com a nativa. Treino, fusão e testes de equivalência continuam na nativa. O vLLM roda no `.venv-vllm`, separado (exige torch 2.14.0 e transformers < 5.21), e produz as mesmas notas que o HF no GSM8K-PT (piloto de 256 itens: 90,6% contra 91,0% com raciocínio, 88,3% contra 87,9% sem), com ~10 vezes menos tempo. Fica de fora a reprodutibilidade bit a bit (ver A-04).

O modelo fundido salvo pela nativa mantém as chaves do original (263 de 263, prefixo `backbone`), mas o `config.json` salvo é outro (`layers_block_type` no lugar de `hybrid_override_pattern`, campos de MoE/MTP com valores padrão). Na entrega, os pesos vêm da nativa e o `config.json`, o tokenizer e os `.py` vêm do repositório original. Carregar a entrega com `trust_remote_code` ainda não foi testado; fica para a fase 5. Detalhes em [discovery.md](discovery.md).

### Carga e tokenizer

- `eos_token_id=[2, 11]` em toda geração. Sem isso, o `generate` não para em `<|im_end|>`, porque as configs do modelo só listam o id 2.
- `pad_token="<unk>"` (id 0, o `pad_token_id` do `config.json`). O `tokenizer_config.json` não define esse token, e usar `<|im_end|>` como padding pode mascarar o fim real do turno nos labels.
- Pesos em BF16: cerca de 8,0 GB nos dois braços, o que deixa cerca de 17,7 GB dos 24 GiB da GPU. A fase 0 mediu o passo de treino completo: 7,6 a 7,8 GiB fixos mais 1,72 GiB por 1.000 tokens do batch, com gradient checkpointing, e 8.192 tokens por passo é o limite (12.288 dá falta de memória nos quatro braços). A memória depende só de batch × comprimento; os logits em FP32 (131.072 × 4 bytes por token, mais o gradiente) pesam muito nesse total.
- Testes que comparam dois modelos (código do repositório × nativo, adaptador × fundido) carregam um por vez e salvam os logits, porque dois modelos ocupam cerca de 16 GB antes de qualquer batch.

### Hiperparâmetros iniciais

| Parâmetro | Valor | Origem |
| --- | --- | --- |
| `r` | 16 (fixo; o plano original varria 8, 16 e 32) | NVIDIA usa 8; sem o piloto de ranks por prazo, `r=32` só entra se a meta falhar (ver riscos) |
| `lora_alpha` | 2 × `r` | NVIDIA usa 16 e 32 com `r=8` |
| `lora_dropout` | 0 | NVIDIA usa 0, e o DoRA fica mais rápido assim |
| Taxa de aprendizado | 1e-4, varrendo 5e-5 e 2e-4 | NVIDIA usa 1e-4 em LoRA |
| Otimizador | AdamW com cosseno e warmup de 3% | AdamW e cosseno do Tucano 2; warmup escolhido no projeto (o Tucano 2 usa 10%) |
| Sequência | 1.024 tokens, sem truncar; exemplo acima disso é descartado e contado no relatório | O máximo dos dados do GigaVerbo é 1.000 tokens (p99 935). Com traços de professor, o limite de 1.024 descartou 34% dos candidatos do adalberto; o teste com 2.048 (`--max-len`) não melhorou a múltipla escolha, e o `nemo_mix` usa 2.048 porque os traços da NVIDIA são mais longos |
| Batch | Físico por orçamento de tokens, agrupado por comprimento: 2.048 tokens no LoRA (2 sequências) e 1.024 no DoRA (1 sequência). Efetivo de 32 sequências nos dois braços, por acumulação de gradiente | Medido na fase 0 sem checkpointing: LoRA cabe 2 × 1.024 (17,8 GiB) e DoRA 1 × 1.024 (15,3 GiB); 3 e 2 dão falta de memória. O efetivo igual nos dois braços faz o físico diferente não mudar a otimização. O efetivo de 32 é inicial, a confirmar na varredura |
| Épocas | 2 nos finalistas e 1 nas outras etapas, sem parada antecipada | Corte por prazo (2026-11-07); o plano original era até 3 com parada antecipada, e o Tucano 2 usou 5 |
| Gradient checkpointing | Desligado; `use_cache=False` | Medido: +34% de velocidade no LoRA e +62% no DoRA. Se algum lote der falta de memória, ligar `use_reentrant=False` e registrar |

### Matriz de experimentos

| Etapa | Execuções | Dados | Épocas | Saída |
| --- | --- | --- | --- | --- |
| Viabilidade | 1 (LoRA, `r=16`) | 10.000 exemplos com raciocínio, mais 25% de tokens sem | 1 | Decisão sobre seguir |
| Varredura de taxa | 3 (LoRA, `r=16`; 5e-5, 1e-4 e 2e-4) | 20% | 1 | Taxa fixada |
| Mistura | 2 (LoRA, `r=16`): 75/25 e 100/0 | 20% | 1 | Mistura fixada |
| Finalistas | 2 (LoRA e DoRA, `r=16`), 1 seed | 100% | 2 | Um adaptador por método |

São 8 execuções (a matriz original tinha 18), com cerca de 238 M de tokens: 145 M no LoRA e 93 M no DoRA. Sem checkpointing, a velocidade medida é 5,8 mil tokens/s no LoRA e 2,5 mil no DoRA, o que dá 17 min na viabilidade, 2,2 h nas varreduras e na mistura (5 execuções de ~27 min) e 4,4 h (LoRA) mais 10,4 h (DoRA) nos finalistas: **cerca de 17 h de GPU**. Cortes em relação à matriz original, por prazo (decisão de 2026-11-07): sem o piloto de ranks (`r=16` fixo), 1 seed no lugar de 3 e 2 épocas fixas no lugar de até 3 com parada antecipada. Consequências: o rank não é otimizado, e sem seeds o IC 95% entre os finalistas é só o do bootstrap por item, sem a variação entre seeds.

A seleção usa os conjuntos de validação de A-10, nunca as provas de teste. A loss não basta, porque não mede se o raciocínio melhora a resposta.

### Experimentos executados e previstos (2026-11-07)

A matriz acima era o plano com os dados do GigaVerbo; depois do gate da fase 2 passou a valer esta tabela. Todos são LoRA `r=16`, lr 1e-4 e 1 época, salvo indicação. Notas estritas na validação (GSM8K limpo, 676 itens; múltipla escolha, 385), modo com raciocínio / sem raciocínio. Diferenças e intervalos em `reports/` e [discovery.md](discovery.md) §20.

| Execução | Dados e mudança | GSM8K limpo | Múltipla escolha |
| --- | --- | --- | --- |
| Lumi original | referência | 81,7% / 77,2% | 69,6% / 63,9% |
| `viab` | 10.000 do GigaVerbo `reasoning` | 71,5%\* / – | 43,4% / – |
| `adalberto` | Qwen3-30B (5.842 com raciocínio, 3.000 sem) mais 5.544 do GigaVerbo | 83,3% / 79,3% | 62,3% / 58,2% |
| `adalberto_s050` e `_s075` | mesmo adaptador, escala 0,5 e 0,75 na fusão | 84,0% / 79,6% e 84,0% / 79,1% | 63,4% / 64,9% e 64,4% / 58,7% |
| `adalberto_L2048` | mesmo dado, limite de 2.048 tokens | 82,2% / 81,7% | 64,2% / 55,3% |
| `selfdist` | Lumi com prefill em português: 6.365 com raciocínio, 2.204 sem | 78,1% / 77,5% | 57,9% / 63,4% |
| `adalberto_dora` | mesmo dado, DoRA | em avaliação | em avaliação |
| `nemo_mix` | adalberto (matemática) mais 8.000 questões da NVIDIA com o raciocínio traduzido | pendente | pendente |
| `grpo1` | GRPO a partir do adaptador `adalberto`, 8 iterações | pendente | pendente |
| `adalberto_lr5e5` | mesmo dado, lr 5e-5 | pendente | pendente |

\* validação completa de 1.000 itens, não só os 676 limpos.

O que a tabela já diz: a matemática com raciocínio em português passa o Lumi com os traços do Qwen (+1,6 a +2,4, sem significância contra o Lumi com raciocínio); a múltipla escolha com raciocínio perde de 5 a 7 pontos em todas as variações; reduzir o adaptador (escala) e subir o limite de tokens não a recuperam; a autodestilação é pior que o Lumi. O gargalo é a múltipla escolha.

### GRPO (RL com recompensa verificável)

Parte do adaptador do SFT e deixa o próprio Lumi gerar; não depende de traços de professor. As iterações rodam em processos separados (a 4090 não comporta o vLLM e o treino juntos): fundir o adaptador, gerar 8 respostas por prompt com o vLLM, treinar (`scripts/run_grpo.sh`).

- **Recompensa:** 1 se a resposta fecha um único `</think>`, termina sozinha, tem o traço em português e a resposta final bate com o gabarito no formato estrito; senão 0. Resposta truncada vale 0 e treina com vantagem negativa.
- **Vantagem:** recompensa menos a média do grupo, dividida pelo desvio-padrão; grupo com recompensas iguais não treina.
- **Perda:** PPO por token com corte assimétrico (0,2 e 0,28), sem KL, normalizada pelos tokens de resposta do minibatch; log-probabilidades antigas calculadas pelo próprio treino antes da atualização; lr 2e-5.
- **Prompts:** 5.612 de matemática (GSM8K-PT `train`, com gabarito) e 3.715 de múltipla escolha (enunciados de ciência traduzidos, letra majoritária como gabarito), metade de cada em cada iteração.
- **Execução:** 256 prompts por iteração, até 2.048 tokens, temperatura 0,8; ~18 min por iteração. Avaliação na validação a cada 3 iterações e completa no fim.
- **Testado:** recompensa, vantagem, perda e seleção de prompts na CPU (`tests/test_grpo.py`), e uma iteração pequena de ponta a ponta na GPU (a razão inicial da perda deu 1,0001). Não testado: as iterações seguintes e o ganho de nota.
- **Riscos:** o gabarito da múltipla escolha é um indicador; grupos sem sinal; custo.

## Requisitos de avaliação

Um candidato só vence se o modo com raciocínio superar o Lumi nos dois modos, sem piorar o modo sem raciocínio nem a segurança.

### Provas

A suíte de raciocínio roda nos dois modos. As 6 notas do cartão são a trava contra regressão e também aparecem no relatório com raciocínio ligado, para a baseline e o vencedor.

| Prova da suíte | Conjunto | Amostra | Licença |
| --- | --- | --- | --- |
| GSM8K-PT | [`Polygl0t/gsm8k-pt`](https://huggingface.co/datasets/Polygl0t/gsm8k-pt), `test` | 1.295 | MIT |
| MMLU-ProX PT | [`li-lab/MMLU-ProX`](https://huggingface.co/datasets/li-lab/MMLU-ProX), `pt`, `test` | 11.759 | MIT |
| ENEM | [`eduagarcia/enem_challenge`](https://huggingface.co/datasets/eduagarcia/enem_challenge) | 1.432 | Não declarada |
| BLUEX | [`nicholasKluge/BLUEX_without_images`](https://huggingface.co/datasets/nicholasKluge/BLUEX_without_images) | 724 | `other` |
| OAB | [`eduagarcia/oab_exams`](https://huggingface.co/datasets/eduagarcia/oab_exams) | 2.210 | Não declarada |

Esses conjuntos só entram na avaliação, nunca no treino. O artigo do Tucano 2 não deixa claro se usou as mesmas versões de ENEM, BLUEX e OAB, então suas notas servem só como ordem de grandeza.

### Requisitos

| ID | Requisito |
| --- | --- |
| A-01 | Medir o Lumi v0.2 antes de treinar, nos dois modos. A referência é a baseline medida aqui, porque o harness muda a nota: o Nemotron 3 Nano 4B tem 40,93 no gráfico do CEIA e 39,63 no repositório do IFEval. |
| A-02 | Rodar os 535 prompts do Portuguese-IFEval com o avaliador oficial e registrar a nota estrita por prompt. Passar a ele só o texto depois de `</think>`, porque a documentação do repositório não diz se a limpeza remove o traço. |
| A-03 | Rodar MMLU PT (ensino médio e graduação), Global MMLU PT, MMMLU PT e INCLUDE PT com versão e amostra fixadas, usando as provas completas da tabela de conjuntos inferidos abaixo. Não usar `Portuguese-MMLU/*` nem as cópias em `cemig-ceia` e `bench-temp-2`: esses conjuntos têm 8.704 e 29.094 questões de teste, o que tornaria a baseline não comparável ao cartão. |
| A-04 | Usar temperatura 0, sem exemplos no prompt, com `enable_thinking` declarado e até 4.096 tokens novos nos dois modos. Sem penalidade de repetição, exceto no IFEval, cujo padrão em Transformers é 1,2. Registrar se isso prejudica os traços. Gerar com o vLLM 0.33.0 (`.venv-vllm`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `max_num_seqs=128`, `max_model_len=8192`), com o mesmo motor e a mesma configuração para a baseline e para todos os candidatos. O vLLM usa batching contínuo e não é reprodutível bit a bit: em duas execuções do mesmo comando, o texto é idêntico em 23% (com raciocínio) a 71% (sem) dos itens, e a nota variou 1,2 ponto em 256 itens (88,3% e 89,4%). Esse ruído não entra no IC 95% do bootstrap por item (A-07), então o relatório registra a variação medida entre duas execuções da baseline. (No HF com lotes fixos, o ruído por composição de lote é ~1,6% das posições; a regra de lote 32 ordenado por prompt vale só se a geração voltar ao HF.) |
| A-05 | Extrair a resposta só depois de `</think>`. Resposta sem `</think>` ou sem alternativa conta como erro, e a taxa de cada caso vai para o relatório. Testar a extração com casos sintéticos antes da baseline (feito: `tests/test_extract.py`, 70 casos). |
| A-06 | No GSM8K-PT, pedir a linha final `#### <número>` e usar `strict-match` como nota. Registrar também `flexible-extract`, porque o artigo do Tucano 2 descreve os dois e não diz qual está nas suas tabelas. Os dois copiam o `gsm8k.yaml` do harness. No dataset, 3 alvos são malformados (`6mph`, `R$ 276.000`, `25%`) e 9 usam `.` de milhar (`2.125`), então a nota estrita não passa de 99,07% para quem responde o inteiro puro; registrar também a leitura PT-BR (`70.000` = 70000) como diagnóstico, sem mudar a nota. |
| A-07 | Calcular a diferença para a baseline com bootstrap pareado por item, dentro de cada prova: 10.000 reamostras e IC de 95% para cada nota, para a Nota de raciocínio e para o Overall. |
| A-08 | Medir a forma do modo com raciocínio: fechamento de `</think>`, resposta não vazia, traço não vazio, idioma do traço, truncamento em 4.096 tokens, tokens por resposta (mediana e p95) e, no GSM8K-PT, se o último número do traço é a resposta. |
| A-09 | Medir recusas nos dois modos em 200 prompts de `harmfull-no-reasoning`, excluindo o prompt que também aparece em `reasoning`. Revisar à mão 50 traços desses prompts, porque o traço pode ter conteúdo nocivo mesmo quando a resposta recusa. |
| A-10 | Selecionar candidatos com dois conjuntos de validação: 1.000 itens do GSM8K-PT `train` sem 8-gramas em comum com o treino final, e 385 questões de Global MMLU PT `dev` (285), MMLU-ProX PT `validation` (70) e INCLUDE PT `validation` (30). Na checagem inicial, 43,6% do `train` tinha 8-gramas em comum com `reasoning`, então a nota serve para comparar candidatos, não como estimativa absoluta. Atualização: mesmo sem 8-gramas em comum, cerca de um terço dos itens do GSM8K estava duplicado em outra tradução nos dados de treino; as comparações usam o subconjunto limpo (A-15). |
| A-11 | Avaliar sempre o modelo fundido em BF16. |
| A-12 | Rodar as provas de teste apenas em: Lumi (suíte e 6 notas, nos dois modos); os 2 candidatos finais (suíte nos dois modos e 6 notas sem raciocínio); e o vencedor (6 notas com raciocínio). |
| A-13 | No vencedor, repetir a validação do GSM8K-PT com `temperature=1.0` e `top_p=0.95`, a recomendação do Nemotron para raciocínio, em 3 seeds. O resultado entra no relatório, sem afetar a seleção. |
| A-14 | Registrar pico de memória, tempo por época e tokens por segundo em cada execução. Medir a memória com `torch.cuda.max_memory_allocated()` e `max_memory_reserved()`. Registrar também os processos do `nvidia-smi` no início e no fim, porque outro usuário na GPU pode distorcer o tempo e causar falta de memória. |
| A-15 | Comparar candidatos no GSM8K-PT de validação só nos 676 itens sem duplicata do treino (`data/val_gsm8k_clean_ids.json`, gerado por `scripts/clean_validation.py` contra o GigaVerbo e o adalberto); o teste do GSM8K-PT está limpo. Os 1.000 itens continuam registrados. |
| A-16 | Reportar a comparação em quatro sistemas com `scripts/report_compare.py`: Nemotron 3 Nano 4B com raciocínio, Lumi com raciocínio, nosso Lumi com raciocínio em inglês (teste posterior) e nosso Lumi com raciocínio em português. Os dois efeitos se leem por diferença: ganho do treino = nosso inglês menos o Lumi; custo do português = nosso inglês menos nosso português. |
| A-17 | Avaliar o Nemotron 3 Nano 4B de referência primeiro na validação (GSM8K limpo e múltipla escolha, nos dois modos; template, prompts e vocabulário conferidos como idênticos aos do Lumi) e, quando a GPU permitir, na suíte de teste, conforme [comparacao.md](comparacao.md). Nenhuma resposta dele entra no treino. |

**Custo estimado:** com o vLLM, o GSM8K-PT inteiro (1.295 itens) leva 233 s com raciocínio e 116 s sem (5,6 e 11,2 itens/s). Extrapolando para as 11 provas (73.095 itens por modo), a baseline do Lumi nos dois modos leva cerca de 5,5 h e a matriz de A-12 cerca de 14 h. É extrapolação de uma prova de matemática; as de múltipla escolha podem ter respostas mais curtas ou mais longas, e a primeira rodada mede isso. No HF as mesmas contas davam ~56 h e ~136 h, porque o `generate` em lote espera a saída mais longa. Medições em [discovery.md](discovery.md).

**Conjuntos e amostras do cartão (inferência)**

O cartão não informa conjuntos, amostras, harness nem parâmetros, e o CEIA ainda não confirmou o que está abaixo. As amostras vêm dos gráficos: nas provas em português, as barras não foram arredondadas, e cada nota é uma fração exata `acertos/n` nos 9 resultados (os 8 modelos de `pt_benchmarks.svg` e o Lumi v0.1). O `n` confirma o tamanho, não o conteúdo nem a revisão.

| Prova | Conjunto inferido | Amostra | Observação |
| --- | --- | --- | --- |
| MMLU PT, ensino médio | `mmlu-pt/mmlu-pt-high-school-full` | 7.187 de 7.233 | Sem as 46 questões com alternativa vazia, bate exato |
| MMLU PT, graduação | `mmlu-pt/mmlu-pt-undergraduate-full` | 19.318 de 19.347 | Sem as 28 com alternativa vazia, sobra 1 questão a mais; a causa não é certa, talvez a do BNDES com a alternativa "NULL" |
| Global MMLU PT | `CohereLabs/Global-MMLU`, `pt`, `test` | 14.042 | Prova completa |
| MMMLU PT | `openai/MMMLU`, `PT_BR`, `test` | 14.042 | Prova completa |
| INCLUDE PT | `CohereLabs/include-base-44`, `Portuguese`, `test` | 551 | Prova completa |
| IFEval PT estrito | Portuguese-IFEval, conjunto PT | 535 | Prova completa; as notas não coincidem com as rodadas publicadas nos 6 modelos em comum, então não vêm delas |

**Critério de vitória**

- IC 95% da diferença da Nota de raciocínio inteiramente acima de 0, com raciocínio ligado, contra o Lumi sem raciocínio e contra o Lumi com raciocínio.
- Na suíte, ao menos 98% das respostas fecham `</think>` e trazem texto depois, e ao menos 90% dos traços estão em português.
- Sem raciocínio, queda de até 1 ponto no Overall, sem nenhuma nota com IC 95% da diferença inteiramente abaixo de 0.
- Taxa de recusa, nos dois modos, não inferior à do Lumi sem raciocínio, salvo decisão registrada.

Se os dois candidatos vencerem, fica o de maior Nota de raciocínio. Se o IC 95% da diferença entre eles contiver 0, fica o LoRA, que é mais simples e mais rápido (com 1 seed, esse IC é o do bootstrap por item e não inclui a variação entre seeds).

O Overall com raciocínio ligado entra no relatório sem trava, porque o custo em formato é conhecido: o Tucano Think perdeu 10 pontos no IFEval-PT contra o Instruct.

As contas abaixo são estimativas. Com 25% a 40% das questões mudando de acerto entre dois sistemas, o IC 95% fica em torno de ±1,1 a ±1,4 na Nota de raciocínio e de ±2,7 a ±3,4 no GSM8K-PT, então um ganho de 3 pontos no GSM8K-PT está no limite do detectável. Na trava de 6 notas, a chance de reprovar sem perda real é de cerca de 14% (1 - 0,975⁶, supondo notas independentes).

Se nenhum candidato cumprir os quatro itens, o relatório declara que não houve vitória e indica a causa provável por prova e por tipo de dado. Se só a trava do modo sem raciocínio falhar, o relatório diz isso e a escolha entre adotar o modelo fundido ou usar o adaptador só com raciocínio ligado fica para decisão. Se o ganho aparecer só em matemática, o relatório também deve dizer isso explicitamente.

## Arquitetura do repositório e configuração

O repositório separa dados, treino e avaliação, e cada execução nasce de um YAML versionado.

```text
Dados de SFT (reasoning filtrado + parcela sem raciocínio)
  -> LoRA (BF16) | DoRA (BF16)
  -> Fusão sobre o Lumi em BF16
  -> Comparação com a baseline do Lumi nos dois modos, no mesmo harness
     sim: entregar adaptador, modelo e relatório
     não: relatório sem vitória, com causa provável por prova
```

```text
lumi-finetune/
├── env.sh                        # caches, nvcc e alocador da GPU
├── pyproject.toml                # dependências com lockfile
├── configs/
│   ├── base.yaml
│   └── arms/{lora,dora}.yaml
├── src/lumi_ft/
│   ├── modeling/load.py          # carga BF16, eos e pad
│   ├── modeling/peft_setup.py    # alvos e checagens
│   ├── data/gigaverbo.py         # seleção, filtros e renderização
│   ├── data/decontaminate.py     # k-gramas contra as provas
│   ├── train/sft.py
│   ├── train/merge.py            # fusão sobre BF16
│   └── eval/{parse,ifeval,mcq,gsm8k,safety,bootstrap,report}.py
├── tests/
├── third_party/portuguese-ifeval/ # fixado por commit
├── data/                         # fora do Git
├── runs/                         # adaptadores, logs e métricas
└── reports/
```

```yaml
# configs/arms/dora.yaml
extends: ../base.yaml
arm: dora
model:
  name: ceia-br/lumi-4b
  revision: 27b30f40b930fd05cee5e1c1f8790f70fa5b816d  # v0.2
  dtype: bfloat16
tokenizer:
  pad_token: "<unk>"
data:
  reasoning_token_share: 0.75
generation:
  eos_token_id: [2, 11]
  max_new_tokens: 4096
  temperature: 0
eval:
  enable_thinking: [true, false]
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  use_dora: true
```

O projeto roda na máquina abaixo, e a fase 0 prova o que depende dela.

| Recurso | Valor |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24 GiB de VRAM (24.564 MiB), uma só |
| CPU | Intel Core i9-14900K, `x86_64`, 24 núcleos e 32 threads |
| Memória | 125 GiB de RAM, separados da VRAM, e 11 GiB de swap |
| Disco | 1,0 TB livres em `/home` (3,3 TB no total) |
| Software | Ubuntu 22.04.5, driver 580.95.05 (CUDA até 13.0), GCC 11.4, uv 0.13 |
| Acesso | Máquina com vários usuários, sem `sudo` e sem acesso ao Docker |

O Python do sistema é o 3.10.12, abaixo do 3.11 exigido pelas versões atuais de Transformers e TRL. O projeto usa o Python 3.12 gerenciado pelo uv, já instalado na máquina.

Os pacotes seguem a tabela abaixo.

| Componente | Versão |
| --- | --- |
| Transformers | ≥ 5.8.1, a versão do `config.json` do Lumi. Fixado no lock: 5.22.2 |
| PEFT | ≥ 0.20. Fixado no lock: 0.21.2 |
| TRL | ≥ 1.17. Fixado no lock: 1.17.1 |
| PyTorch | 2.14.1+cu130, fixado no `pyproject.toml` e no `uv.lock`. A 2.15.0 só tem builds cu132 e cu134, acima do CUDA 13.0 do driver |
| `nvcc` | `cuda-nvcc` 13.0.88 do conda-forge, instalado com micromamba dentro de `barros` |
| vLLM 0.33.0 | Só para a avaliação, no `.venv-vllm` separado (torch 2.14.0, transformers 5.20.2), instalado por `scripts/install_vllm.sh`; rodar com `VLLM_USE_FLASHINFER_SAMPLER=0` |
| `mamba-ssm` 2.3.2.post1 e `causal-conv1d` 1.7.0 | Só têm código-fonte no PyPI, então são compilados com o `nvcc` acima por `scripts/build_kernels.sh` e ficam fora do lockfile |

Os `setup.py` dos dois kernels usam `CUDAExtension` e leem a versão do `nvcc`. A ordem importa: instalar o PyTorch primeiro e compilar com `--no-build-isolation`, para os kernels não serem compilados contra outro PyTorch.

A rota foi executada na fase 0 e funcionou, com duas correções. O `mamba-ssm` compila com `-std=c++17`, mas os headers do PyTorch 2.14 exigem C++20 (o g++ 11 da máquina basta). E o `setup.py` dele gera código para 9 arquiteturas de GPU, então a cópia local patchada gera só a sm_89. O patch está em `scripts/patches/mamba-ssm-sm89-cxx20.patch`. O binário só roda em GPUs sm_89 (RTX 4090, L40, RTX 6000 Ada); em outra GPU, é preciso ajustar o patch e recompilar. O `causal-conv1d` foi compilado sem patch, para todas as arquiteturas. Os dois módulos importam e rodaram na GPU, mas `is_fast_path_available` no modelo ainda não foi confirmado.

O `env.sh` define `CUDA_HOME`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, para reduzir a fragmentação com sequências de tamanhos diferentes, e `UV_NO_SYNC=1`, para o `uv run` não remover os kernels, que não estão no lockfile. Para alinhar o ambiente ao lock, usar `uv sync --frozen --inexact`.

Jobs longos rodam em tmux, desacoplados do terminal, com `scripts/detach.sh` (opção `--gpu` recusa lançar se a GPU estiver ocupada).

Cada execução grava em `runs/<id>/` o YAML resolvido, o commit, a seed, o hash dos dados, as métricas e o adaptador.

## Fases e critérios de aceite

A execução tem seis fases e para em cada gate, onde os resultados são apresentados antes de seguir.

### Regras de execução

Estas regras valem para todas as fases.

1. Parar em cada gate e apresentar o relatório.
2. Pedir aprovação antes de qualquer execução com mais de 1 hora de GPU.
3. Nunca usar prompts de benchmark no treino; os testes devem bloquear isso.
4. Nunca alterar metas, métricas ou limiares para um candidato passar.
5. Não publicar nada no Hugging Face sem pedido explícito.
6. Se o código do Lumi divergir deste PRD, registrar a divergência e perguntar antes de seguir.
7. Declarar `enable_thinking` em todo treino e avaliação, porque o padrão do template é ligado.
8. Carregar `env.sh` antes de baixar, instalar ou executar qualquer coisa, para manter os caches dentro de `barros`.
9. Antes de cada execução com GPU, conferir no `nvidia-smi` se ela está livre, porque a máquina é compartilhada. Nunca encerrar processos de outros usuários: se a GPU estiver ocupada, avisar e esperar.

### Fase 0: fundação e baseline

- [x] Criar repositório, lockfile e ambiente com Python 3.12, instalando o `nvcc` sem `sudo` e compilando `mamba-ssm` e `causal-conv1d` contra o PyTorch fixado.
- [x] Confirmar que `torch.cuda.is_available()` e `torch.cuda.is_bf16_supported()` são verdadeiros.
- [x] Confirmar que `is_fast_path_available` é verdadeiro, porque o código cai no caminho lento só com um aviso.
- [x] Baixar o Lumi v0.2 no commit `27b30f40` e ler `modeling_nemotron_h.py`.
- [x] Comparar logits do código do repositório e da implementação nativa em um batch fixo, e escolher uma.
- [x] Listar `named_modules()` e confirmar `TARGET_REGEX`.
- [x] Provar nos dois braços que todo parâmetro `lora_*` treinável recebe gradiente não nulo após 1 passo.
- [x] Provar que um adaptador recém-criado não altera os logits.
- [x] Provar que adaptador ativo e modelo fundido geram logits equivalentes.
- [x] Provar que a geração para em `<|im_end|>` com `eos_token_id=[2, 11]`.
- [x] Testar a extração de resposta conforme A-05 e A-06.
- [x] Medir pico de memória, maior batch viável e tokens por segundo por braço, conforme A-14, e fixar o batch do treino no pior caso (DoRA, `r=32`, sequências mais longas).
- [x] Medir o maior batch viável na geração, no pior caso de 4.096 tokens novos, porque o cache cresce com a resposta. Resultado: 56 com prompt de 1.024 tokens, 64 com 512 e 96 com 256; o limite é o estado do Mamba (~82 MiB por sequência) e o prefill, não o cache de atenção.
- [x] Instalar o vLLM em ambiente separado e validar contra o HF no GSM8K-PT (piloto de 256 itens; ver A-04).
- [ ] Medir a baseline do Lumi nos dois modos, conforme A-01 a A-09, inclusive a forma do raciocínio. Feito só o GSM8K-PT inteiro (88,9% com raciocínio, 85,2% sem, no vLLM); faltam as outras 10 provas e A-09.

**Gate:** todos os testes passam, e a baseline sem raciocínio está a até 1 ponto do cartão ou a diferença está documentada. Se o Lumi com raciocínio já cumprir as metas de forma ou de nota, apresentar isso antes de seguir, porque o problema a resolver muda.

### Fase 1: dados

- [x] Implementar seleção, filtros, renderização e remoção de sobreposição, conforme D-01 a D-09.
- [x] Criar testes comparando exemplos renderizados com o template do Lumi, nos dois modos.
- [x] Provar que o prompt é prefixo exato do exemplo nos dois modos (o treino usa laço próprio, sem TRL).
- [x] Verificar que `labels` cobrem `</think>` e `<|im_end|>`, e que tudo fora da resposta recebe `-100`.
- [x] Montar os conjuntos de validação conforme A-10 e registrar os hashes conforme D-11 (`reports/data_manifest.json`, `reports/val_manifest.json`).
- [x] Relatar quantas linhas cada filtro removeu, por `task_type` (discovery.md §13).
- [ ] Fazer a revisão humana de D-10.

**Gate:** 100 amostras renderizadas revisadas, sem erro de template ou máscara, e revisão de D-10 concluída.

### Fase 2: viabilidade

Um treino pequeno mostra se o caminho funciona antes de gastar GPU com a matriz completa.

- [x] Treinar 1 LoRA com `r=16` e taxa 1e-4 em 10.000 exemplos com raciocínio, estratificados por `task_type`, mais 25% de tokens sem raciocínio (486 passos, 21,7 min).
- [x] Comparar adaptador e Lumi, nos dois modos, nos conjuntos de validação de A-10, com IC 95% da diferença (`reports/compare_viab_vs_base.md`).
- [x] Medir a forma do raciocínio conforme A-08 (fecha `</think>`, idioma do traço, truncamento, tokens).
- [ ] Ler 50 respostas e registrar o que os números não mostram.

**Gate:** ao menos 95% das respostas fecham `</think>`, ao menos 90% dos traços estão em português e a nota na validação do GSM8K-PT tem IC 95% da diferença acima de 0 contra o Lumi sem raciocínio. Se falhar, parar e decidir entre mais dados, traços gerados e verificados (decidido em 2026-11-07: dentro deste PRD, ver a fase 2b) ou encerrar.

**Resultado de 2026-11-07: o gate falhou na nota e passou na forma.** Forma: 99,7% das respostas fecham `</think>` e trazem texto (99,2% na múltipla escolha; o Lumi: 95,4% e 86,2%), 99,7% dos traços em português (o Lumi: 0%) e truncamento de 0,3% (o Lumi: 4,7% e 13,8%). Nota com raciocínio contra o Lumi sem raciocínio, GSM8K-PT validação: 71,5% contra 81,4% no strict (−9,9, IC 95% [−12,7, −7,1]) e 80,3% contra 81,5% no flexível (−1,2, [−3,4, +1,0]); múltipla escolha: 43,4% contra 63,9% (−20,5) no strict e 53,0% contra 63,4% (−10,4) no flexível. Parte da queda do strict é formato (a resposta sai em prosa depois do `</think>`, sem `#### N` ou `Resposta: X`), mas não toda: com a leitura mais generosa (qualquer letra no texto), a múltipla escolha fica em 55,1% contra 69,6% do Lumi. O modo sem raciocínio também piorou (GSM8K strict −20,2, flexível −3,9; múltipla escolha flexível −6,8). Decisão do usuário em 2026-11-07: ampliar o escopo (fase 2b); ver [discovery.md](discovery.md) §15.

### Fase 2b: dados de professor e GRPO (novo escopo)

Entre o gate da fase 2 e as varreduras: descobrir uma receita que passe o Lumi também na múltipla escolha, ainda com LoRA `r=16`.

- [x] Remover da validação as duplicatas em outra tradução e passar a comparar nos 676 itens limpos (A-15).
- [x] Treinar com traços do Qwen3-30B (`adalberto`): matemática passa o Lumi, múltipla escolha perde 5 a 7 pontos.
- [x] Testar a escala do adaptador (0,5 e 0,75) e o limite de 2.048 tokens: nenhum recupera a múltipla escolha.
- [x] Testar a autodestilação do Lumi (`selfdist`): pior que o Lumi.
- [x] Levantar datasets de raciocínio em português e no estilo do Nemotron; descartar o Bode-reasoning (100% de 8-grama nas provas).
- [x] Piloto de tradução do raciocínio pelo Lumi: 86% aceito, 94,7% dos números preservados (média), vocabulário técnico com erros.
- [x] Montar os pacotes da Thor (Qwen3-30B e Nemotron 3 Nano 30B-A3B) e a entrega em `thor_out.tar.gz` e `thor2_out.tar.gz`.
- [x] Escrever o GRPO e passar o teste de fumaça.
- [ ] Avaliar o DoRA com os dados do `adalberto`, o `nemo_mix`, o GRPO (`grpo1`) e o lr 5e-5.
- [ ] Avaliar o Nemotron 3 Nano 4B de referência (A-17) e montar a tabela de A-16.
- [ ] Processar as entregas da Thor, traduzir e treinar com elas.
- [ ] Medir o diagnóstico português x inglês no MMLU-ProX (decide se o CPT reabre).
- [ ] Rodar a validação nas provas de teste do candidato escolhido, junto com o Lumi (A-12).

**Gate:** um candidato com Nota de raciocínio acima do Lumi em validação nos dois modos, ou o relatório declarando que não houve vitória, com a causa provável por prova e por tipo de dado. Se o ganho ficar só em matemática, dizer isso.

### Fase 3: varreduras

- [ ] Rodar a varredura de taxa de aprendizado.
- [ ] Comparar as misturas 75/25 e 100/0, inclusive no modo sem raciocínio.
- [ ] Comparar nota na validação, forma do raciocínio, loss, memória e tempo.

(O piloto de ranks foi cortado por prazo; ver a matriz de experimentos.)

**Gate:** taxa e mistura escolhidas, com justificativa.

### Fase 4: finalistas

- [ ] Rodar as 2 execuções finalistas (LoRA e DoRA, `r=16`, 2 épocas).
- [ ] Escolher o melhor adaptador de cada método na validação.
- [ ] Fundir os 2 candidatos sobre o Lumi em BF16 e repetir a checagem de equivalência.

**Gate:** 2 candidatos fundidos e checados.

### Fase 5: avaliação e entrega

- [ ] Rodar o benchmark conforme A-02 a A-09 e a matriz de A-12.
- [ ] Rodar a análise de sensibilidade de A-13 no vencedor.
- [ ] Gerar `reports/final.md` com tabela por método, por prova e por modo, IC 95%, forma do raciocínio, memória e tempo.
- [ ] Registrar qual método venceu ou declarar que nenhum venceu.
- [ ] Rascunhar um model card com base, dados, limitações, o uso em um só turno e o aviso de que o modo foi treinado fora do pipeline oficial do CEIA.

**Gate:** critério de vitória avaliado e relatório reproduzível a partir de `runs/`.

## Riscos, restrições e decisões em aberto

O maior risco é de resultado: o Tucano 3.7B Think usou os mesmos dados e só ganhou 2,89 pontos no GSM8K-PT, e ainda perdeu nas outras provas. O maior risco técnico continua sendo o código próprio do Lumi se comportar de forma diferente da implementação nativa.

| Risco | Efeito | Mitigação |
| --- | --- | --- |
| Traços curtos e sem checagem limitam o ganho | Modelo imita o formato sem acertar mais | Gate da fase 2 antes da matriz completa; sem ganho, decidir sobre novos dados |
| Dados concentrados em matemática | Ganho só no GSM8K-PT | Relatório por prova; H3 explícita |
| Código do Lumi diverge do nativo | Alvos errados, logits errados ou geração quebrada | Resolvido na fase 0: as duas calculam o mesmo sem padding; o código do repositório erra com padding à esquerda e quebra no `generate`, então o projeto usa a nativa |
| Linhas zeradas na `in_proj` quebram o DoRA | NaN nos logits e nos gradientes | Patch local do DoRA, testado em CPU e no modelo; ver seção própria |
| Kernels não compilam (sem `nvcc` nativo e sem `sudo`) | Treino no caminho lento | Rota do conda-forge provada na fase 0 (compilação e importação). Falta confirmar o caminho rápido no modelo; se falhar, usar o caminho PyTorch nos dois braços |
| 24 GiB não bastam para o batch desejado | Falta de memória ou treino mais lento | Batch físico medido no pior caso, acumulação de gradiente e, se preciso, calcular `lm_head` só nas posições com label |
| Outro usuário ocupa a GPU | Falta de memória, execução interrompida ou tempo distorcido | Regra 9, processos registrados em A-14, checkpoint por época e respostas salvas por item |
| Modo sem raciocínio regride | Modelo fundido piora o uso atual | Parcela de 25%, teste da mistura e trava do critério de vitória |
| Traço não fecha ou entra em loop | Resposta perdida e custo alto | D-08, parada correta, limite de 4.096 tokens e métricas de A-08 |
| Traço em inglês | Perde o objetivo de raciocinar em português | D-09, A-08 e gate da fase 2; ver decisão em aberto |
| Harness erra no modo com raciocínio | Nota falsa, como pode ter ocorrido no BELEBELE do Qwen3-4B | Testes de extração de A-05 e taxas de erro no relatório |
| Baixo rank não basta | Meta não atingida | Testar `r=32` e, se falhar, abrir PRD de treino completo |
| Traço expõe conteúdo nocivo ou reduz recusas | Perda de segurança | A-09, com revisão humana e trava do critério de vitória |
| Sobreposição parafraseada com as provas | Nota inflada | D-05 e revisão dos casos limítrofes; validação usada só para comparar candidatos |
| Ajuste indireto ao benchmark | Resultado otimista | Seleção na validação e benchmark só nos finalistas |
| CEIA lança o modo oficial | Trabalho perde utilidade prática | Perguntar o prazo antes da fase 2 |
| Licença impede derivados | Modelo não pode ser publicado | Confirmar antes de qualquer publicação |
| Estilo do professor diferente do Lumi | O modelo aprende um modo de raciocinar que não é o dele e perde na múltipla escolha | Professor da mesma família (Nemotron) e traços no estilo do GPT-OSS; comparar estilo e nota |
| Tradução do raciocínio com erros técnicos | Terminologia errada nos exemplos | D-15, leitura de amostras e troca do tradutor (Qwen3-30B na Thor) se o treino mostrar que a tradução limita |
| Gabarito da múltipla escolha por maioria dos traços do R1 | Recompensa e rótulo errados em parte dos itens | Exigir maioria de 75%; usar só como prompts de GRPO e SFT, nunca na avaliação |
| Dados de professor sem licença ou gerados por modelo comercial | Modelo não publicável | D-16; o adalberto fica só em experimento |
| Domínio dos dados da NVIDIA (química e física de pós-graduação) mais estreito que as provas | Pouca transferência para ENEM, OAB e conhecimentos gerais | Misturar com matemática, incluir as questões GPQA (mais variadas) e medir por prova |
| GRPO instável ou caro | Sem ganho e horas de GPU gastas | Teste de fumaça, avaliação a cada 3 iterações, cancelamento a qualquer momento |
| Fila de jobs reencadeada errado | Jobs disparam com a GPU ocupada e falham | Ao inserir um job, matar os dependentes do mais distante para o mais próximo e conferir `grep has-session runs/jobs/*.run.sh` |
| Dados parcialmente copiados das provas | Nota inflada por contaminação | D-05 e D-12; o Bode-reasoning foi descartado por isso |

### Decisões em aberto

- [ ] Qual harness e quais parâmetros o CEIA usou no cartão (temperatura, few-shot, prompts, limite de tokens e raciocínio nos modelos comparados)? O cartão não documenta nada disso. Pedir também o notebook `colab_graficos_llms-2.ipynb`, citado na primeira versão do cartão.
- [x] Vale usar [`adalberto-temp/gsm8k-sft-thinking-pt`](https://huggingface.co/datasets/adalberto-temp/gsm8k-sft-thinking-pt)? Decidido em 2026-11-07: sim, só como experimento, com o filtro D-12 e a validação limpa (A-15). Sem licença declarada, não entra em nada que seja publicado sem decisão registrada.
- [ ] Qual é o ponto de partida do GRPO? Hoje é o adaptador `adalberto`; se o `nemo_mix` ou o DoRA vencerem, trocar antes de o `grpo1` começar.
- [ ] O raciocínio em português é requisito fixo? A comparação em quatro colunas (A-16) mede o custo do idioma; a decisão fica para depois do teste com raciocínio em inglês, conduzido pelo usuário.
- [ ] Reabrir o CPT? Depende do diagnóstico português x inglês no MMLU-ProX.


### Fontes

Nesta versão, o `chat_template.jinja` e as configs do Lumi foram lidos por completo, e o `modeling_nemotron_h.py` só nos trechos de módulos e kernels. Do artigo do Tucano 2, foram lidas as seções sobre dados, pós-treino e avaliação, com as tabelas dos modelos Think. Os traços foram avaliados só em amostras pequenas, sem revisão sistemática. Na adaptação à RTX 4090, os dados da máquina vieram dela própria; os `setup.py` de `mamba-ssm` 2.3.2.post1 e `causal-conv1d` 1.7.0 foram lidos sem executar, e as versões vieram do PyPI, do índice do PyTorch e do conda-forge. Nada foi instalado nem medido na GPU.

- [Cartão do CEIA Lumi-4B](https://huggingface.co/ceia-br/lumi-4b), incluindo a [versão antiga](https://huggingface.co/ceia-br/lumi-4b/blob/1c8928e1/README.md) com o MMLU-ProX PT
- [Cartão do Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- [GigaVerbo-v2 SFT](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-sft) e [Preferences](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-preferences)
- [Tucano 2 Cool](https://arxiv.org/abs/2603.03543) e [harness do Tucano 2](https://github.com/Polygl0t/lm-evaluation-harness)
- [Portuguese-IFEval](https://github.com/ark-aiml-research/Portuguese-IFEval)
- [GSM8K-PT](https://huggingface.co/datasets/Polygl0t/gsm8k-pt), [MMLU-ProX](https://huggingface.co/datasets/li-lab/MMLU-ProX), [ENEM](https://huggingface.co/datasets/eduagarcia/enem_challenge), [BLUEX](https://huggingface.co/datasets/nicholasKluge/BLUEX_without_images) e [OAB](https://huggingface.co/datasets/eduagarcia/oab_exams)
- [MMLU PT, ensino médio](https://huggingface.co/datasets/mmlu-pt/mmlu-pt-high-school-full) e [graduação](https://huggingface.co/datasets/mmlu-pt/mmlu-pt-undergraduate-full)
- [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), [MMMLU](https://huggingface.co/datasets/openai/MMMLU) e [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44)
- [`nemotron_h` no Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
- [DoRA no PEFT](https://huggingface.co/docs/peft/main/package_reference/lora_variant_dora)
- [Nemotron 3 Nano 4B no Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/nemotron/nemotron3-nano-4b.html)
- [Guia do Nemotron 3 na Unsloth](https://unsloth.ai/docs/models/tutorials/nemotron-3)
- [Nemotron-SFT-Multilingual-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2), [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1), [Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) e [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)
- [Nemotron 3 Nano 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) e o [blog da NVIDIA sobre o Nemotron 3 Nano](https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models)
- [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507), [`open-r1/Mixture-of-Thoughts`](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) e [Long Chain-of-Thought Reasoning Across Languages](https://arxiv.org/html/2508.14828v1)
