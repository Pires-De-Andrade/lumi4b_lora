# PRD: SFT do Lumi-4B com LoRA ou DoRA para adicionar raciocínio

Nov 6, 2026 · @gustavobarrosss

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

Este projeto cobre apenas um SFT por adaptador sobre os pesos do Lumi-4B, comparando LoRA e DoRA no modo com raciocínio e na preservação do modo sem raciocínio.

**Dentro do escopo**

- Preparar o subset `reasoning` do GigaVerbo-v2 e uma parcela de exemplos sem raciocínio.
- Treinar e comparar LoRA e DoRA.
- Medir o Lumi e os candidatos nos dois modos, no mesmo harness.
- Entregar o adaptador vencedor, o modelo fundido em BF16 e o relatório.

**Fora do escopo**

- QLoRA e QDoRA: nesta máquina a memória não é limite, e a quantização não promete ganho de qualidade.
- APO, RL e RLVR. São os próximos passos naturais se o SFT não bastar, e cada um pede outro PRD.
- Gerar traços novos. Se a fase 2 mostrar que os dados limitam o resultado, isso vira outro PRD.
- Melhorar o IFEval: aqui ele é só uma trava contra regressão.
- Raciocínio em conversa com vários turnos: todos os 78.249 exemplos de `reasoning` têm um só turno. O model card registra essa limitação.
- Pré-treino continuado dos pesos completos, tool calling, contexto longo, exportação GGUF e publicação no Hugging Face.

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

## Requisitos de treino

Os dois métodos usam os mesmos módulos-alvo e diferem apenas no `use_dora`. O LoRA é a referência; o DoRA testa se há ganho em rank baixo, com mais custo de tempo.

| Braço | Pesos-base | `use_dora` | Kernel Mamba | Papel |
| --- | --- | --- | --- | --- |
| LoRA | BF16 | `False` | Kernels permitidos | Referência |
| DoRA | BF16 | `True` | Kernels permitidos | Testar ganho em rank baixo |

Os kernels são os pacotes `mamba-ssm` e `causal-conv1d`. O código do Lumi só ativa o caminho rápido com os dois instalados e, sem eles, cai no caminho PyTorch apenas com um aviso.

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

Esses nomes existem tanto na implementação nativa do Transformers quanto no código próprio do Lumi, usado via `auto_map`. A fase 0 ainda precisa confirmá-los com `named_modules()`.

### Carga e tokenizer

- `eos_token_id=[2, 11]` em toda geração. Sem isso, o `generate` não para em `<|im_end|>`, porque as configs do modelo só listam o id 2.
- `pad_token="<unk>"` (id 0, o `pad_token_id` do `config.json`). O `tokenizer_config.json` não define esse token, e usar `<|im_end|>` como padding pode mascarar o fim real do turno nos labels.
- Pesos em BF16: cerca de 8,0 GB nos dois braços. Os logits custam cerca de 0,5 GB por 1.000 tokens no batch, antes dos gradientes (131.072 × 4 bytes por token). A fase 0 mede os valores reais.

### Hiperparâmetros iniciais

| Parâmetro | Valor | Origem |
| --- | --- | --- |
| `r` | 8, 16 e 32 | NVIDIA usa 8; mudar o comportamento pode exigir mais |
| `lora_alpha` | 2 × `r` | NVIDIA usa 16 e 32 com `r=8` |
| `lora_dropout` | 0 | NVIDIA usa 0, e o DoRA fica mais rápido assim |
| Taxa de aprendizado | 1e-4, varrendo 5e-5 e 2e-4 | NVIDIA usa 1e-4 em LoRA |
| Otimizador | AdamW com cosseno e warmup de 3% | AdamW e cosseno do Tucano 2; warmup escolhido no projeto (o Tucano 2 usa 10%) |
| Sequência | 2.048 tokens, sem truncar | Acima do máximo de 1.000 tokens dos dados |
| Épocas | Até 3, com parada antecipada | Escolha do projeto; o Tucano 2 usou 5 |
| Gradient checkpointing | `use_reentrant=False`, `use_cache=False` | Requisito de memória |

### Matriz de experimentos

| Etapa | Execuções | Dados | Saída |
| --- | --- | --- | --- |
| Viabilidade | 1 (LoRA, `r=16`) | 10.000 exemplos com raciocínio, mais 25% sem | Decisão sobre seguir |
| Varredura de taxa | 3 (LoRA, `r=16`) | 20% | Taxa fixada |
| Mistura | 2 (LoRA, `r=16`): 75/25 e 100/0 | 20% | Mistura fixada |
| Piloto | 6 (2 métodos × 3 ranks) | 20% | Melhor rank por método |
| Finalistas | 6 (2 métodos × 3 seeds) | 100% | Melhor adaptador por método |

São 18 execuções e, no pior caso, cerca de 1,2 bilhão de tokens de treino. A fase 0 mede a velocidade e transforma isso em horas de GPU.

A seleção usa os conjuntos de validação de A-10, nunca as provas de teste. A loss não basta, porque não mede se o raciocínio melhora a resposta.

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
| A-04 | Usar temperatura 0, sem exemplos no prompt, com `enable_thinking` declarado e até 4.096 tokens novos nos dois modos. Sem penalidade de repetição, exceto no IFEval, cujo padrão em Transformers é 1,2. Registrar se isso prejudica os traços. |
| A-05 | Extrair a resposta só depois de `</think>`. Resposta sem `</think>` ou sem alternativa conta como erro, e a taxa de cada caso vai para o relatório. Testar a extração com casos sintéticos antes da baseline. |
| A-06 | No GSM8K-PT, pedir a linha final `#### <número>` e usar `strict-match` como nota. Registrar também `flexible-extract`, porque o artigo do Tucano 2 descreve os dois e não diz qual está nas suas tabelas. |
| A-07 | Calcular a diferença para a baseline com bootstrap pareado por item, dentro de cada prova: 10.000 reamostras e IC de 95% para cada nota, para a Nota de raciocínio e para o Overall. |
| A-08 | Medir a forma do modo com raciocínio: fechamento de `</think>`, resposta não vazia, traço não vazio, idioma do traço, truncamento em 4.096 tokens, tokens por resposta (mediana e p95) e, no GSM8K-PT, se o último número do traço é a resposta. |
| A-09 | Medir recusas nos dois modos em 200 prompts de `harmfull-no-reasoning`, excluindo o prompt que também aparece em `reasoning`. Revisar à mão 50 traços desses prompts, porque o traço pode ter conteúdo nocivo mesmo quando a resposta recusa. |
| A-10 | Selecionar candidatos com dois conjuntos de validação: 1.000 itens do GSM8K-PT `train` sem 8-gramas em comum com o treino final, e 385 questões de Global MMLU PT `dev` (285), MMLU-ProX PT `validation` (70) e INCLUDE PT `validation` (30). Na checagem inicial, 43,6% do `train` tinha 8-gramas em comum com `reasoning`, então a nota serve para comparar candidatos, não como estimativa absoluta. |
| A-11 | Avaliar sempre o modelo fundido em BF16. |
| A-12 | Rodar as provas de teste apenas em: Lumi (suíte e 6 notas, nos dois modos); os 2 candidatos finais (suíte nos dois modos e 6 notas sem raciocínio); e o vencedor (6 notas com raciocínio). |
| A-13 | No vencedor, repetir a validação do GSM8K-PT com `temperature=1.0` e `top_p=0.95`, a recomendação do Nemotron para raciocínio, em 3 seeds. O resultado entra no relatório, sem afetar a seleção. |
| A-14 | Registrar pico de memória, tempo por época e tokens por segundo em cada execução. Medir a memória com `torch.cuda.max_memory_allocated()`, porque o `nvidia-smi` não a informa nesta máquina. |

**Custo estimado:** um sistema com raciocínio nas 11 provas (73.095 itens, cerca de 500 tokens por resposta) gera cerca de 37 milhões de tokens, ou 5 a 10 horas a 1.000 ou 2.000 tokens por segundo. A matriz de A-12 soma cerca de 13 a 26 horas, com o modo sem raciocínio incluído. A conta é grosseira, porque o batch pode acelerar bastante, e a fase 0 mede a velocidade real.

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

Se os dois candidatos vencerem, fica o de maior Nota de raciocínio. Se o IC 95% da diferença entre eles contiver 0, fica o LoRA, que é mais simples e mais rápido.

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
├── env.sh                        # caches dentro do projeto
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
| Máquina | NVIDIA Jetson AGX Thor, `aarch64`, 14 núcleos |
| Memória | 122 GiB compartilhados entre CPU e GPU, sem swap |
| Software | CUDA 13.2, driver 595.78, Python 3.12.3 |

Os pacotes seguem a tabela abaixo.

| Componente | Versão |
| --- | --- |
| Transformers | ≥ 5.8.1, a versão do `config.json` do Lumi |
| PEFT | ≥ 0.20 |
| TRL | ≥ 1.17 |
| PyTorch | Fixar na fase 0 |
| `mamba-ssm` e `causal-conv1d` | Só têm código-fonte no PyPI, então precisam compilar em ARM64 |

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

### Fase 0: fundação e baseline

- [ ] Criar repositório, lockfile e ambiente, compilando `mamba-ssm` e `causal-conv1d` para ARM64.
- [ ] Confirmar que `is_fast_path_available` é verdadeiro, porque o código cai no caminho lento só com um aviso.
- [ ] Baixar o Lumi v0.2 no commit `27b30f40` e ler `modeling_nemotron_h.py`.
- [ ] Comparar logits do código do repositório e da implementação nativa em um batch fixo, e escolher uma.
- [ ] Listar `named_modules()` e confirmar `TARGET_REGEX`.
- [ ] Provar nos dois braços que todo parâmetro `lora_*` treinável recebe gradiente não nulo após 1 passo.
- [ ] Provar que um adaptador recém-criado não altera os logits.
- [ ] Provar que adaptador ativo e modelo fundido geram logits equivalentes.
- [ ] Provar que a geração para em `<|im_end|>` com `eos_token_id=[2, 11]`.
- [ ] Testar a extração de resposta conforme A-05 e A-06.
- [ ] Medir pico de memória, maior batch viável e tokens por segundo por braço, conforme A-14.
- [ ] Medir a baseline do Lumi nos dois modos, conforme A-01 a A-09, inclusive a forma do raciocínio.

**Gate:** todos os testes passam, e a baseline sem raciocínio está a até 1 ponto do cartão ou a diferença está documentada. Se o Lumi com raciocínio já cumprir as metas de forma ou de nota, apresentar isso antes de seguir, porque o problema a resolver muda.

### Fase 1: dados

- [ ] Implementar seleção, filtros, renderização e remoção de sobreposição, conforme D-01 a D-09.
- [ ] Criar testes comparando exemplos renderizados com o template do Lumi, nos dois modos.
- [ ] Provar que o prompt é prefixo exato do exemplo nos dois modos, inclusive no caminho do TRL.
- [ ] Verificar que `labels` cobrem `</think>` e `<|im_end|>`, e que tudo fora da resposta recebe `-100`.
- [ ] Montar os conjuntos de validação conforme A-10 e registrar os hashes conforme D-11.
- [ ] Relatar quantas linhas cada filtro removeu, por `task_type`.
- [ ] Fazer a revisão humana de D-10.

**Gate:** 100 amostras renderizadas revisadas, sem erro de template ou máscara, e revisão de D-10 concluída.

### Fase 2: viabilidade

Um treino pequeno mostra se o caminho funciona antes de gastar GPU com a matriz completa.

- [ ] Treinar 1 LoRA com `r=16` e taxa 1e-4 em 10.000 exemplos com raciocínio, estratificados por `task_type`, mais 25% de tokens sem raciocínio.
- [ ] Comparar adaptador e Lumi, nos dois modos, nos conjuntos de validação de A-10, com IC 95% da diferença.
- [ ] Medir a forma do raciocínio conforme A-08.
- [ ] Ler 50 respostas e registrar o que os números não mostram.

**Gate:** ao menos 95% das respostas fecham `</think>`, ao menos 90% dos traços estão em português e a nota na validação do GSM8K-PT tem IC 95% da diferença acima de 0 contra o Lumi sem raciocínio. Se falhar, parar e decidir entre mais dados, traços gerados e verificados (outro PRD) ou encerrar.

### Fase 3: piloto

- [ ] Rodar a varredura de taxa de aprendizado.
- [ ] Comparar as misturas 75/25 e 100/0, inclusive no modo sem raciocínio.
- [ ] Rodar as 6 execuções do piloto.
- [ ] Comparar nota na validação, forma do raciocínio, loss, memória e tempo.

**Gate:** taxa, mistura e um rank por método escolhidos, com justificativa.

### Fase 4: finalistas

- [ ] Rodar as 6 execuções finalistas.
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
| Código do Lumi diverge do nativo | Alvos errados ou adaptador sem efeito | Testes de gradiente e equivalência da fase 0 |
| Kernels não compilam em ARM64 | Treino no caminho lento | Medir na fase 0 e, se preciso, usar o caminho PyTorch nos dois braços |
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

### Decisões em aberto

- [ ] Qual harness e quais parâmetros o CEIA usou no cartão (temperatura, few-shot, prompts, limite de tokens e raciocínio nos modelos comparados)? O cartão não documenta nada disso. Pedir também o notebook `colab_graficos_llms-2.ipynb`, citado na primeira versão do cartão.
- [ ] Vale usar [`adalberto-temp/gsm8k-sft-thinking-pt`](https://huggingface.co/datasets/adalberto-temp/gsm8k-sft-thinking-pt)? São 56.296 respostas do Qwen3-30B-A3B-Thinking para 7.225 problemas do GSM8K, geradas com o traço antes da resposta, e com checagem pedida no prompt. A mediana é de 236 palavras, e o p99 chega a 1.819. O dataset não declara licença, cerca de 6% dos traços parecem conter trechos em inglês e usá-lo exigiria trocar a validação de A-10.


### Fontes

Nesta versão, o `chat_template.jinja` e as configs do Lumi foram lidos por completo, e o `modeling_nemotron_h.py` só nos trechos de módulos e kernels. Do artigo do Tucano 2, foram lidas as seções sobre dados, pós-treino e avaliação, com as tabelas dos modelos Think. Os traços foram avaliados só em amostras pequenas, sem revisão sistemática.

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
