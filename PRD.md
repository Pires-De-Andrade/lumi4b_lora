# PRD: Fine-tuning do Lumi-4B com LoRA, DoRA e QLoRA

Nov 6, 2026 · @gustavo

## Resumo e objetivo

Treinar adaptadores LoRA, DoRA e QLoRA sobre o `ceia-br/lumi-4b` v0.2 para superar o Overall de referência (52,99) e o IFEval PT (35,33), comparando os três métodos na mesma avaliação pareada.

A entrega final é um adaptador vencedor, sua versão fundida em BF16 e um relatório reproduzível. Cada +6 pontos em uma nota vale +1 no Overall, porque ele é a média simples de 6 notas.

| Métrica | Referência | Meta mínima | Meta alvo |
| --- | --- | --- | --- |
| Overall | 52,99 no cartão | Acima da baseline medida, com IC 95% da diferença acima de 0 | Baseline medida + 2,0 (cerca de 55,0 pelo cartão) |
| IFEval PT estrito | 35,33 no cartão | ≥ 39,63 | ≥ 47,29 |
| MMLU PT, Global MMLU PT, MMMLU PT e INCLUDE PT | 47,63 a 61,73 | Queda média de até 1 ponto, sem queda significativa em nenhuma nota | Ganho em ao menos uma nota |

As metas de IFEval são resultados publicados: 39,63 é o Nemotron 3 Nano 4B e 47,29 é o Gemma 4 E4B, na melhor de 3 rodadas (a média é 46,60). A meta alvo do Overall equivale a levar o IFEval a 47,29 com as outras notas estáveis.

## Contexto e evidências

O Lumi-4B é mais fraco em seguir instruções, e a hipótese central é que isso melhora com dados de restrições verificáveis, e não com mais texto em português.

| Item | Valor |
| --- | --- |
| Versão | v0.2, lançada em 6 de novembro de 2026, no commit `27b30f40` |
| Classe | `NemotronHForCausalLM`, BF16, 7,95 GB |
| Camadas | 42: 21 Mamba, 17 MLP e 4 de atenção (índices 12, 17, 24 e 32) |
| Vocabulário e cabeça | 131.072 tokens, sem pesos amarrados: embeddings e `lm_head` somam cerca de 20% dos parâmetros |
| Contexto máximo | 262.144 tokens |
| Template | `enable_thinking` ligado por padrão, tool calls em XML |
| Raciocínio | O cartão manda mantê-lo desligado nesta versão |
| Treino | Pré-treino contínuo de 50B tokens, com destilação da Nemotron 3.5 Lightning, seguido de SFT |
| Dados de treino | Listados no cartão; os conjuntos do CEIA não têm link público |
| Modelo-base | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| Licença | Não documentada no cartão |

O cartão atual traz as notas só em gráfico, com 6 provas: em relação à v0.1, entraram MMMLU PT e INCLUDE PT, e saiu MMLU-ProX PT. A tabela abaixo usa os valores das barras de `lumi_versoes.svg`. Os valores da v0.1 batem com o cartão antigo nas provas em comum.

| Prova | Nemotron 3 Nano 4B | Lumi v0.1 | Lumi v0.2 |
| --- | --- | --- | --- |
| MMLU PT, ensino médio | 42,67 | 45,72 | 47,63 |
| MMLU PT, graduação | 44,02 | 48,80 | 50,75 |
| Global MMLU PT | 55,46 | 59,36 | 61,73 |
| MMMLU PT | 54,87 | 59,31 | 61,50 |
| INCLUDE PT | 54,08 | 58,44 | 60,98 |
| IFEval PT estrito | 40,93 | 33,83 | 35,33 |
| Overall | 48,67 | 50,91 | 52,99 |

A v0.1 tirou 35,7 no IFEval PT em teste independente, que não cobre a v0.2. O Lumi não aparece nos resultados publicados do benchmark, então sua taxa por classe ainda é desconhecida. A tabela abaixo mostra a taxa de acerto de três modelos próximos, em % (rodada `eval_open_2`).

| Classe de instrução | Nemotron 3 Nano 4B | Qwen3-4B | Tucano2 3.7B Instruct |
| --- | --- | --- | --- |
| Mesóclise | 1,6 | 1,6 | 3,2 |
| Crase | 16,4 | 49,2 | 59,0 |
| Contagem de cedilha | 17,9 | 5,4 | 3,6 |
| Uso de "porque" | 18,2 | 6,8 | 9,1 |
| Sem til | 30,6 | 14,5 | 1,6 |
| Tratamento por vós | 43,2 | 79,5 | 65,9 |
| Sem vírgula | 53,6 | 62,5 | 10,7 |
| JSON | 81,5 | 74,1 | 18,5 |

Três hipóteses guiam o projeto:

- **H1:** o Lumi perdeu disciplina de formato durante o treino em português. Indício: o Nemotron 3 Nano 4B, modelo-base segundo o cartão, tem 40,93 no IFEval PT pelo gráfico do CEIA e 39,63 no repositório do benchmark, acima dos 35,33 do Lumi.
- **H2:** as seis restrições próprias do português são o maior potencial. Elas aparecem em 47,7% dos prompts, e passar em todas levaria a Nemotron de 39,63 a 67,48, um teto teórico.
- **H3:** o `gigaverbo-v2-sft` sozinho não resolve o IFEval. O Tucano2 treinou com ele e fez 23,93, enquanto o Qwen3-4B fez 41,87. Em conhecimento e raciocínio, porém, o Tucano2 lidera (56,22 contra 42,33).

O SFT do Lumi já usou dados de seguir instruções, inclusive em português (`Nemotron-SFT-Instruction-Following-Chat` v2 e v3 e `CEIA-Portuguese-Instruction-Following-Chat`), e a nota segue abaixo da base. Por isso, a fase 2 testa a viabilidade antes da geração completa.

## Escopo

Este projeto cobre apenas o pós-treino por adaptadores sobre os pesos do Lumi-4B: refino supervisionado (SFT) seguido de otimização de preferência (APO).

**Dentro do escopo**

- Preparar os subsets do GigaVerbo-v2 no formato do Lumi.
- Gerar um dataset novo de restrições verificáveis em português.
- Treinar e comparar LoRA, DoRA e QLoRA, com QDoRA como braço opcional.
- Avaliar todos os finalistas contra o Lumi no mesmo harness.
- Entregar o adaptador vencedor, o modelo fundido em BF16 e o relatório.

**Fora do escopo**

- Pré-treino continuado dos pesos completos. Se os adaptadores não atingirem a meta, isso vira outro PRD.
- Tool calling: não entra no Overall e o subset `function_call` pode exigir conversão de formato.
- Contexto longo, exportação GGUF e publicação no Hugging Face.
- Modo de raciocínio: o cartão manda mantê-lo desligado nesta versão.

## Requisitos de dados

O treino usa três fontes: subsets selecionados do GigaVerbo-v2, um dataset novo de restrições e pares de preferência. Tudo deve ser renderizado pelo template do próprio Lumi, com `enable_thinking=False`.

| Fonte | Uso | Regra de seleção | Licença |
| --- | --- | --- | --- |
| [`Polygl0t/gigaverbo-v2-sft`](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-sft) | SFT | `instruct_score` ≥ 4,0, sem `function_call` e sem `reasoning` | Apache-2.0 |
| `lumi-constraints-pt` (novo) | SFT e APO | Aprovado por validador e por filtro de qualidade | Definir na fase 2 |
| [`Polygl0t/gigaverbo-v2-preferences`](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-preferences) | APO | 10.521 pares de `harmless-no-reasoning` e depois 4.267 de `harmfull-no-reasoning`; os subsets `*-reasoning` ficam de fora | Apache-2.0 |

A mistura inicial do SFT abaixo é uma hipótese de partida, a revisar depois da baseline por classe.

| Fonte | Participação nos tokens |
| --- | --- |
| `lumi-constraints-pt` | 40% |
| `math_cot` e `math` | 20% |
| `structured`, `system_prompts`, `rewriting` e `summarization` | 20% |
| `general` | 15% |
| `retrieval` | 5% |

### Gerador do `lumi-constraints-pt`

1. **Prompts:** combinar temas inéditos com 1 a 3 instruções do registro `pt:` do [Portuguese-IFEval](https://github.com/ark-aiml-research/Portuguese-IFEval) (Apache-2.0). Guardar tema e instruções em campos separados, porque D-05 compara só o tema. As personas de [`nvidia/Nemotron-Personas-Brazil`](https://huggingface.co/blog/nvidia/nemotron-personas-brazil) (CC BY 4.0) podem servir de sementes.
2. **Respostas:** gerar N candidatas por prompt com um modelo de licença permissiva, como o Qwen3-8B. Para mesóclise, cedilha, "sem til" e "porque", usar exemplos no prompt, porque a taxa de acerto do Qwen3-8B fica em 3,2%, 5,4%, 7,0% e 8,3%, na média das 3 rodadas publicadas.
3. **Validação:** rodar os validadores determinísticos do repositório.
4. **Qualidade:** filtrar as aprovadas com [`Polygl0t/portuguese-qwen3-4b-instruct-quality-classifier`](https://huggingface.co/Polygl0t/portuguese-qwen3-4b-instruct-quality-classifier).
5. **Saída:** aprovadas viram exemplos de SFT. Uma aprovada e uma reprovada do mesmo prompt viram um par `chosen` e `rejected`.

**Custo estimado:** os 80.000 exemplos de D-03 exigem cerca de 460 mil respostas e 140 milhões de tokens, pelas taxas publicadas do Qwen3-8B e com cerca de 300 tokens por resposta. A 1.000 ou 2.000 tokens por segundo, são de 1 a 2 dias de GPU. A conta é grosseira: ignora o ganho com exemplos no prompt e os prompts com várias restrições. A velocidade real é medida na fase 2.

### Requisitos

| ID | Requisito |
| --- | --- |
| D-01 | Renderizar com `tokenizer.apply_chat_template` do Lumi, sem formatação manual e com `enable_thinking=False`. Nos trainers do TRL, passar o argumento pela coluna `chat_template_kwargs`. A loss cobre apenas tokens do assistente. |
| D-02 | Todo exemplo deve conter o prefixo `<think></think>` que o template adiciona. Descartar exemplos com `<think>` não vazio, como os do subset `reasoning`. |
| D-03 | Mínimo de 5.000 exemplos aprovados por restrição do português e 2.000 pelas demais classes. São 6 e 25 classes, o que soma 80.000 exemplos. |
| D-04 | Mínimo de 20.000 pares novos para APO. |
| D-05 | Remover sobreposição com os benchmarks por k-gramas de 8 a 32 tokens, incluindo os 535 prompts do IFEval. No `lumi-constraints-pt`, comparar só o tema, porque 52,7% dos prompts do IFEval repetem ao menos 8 palavras seguidas do texto das instruções do registro. |
| D-06 | Reservar 2 classes de instrução fora de um treino de controle, para medir generalização. |
| D-07 | Revisão humana de 200 amostras por restrição do português, pois os validadores só checam a forma do texto. |
| D-08 | Nunca usar o `rejected` dos subsets `harmfull-*` como exemplo de SFT: ele contém respostas nocivas por design. O nome tem dois "l", e um teste deve provar que o bloqueio encontra esses subsets. |
| D-09 | Separar 2% para validação interna, estratificado por classe, e registrar hashes em `data_manifest.json`. |
| D-10 | Não gerar respostas com APIs cujos termos proíbam treinar outros modelos. |

## Requisitos de treino

Os três métodos usam os mesmos módulos-alvo e diferem na carga dos pesos e no `use_dora`. O QLoRA reduz memória, mas não promete ganho de qualidade. Nesta máquina a memória não é limite, então ele fica como braço de comparação do estudo e só precisa empatar com o LoRA.

| Braço | Pesos-base | `use_dora` | Kernel Mamba | Papel |
| --- | --- | --- | --- | --- |
| LoRA | BF16 | `False` | Kernels permitidos | Referência |
| DoRA | BF16 | `True` | Kernels permitidos | Testar ganho em rank baixo, com mais custo de tempo |
| QLoRA | 4-bit NF4, double quant, cómputo BF16 | `False` | Caminho PyTorch, sem kernels | Comparação do estudo; a economia de memória não é necessária nesta máquina |
| QDoRA (opcional) | Igual ao QLoRA | `True` | Caminho PyTorch, sem kernels | Só se DoRA vencer LoRA e a memória virar limite |

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
    use_dora=False,  # True nos braços DoRA e QDoRA
)
```

- **`out_proj` do Mamba fica de fora.** Com `mamba-ssm`, o treino passa `self.out_proj.weight` direto ao kernel e ignora o `forward` do módulo. A [receita LoRA da NVIDIA](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_nano_4b_squad_peft.yaml) exclui `*.out_proj` por esse motivo.
- **`conv1d` fica de fora.** O peso também é lido direto, e o PEFT não funde convoluções com `groups > 1`.
- **`embeddings` e `lm_head` ficam de fora.** O adaptador deve mudar o comportamento sem tocar no vocabulário.

Esses nomes existem tanto na implementação nativa do Transformers quanto no código próprio do Lumi, usado via `auto_map`. A fase 0 ainda precisa confirmá-los com `named_modules()`.

### Regras do QLoRA

- Com `mamba-ssm`, o kernel receberia o peso empacotado de 4 bits do `out_proj`. Por isso, o padrão é rodar sem esse pacote, no caminho PyTorch, que o Transformers descreve como correto e bem mais lento.
- Se a fase 0 escolher a implementação nativa, não usar `use_kernels=True`, que ativa kernels do Hub.
- A alternativa é `llm_int8_skip_modules=["lm_head", "out_proj"]`, que mantém os `out_proj` sem quantizar. O `lm_head` precisa constar porque essa lista substitui os padrões.
- `prepare_model_for_kbit_training` converte para FP32 tudo que não está em 4 bits, o que dobra a memória de embeddings e `lm_head`.
- A fusão final é sempre sobre o Lumi em BF16, e a avaliação usa esse modelo fundido, porque ele difere dos pesos 4-bit usados no treino.

| Configuração | Pesos em memória (estimativa) |
| --- | --- |
| LoRA e DoRA em BF16 | 8,0 GB |
| QLoRA, mantendo embeddings e `lm_head` em BF16 | 3,2 GB |
| QLoRA com `prepare_model_for_kbit_training` | 4,9 GB |
| QLoRA com `out_proj` fora da quantização e cast para FP32 | 6,6 GB |

Em todos os braços, os logits custam cerca de 0,5 GB por 1.000 tokens no batch, antes dos gradientes (131.072 × 4 bytes por token). A fase 0 mede os valores reais.

### Hiperparâmetros iniciais

| Parâmetro | Valor | Origem |
| --- | --- | --- |
| `r` | 8, 16 e 32 | NVIDIA usa 8 |
| `lora_alpha` | 2 × `r` | NVIDIA usa 16 e 32 com `r=8` |
| `lora_dropout` | 0 | NVIDIA usa 0, e o DoRA fica mais rápido assim |
| Taxa de aprendizado do SFT | 1e-4, varrendo 5e-5 e 2e-4 | NVIDIA usa 1e-4 em LoRA |
| Otimizador | AdamW com cosseno e warmup de 3% | Tucano 2, warmup escolhido no projeto |
| Sequência | 2.048 tokens | NVIDIA e Tucano 2 |
| Épocas do SFT | Até 3, com parada antecipada | Escolha do projeto; o Tucano 2 usou 5 |
| Gradient checkpointing | `use_reentrant=False`, `use_cache=False` | Requisito de memória |
| APO | `loss_type=["apo_zero"]`, beta 0,5, taxa 1e-5, 2 épocas | Tucano 2 e TRL; épocas escolhidas no projeto |

### Matriz de experimentos

| Etapa | Execuções | Dados | Saída |
| --- | --- | --- | --- |
| Viabilidade | 1 (LoRA, `r=16`) | 3.000 exemplos de 3 restrições | Decisão sobre a geração completa |
| Varredura de taxa | 3 (LoRA, `r=16`) | 20% | Taxa fixada |
| Piloto | 9 (3 métodos × 3 ranks) | 20% | Melhor rank por método |
| Finalistas | 9 (3 métodos × 3 seeds) | 100% | Melhor adaptador por método |
| Controle de generalização | 1 | 100%, sem 2 classes | Ganho nas classes não vistas |
| APO | 3 (1 por método) | Preferências | Candidatos finais |

A seleção usa a taxa de aprovação na validação interna, nunca o benchmark.

### APO com adaptadores

- Passar ao `DPOTrainer` o `PeftModel` já treinado, sem `peft_config` nem `ref_model`. Assim, o TRL cria um adaptador `ref` congelado e a referência é o modelo pós-SFT.
- Com um `peft_config` novo, a referência passaria a ser o Lumi sem adaptador, o que muda o sentido do APO.
- Passar o modelo já carregado, e não o nome, porque o TRL usa FP32 quando recebe uma string sem `dtype`.
- Manter `sync_ref_model=False`, pois o TRL não suporta essa opção com PEFT.

## Requisitos de avaliação

Um candidato só vence se superar o Lumi no mesmo harness, com intervalo de confiança acima de zero e sem perder conhecimento.

| ID | Requisito |
| --- | --- |
| A-01 | Medir as 6 notas do cartão no Lumi v0.2 antes de treinar. A referência é a baseline medida aqui, porque o harness muda a nota: o Nemotron 3 Nano 4B tem 40,93 no gráfico do CEIA e 39,63 no repositório do IFEval. |
| A-02 | Rodar os 535 prompts do Portuguese-IFEval com o avaliador oficial e registrar a nota estrita por prompt e a taxa por classe. |
| A-03 | Rodar MMLU PT (ensino médio e graduação), Global MMLU PT, MMMLU PT e INCLUDE PT com versão e amostra fixadas, usando as provas completas da tabela de conjuntos inferidos abaixo, pois mais questões reduzem o ruído. Não usar `Portuguese-MMLU/*` nem as cópias em `cemig-ceia` e `bench-temp-2`: esses conjuntos têm 8.704 e 29.094 questões de teste, o que tornaria a baseline não comparável ao cartão. |
| A-04 | Usar temperatura 0, o mesmo limite de tokens e `enable_thinking=False`. |
| A-05 | Calcular a diferença para a baseline com bootstrap pareado por item, dentro de cada prova: 10.000 reamostras e IC de 95% para cada nota e para o Overall. |
| A-06 | Avaliar sempre o modelo fundido em BF16, inclusive no braço QLoRA. |
| A-07 | Rodar o benchmark apenas na baseline, nos 3 candidatos finais e no controle de generalização. |
| A-08 | Registrar pico de memória, tempo por época e tokens por segundo em cada execução. Medir a memória com `torch.cuda.max_memory_allocated()`, porque o `nvidia-smi` não a informa nesta máquina. |
| A-09 | Medir recusas em 200 prompts de `harmfull-no-reasoning`, reservados fora de todo treino, para detectar perda de segurança. |

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

As notas do teste independente da v0.1 nas provas de múltipla escolha usam 1.000 questões por prova, então não são comparáveis às do cartão.

**Critério de vitória**

- IC 95% da diferença do Overall inteiramente acima de 0.
- IFEval PT estrito ≥ 39,63.
- Queda média de até 1 ponto nas 5 notas de múltipla escolha, sem nenhuma nota com IC 95% da diferença inteiramente abaixo de 0.
- Taxa de recusa não inferior à do Lumi, salvo decisão registrada.

A trava pela média evita reprovar por acaso: com as provas completas inferidas acima, um limite de 1 ponto por nota reprovaria de 15% a 33% dos modelos sem perda real, quase sempre por causa do INCLUDE PT, que tem só 551 questões, e esta regra reprova cerca de 12%. Com as outras notas estáveis, o primeiro critério pede IFEval de 39,2 a 41,2, faixa próxima do mínimo de 39,63, então qualquer um dos dois pode ser o mais exigente. Essas contas são estimativas e supõem de 5% a 20% das respostas mudando nas provas de múltipla escolha e de 15% a 25% no IFEval.

Se nenhum candidato cumprir os quatro itens, o relatório declara que não houve vitória e indica a causa provável por classe. Se o ganho aparecer só nas classes treinadas, o relatório deve dizer isso explicitamente, com o resultado do controle de generalização.

## Arquitetura do repositório e configuração

O repositório separa dados, treino e avaliação, e cada execução nasce de um YAML versionado.

```text
Dados de SFT (GigaVerbo-v2 SFT + lumi-constraints-pt)
  -> LoRA (BF16) | DoRA (BF16) | QLoRA (4-bit NF4)
  -> APO por método (pares de preferência, adaptador ref congelado)
  -> Fusão sobre o Lumi em BF16
  -> Comparação com a baseline do Lumi no mesmo harness
     sim: entregar adaptador, modelo e relatório
     não: relatório sem vitória, com causa provável por classe
```

Cada braço usa os mesmos dados, passa pelo APO e é fundido em BF16 antes da comparação com a baseline.

```text
lumi-finetune/
├── env.sh                        # caches dentro do projeto
├── pyproject.toml                # dependências com lockfile
├── configs/
│   ├── base.yaml
│   ├── apo.yaml
│   └── arms/{lora,dora,qlora,qdora}.yaml
├── src/lumi_ft/
│   ├── modeling/load.py          # carga BF16 e 4-bit
│   ├── modeling/peft_setup.py    # alvos e checagens
│   ├── data/gigaverbo.py         # seleção e renderização
│   ├── data/constraints.py       # gerador do dataset novo
│   ├── data/decontaminate.py     # k-gramas contra benchmarks
│   ├── train/sft.py
│   ├── train/apo.py
│   ├── train/merge.py            # fusão sobre BF16
│   └── eval/{ifeval,mcq,bootstrap,report}.py
├── tests/
├── third_party/portuguese-ifeval/ # fixado por commit
├── data/                         # fora do Git
├── runs/                         # adaptadores, logs e métricas
└── reports/
```

```yaml
# configs/arms/qlora.yaml
extends: ../base.yaml
arm: qlora
model:
  name: ceia-br/lumi-4b
  revision: 27b30f40b930fd05cee5e1c1f8790f70fa5b816d  # v0.2
  dtype: bfloat16
template:
  enable_thinking: false
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  use_dora: false
env:
  forbid_packages: [mamba-ssm, causal-conv1d]
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
| bitsandbytes e PyTorch | Fixar na fase 0. Há wheel `aarch64` do bitsandbytes, mas o teste de gradiente precisa provar que ele funciona nesta máquina |
| `mamba-ssm` e `causal-conv1d` | Apenas nos ambientes LoRA e DoRA. Só têm código-fonte no PyPI, então precisam compilar em ARM64 |

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
6. Não instalar `mamba-ssm` nem `causal-conv1d` no ambiente do QLoRA.
7. Se o código do Lumi divergir deste PRD, registrar a divergência e perguntar antes de seguir.
8. Usar `enable_thinking=False` em treino, APO e avaliação.
9. Carregar `env.sh` antes de baixar, instalar ou executar qualquer coisa, para manter os caches dentro de `barros`.

### Fase 0: fundação e baseline

- [ ] Criar repositório, lockfile e dois ambientes, com e sem `mamba-ssm` e `causal-conv1d`, compilando os pacotes para ARM64.
- [ ] Confirmar no ambiente com kernels que `is_fast_path_available` é verdadeiro, porque o código cai no caminho lento só com um aviso.
- [ ] Baixar o Lumi v0.2 no commit `27b30f40` e ler `modeling_nemotron_h.py`.
- [ ] Comparar logits do código do repositório e da implementação nativa em um batch fixo, e escolher uma.
- [ ] Listar `named_modules()` e confirmar `TARGET_REGEX`.
- [ ] Provar nos três braços que todo parâmetro `lora_*` treinável recebe gradiente não nulo após 1 passo.
- [ ] Provar que um adaptador recém-criado não altera os logits.
- [ ] Provar que, sobre o Lumi em BF16, adaptador ativo e modelo fundido geram logits equivalentes.
- [ ] Medir pico de memória, maior batch viável e tokens por segundo por braço, conforme A-08.
- [ ] Medir a baseline do Lumi conforme A-01 a A-04, com taxa por classe.

**Gate:** todos os testes passam, e a baseline está a até 1 ponto do cartão ou a diferença está documentada.

### Fase 1: dados existentes

- [ ] Implementar seleção, filtros e renderização conforme D-01, D-02 e D-09.
- [ ] Criar testes comparando exemplos renderizados com o template do Lumi.
- [ ] Provar que todo prompt renderizado termina em `<think></think>`, inclusive no caminho do TRL.
- [ ] Verificar que tokens fora do assistente recebem `labels = -100`.
- [ ] Preparar pares de preferência, com o bloqueio de D-08.

**Gate:** 100 amostras renderizadas revisadas, sem erro de template ou máscara.

### Fase 2: viabilidade e gerador de restrições

Primeiro, um teste pequeno mostra se os dados de restrições movem a taxa de aprovação. Só depois vem a geração completa.

**Etapa A: teste de viabilidade**

- [ ] Implementar geração, validação e filtro de qualidade.
- [ ] Gerar 3.000 exemplos aprovados, com D-05 aplicado, 1.000 em cada uma de 3 restrições do português escolhidas pela baseline por classe.
- [ ] Treinar 1 LoRA com `r=16` e taxa 1e-4 nesses exemplos.
- [ ] Comparar adaptador e Lumi em 200 prompts novos por classe, com IC 95% da diferença.
- [ ] Medir os tokens por segundo do gerador e estimar as horas da geração completa.

**Gate A:** o adaptador supera o Lumi em ao menos 2 das 3 classes, com IC 95% da diferença acima de 0. Se falhar, parar e decidir se a geração completa ainda vale o custo.

**Etapa B: geração completa**

- [ ] Gerar o dataset completo.
- [ ] Rodar a remoção de sobreposição com benchmarks.
- [ ] Gerar os pares `chosen` e `rejected`.
- [ ] Preparar a amostra para revisão humana.

**Gate B:** D-03, D-04, D-05 e D-07 atendidos, com taxa de aprovação por classe no relatório.

### Fase 3: piloto

- [ ] Rodar a varredura de taxa de aprendizado.
- [ ] Rodar as 9 execuções do piloto.
- [ ] Comparar aprovação na validação, loss, memória e tempo.

**Gate:** um rank escolhido por método, com justificativa.

### Fase 4: finalistas e APO

- [ ] Rodar as 9 execuções finalistas e o controle de generalização.
- [ ] Rodar o APO no melhor adaptador de cada método.
- [ ] Provar que `chosen_logratios` e `rejected_logratios` ficam perto de 0 no início do APO.
- [ ] Fundir os 3 candidatos sobre o Lumi em BF16.

**Gate:** 3 candidatos fundidos e checados, mais o modelo de controle.

### Fase 5: avaliação e entrega

- [ ] Rodar o benchmark completo conforme A-02 a A-09.
- [ ] Gerar `reports/final.md` com tabela por método e por classe, IC 95%, memória e tempo.
- [ ] Registrar qual método venceu ou declarar que nenhum venceu.
- [ ] Rascunhar um model card com base, dados, limitações e o ganho nas classes treinadas.

**Gate:** critério de vitória avaliado e relatório reproduzível a partir de `runs/`.

## Riscos, restrições e decisões em aberto

O maior risco técnico é o código próprio do Lumi se comportar de forma diferente da implementação nativa. Os nomes dos módulos já foram conferidos, mas gradientes e logits ainda precisam de prova na fase 0.

| Risco | Efeito | Mitigação |
| --- | --- | --- |
| Código do Lumi diverge do nativo | Alvos errados ou adaptador sem efeito | Testes de gradiente e equivalência da fase 0 |
| Kernels não compilam em ARM64 | Todos os braços ficam no caminho lento | Medir na fase 0 e, se preciso, usar o caminho PyTorch em todos |
| Caminho PyTorch lento no QLoRA | Custo alto de GPU | Medir na fase 0 e, se preciso, usar `llm_int8_skip_modules` |
| Dados de restrições não movem a nota | Custo alto sem ganho | Teste de viabilidade da fase 2, antes da geração completa |
| Baixo rank não corrige a fraqueza | Meta não atingida | Testar `r=32` e, se falhar, abrir PRD de treino completo |
| Treino nas classes e na redação das instruções da prova infla a nota | Ganho não generaliza | Controle com 2 classes reservadas e declaração no relatório |
| Validador aprova texto ruim | Modelo aprende truques de forma | Filtro de qualidade e revisão humana |
| Treino reduz conhecimento | Queda nas notas de múltipla escolha | Trava do critério de vitória; a medida interna está em aberto |
| Ajuste indireto ao benchmark | Resultado otimista | Seleção na validação interna e benchmark só nos finalistas |
| Licença impede derivados | Modelo não pode ser publicado | Confirmar antes de qualquer publicação |

O cartão manda manter o raciocínio desligado nesta versão, então este PRD não treina esse modo. A recomendação da Unsloth, de ao menos 75% de exemplos com raciocínio em modelos Nemotron Nano, só volta a importar quando o CEIA liberar esse modo.

### Decisões em aberto

- [ ] Qual é o orçamento de tempo de GPU? Este PRD assume a Jetson AGX Thor deste projeto.
- [ ] Qual é a licença do Lumi, e derivados podem ser publicados?
- [ ] O CEIA confirma os conjuntos e as amostras inferidos em "Requisitos de avaliação", e quais revisões usou?
- [ ] Qual harness e quais parâmetros o CEIA usou no cartão (temperatura, few-shot, prompts, limite de tokens e raciocínio nos modelos comparados)? O cartão não documenta nada disso. Pedir também o notebook `colab_graficos_llms-2.ipynb`, citado na primeira versão do cartão.
- [ ] Qual licença terá o `lumi-constraints-pt`?
- [ ] O que fazer se o APO piorar o modelo, já que hoje todo candidato passa por ele?
- [ ] Como desempatar se mais de um candidato vencer?
- [ ] Qual medida interna de conhecimento acompanha a seleção, que hoje só olha o validador?
- [ ] Quais 2 classes ficam de fora no controle? Declarar antes de treinar.

### Fontes

Nesta versão, o `modeling_nemotron_h.py` do Lumi foi lido só nos trechos de módulos e kernels, e o apêndice L do artigo do Tucano 2 não foi lido.

- [Cartão do CEIA Lumi-4B](https://huggingface.co/ceia-br/lumi-4b)
- [Teste independente em GGUF](https://huggingface.co/Mateus-Nery/lumi-4b-GGUF), da v0.1
- [Portuguese-IFEval](https://github.com/ark-aiml-research/Portuguese-IFEval), incluindo os resultados publicados em `experiments/`
- [MMLU PT, ensino médio](https://huggingface.co/datasets/mmlu-pt/mmlu-pt-high-school-full) e [graduação](https://huggingface.co/datasets/mmlu-pt/mmlu-pt-undergraduate-full)
- [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), [MMMLU](https://huggingface.co/datasets/openai/MMMLU) e [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44)
- [Tucano 2 Cool](https://arxiv.org/abs/2603.03543)
- [`nemotron_h` no Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
- [DoRA no PEFT](https://huggingface.co/docs/peft/main/package_reference/lora_variant_dora)
- [Quantização no PEFT](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md)
- [DPO no TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [Nemotron 3 Nano 4B no Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/nemotron/nemotron3-nano-4b.html)
- [Guia do Nemotron 3 na Unsloth](https://unsloth.ai/docs/models/tutorials/nemotron-3)
