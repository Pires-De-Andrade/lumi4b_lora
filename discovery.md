# Descobertas da fase 0

Nov 6, 2026 · RTX 4090 · registro de evidências, separado do [PRD](PRD.md). O PRD guarda decisões e requisitos; este arquivo guarda o que foi medido, como e com que limites.

Tudo abaixo foi medido nesta máquina, com Transformers 5.22.2, PEFT 0.21.2, PyTorch 2.14.1+cu130, `mamba-ssm` 2.3.2.post1 e `causal-conv1d` 1.7.0, sobre o Lumi v0.2 (commit `27b30f40`), em BF16. Os resultados brutos ficam em `runs/phase0/*.json` e `runs/jobs/*.log`, que não vão para o Git.

## 1. Ambiente e kernels

- A máquina não tinha `nvcc`, `sudo` nem acesso ao Docker. O `nvcc` 13.0.88 veio do conda-forge, via micromamba, dentro do projeto. O driver (580.95) suporta CUDA até 13.0, então o PyTorch ficou em 2.14.1+cu130; a 2.15.0 só tem builds cu132 e cu134.
- O `causal-conv1d` compilou sem alterações.
- O `mamba-ssm` falhou na primeira tentativa: `C++20 or later compatible compiler is required to use ATen`. O `setup.py` compila com `-std=c++17`, e os headers do PyTorch 2.14 exigem C++20. O g++ 11 da máquina basta, e a correção foi trocar a flag.
- O `setup.py` do `mamba-ssm` fixa 9 arquiteturas, sem opção de ambiente para limitar. A cópia local patchada compila só a sm_89. O binário resultante só roda em GPUs sm_89; em outra GPU é preciso ajustar o patch e recompilar. O patch está em `scripts/patches/mamba-ssm-sm89-cxx20.patch`.
- Os dois módulos importam, e o kernel do `causal_conv1d` rodou em BF16 na GPU com saída finita.
- A confirmação de que o *modelo* usa os kernels veio do profiler (seção 3), e não só da importação.

## 2. Estrutura do modelo

- 42 camadas: 21 Mamba, 17 MLP e 4 de atenção, nos índices 12, 17, 24 e 32. Bate com o PRD.
- 3,974 bilhões de parâmetros. 93 camadas `Linear`.
- O `TARGET_REGEX` casa exatamente 71 módulos: 21 `in_proj`, 17 `up_proj`, 17 `down_proj` e 4 de cada projeção de atenção. Ficam de fora os 21 `out_proj` e o `lm_head`, e os `conv1d` e embeddings não são `Linear`.
- Nomes: o código do repositório usa `backbone.layers.N.mixer.*`, e a implementação nativa usa `model.layers.N.mixer.*`. O regex vale para os dois.
- No caminho de treino, o código do repositório passa `out_proj.weight` e `conv1d.weight` direto ao kernel fundido `mamba_split_conv1d_scan_combined`. Isso confirma a razão do PRD para excluir esses dois módulos.
- Os arquivos Python do repositório (`modeling_nemotron_h.py`, `configuration_nemotron_h.py` e `nano_v3_reasoning_parser.py`) foram examinados por padrão de busca: sem rede, `exec`, `eval`, `subprocess`, `pickle` nem `os.system`. Isso não é uma auditoria completa; li só os trechos de módulos, kernels e geração.

## 3. Código do repositório × implementação nativa

### Resultado

| Comparação | Diferença média nos logits | Máxima | Top-1 igual |
| --- | --- | --- | --- |
| remote × nativa, sem padding (4 prompts) | 0,024 | 0,44 | 100% |
| remote: sem padding × batch com padding à direita | 0,029 | 0,63 | 100% |
| remote: sem padding × batch com padding à **esquerda** | **0,307** | **9,75** | 95,6% |
| nativa: sem padding × padding à direita | 0,030 | 0,52 | 99,1% |
| nativa: sem padding × padding à esquerda | 0,031 | 0,52 | 99,1% |
| remote × nativa, batch de 12 com padding à esquerda | 0,379 | 9,63 | 94,3% (332 posições) |

A divergência de 0,379 vista primeiro era, portanto, o padding à esquerda do código do repositório, e não uma diferença entre as implementações.

### Causa

A causa do erro com padding à esquerda **não foi isolada**. A hipótese é que o caminho rápido do código do repositório não zera os tokens de pad antes do estado do Mamba: no `forward()`, a multiplicação pela máscara vem depois do desvio para `cuda_kernels_forward`. Não li esse método por inteiro, então ele pode tratar a máscara de outro jeito. É hipótese de leitura de código, não um experimento.

### Geração

- No código do repositório, `generate` falha no Transformers 5.22.2: `TypeError: 'NoneType' object is not subscriptable`, em `prepare_inputs_for_generation` (linha 1531). O `cache_position` chega nulo, porque esse método é uma cópia do Jamba, no estilo antigo. Não testei versões anteriores do Transformers.
- Na nativa, `generate` funciona. Com `eos_token_id=[2, 11]` parou em 3 tokens (`olá<|im_end|>`, último id 11). Só com `eos_token_id=2`, gerou os 128 tokens permitidos. Isso confirma a afirmação do PRD sobre o `eos`.

### Kernels na nativa

O código nativo escolhe, por função, kernels do Hub (se pedidos), depois os pacotes `mamba-ssm` e `causal-conv1d` (se importáveis) e só depois PyTorch puro, com aviso de lentidão. Evidências de que usa os pacotes:

- Nenhum log contém o aviso de fallback.
- O profiler de um forward em modo `eval` mostrou `_state_passing_fwd_kernel` (Triton, do `mamba-ssm`) e `causal_conv1d_channellast_fwd_kernel` (CUDA, do `causal-conv1d`).

A nativa não tem a variável `is_fast_path_available`; o código do repositório tem, e vale `True`.

## 4. DoRA e as linhas zeradas da `in_proj`

- A `in_proj` tem **213 linhas com norma exatamente zero** (média de cerca de 10 por camada; a camada 0 tem 25 e a camada 2 tem 17). Nenhum outro tipo de módulo-alvo tem linhas zeradas.
- Só classifiquei as posições das camadas 0 e 2: nas duas, os primeiros índices listados caem no trecho `xBC` da saída da `in_proj` (a ordem é `z`, `xBC`, `dt`). As demais camadas não foram classificadas.
- Sintoma sem o patch: com o DoRA, os logits viram `nan` e os 213 parâmetros treináveis ficam sem gradiente válido, já no primeiro passo. O LoRA no mesmo teste passou, com diferença exatamente 0.
- Causa: o DoRA do PEFT inicializa a magnitude com a norma da linha (0) e calcula `magnitude / norma` sem epsilon. É 0/0 nessas linhas. O mesmo cálculo acontece na fusão.
- Verificação da causa: com `lora_B` aleatório (diferente de zero), a norma deixa de ser zero e os logits ficam finitos. Isso é coerente com a explicação.
- Correção (`src/lumi_ft/modeling/peft_patch.py`): onde a norma é exatamente 0, o divisor vale 1. O patch fica desligado durante `update_layer`, para a magnitude inicial continuar sendo a norma verdadeira. Se o patch valesse também ali, as linhas mortas nasceriam com magnitude 1 e virariam canais ativos.
- Efeito: nessas linhas o fator é 0/1 = 0, então elas continuam produzindo zero. O DoRA **não** as ativa, ao contrário do LoRA, que pode. É uma diferença real entre os dois braços.
- Os testes unitários em CPU mostram que: sem o patch há NaN; com o patch a saída de um adaptador novo é igual à base; em modelos sem linhas zeradas a saída é bit a bit igual com e sem o patch; os gradientes ficam finitos; a fusão bate com o adaptador ativo; e as linhas mortas continuam zero após a fusão.
- Limite: o `unmerge` dividiria por um fator 0 nessas linhas. O projeto não o usa.

## 5. Equivalência: adaptador recém-criado, fusão e piso de ruído

### Como o critério foi definido (e mudou)

1. Primeiro escolhi limites à mão: diferença máxima de 1% (adaptador novo) e 2% (fusão) da escala dos logits, e top-1 igual em 98%.
2. Esses limites falharam por razões minhas: a "guarda" que exigia efeito máximo 10× maior que a diferença da fusão era arbitrária; depois troquei por médias, KL e top-1 igual em 99%, número que escolhi **depois** de ver o LoRA passar com folga.
3. O DoRA ficou fora desse limite (top-1 igual em 98,87%).
4. Em vez de afrouxar o número, passei a medir o **piso de ruído do próprio modelo em BF16**: o mesmo batch processado em batch com padding × cada sequência sozinha (matematicamente equivalentes). O critério passou a ser: nada pior que **2× esse piso** em diferença média, KL e discordância do top-1 (com folga de 1 posição). Os números do critério vêm da medição, e não de uma escolha.

Os critérios de 1 e 2 foram abandonados; os números abaixo são os do critério 4. O PRD não define um número para "equivalentes".

### Resultados (batch de 16 prompts, 444 posições reais, implementação nativa)

| Medida | Diferença média | Top-1 discorda | KL médio |
| --- | --- | --- | --- |
| **Piso de ruído BF16** (batch × sequência sozinha) | 0,0251 | 1,58% | 4,8e-4 |
| LoRA recém-criado × base | 0,0 | 0,0% | 0 |
| DoRA recém-criado × base (com patch) | 0,0280 | 1,35% | 6,2e-4 |
| LoRA fundido × adaptador ativo | 0,0282 | 0,23% | 4,1e-4 |
| DoRA fundido × adaptador ativo | 0,0300 | 0,45% | 5,1e-4 |

- Para a fusão, `lora_B` foi sorteado (desvio 0,02), para o adaptador ter efeito real. O efeito médio do adaptador ficou em 0,21 (LoRA) e 0,22 (DoRA), mais de 5× a diferença da fusão. Isso garante que o teste não é vazio.
- A fusão e o DoRA recém-criado ficam dentro do ruído do modelo. O DoRA recém-criado tem diferença média de 0,028 mesmo sem mudar nada em teoria, porque a norma é calculada em BF16.
- **Limites:** os testes usam adaptadores sorteados e poucos passos, e não um adaptador treinado de verdade. A equivalência com pesos treinados deve ser repetida na fase 4, como o PRD prevê.

### Efeito colateral para a avaliação

Só mudar a composição do batch troca o top-1 em cerca de **1,6%** das posições, mesmo em temperatura 0. A avaliação precisa fixar o tamanho do batch e a ordem dos itens, senão parte da diferença entre dois sistemas será ruído numérico.

## 6. Gradientes

- LoRA: 142 parâmetros treináveis. No primeiro passo, os 71 `lora_A` têm gradiente zero (esperado, porque `lora_B` começa em zero). Depois de um passo do otimizador, nenhum parâmetro fica sem gradiente válido.
- DoRA (com patch): 213 parâmetros treináveis (`A`, `B` e magnitude por módulo). Sem gradiente válido: 0 no segundo passo. No primeiro passo, os 71 `lora_A` também têm gradiente zero.
- A pergunta do PRD ("gradiente não nulo após 1 passo") foi interpretada como "após um passo do otimizador, no passo seguinte", porque no primeiro backward o gradiente de `lora_A` é zero por construção.
- O teste verifica o tensor inteiro (`abs().max() > 0`). Nas linhas mortas do DoRA, as entradas individuais da magnitude e de `lora_B` têm gradiente zero, e o teste não as distingue.

## 7. Parada e `eos`

Com `eos_token_id=[2, 11]` a geração termina em `<|im_end|>`. Só com o `eos=2` das configs, segue gerando até o limite. O último token foi o 13 (`</think>`) na rodada de 128 tokens, o que mostra que o modelo continuou gerando em vez de parar.

## 8. Lições de processo

- `pkill -f` casou com a linha de comando do próprio shell e o encerrou. Usar `pkill -x`, por nome exato.
- O `exit` de um job não vale se o comando termina em `| grep`: o grep mascara a falha. Os jobs não devem ter pipe de filtro; filtrar só na leitura do log.
- O servidor do tmux sai quando a última sessão termina. `tmux ls` vazio não significa falha: ver `runs/jobs/<nome>.exit` e `.log`.
- Aspas aninhadas no `tmux new-session` expandiram `${PIPESTATUS[0]}` cedo demais. O lançador agora grava um script em arquivo.

## 9. Extração de resposta (A-05 e A-06)

Código em `src/lumi_ft/eval/parse.py`; 70 testes sintéticos em `tests/test_extract.py` (só CPU, 0,04 s).

- **`</think>` e `<think>` não são tokens especiais** (id 12 e 13); `<|im_end|>` é. Então `skip_special_tokens=True` mantém as tags e apaga `<|im_end|>`, e `False` mantém tudo. A extração aceita os dois e corta no primeiro `<|im_end|>`.
- **Regra:** a resposta é o texto depois do primeiro `</think>`. Com raciocínio ligado, a saída gerada começa dentro do traço (o prompt termina em `<think>\n`), então sem `</think>` não existe resposta (`no_close`), mesmo que o traço contenha `Resposta: B` ou `#### 7`. Sem raciocínio, o prompt já termina em `<think></think>` e a saída inteira é a resposta; um `<think>` aberto e nunca fechado também é `no_close`. Status: `ok`, `no_close`, `empty_answer`, `no_option`, `invalid_option`, `no_number`; `rates()` dá a taxa de cada um para o relatório.
- **Múltipla escolha:** a nota é a forma estrita, a última linha `Resposta: <letra>` (aceita negrito, parênteses e ponto final). A forma flexível (`a resposta correta é a letra B`, `gabarito: C`, ou a resposta ser só a letra) é registrada ao lado. Nenhuma letra solta vale, porque `A`, `E` e `O` são palavras em português: `As alternativas A e B estão erradas` não extrai nada, e `Resposta: A resposta correta é B` extrai `B`. O prompt de múltipla escolha ainda não existe e precisa pedir a linha `Resposta: <letra>`.
- **GSM8K-PT:** `strict-match` (primeiro `#### <número>`) e `flexible-extract` (último número do texto) copiam o `gsm8k.yaml` do fork [Polygl0t/lm-evaluation-harness](https://github.com/Polygl0t/lm-evaluation-harness), inclusive as regexes de normalização (`,`, `$`, prefixo até `#### ` e `.` final), para a nota ser comparável. O `flexible-extract` herda uma limitação do harness: `[$0-9.,]{2,}` aceita `...` como número (fixada em teste).
- **Checagem em dados reais** (`Polygl0t/gsm8k-pt`, `test`, 1.295 itens): lendo as próprias referências como se fossem saídas do modelo, 1.292 batem. Os outros 3 têm alvo malformado no dataset (`#### 6mph`, `#### R$ 276.000`, `#### 25%`; índices 18, 225 e 1072) e nenhuma extração os acerta. Além disso, 9 alvos usam `.` como separador de milhar (`2.125`, `114.200`, `1.600`, `65.960`, `6.250`, `14.000`, `6.600`, `1.875`, `40.000`; índices 143, 196, 495, 598, 803, 813, 977, 988 e 1182): quem responde o inteiro correto (`2125`) perde na nota estrita, e quem imita o formato (`2.125`) acerta. **A nota estrita fica limitada a 1.283/1.295 = 99,07% para respostas escritas como inteiro puro.** O efeito é igual para todos os sistemas, mas pode pesar diferente se um modo escrever milhares com ponto mais que o outro. `gsm8k_correct_ptbr` (aceita `70.000` = 70000 e `3,5` = 3.5) é diagnóstico para medir quanto da nota perdida vem só do formato; a nota continua sendo a do harness.
- **Limites:** o harness e os prompts ainda não foram implementados. Os testes provam a extração, não a nota; o comportamento real do Lumi (por exemplo, `70.000` ou `R$ 70.000` no `####`) só aparece na baseline.

## 10. Modelo fundido e formato de chaves

`scripts/check_merged_keys.py` carrega a nativa na CPU, injeta pesos aleatórios nos 71 `lora_B`, funde com `merge_and_unload`, salva com `save_pretrained` e compara com o `model.safetensors` original (resultado em `runs/phase0/merged_keys.json`).

- **Chaves iguais ao original:** 263 de 263, prefixos `backbone` e `lm_head`, mesmas formas e dtypes (BF16), nenhum `lora_`/`base_layer` sobrando. A nativa renomeia `backbone`→`model` ao carregar e desfaz ao salvar.
- **Valores:** exatamente 71 tensores mudaram (os `in_proj`, `up_proj`, `down_proj`, `q/k/v/o_proj` alvo); o salvo é igual, bit a bit, ao fundido em memória. A nativa recarrega o diretório sem chaves faltando nem sobrando.
- **O `config.json` salvo é diferente do original:** a nativa troca `hybrid_override_pattern` por `layers_block_type`, remove `num_hidden_layers` e acrescenta campos de MoE/MTP com valores padrão (`n_routed_experts: 8`, `mamba_ssm_cache_dtype: float32` etc.). O `auto_map` permanece. Quem carregar o modelo com o código do repositório precisa do `config.json` original, não do salvo. **Para entregar o modelo fundido: pesos salvos pela nativa, mais o `config.json`, o `generation_config.json` (com `eos_token_id: [2, 11]` se for decidido), o tokenizer e os `.py` copiados do repositório original.**
- **Não testado:** carregar o diretório entregue com `trust_remote_code=True` e gerar. As chaves são idênticas, então o carregamento do código do repositório deve funcionar, mas isso é inferência. Vale um teste na fase 5 com o modelo realmente treinado.

## 11. Memória e velocidade (A-14)

Medido em 2026-11-06 na RTX 4090 sem outro usuário (o `nvidia-smi` mostra só o processo da medição no início e no fim). Dados brutos em `runs/phase0/train_resources.jsonl` e `runs/phase0/gen_resources.jsonl` (gitignored); scripts `scripts/measure_train.py` e `scripts/measure_generate.py`, retomáveis.

### Treino

Passo completo (forward com loss, backward, AdamW), implementação nativa, pesos BF16, autocast BF16 como no Trainer com `bf16=True`, adaptador em FP32, gradient checkpointing `use_reentrant=False`, `use_cache=False`. Tokens aleatórios de comprimento cheio, sem padding, loss em todas as posições. Três passos cronometrados depois de um de aquecimento.

| Batch (sequências × tokens) | LoRA r=16: pico / tokens/s | DoRA r=32: pico / tokens/s |
| --- | --- | --- |
| 1 × 1.024 | 9,4 GiB / 4.504 | 9,6 GiB / 1.536 |
| 2 × 1.024 | 11,1 GiB / 4.304 | 11,3 GiB / 1.977 |
| 4 × 1.024 | 14,7 GiB / 4.108 | 14,9 GiB / 2.294 |
| 8 × 1.024 | 21,7 GiB / 4.046 | 21,9 GiB / 2.509 |
| 4 × 2.048 | 21,7 GiB / 4.025 | 21,9 GiB / 2.502 |
| 12 × 1.024, 6 × 2.048 | falta de memória | falta de memória |

- **A memória depende só de batch × comprimento:** 7,6 GiB (LoRA r=16) a 7,8 GiB (DoRA r=32) fixos mais 1,72 GiB por 1.000 tokens, igual nos quatro braços. 8 × 1.024 e 4 × 2.048 dão o mesmo pico. O limite é 8.192 tokens por passo; 12.288 dá falta de memória em todos os braços. Os logits em FP32 e o gradiente deles pesam muito nessa inclinação.
- **A GPU tem 23,1 GiB livres depois do contexto CUDA.** Com 8.192 tokens o pico reservado é 22,3 GiB, ou seja, menos de 1 GiB de folga. Por isso o batch físico fixado é 4 sequências: com os dados em até 1.000 tokens, ficam em cerca de 15 GiB; no teto de 2.048 tokens por sequência, continuam cabendo (21,9 GiB).
- **`r` quase não muda nada:** 0,2 GiB de memória entre r=16 e r=32; a velocidade é igual até a terceira casa.
- **LoRA já é limitado por computação:** a vazão cai de 4.504 (batch 1) para 4.046 tokens/s (batch 8). 4.100 tokens/s equivalem a ~100 TFLOP/s efetivos (6 × parâmetros × tokens, com o forward refeito pelo checkpointing).
- **DoRA custa 1,8 vez o tempo do LoRA por token com batch 4** (2.294 contra 4.108 tokens/s), e sobe até 2.509 com batch 8 porque amortiza o cálculo da norma de pesos. Com batch 1 é 2,9 vezes mais lento.
- **Horas de GPU no pior caso** (18 execuções, 3 épocas sem parada antecipada, dados sem filtrar: 34,8 M de tokens de raciocínio são 75% de uma época de 46,4 M): LoRA 658 M de tokens = 44,5 h; DoRA 501 M = 60,7 h; **total 105 h (4,4 dias contínuos)**. Parada antecipada e os filtros D-06 reduzem isso; a conta usa a velocidade com batch 4.
- **Limites desta medição:** não inclui o TRL (métricas de entropia e acurácia por token, collator) nem sequências reais com padding e máscara de labels. Medir um passo do `SFTTrainer` no começo da fase 3, antes de confiar na folga.

### Geração

Implementação nativa, modelo BF16 sem adaptador (o fundido tem a mesma forma), greedy, `eos_token_id=[2, 11]`, prompt com o gabarito do Lumi com raciocínio ligado (`<think>\n`) em volta de ids aleatórios, **4.096 tokens novos forçados** (`min_new_tokens=4096`), para o cache chegar ao máximo.

| Prompt | Batch | Pico reservado | ms/passo | Tokens/s (decode) |
| --- | --- | --- | --- | --- |
| 1.024 | 8 | 9,8 GiB | 12,7 | 628 |
| 1.024 | 32 | 16,9 GiB | 21,7 | 1.472 |
| 1.024 | 48 | 21,6 GiB | 28,1 | 1.709 |
| 1.024 | 56 | 22,7 GiB | 30,7 | 1.823 |
| 1.024 | 64 | falta de memória no prefill | | |
| 512 | 32 | 14,9 GiB | 20,9 | 1.534 |
| 512 | 64 | 22,6 GiB | 31,9 | 2.008 |
| 512 | 96 | falta de memória no prefill | | |
| 256 | 32 | 13,2 GiB | 20,4 | 1.568 |
| 256 | 64 | 21,6 GiB | 31,0 | 2.064 |
| 256 | 96 | 23,0 GiB | 44,6 | 2.151 |
| 256 | 128 | falta de memória no prefill | | |

- **O limite é o estado do Mamba e o prefill, não o cache de atenção.** Cada sequência custa cerca de 82 MiB fixos (ajuste das medidas; o estado SSM em FP32 calculado da config dá 78,75 MiB: 96 cabeças × 80 × 128 × 4 bytes × 21 camadas) mais cerca de 0,19 MiB por token do prompt durante o prefill; o cache de atenção custa só 16 KiB por token (4 camadas). Os três comprimentos seguem essa fórmula (0,27, 0,176 e 0,129 GiB por sequência em 1.024, 512 e 256 tokens). `logits_to_keep=1` já vale na geração.
- **Maior batch que cabe com 4.096 tokens novos:** 56 (prompt de 1.024), 64 (512) e 96 (256), todos a menos de 1 GiB do limite. A vazão satura em 1,8 a 2,2 mil tokens/s.
- **Batch recomendado para a avaliação: 32**, com 16,9 GiB de pico reservado no pior caso de 1.024 tokens e 1.470 a 1.570 tokens/s. Prompts mais longos pedem batch menor, por uma regra determinística que só depende do prompt (por exemplo, `batch = min(32, 32.768 // maior_prompt_do_lote)`), ordenando os prompts por comprimento. Isso mantém o tamanho e a ordem fixos, como A-04 exige. Falta medir os comprimentos de prompt reais das provas; o ENEM pode passar de 1.024 tokens.
- **O custo real é maior que tokens ÷ vazão.** O `generate` em batch só termina quando a sequência mais longa do lote termina, então o tempo de um lote é (maior saída do lote) × ms/passo, e uma saída que entra em loop até 4.096 tokens segura as outras 31. Se 3% das saídas fossem truncadas (hipótese), 62% dos lotes de 32 conteriam ao menos uma. A conta da seção "Custo estimado" do PRD (tokens úteis ÷ 1.000 a 2.000 tokens/s) ignora isso. Cenários, não medidas, a 21,7 ms/passo (batch 32) e 5.113 lotes com raciocínio e 6.853 sem (matriz A-12: 163.610 itens com raciocínio e 219.285 sem):

| Maior saída média por lote | Com raciocínio (A-12) | Sem raciocínio (A-12) |
| --- | --- | --- |
| 300 a 800 tokens | | 12 a 33 h |
| 1.000 tokens | 31 h | |
| 2.000 tokens | 62 h | |
| 4.096 tokens (todo lote tem ao menos um truncado) | 126 h | |

  A soma dos dois lados do A-12 fica entre cerca de 43 h e 159 h, contra 13 a 26 h no PRD. A distribuição de comprimentos só sai da baseline; vale rodar primeiro um piloto de algumas centenas de itens para medi-la. Se o piloto confirmar a cauda, as saídas são ficar com esse custo, reduzir o orçamento de A-12, ou trocar o `generate` por um servidor com batching contínuo (o vLLM tem suporte a NemotronH), o que exige reinstalar e repetir a validação de equivalência da seção 5.

## 12. vLLM e outras otimizações

Medido em 2026-11-07, motivado por prazo (projeto inteiro em 2 dias). Scripts: `scripts/install_vllm.sh`, `scripts/run_gsm8k_vllm.py`; dados em `runs/pilot_gsm8k_vllm/` e `runs/baseline_gsm8k_vllm/` (gitignored).

### Instalação

- **Ambiente separado `.venv-vllm`.** O vLLM 0.33.0 fixa `torch==2.14.0` e `transformers>=5.16.1,<5.21`; o `.venv` tem torch 2.14.1, transformers 5.22.2 e os kernels compilados contra ele. Instalar no `.venv` quebraria o lockfile e os kernels. O torch 2.14.0 do PyPI é CUDA 13.0.3 (`cuda-toolkit==13.0.3`), compatível com o driver 580; o vLLM depende de `nvidia-cutlass-dsl[cu13]`. Resultado: torch 2.14.0+cu130, vLLM 0.33.0, transformers 5.20.2.
- **O vLLM aceitou o Lumi sem `trust_remote_code`** (arquitetura `NemotronH`). Usa os próprios kernels Triton de Mamba2 (`mamba_mixer2`), não o `mamba-ssm` compilado.
- **Falha corrigida:** o kernel de amostragem do FlashInfer é compilado na hora com o `nvcc` do conda, que não tem `curand.h` (`fatal error: curand.h`). Solução: `VLLM_USE_FLASHINFER_SAMPLER=0` (amostrador do PyTorch, que serve para greedy e para temperatura/top-p).
- **Cache de KV:** com `gpu_memory_utilization=0.88`, 12,0 GiB de cache (245 mil tokens); o estado do Mamba (~82 MiB por sequência) sai do mesmo pool, então `max_num_seqs=128` é o que cabe com folga.

### Equivalência com o HF (piloto de 256 itens, mesmos prompts e pontuação)

| | HF | vLLM |
| --- | --- | --- |
| Com raciocínio: nota estrita | 91,0% | 90,6% (6 só o HF acerta, 5 só o vLLM) |
| Sem raciocínio: nota estrita | 87,9% | 88,3% (3 só o HF, 4 só o vLLM) |
| Fecha `</think>`, traço em inglês | 98,4%, 252/252 | 98,0%, 251/251 |

Mesmo comportamento. **Mas a saída do vLLM não é reprodutível entre execuções** (batching contínuo: a composição do lote varia com o agendamento): em duas execuções do mesmo comando, o texto é idêntico em só 23% dos itens com raciocínio e 71% sem; a nota sem raciocínio foi 88,3% e 89,4% (3 itens de 256). A diferença entre HF e vLLM tem o mesmo tamanho dessa variação entre execuções do próprio vLLM. Consequências: (1) a regra de A-04 de fixar tamanho e ordem do lote não se aplica ao vLLM; (2) o ruído por execução (~±1 ponto em n=256, bem menor nas provas grandes) não entra no IC 95% do bootstrap por item; (3) toda a avaliação, baseline incluída, deve passar pelo mesmo motor.

### Velocidade

| | HF (lotes de 32) | vLLM (`max_num_seqs=128`) |
| --- | --- | --- |
| 256 itens, com raciocínio | 527 s | 64 s (8,2×) |
| 256 itens, sem raciocínio | 178 s | 14 s (12×; 45 s no 2º run, por um item de 4.096 tokens sozinho) |
| 1.295 itens (GSM8K-PT inteiro), com raciocínio | ~59 min (extrapolado) | 233 s, 5,55 itens/s, 2.938 tokens/s |
| 1.295 itens, sem raciocínio | | 116 s, 11,15 itens/s, 2.441 tokens/s |

Com batching contínuo, uma saída em loop não segura as outras: o custo da cauda cai para os tokens que ela gasta. Em lotes pequenos, o retardatário ainda domina o tempo total (por isso o piloto de 256 subestima a vazão em regime).

### Baseline do GSM8K-PT inteiro (vLLM, uma execução, greedy, 4.096 tokens)

| | Com raciocínio | Sem raciocínio |
| --- | --- | --- |
| Nota estrita / flexível / leitura PT-BR | 88,88% / 88,88% / 89,42% | 85,17% / 85,17% / 85,56% |
| Truncadas em 4.096 | 3,0% | 1,4% |
| Fecha `</think>` e responde | 97,0% | – |
| Idioma do traço (heurística) | en 1.255, misto 1 | – |
| Último número do traço é a resposta | 91,6% | – |

Diferença pareada (com − sem raciocínio): +3,71 pontos, IC 95% [+1,85, +5,56] (bootstrap, 10.000 reamostras): o raciocínio ajuda de forma estatisticamente clara no GSM8K-PT, ao contrário do que o piloto de 256 itens deixava em aberto. O Lumi com raciocínio, a referência a superar, está em 88,9% (no piloto de 256 saiu 91%: ruído de amostra).

### Custos reprojetados (extrapolação do GSM8K; as provas de múltipla escolha podem diferir)

| | HF (estimado) | vLLM |
| --- | --- | --- |
| Baseline do Lumi, 11 provas, dois modos (73.095 itens cada) | ~56 h | ~5,5 h (3,7 h com + 1,8 h sem) |
| Matriz A-12 inteira | ~136 h | ~13,7 h (8,2 h + 5,5 h) |

### Treino sem gradient checkpointing (medido com `measure_train.py --ckpt none`)

| Braço | Batch (× 1.024 tokens) | Pico | Tokens/s | Com checkpointing |
| --- | --- | --- | --- | --- |
| LoRA r=16 | 1 | 12,7 GiB | 6.049 | 4.504 |
| LoRA r=16 | 2 | 17,8 GiB | 5.785 | 4.304 |
| LoRA r=16 | 3 | falta de memória | | |
| DoRA r=16 | 1 | 15,3 GiB | 2.485 | 1.541 |
| DoRA r=16 | 2 | falta de memória | | |

+34% no LoRA (batch 2) e +62% no DoRA (batch 1). O batch físico diferente entre os braços não muda a otimização se o batch efetivo for o mesmo, com acumulação de gradiente. Uma época completa (46,4 M de tokens): LoRA 2,2 h, DoRA 5,2 h. A matriz de 18 execuções do PRD (3 épocas, sem parada antecipada) cairia de ~105 h para ~88 h (LoRA 32 h + DoRA 56 h), ainda longe de caber em 2 dias.

### Não testado

Fused cross-entropy (Liger) para liberar memória dos logits e permitir batch maior; `torch.compile`; empacotamento de sequências; FP8 (treino ou pesos no vLLM, que mudaria a numérica contra o BF16 de A-11); ajuste fino de `max_num_seqs` e `gpu_memory_utilization`; vLLM com adaptador LoRA sem fundir. O modelo fundido para o vLLM precisa do `config.json` original (a seção 10 mostra que o salvo pela nativa difere).

## 13. Pendências da fase 0

1. Baseline do Lumi nos dois modos (precisa de aprovação, por passar de 1 hora de GPU). Antes dela, falta escrever o harness (prompts, laço de geração em lotes ordenados, gravação por item) e um piloto para medir o comprimento das saídas.
2. Medir um passo do `SFTTrainer` (fase 3) e carregar a entrega fundida com `trust_remote_code` (fase 5).
3. Isolar a causa do erro de padding à esquerda no código do repositório, se for útil. Não é necessário para seguir com a nativa.

## 14. Como reproduzir

```bash
source env.sh
scripts/build_kernels.sh                       # compila os kernels (não foi executado de ponta a ponta; o build real seguiu os mesmos passos à mão)
LUMI_IMPL=native python -m pytest tests/       # 83 testes (13 de modelo na GPU, 70 de extração na CPU)
PYTHONPATH=src python scripts/compare_impls.py # remote x nativa, geração
PYTHONPATH=src python scripts/diag_impls.py    # sensibilidade ao padding
scripts/detach.sh merged-keys PYTHONPATH=src python scripts/check_merged_keys.py   # formato de chaves do fundido (CPU)
scripts/detach.sh --gpu measure-train PYTHONPATH=src python scripts/measure_train.py       # ~5 min
scripts/detach.sh --gpu measure-gen PYTHONPATH=src python scripts/measure_generate.py     # ~5 min; outros comprimentos: --prompt-len 512 --ladder ...
scripts/detach.sh --gpu pilot-gsm8k PYTHONPATH=src python scripts/run_gsm8k.py --n 256 --out runs/pilot_gsm8k   # HF, ~12 min
scripts/detach.sh install-vllm scripts/install_vllm.sh                                      # cria .venv-vllm
scripts/detach.sh --gpu gsm8k-vllm "VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/run_gsm8k_vllm.py --n 0 --out runs/baseline_gsm8k_vllm"   # ~6 min
PYTHONPATH=src python scripts/chat.py                                                       # REPL para conversar com o Lumi
```

`LUMI_IMPL=remote` roda a suíte no código do repositório; nesse caso o teste de geração falha, como descrito na seção 3.
