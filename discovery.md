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

## 13. Fase 1: dados

Código em `src/lumi_ft/data/` e `scripts/build_dataset.py`, `scripts/build_validation.py`; testes em `tests/test_data.py` (29). Saída em `data/` (fora do Git) e `reports/data_manifest.json`, `reports/val_manifest.json` (hashes).

### Template do Lumi (D-01 a D-03)

- Com raciocínio, o prompt (`...<|im_start|>assistant\n<think>\n`) é prefixo exato do exemplo renderizado, em texto e em ids. O template escreve `traço\n</think>\nresposta<|im_end|>\n`; o dataset traz `</think>\n\n`, então o traço e a resposta vão em `reasoning_content` e `content` (D-02).
- Sem raciocínio, o prompt termina em `<think></think>` e o exemplo é `prompt + resposta<|im_end|>\n`. Com `enable_thinking=True` e sem `reasoning_content`, o template também escreve `<think></think>`.
- A loss cobre do fim do prompt até o último `<|im_end|>` (id 11), inclusive; o `\n` depois dele fica de fora. `render_example` falha se o prompt não for prefixo; 0 falhas de renderização nos 78 mil exemplos.
- Cada exemplo guarda só `input_ids`, `n_prompt` e `loss_end`; os labels saem daí (`train/sft.py: labels_for`, igual a `render_example`, testado).

### Funil do subset `reasoning` (78.249 linhas)

| Etapa | Restam | Removidas | Observação |
| --- | --- | --- | --- |
| Um turno, formato `<think>...</think>` válido | 78.249 | 0 | todas |
| Prompts duplicados (D-06; fica o de maior `instruct_score`) | 75.069 | 3.180 | o PRD contou 3.112 linhas duplicadas |
| Resíduo do pipeline ("contexto fornecido", "resposta fornecida") | 67.618 | 7.451 | 9,9%; o PRD estimou 11,4% com regex mais larga |
| Último número do traço fora da resposta (D-07, só `math_reasoning`) | 60.346 | 7.272 | 11,6% da matemática: em geral o traço só repete o enunciado |
| Traço em inglês ou misto (D-09, `shape.trace_language`) | 60.269 | 77 | 0,1%: os traços do dataset já são em português |
| 8-gramas em comum com as provas (D-05) | 58.424 | 1.845 | 3,1% |
| Mais de 1.024 tokens renderizados (D-08, descarta, nunca trunca) | 57.754 | 670 | o limite inclui o prompt e o system |

Por tipo: `math_reasoning` 66.067 → 46.528 e `reasoning` 12.182 → 11.226.

- **Descontaminação:** índice de 2.964.512 8-gramas de palavras de 46.440 itens: GSM8K-PT `test`, MMLU-ProX PT (`test` e `validation`), ENEM, BLUEX, OAB, Global-MMLU PT (`test` e `dev`), MMMLU PT, INCLUDE PT (`test` e `validation`). O `train` do GSM8K-PT fica de fora de propósito (é a fonte da validação). **Faltam o MMLU PT (ensino médio e graduação) e o IFEval PT**, cujos ids o PRD não dá; o MMLU PT é tradução do MMLU como o Global-MMLU e o MMMLU, então a maior parte do risco está coberta, mas não é a mesma coisa.
- **Detector de idioma:** a heurística `trace_language` (lista de palavras funcionais), não um detector de verdade. Como removeu só 77, o efeito é pequeno em qualquer caso.
- **Revisão dos filtros (D-06):** olhei 3 amostras de cada filtro numa prévia de 3.000 linhas (resíduo, último número, descontaminação): faziam o que se espera. Uma revisão maior dos removidos está em `runs/data/removed_samples.json` (15 por filtro) e não foi feita.

### Mistura e conjuntos

- **Sem raciocínio** (25% dos tokens, sorteio de semente fixa, `instruct_score` ≥ 4,0, um turno, sem tags `<think>`, sem prompts duplicados, sem 8-gramas das provas, até 1.024 tokens): general 4,40 M, structured 0,73 M, system_prompts 0,73 M, rewriting 0,73 M, summarization 0,73 M, math 0,92 M e retrieval 0,92 M de tokens, todos no orçamento. Um shard de cada subset basta.
- **`train.parquet`:** 87.927 exemplos, 36,0 M de tokens, 75,0% deles com raciocínio. **`valid.parquet`** (2% por estrato, D-11): 1.798 exemplos, 0,7 M de tokens. **Viabilidade:** 15.544 exemplos (10.000 com raciocínio mais os sem raciocínio até 25% dos tokens), 6,35 M de tokens. Os subconjuntos de 20% são `bucket < 2.000` (hash estável do id).
- **Validação de A-10** (`data/val_gsm8k.jsonl`, `data/val_mcq.jsonl`): 1.000 itens do GSM8K-PT `train` escolhidos entre os 2.718 de 7.463 (36,4%) sem 8-gramas em comum com o treino final, mais 385 de múltipla escolha (Global-MMLU `dev` 285, MMLU-ProX `validation` 70, INCLUDE `validation` 30), nenhum com 8-grama no treino.
- **Revisão humana (D-10):** `reports/review_d10.md` tem 200 traços (100 de cada tipo). **Ninguém revisou ainda.** O PRD exige a revisão antes do treino completo.

### Provas de múltipla escolha

Carregadores em `eval/mcq.py`, testados contra as contagens do PRD: MMLU-ProX PT 11.759, Global-MMLU PT 14.042, MMMLU PT 14.042, INCLUDE PT 551, OAB 2.210. **ENEM 1.431 de 1.432 e BLUEX 722 de 724:** os itens cujo `answerKey` não está entre os rótulos das alternativas saem (não dá para pontuar). O MMLU-ProX tem de 3 a 10 alternativas (9.826 de 11.759 com 10); o prompt usa as letras A a J conforme o item.

## 14. Treino: laço, memória e retomada

Código em `src/lumi_ft/train/sft.py` e `scripts/train_sft.py`; testes em `tests/test_train.py` (15). Laço próprio, sem TRL, para controlar memória e acumulação.

- **Loss normalizada pelo total de tokens-alvo do batch efetivo**: a soma por micro-batch dividida pelo total do batch de 32 sequências. Testado: a loss acumulada é igual à de um batch único, qualquer que seja o fatiamento (LoRA com micro-batches de 2, DoRA de 1). Os logits não são copiados (desloca-se o label, não o logit).
- **Micro-batches por orçamento de tokens** (2.048 no LoRA, 1.024 no DoRA), ordenados por comprimento dentro do batch efetivo: pares de comprimento parecido, pouco padding.
- **Retomável:** `ckpt/last.pt` (adaptador, otimizador, passo) a cada 200 passos e ao fim da época; a ordem dos dados só depende de (semente, época). Testado: um run interrompido no passo 3 e retomado do 2 reproduz o passo 3 com a mesma loss (0,7368) e o mesmo lr.
- **Medido nos dados reais (smoke de 3 a 5 passos):** LoRA r=16: loss inicial 0,58 a 0,74 por token (o modelo base já prevê bem esses traços), 5,6 a 5,7 mil tokens/s (a medição sintética deu 5,8), pico de 18,9 GiB alocados e 19,8 GiB reservados de 23,1 livres. DoRA r=16: loss 0,63, 0,58 e 0,73 nos mesmos passos (igual ao LoRA, como deve ser no começo), sem NaN (o patch funciona no laço real), 15,9 GiB, ~2,1 mil tokens/s (abaixo dos 2,5 mil sintéticos: micro-batches de 1 sequência).
- **Horas por época completa** (36,0 M de tokens): LoRA ~1,8 h; DoRA ~4,6 h a 2,1 mil tokens/s.

## 15. Fase 2: viabilidade (gate falhou na nota)

LoRA `r=16`, lr 1e-4, 1 época nos 15.544 exemplos de viabilidade (10.000 com raciocínio): 486 passos, 21,7 min, pico de 19,0 GiB. Loss de treino 0,53 (primeiro décimo) para 0,40 (último); loss de validação 0,443. Fundido em BF16 com `scripts/merge_adapter.py` (263 chaves, 71 tensores alterados) e avaliado no vLLM nos dois conjuntos de validação (1.000 itens do GSM8K-PT `train`, 385 de múltipla escolha). Dados brutos em `runs/eval/{base,viab}/`; comparações em `reports/compare_viab_vs_base.md`.

| | Lumi, com raciocínio | Viabilidade, com raciocínio | Lumi, sem | Viabilidade, sem |
| --- | --- | --- | --- | --- |
| GSM8K val, nota estrita | 85,7% | 71,5% | 81,4% | 61,2% |
| GSM8K val, nota flexível | 85,7% | 80,3% | 81,5% | 77,6% |
| Múltipla escolha val, estrita | 69,6% | 43,4% | 63,9% | 43,1% |
| Múltipla escolha val, flexível | 69,4% | 53,0% | 63,4% | 56,6% |
| Múltipla escolha val, letra em qualquer lugar | 69,6% | 55,1% | 63,9% | 56,6% |
| Fecha `</think>` e responde (GSM8K / múltipla escolha) | 95,4% / 86,2% | 99,7% / 99,2% | | |
| Traço em português (heurística) | 0% | 99,7% / 100% | | |
| Truncadas em 4.096 (GSM8K / múltipla escolha) | 4,7% / 13,8% | 0,3% / 0,8% | 2,0% / 1,0% | 3,3% / 2,3% |

**O que funcionou (forma):** o traço sai em português, `</think>` fecha e há texto depois quase sempre, e os loops que truncavam o Lumi quase sumiram. As metas de forma do PRD (≥ 95% e ≥ 90%) estão cumpridas.

**O que falhou (nota):** com raciocínio, a diferença contra o Lumi sem raciocínio no GSM8K de validação é −9,9 pontos no strict (IC 95% [−12,7, −7,1]) e −1,2 no flexível ([−3,4, +1,0]); na múltipla escolha, −20,5 e −10,4. O gate exige IC acima de 0.

**Quanto é formato e quanto é capacidade:**
- Formato: o modelo escreve `Resposta: B` ou o número dentro do traço e, depois do `</think>`, responde em prosa ("a resposta correta é B", "60 dias."). A extração só olha depois do `</think>` (A-05) e exige `#### N` ou `Resposta: <letra>` (A-06), então o strict falha em 10,7% (GSM8K) e 26% (múltipla escolha) com raciocínio, e 23% e 32% sem. Os dados de treino não têm esses marcadores, e o SFT enfraqueceu o seguimento da instrução de formato (o Lumi segue em ~98% dos casos que terminam). Isso também derrubou o modo sem raciocínio: strict −20 pontos nas duas provas.
- Capacidade: com a leitura mais generosa (qualquer letra em qualquer lugar do texto), a múltipla escolha com raciocínio é 55,1% contra 69,6% do Lumi, e sem raciocínio 56,6% contra 63,9%. No GSM8K (flexível), 80,3% contra 85,7% com raciocínio e 77,6% contra 81,5% sem. Ou seja, há perda real de acerto, maior fora de matemática. O treino tem 84% de `math_reasoning`, traços mais curtos que os do Lumi (GSM8K de validação: mediana de 285 tokens contra 358 do Lumi no GSM8K de teste, média de 530), e, no dataset, 4.007 dos 12.182 exemplos `reasoning` usam prompts de `harmfull-reasoning` (as amostras lidas são recusas). É o risco que o PRD já registrou (o Tucano 3.7B Think, com os mesmos dados, ganhou 2,89 pontos no GSM8K-PT e perdeu nas outras provas).
- Limites desta leitura: uma execução, 1 época, uma taxa; não separei quanto da perda vem do idioma do traço, do comprimento menor ou da deriva do LoRA sobre dados diferentes da distribuição do Lumi. Li ~5 saídas, não 50 (item da fase 2 não cumprido).

### Teste de prefill e de instrução no Lumi original (validação, só com raciocínio)

Para saber se o Lumi raciocina em português sem treino. `scripts/eval_vllm.py --prefill` força o começo do traço; `--extra` acrescenta uma frase ao prompt. Dados em `runs/eval/base_ptprefill/` e `runs/eval/base_ptinstr/`.

| | Lumi | + prefill "Vou raciocinar em português." | + "Pense e raciocine em português." no prompt |
| --- | --- | --- | --- |
| GSM8K val, nota estrita | 85,7% | 48,1% | 85,2% |
| Múltipla escolha val, nota estrita | 69,6% | 61,3% | 70,1% |
| Idioma do traço (GSM8K) | inglês 954/954 | português 570/570 (nos que fecham) | inglês 956/957 |
| Fecha `</think>` e responde (GSM8K / múltipla escolha) | 95,4% / 86,2% | 57,0% / 79,7% | 95,7% / 87,8% |

- **A instrução no prompt não muda o idioma do traço:** o Lumi segue pensando em inglês, e por isso a nota não muda.
- **O prefill muda o idioma, mas quebra a estrutura.** No GSM8K, 361 das 1.000 saídas terminaram sozinhas **sem fechar o `</think>`**, com a solução e o `#### N` escritos dentro do traço (329 delas certas), e 69 foram truncadas. Das 570 que fecham, 481 estão certas (84,4%). Lendo o texto inteiro, inclusive o que está dentro do traço, o acerto é **81,3%** (o Lumi: 85,7%): raciocinar em português preservou quase toda a capacidade, e o que se perde é o formato. Na múltipla escolha, 18,4% truncam e 20% não fecham o traço.
- **Uma execução, uma frase de prefill**; não testei outras frases. A leitura "texto inteiro" é minha e só serve para diagnóstico, não é a nota do PRD.

**Opções que o PRD prevê (gate):** mais dados e ajustes (a varredura de taxa e mistura da matriz reduzida), traços gerados e verificados (outro PRD), ou encerrar.

## 16. Contaminação da validação e experimento com o dataset do adalberto

### A validação do GSM8K estava contaminada (o teste não)

O filtro de 8-gramas só pega a mesma tradução. Os dados do GigaVerbo e o dataset do adalberto trazem problemas do GSM8K `train` em **outras traduções**. Comparando a assinatura dos números do enunciado (`teacher.number_signature`) e a sobreposição de palavras (Jaccard):

| | Mesma assinatura de números | Jaccard ≥ 0,3 (duplicata provável) |
| --- | --- | --- |
| Validação do GSM8K (1.000) contra o treino de matemática do GigaVerbo | 599 | ~325 (112 com ≥ 0,5) |
| **Teste** do GSM8K-PT (1.295) contra todo o treino | 473 | **0** (todas < 0,3: coincidência de números) |

- **O teste está limpo**; a validação não. `scripts/clean_validation.py` gera o subconjunto limpo (676 itens, `data/val_gsm8k_clean_ids.json`, contra o GigaVerbo e o adalberto); o `compare_runs.py --only-ids` o usa.
- **Efeito no viab:** na validação limpa, 67,2% (estrito, com raciocínio) contra 71,5% na completa; o Lumi, 81,7% contra 85,7%. A contaminação inflava o candidato em ~4 pontos, e a conclusão da fase 2 não muda.
- O pacote da Thor (`handoff/thor`) foi gerado antes disso. Os problemas dele não têm 8-grama da validação, mas podem ter duplicatas por tradução; `scripts/prepare_thor.py` as tira quando o resultado voltar.

### Dataset `adalberto-temp/gsm8k-sft-thinking-pt`

56.296 respostas do Qwen3-30B-A3B-Thinking para 7.225 problemas do GSM8K `train` (mediana de 8 respostas por problema, 404 palavras por resposta), **sem licença declarada** (só para experimento). Com o prompt dele, o Qwen já pensa em português. Só 207 problemas casam exatamente com o GSM8K-PT `train`; 805 dos 1.000 itens da nossa validação têm 8-grama em comum com ele.

- **Seleção** (`scripts/prepare_adalberto.py`): problema com resposta majoritária em ≥ 75% das respostas; resposta igual à maioria (verificação por concordância, não por gabarito); um único `</think>`; traço em português (heurística); última linha `Resposta final: X` trocada por `#### X`. Fora, por contaminação: 81 problemas com 8-grama do teste, 807 com 8-grama da validação e 706 com os mesmos números de um item da validação. Sobram 5.627 problemas e 41.143 respostas boas (2.678 descartadas por traço fora do português).
- **Treino** (`scripts/build_teacher_dataset.py`): prompt = o da avaliação (instrução `#### N`), sem system. 5.842 exemplos com raciocínio (4,6 M de tokens; o limite de 1.024 tokens descartou 9.680 candidatos, 34%), 3.000 sem raciocínio no mesmo formato (1,5 M) e as 5.544 linhas sem raciocínio do GigaVerbo da viabilidade: 14.386 linhas, 7,7 M de tokens. LoRA `r=16`, lr 1e-4, 1 época, 450 passos, 25,7 min, loss de treino 0,42 → 0,21.

| Validação | Lumi | Adalberto | Diferença (IC 95%) |
| --- | --- | --- | --- |
| GSM8K limpo (676), com raciocínio | 81,7% | **83,3%** | +1,6 (−1,0, +4,3) |
| GSM8K limpo, com raciocínio, contra o Lumi **sem** raciocínio (77,2%) | | | **+6,1 (+3,4, +8,7)** |
| GSM8K limpo, sem raciocínio | 77,2% | 79,3% | +2,1 (−0,7, +4,9) |
| Múltipla escolha (385), com raciocínio | 69,6% | 62,3% | **−7,3 (−11,7, −2,9)** |
| Múltipla escolha, sem raciocínio (estrita) | 63,9% | 58,2% | **−5,7 (−10,6, −0,8)** |
| Múltipla escolha, sem raciocínio (flexível) | 63,4% | 63,9% | igual |
| Traço em português (GSM8K / múltipla escolha) | 0% | 968 de 973 / 332 de 334 | |
| Fecha `</think>` e responde (GSM8K / múltipla escolha) | 95,4% / 86,2% | 97,3% / 86,8% | |
| Truncadas em 4.096 (GSM8K / múltipla escolha) | 4,7% / 13,8% | 3,2% / 13,8% | |

- **Em matemática funcionou:** traço em português e nota igual ou melhor que a do Lumi nos dois modos; o modo com raciocínio supera o Lumi sem raciocínio com IC acima de 0 (a meta da fase 2).
- **Fora de matemática não:** a múltipla escolha com raciocínio perde 7,3 pontos. Sem raciocínio, a perda estrita (5,7) é de formato: a nota flexível é igual à do Lumi. Os 13,8% de truncamento na múltipla escolha são os loops do Lumi, que estes dados não corrigem.
- **Limites:** uma execução, uma taxa, 1 época; os problemas vêm do GSM8K `train` (mesma distribuição do teste, então o ganho em matemática pode ser em domínio); a licença do dataset é desconhecida; as provas de teste ainda não foram rodadas.

## 17. Pacote da Thor e autodestilação do Lumi

- **`handoff/thor/`:** `PLANO_THOR.md` (para o Claude Code da Thor), `gen_traces.py` (cliente de `/v1/completions`, retomável), `filter_traces.py`, `pack.sh` e 50.908 problemas embaralhados (5.612 do GSM8K-PT `train` com gabarito e 45.296 do GigaVerbo com referência fraca). Professor: Qwen3-30B-A3B-Thinking-2507, com o começo do raciocínio forçado em português. Testado só contra um servidor falso (`tests/test_thor_pkg.py`); o motor real ainda não.
- **Autodestilação do Lumi** (`scripts/gen_selfdistill.py`; resultado na §20.1, pior que o Lumi): 14.000 problemas com raciocínio (prefill "Vou raciocinar em português.", T=0,6) e 3.000 sem raciocínio, aceitando só o que fecha `</think>`, está em português e acerta a referência. Depois, `build_teacher_dataset.py`, treino e avaliação (`scripts/run_experiment.sh`).

## 18. Pendências

1. Baseline do Lumi na suíte de teste (`eval_vllm.py --name base --tasks suite`, depois `card`), pausada para liberar a GPU. Os candidatos também precisam passar pela suíte de teste, não só pela validação.
2. Resultados da fila da GPU: GRPO (`grpo1`, rodando desde 20:11) e `adalberto_lr5e5`. Já saíram: DoRA e Nemotron 4B de referência (§20.10), diagnóstico português x inglês (§20.11), escala 0,5 e 0,75 e limite de 2.048 tokens (§20.6).
3. Tradução em escala: o piloto (§20.7) foi aprovado; a tradução dos exemplos do pacote 2 e do pacote 3 será feita na Thor com o Nemotron 30B (§20.11, `handoff/thor3/`).
4. Entregas da Thor: pacote 1 (`thor_out.tar.gz` → `data/thor/`, `prepare_thor.py`) e pacotes 2 e 3 (`thor3_out.tar.gz` → `data/thor3/`; `scripts/prepare_thor_translated.py` gera os registros em português e o par em inglês, depois `filter_translated.py`), depois `build_teacher_dataset.py`, treino e avaliação. Os pacotes 1 e 2 estão rodando na Thor; o pacote 3 está pronto para ir.
5. ~~Registrar no PRD o que só está aqui~~: feito em 2026-11-07 (§20.9).
6. Revisão humana de `reports/review_d10.md` (D-10), se os dados do GigaVerbo continuarem no treino.
7. Provas do cartão que faltam: MMLU PT ×2 e IFEval PT (ids desconhecidos). Licença do dataset do adalberto (não declarada: só experimento).
8. Medir um passo do `SFTTrainer` (fase 3) e carregar a entrega fundida com `trust_remote_code` (fase 5).

## 19. Como reproduzir

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
scripts/detach.sh merge-s PYTHONPATH=src python scripts/merge_adapter.py --adapter runs/train/NOME/adapter --out runs/merged/NOME_s050 --scale 0.5   # CPU
scripts/detach.sh --gpu eval-x scripts/eval_merged.sh NOME_s050                             # avalia fundidos na validação
PYTHONPATH=src python scripts/prepare_mot_science.py                                       # data/mot_science_en.jsonl
scripts/detach.sh --gpu translate "VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/translate_traces.py --n 200 --out data/mot_pilot"   # piloto
PYTHONPATH=src python scripts/filter_translated.py data/mot_pilot                          # descontamina
```

`LUMI_IMPL=remote` roda a suíte no código do repositório; nesse caso o teste de geração falha, como descrito na seção 3.

## 20. Resultado do selfdist, levantamento de datasets e ordem seguida

### 20.1 Autodestilação do Lumi (`selfdist`): pior que o Lumi

- **Geração** (`gen_selfdistill.py`, ~1 h): 14.000 problemas com raciocínio forçado em português, 6.742 aceitos (descartados: 4.125 sem um único `</think>`, 1.938 resposta errada ou sem `#### N`, 1.158 truncados, 37 com traço fora do português); 3.000 sem raciocínio, 2.212 aceitos. 603 problemas saíram por duplicar a validação.
- **Treino:** 6.365 com raciocínio + 2.204 sem raciocínio + 5.544 linhas do GigaVerbo (as mesmas do adalberto) = 14.113 linhas, 5,59 M de tokens. LoRA r=16, lr 1e-4, 1 época, 441 passos, ~18 min. O construtor ganhou `--disjoint-inputs`: sem ele a divisão 70/30 por hash descartava 71% dos exemplos sem raciocínio, porque aqui os problemas de cada modo já eram conjuntos separados.
- **Resultado** (validação, nota estrita; GSM8K = 676 itens limpos, múltipla escolha = 385):

| Modo | Prova | Lumi | `selfdist` | Diferença (IC 95%) | `adalberto` |
| --- | --- | --- | --- | --- | --- |
| thinking | GSM8K | 81,7% | 78,1% | −3,6 (−6,5, −0,7) | 83,3% |
| thinking | múltipla escolha | 69,6% | 57,9% | −11,7 (−16,4, −7,0) | 62,3% |
| sem thinking | GSM8K | 77,2% | 77,5% | +0,3 (−1,8, +2,4) | |
| sem thinking | múltipla escolha | 63,9% | 63,4% | −0,5 (−4,7, +3,6) | |

  Contra o `adalberto` (thinking): GSM8K −5,2 (−8,0, −2,5); múltipla escolha −4,4 (−8,8, 0,0).
- **Forma:** fecha `</think>` e responde 92,2% (GSM8K) e 79,2% (múltipla escolha); traço em português 99,6% e 93,8%; truncadas 7,8% e 19,7% (o Lumi original trunca 13,8% na múltipla escolha); mediana de 384 e 475 tokens.
- **Leitura:** aprendeu o idioma e a forma, mas perdeu nota no modo thinking; o modo sem thinking ficou igual ao Lumi. **Hipótese, não testada:** os traços do próprio Lumi com começo forçado em português são piores que o raciocínio livre dele em inglês, então o aluno aprende a forma sem ganhar qualidade. O professor precisa ser mais forte que o aluno (Qwen3-30B na Thor, ou traços do R1 traduzidos).

### 20.2 Levantamento de datasets (buscas em 07/11/2026)

O gargalo é a múltipla escolha (−7,3 do adalberto contra o Lumi pensando); adalberto, selfdist e Thor são quase só matemática do GSM8K.

| Dataset | O que é | Veredito |
| --- | --- | --- |
| `recogna-nlp/Bode-reasoning` | 10.287 de treino, raciocínio em português; ~38% são provas | **As provas não servem:** os 3.869 itens de OAB (2.001), BLUEX (605) e ENEM (1.263) têm 8-grama igual ao das nossas provas (100%). Resto: reasoning sintético traduzido (4.337), EduBench proprietário (1.761), matemática (320), 1% a 7% de sobreposição, perguntas abertas, sem licença declarada |
| `xandykati98/PTBR-Opus-4.6-Reasoning-3000x-filtered` | 2.326 exemplos com `thinking` e `solution` em português; Apache-2.0 declarada | Pequeno; os traços vêm do Claude Opus 4.6, conferir os termos do provedor antes de treinar |
| `cnmoro/reasoning-v1-20m-portuguese` | 20 M, Apache-2.0, traduzido de um R1-Distill-70B | Perguntas abertas sem gabarito, tradução com erros ("Algumente", "Oligoligo"): arriscado |
| `AxionLab-Co/ThinkSet-PTBR` | aritmética sintética trivial | Descartar |
| `rhaymison/orca-math-portuguese-64k` | 64 mil problemas com resposta, Apache-2.0 | Só como fonte de problemas; sem traço |

Não há tradução pronta para português de OpenThoughts, Mixture-of-Thoughts ou OpenR1 (busca na web e no Hub). Um [estudo de traços longos em vários idiomas](https://arxiv.org/html/2508.14828v1) fez SFT com traços ingleses traduzidos (francês, japonês, letão, suaíli) e funcionou; português não foi testado.

**Fonte escolhida: `open-r1/Mixture-of-Thoughts`, subconjunto `science`.** 172.514 linhas, 53.195 questões distintas, todas de múltipla escolha (`A: ...`, `B: ...`), 2 a 4 traços do R1 por questão, resposta em `\boxed{letra}`. Não há gabarito: usei a letra majoritária (≥ 75% de ao menos 3 traços; é um indicador, não verdade). Seleção (`scripts/prepare_mot_science.py`, `lumi_ft/data/mot_science.py`): 37.231 questões e 71.422 traços (até 3.000 tokens), ~99 M de tokens, em `data/mot_science_en.jsonl`. No arquivo 0, a mediana é 1.341 tokens: 35% cabem em 1.024, 68% em 2.048 e 88% em 4.096.

### 20.3 Ordem seguida (aprovada pelo usuário)

1. `selfdist`: feito (acima).
2. Testes baratos, em fila (`queue-gpu`): escala do adaptador do adalberto em 0,5 e 0,75 (`merge_adapter.py --scale`, só LoRA; `lumi_ft/modeling/lora_scale.py`) e adalberto com limite de 2.048 tokens (`--max-len`; mesmos 5.842 + 3.000 exemplos, 346 descartados em vez de 9.680; 8,8 M de tokens). O limite de 1.024 tokens descartou 34% dos candidatos do adalberto, só ficaram os problemas curtos.
3. Traduzir os traços: `scripts/translate_traces.py` (Lumi no vLLM, T=0, em blocos de parágrafos; 70% dos problemas com raciocínio, 30% só resposta) e `scripts/filter_translated.py` (8-grama contra as provas e a validação). Um bloco ruim derruba o registro. Registros de múltipla escolha levam o campo `prompt` pronto (`teacher.build_rows`). **Ainda não rodado:** falta o piloto de 200 questões.
4. GRPO (RL com recompensa verificável): só se sobrar tempo.

Em paralelo, a Thor gera traços com o Qwen3-30B (pacote em `handoff/thor_pacote.tar.gz`); o resultado ainda não voltou.

### 20.4 Caches

Alguns comandos interativos foram rodados sem `env.sh` e o Hugging Face gravou 0,77 GB em `~/.cache/huggingface` (compartilhado com outros projetos). Os 14 repositórios criados hoje foram copiados e comparados byte a byte para `.cache/huggingface` do projeto, e as cópias soltas foram apagadas; o cache de outros projetos não foi tocado. Docker não está disponível (sem permissão no socket, sem sudo). Regra: todo comando em `peft_lora_study` carrega `env.sh`.

### 20.5 GRPO (código pronto, ainda não rodado na GPU)

Desenho (iterações em processos separados, porque a 4090 não comporta o vLLM e o treino ao mesmo tempo): `scripts/run_grpo.sh NOME --init <adaptador do SFT> --iters N` repete: (1) funde o adaptador (`merge_adapter.py`, com `--fast` depois da primeira), (2) `scripts/grpo_rollout.py` gera 8 respostas por prompt com o vLLM (T=0,8, até 2.048 tokens, modo thinking sem começo forçado) e calcula recompensa e vantagem, (3) `scripts/grpo_train.py` atualiza o adaptador. Estimativa: ~18 min por iteração (256 prompts × 8 respostas), 10 iterações em ~3 h. Avaliação do modelo que entra na iteração N a cada `--eval-every`, e completa no fim.

- **Recompensa** (`lumi_ft/train/grpo.py`): 1 só se a resposta fecha um único `</think>`, termina sozinha, o traço está em português e a resposta final bate com o gabarito no formato estrito (`#### N` ou `Resposta: <letra>`); senão 0. Truncada vale 0 e treina com vantagem negativa (ensina a parar).
- **Vantagem:** recompensa menos a média do grupo, dividida pelo desvio-padrão (`--adv std`, GRPO original) ou sem dividir (`mean`, Dr. GRPO). Grupo com recompensas iguais não treina.
- **Perda:** PPO por token com corte assimétrico (0,2 / 0,28), sem KL, normalizada pelos tokens de resposta do minibatch. As log-probabilidades antigas vêm de uma passada sem gradiente do próprio treino (a razão começa em 1, sem depender da diferença numérica entre vLLM e Transformers). lr 2e-5 constante, AdamW com o estado guardado entre iterações, 64 respostas por passo.
- **Prompts** (`scripts/make_grpo_prompts.py`): 5.612 de matemática (GSM8K-PT `train`, gabarito) e as questões de ciência de `mot_science_en.jsonl` traduzidas só no enunciado (`scripts/translate_questions.py`, 4.000, descontaminadas por `filter_translated.py`), metade de cada em cada iteração. O gabarito da múltipla escolha é a letra majoritária dos traços do R1, não a verdade.
- **Testes:** 9 na CPU (`tests/test_grpo.py`: recompensas, vantagens, seleção de prompts, token de parada, log-probabilidades, gradiente da perda com e sem corte). **Não testado:** `grpo_rollout.py`, `grpo_train.py` e `run_grpo.sh` de ponta a ponta. Um teste de fumaça (1 iteração, 16 prompts × 4 respostas) está na fila (`queue-prep`). Ponto a conferir: se o vLLM inclui o token de parada na resposta (`appended_stop_token` em `rollouts.stats.json`).

### 20.6 Escala do adaptador, limite de 2.048 tokens e estilo de raciocínio

Validação, nota estrita; GSM8K limpo (676) e múltipla escolha (385); diferença contra o Lumi original no mesmo modo.

| Modelo | GSM8K thinking | MCQ thinking | GSM8K sem thinking | MCQ sem thinking |
| --- | --- | --- | --- | --- |
| Lumi original | 81,7% | 69,6% | 77,2% | 63,9% |
| `adalberto` (limite 1.024 tokens) | 83,3% | 62,3% | 79,3% | 58,2% |
| `adalberto` escala 0,5 | 84,0% | 63,4% | 79,6% | 64,9% |
| `adalberto` escala 0,75 | 84,0% | 64,4% | 79,1% | 58,7% |
| `adalberto_L2048` (limite 2.048 tokens) | 82,2% | 64,2% | 81,7% | 55,3% |

- **Escala 0,5 e 0,75:** mantêm o ganho em matemática (+2,4) e não recuperam a múltipla escolha com thinking (−6,2 e −5,2, IC acima de zero na perda). A variação da múltipla escolha sem thinking (64,9% e 58,7%) é compatível com ruído de 385 itens.
- **Limite de 2.048 tokens:** só 346 exemplos descartados (antes 9.680), mesmos 5.842 + 3.000 exemplos, 8,8 M de tokens. **Não resolveu:** múltipla escolha com thinking 64,2% (−5,5 contra o Lumi, +1,8 contra o `adalberto`, sem significância) e **piorou** a múltipla escolha sem thinking (55,3%, −8,6 contra o Lumi). A matemática sem thinking subiu (81,7%, +4,4). O limite de tokens não é a alavanca.
- **Estilo de raciocínio** (GSM8K, palavras no traço; marcadores por 1.000 palavras): Lumi original mediana 96 palavras, abre com "We need to", "wait" 0,2, "let me/vamos" 2,3; Qwen3-30B (adalberto) mediana 177, abre com "Certo, vamos", "wait" 1,9, "vamos" 13,4; DeepSeek-R1 (ciência) mediana 610, abre com "Okay, so", "wait" 7,6. O Lumi pensa de forma curta e em forma de plano, diferente do Qwen e do R1. O cartão do Lumi diz que ele parte do Nemotron 3 Nano 4B (CPT de 50 B de tokens com destilação, SFT em português, thinking "deve ficar desligado", sem RL). Hipótese, não testada: traços de estilo diferente ensinam ao Lumi um modo de raciocinar que ele não tem.
- **Próximo passo escolhido:** professor da mesma família, o Nemotron 3 Nano 30B-A3B (suporta inglês, espanhol, francês, alemão, japonês e italiano, não português): gera em inglês, sem forçar idioma, e traduzimos aqui. Pacote `handoff/thor2/` (plano, cliente, filtro, `pack.sh`, 21.612 prompts: 9.612 de matemática e 12.000 de múltipla escolha), gerado por `scripts/make_thor2_prompts.py` e testado em `tests/test_thor2_pkg.py`. Falta no lado local: converter a saída (`thor2_out.tar.gz`) e traduzir (matemática: só o traço; múltipla escolha: enunciado, alternativas, traço e resposta) com a qualidade que o piloto de tradução mostrar.

### 20.7 Piloto de tradução (200 questões de ciência, Lumi como tradutor)

`scripts/translate_traces.py --n 200` (vLLM, T=0, modo sem thinking, blocos de parágrafos de até 1.800 caracteres), depois `filter_translated.py` (8-grama). 88 s para 200 registros (849 pedidos, 2.848 tok/s), ou seja ~90 min para 12.000.

| Medida | Resultado |
| --- | --- |
| Aceitos pelas conferências | 172 de 200 (86%): com raciocínio 125 de 150, só resposta 47 de 50 |
| Rejeitados (28) | 16 tamanho incompatível, 10 não terminou, 2 alternativas não voltaram |
| 8-grama contra as provas | 1 removido (171 ficam) |
| Tamanho PT/EN (caracteres) | mediana 1,04; p5 0,98, p95 1,12 |
| Números do original presentes na tradução | média 94,7%, mediana 100% (normalizando `3.5` e `3,5`; sem normalizar dava 56%, erro da medida); 16 itens abaixo de 90%, 5 abaixo de 50%; dos 123 números perdidos, 99 têm 1 dígito (provavelmente subscritos como "ZnCl₂" e enumerações; não conferido item a item) |
| Inglês que sobrou (palavras funcionais) | média 1,2%; nenhum item acima de 15% |
| Parágrafos | diferença mediana 0; 2 itens com diferença maior que 2 |

- **Defeitos:** 120 dos 125 traços abrem com "Okay, então…" (o "Okay" fica sem traduzir; vem do estilo do R1), "Hmm" vira "Hum"; terminologia técnica errada em química ("S-metil benzotiocato" em vez de "benzotioato de S-metila"); erros leves de gênero ("o integral de linha").
- **Leitura:** serve para treino (sentido, números e estrutura preservados); o ponto fraco é o vocabulário técnico, menos relevante em matemática. **Não medido:** correção de sentido (só 3 exemplos lidos) e se treinar com o traço traduzido melhora o modelo. Se o treino mostrar que a tradução limita, trocar o tradutor (Qwen3-30B na Thor).
- Os mesmos scripts traduziram os 4.000 enunciados dos prompts de múltipla escolha do GRPO: 3.739 aceitos (93,5%), 3.715 depois do 8-grama.

### 20.8 GRPO: teste de fumaça e motivo

Teste de fumaça (`run_grpo.sh smoke`, 1 iteração, 16 prompts × 4 respostas, 512 tokens no máximo, adaptador `adalberto`): **a cadeia inteira funcionou** (fundir, gerar com o vLLM, calcular recompensa, treinar, salvar).

| Medida | Valor |
| --- | --- |
| Respostas geradas / com vantagem diferente de zero | 64 / 12 |
| Recompensa média (matemática, múltipla escolha) | 0,125 e 0,031 (quase tudo truncou: o teste usou só 512 tokens, 59 de 64 respostas) |
| `appended_stop_token` | 0: o vLLM já inclui o token de parada na resposta |
| Razão da perda no início do treino / fração cortada | 1,0001 / 0 (a passada "antiga" e a nova concordam) |
| Pico de memória | 20,8 GiB alocados, 22,5 GiB reservados (cabe na 4090, sem folga) |
| Velocidade | geração 2.896 tok/s; passada antiga 7 s; treino 12 s |

- **Não testado ainda:** a iteração 1 em diante (o estado do AdamW de `ckpt.pt` carregando na próxima iteração, a avaliação intermediária). Conferido na CPU: `merge_adapter.py --fast` leva 8 s; o `ckpt.pt` tem estado para 142 parâmetros (71 módulos × A e B), como esperado. O teste de fumaça não diz nada sobre ganho de nota.
- **Por que GRPO:** o SFT imita um professor, e nenhuma fonte de professor cobre a múltipla escolha no estilo do Lumi (adalberto e selfdist perderam na múltipla escolha). O GRPO parte do adaptador do SFT e deixa o próprio Lumi gerar, recompensando só a resposta que fecha um único `</think>`, não trunca, tem traço em português e acerta o gabarito. Otimiza direto o que o critério de vitória mede e não depende de traços de professor. Riscos: o gabarito da múltipla escolha é a letra majoritária do R1 (indicador, não verdade), grupos sem sinal (todas certas ou todas erradas) não ensinam, e é caro (~18 min por iteração).
- **Execução:** `grpo1` (8 iterações a partir de `runs/train/adalberto/adapter`, 256 prompts × 8 respostas, até 2.048 tokens, T 0,8, lr 2e-5), na fila depois do DoRA.

### 20.9 PRD e comparação atualizados (2026-11-07)

- **PRD:** nova seção "Atualização de escopo" (metas, critério de vitória e limiares **não mudaram**, regra 4); escopo ampliado com traços de professor, tradução do raciocínio e GRPO; Fora do escopo agora lista só APO, QLoRA, o CPT (avaliado e adiado) e o raciocínio em inglês (teste posterior do usuário); requisitos novos D-12 a D-16 (duplicatas em outra tradução, aceitação de traços de professor, prompt de treino, tradução, licenças) e A-15 a A-17 (validação limpa, comparação em quatro sistemas, Nemotron 4B de referência); tabela de fontes de professor; tabela dos experimentos executados e previstos; seção do GRPO; fase 2b com checklist e gate; 9 riscos novos; decisão sobre o adalberto fechada e 3 decisões novas em aberto (ponto de partida do GRPO, raciocínio em português como requisito, reabrir o CPT).
- **`comparacao.md`:** quatro sistemas (Nemotron 3 Nano 4B, Lumi, nosso Lumi em português e nosso Lumi em inglês); ganho do treino e custo do português como diferenças entre colunas; avaliação primeiro na validação (15 a 20 min) e depois na suíte; template, prompts e vocabulário do Nemotron 4B conferidos como idênticos aos do Lumi; esclarecido que "nenhuma resposta do Nemotron entra no treino" vale para a referência (4B), não para o professor de 30B; tabela do estado atual.
- **Ferramenta:** `scripts/report_compare.py` gera a tabela de comparação (nota, diferença com IC 95% contra uma referência e forma da saída) para qualquer conjunto de sistemas e provas.

### 20.10 DoRA e Nemotron 4B de referência (07/11/2026, 19:35)

- **DoRA = LoRA.** GSM8K limpo (676) 84,0% / 79,1% (com / sem thinking) contra 83,3% / 79,3% do LoRA; múltipla escolha (385) 62,6% / 57,7% contra 62,3% / 58,2%. A diferença é de 0,3 a 0,7 ponto, dentro do ruído; o GRPO continua partindo do `adalberto` LoRA.
- **Nemotron 3 Nano 4B de referência** (thinking, traço em inglês): GSM8K 85,2% (+3,6 sobre o Lumi, IC +1,0 a +6,1), múltipla escolha 68,3% (−1,3, empate). Sem thinking: 63,5% (−13,8) e 58,7% (−5,2). Acerto estrito e flexível são iguais, então não é formato. O Lumi empata com o pai na múltipla escolha; o nosso treino com traço em português perde 7 pontos.
- **Fontes dos prompts do pacote 2:** Mixture-of-Thoughts `science` (origem: nvidia/Llama-Nemotron-Post-Training-Dataset, fora da lista de SFT do Lumi), GSM8K-PT `train` e GigaVerbo (4.000 de matemática; o GigaVerbo-v2 está no pré-treino do Lumi, como texto cru).
- **Fila da GPU** (19:39, reordenada a pedido do usuário): diagnóstico português x inglês no Lumi base (`queue-diag`, ~20 min), depois `grpo1` (~3 h) e `adalberto_lr5e5`. O `grpo1` tinha começado às 19:34; parei na iteração 0 (sem rollouts salvos, o script é retomável) para o diagnóstico rodar antes.

### 20.11 Diagnóstico português x inglês e pacote 3 da Thor (07/11/2026, 20:12)

- **Diagnóstico** (Lumi original, as mesmas 1.200 questões do teste do MMLU-ProX nos dois idiomas, instrução no idioma da questão, marcador `Resposta:` igual; bootstrap pareado, 10.000 reamostras):

| | Português | Inglês | PT − EN (IC 95%) |
| --- | --- | --- | --- |
| Com thinking, estrita | 52,2% | 51,8% | +0,4 (−2,1 a +2,9) |
| Com thinking, flexível | 52,1% | 53,4% | −1,3 (−3,8 a +1,1) |
| Sem thinking, estrita | 48,7% | 44,8% | +3,8 (+0,8 a +6,8) |
| Sem thinking, flexível | 47,1% | 48,5% | −1,4 (−4,2 a +1,4) |

  O idioma da questão não muda o acerto do Lumi (empate na nota flexível nos dois modos; a vantagem do português na estrita sem thinking é de formato: em inglês ele escreve menos a linha `Resposta: X`). O Lumi raciocina em inglês nos dois casos, então isso não mede o custo de **pensar** em português; mede que a perda de 7 pontos dos nossos treinos não vem da questão estar em português, e sim do treino (traços em português, estilo do professor ou os dados). Truncadas com thinking: 27,5% (PT) e 26,1% (EN) em 4.096 tokens (questões de 10 alternativas).
- **Pacote 3 da Thor** (`handoff/thor3/`, `handoff/thor3_pacote.tar.gz`, sha256 `c5b2b18204e5dbf5626d90e16042af48778417d854e867ddffbcde44a87cf380`), a pedido do usuário (traduzir na Thor com o 30B e ampliar o domínio):
  - `translate_thor.py`: tradução com o próprio Nemotron 30B (`/v1/completions`, `enable_thinking=False`, T=0), em blocos de até 3.500 caracteres. Confere cada pedaço: terminou, sem `<think>`, tamanho 0,8 a 1,7× (0,4 a 2,6× abaixo de 200 caracteres), números (sequências de dígitos) preservados com até 10% de falta e "respondeu em vez de traduzir" (`\boxed`, `Resposta:`, `####`, `Answer:` que não estavam no original). Só os 70% de problemas com `bucket(id) < 7000` têm o raciocínio traduzido (a mesma divisão de `build_teacher_dataset.py`). O registro guarda o original em inglês.
  - Prompts novos (`scripts/make_thor3_prompts.py`, `lumi_ft/data/thor3.py`): 7.000 do WebInstruct-verified (Apache-2.0; das 27.764 de múltipla escolha, 18.360 com alternativas separáveis; nenhuma área acima de 20%) e 5.000 do MedMCQA (Apache-2.0; só `single`, sem imagem e sem o defeito de OCR "aery"; nenhuma área acima de 8%). Fora 925 + 2.007 com 8-grama em comum com o MMLU-ProX (teste e validação) ou o MMLU (teste) em inglês, e 1.160 + 19.489 enunciados repetidos. Nenhuma das duas fontes está na lista de treino do Lumi.
  - Do nosso lado: `scripts/prepare_thor_translated.py` escreve `think.jsonl` (raciocínio traduzido), `nothink.jsonl` (os outros 30%, disjuntos) e `think_en.jsonl` (o par com o raciocínio original em inglês, mesmo prompt e mesma resposta em português: o experimento "raciocínio em inglês" com os mesmos problemas). `filter_translated.py` agora também descontamina `think_en`.
  - Testes: `tests/test_thor3_pkg.py` (27; tradução de ponta a ponta com servidor falso, retomada, conversão, parsers). Não testado: a tradução com o servidor real (é o passo 1 do plano da Thor).
- **GRPO, ciclo 0** (`grpo1`, 256 questões × 8 respostas, 11 min, 2,2 M de tokens, limite de 2.048): recompensa média 0,87 em matemática e 0,54 em múltipla escolha; truncadas 3% e 23%; sinal (grupos com respostas diferentes) 23% e 66%. O treino do ciclo 0 estourou a memória (OOM no `backward`, 22 GiB, com uma só resposta de ~2.350 tokens e sem gradient checkpointing); a fila disparou o `adalberto_lr5e5` no lugar, que parei no passo ~170 e apaguei. Correção: `grpo_train.py --ckpt full` (padrão), pico de 12,3 GiB; o `grpo1` recomeçou às 20:36 aproveitando as respostas do ciclo 0. O limite de 2.048 tokens na geração (contra 4.096 na avaliação) é por tempo e memória do treino e porque a truncada vale 0 (pressiona a terminar); é uma escolha, não um requisito.
- **Fila da GPU** (20:36): `grpo1` (~3 h), depois `adalberto_lr5e5`.

### 20.12 Entropia do GRPO e salvaguardas (07/11/2026, 23:30)

- **`grpo1` terminado** (8 ciclos, avaliação gulosa na validação limpa; ● = IC inclui zero):

| | GSM8K com / sem thinking | Múltipla escolha com / sem thinking |
| --- | --- | --- |
| Lumi | 81,7 / 77,2 | 69,6 / 63,9 |
| adalberto (ponto de partida) | 83,3 / 79,3 | 62,3 / 58,2 |
| `grpo1` após 3 ciclos | 83,3 / – | 64,4 / – |
| `grpo1` após 6 ciclos | 83,9 / – | 65,5 / – |
| `grpo1` final (8 ciclos) | 82,8 ● / 77,5 ● | 66,8 ● (−2,9; −7,5 a +2,1) / 61,6 ● (−2,3) |

  Forma do `grpo1` final: fecha `</think>` em 98,4% (GSM8K) e 91,7% (múltipla escolha), traço em português 99,2%, truncadas 8,6% na múltipla escolha, mediana 904 tokens. Pelo critério do PRD ainda não vence: a diferença para o Lumi é empate estatístico, a queda sem thinking (−2,3) passa de 1 ponto e 91,7% fecha < 98%.
- **A entropia do modelo nas próprias respostas explode** (`scripts/run_grpo_entropy.sh`, 512 respostas sorteadas por ciclo, modelo que gerou cada ciclo, nats por token, temperatura 1): 

| Ciclo | todas | múltipla escolha | matemática | logp média do token amostrado |
| --- | --- | --- | --- | --- |
| 0 (adalberto) | 0,42 | 0,54 | 0,24 | −0,31 |
| 1 | 0,49 | 0,65 | 0,26 | −0,36 |
| 2 | 0,55 | 0,75 | 0,27 | −0,41 |
| 3 | 0,71 | 0,99 | 0,29 | −0,51 |
| 4 | 0,97 | 1,31 | 0,53 | −0,69 |
| 5 | 1,56 | 2,03 | 0,90 | −1,10 |
| 6 | 2,14 | 2,89 | 1,02 | −1,62 |
| 7 | 1,98 | 2,31 | 1,61 | −1,60 |

  Acompanha as respostas sem `</think>` (11 → 223 de 1.024 na múltipla escolha do ciclo 0 ao 6) e o lixo no fim das respostas. A avaliação gulosa continua boa porque o caminho mais provável se mantém. A hipótese (negativos longos sem KL) fica de pé como candidata, mas não está isolada.
- **Salvaguardas implementadas** (CPU: 12 testes em `tests/test_grpo.py`): entropia e logp média por ciclo em `train_log.jsonl`; `scripts/grpo_guard.py` + `run_grpo.sh --max-noclose` (para se > X das respostas de um tipo terminarem sem um único `</think>`); `run_grpo.sh --top-p`, `--mask-truncated 1` (filtro de truncadas do DAPO) e `--ratio seq` (variante GSPO, desligada por padrão).
- **Fila da GPU** (23:30): avaliação do `grpo1` com 4 atualizações (`grpo1_u4`, para comparar com o `grpo2` no mesmo número de passos), depois `grpo2` (4 ciclos, T 0,6, top-p 0,95, `--mask-truncated 1`, `--max-noclose 0,05`), depois `adalberto_lr5e5`.

### 20.13 `grpo2`, teste do KL e triagem por camadas (08/11/2026, 01:05)

- **`grpo2`** (4 ciclos, T 0,6, top-p 0,95, `--mask-truncated 1`, guarda `--max-noclose 0,05`; ● = IC inclui zero), validação limpa com / sem thinking: GSM8K 82,7 / 79,6 ●, múltipla escolha **66,0** (−3,6; −8,1 a +0,8) ● / 59,5 (−4,4) ●. Treino estável: entropia 0,42 → 0,34 → 0,27 → 0,23 (as respostas vêm de uma geração mais fria, então o número não é comparável ao do `grpo1`), nenhuma resposta sem `</think>`. **Piorou a forma:** trunca 16,1% (adalberto 13,8%, `grpo1` com 4 atualizações 9,6%) e fecha `</think>` em 84,2%. Leitura (hipótese): o filtro de truncadas tira da perda as respostas que ensinavam a parar, e as respostas usadas por ciclo caem de ~900 para ~600. Com 4 atualizações a nota é parecida com a do `grpo1` (66,0 contra 64,9).
- **Teste do KL** (`grpo_train_kl.py`, k3 por token, referência = adaptador do SFT): 0,0003 com política = referência e 0,129 com a política do ciclo 6 do `grpo1`.
- **Fila refeita** (01:02): `grpo4` (T 0,6, top-p 0,95, **sem** o filtro, KL 0,04), triagem por camadas do `adalberto`, `grpo3` (Dr. GRPO, 70% de múltipla escolha, sem o filtro), `raft1` (só respostas certas), `gspo1` (razão por sequência, sem o filtro) e `adalberto_lr5e5`. Todos com 4 ciclos, a partir do `adalberto`, guarda a 5%.
- **Triagem por camadas** (`scripts/merge_adapter_layers.py`): funde só parte do adaptador, sem treinar (zera o efeito do resto): camadas 0-20, camadas 21-41, sem MLP (Mamba e atenção) e só MLP. Avaliação só com thinking na validação. Aproximada: zerar depois do treino não equivale a treinar só naquelas camadas. A hipótese a testar é a do MLP (fatos nos MLPs; o adaptador mexe nos 17).
- **Código:** `grpo_train_kl.py` e `run_grpo_kl.sh` são cópias com KL e `--limit`; a ser unificado com `grpo_train.py` e `run_grpo.sh` quando nenhum job os usar. `run_grpo.sh` ganhou `--top-p`, `--mask-truncated`, `--max-noclose`, `--ratio seq` (GSPO) e `--positives-only`; `grpo_status.py` resume um GRPO por ciclo. Lição: não editar um `.sh` enquanto ele roda (o `grpo1` saiu com código 1 por isso; resultados intactos).

### 20.14 Resultados da noite e triagem por camadas (08/11/2026, 09:30)

- **Rodadas de GRPO** (4 ciclos, a partir do `adalberto`, T 0,6, top-p 0,95; validação limpa, com / sem thinking; ▲ = IC acima de 0, ▼ = abaixo):

| Rodada | GSM8K | Múltipla escolha | Entropia (ciclo 3) |
| --- | --- | --- | --- |
| `grpo3` (Dr. GRPO, 70% MCQ, sem filtro) | 84,6 ▲ / 79,1 | 66,5 / **43,1 ▼** | 0,89 (sobe) |
| `grpo4` (KL 0,04, sem filtro) | 84,3 ▲ / 79,7 | 62,9 ▼ / **43,1 ▼** | 0,75 (sobe) |
| `raft1` (só respostas certas) | 82,8 / 78,4 | 61,6 ▼ / 59,2 | 0,18 (cai) |
| `gspo1` (razão por sequência) | 84,3 / 79,9 | **67,8** (−1,8; IC −6,2 a +2,6) / 59,5 | 0,45 (estável); 40 a 50% dos tokens cortados |
| `adalberto_lr5e5` (SFT, lr 5e-5) | 83,1 / 80,2 ▲ | 63,4 ▼ / 57,9 ▼ | – |

  O KL de 0,04 não segurou a entropia (KL médio 0,04 por token no ciclo 3). O `grpo3` e o `grpo4` derrubaram a múltipla escolha **sem thinking** para 43,1% (o RL só treina o modo com thinking e não protege o outro). Nenhuma rodada passa o critério do PRD. O `gspo1` é o mais estável e o melhor com thinking; continua até 8 ciclos (`queue-gspo1b`, resumindo do ciclo 4; a avaliação de 4 ciclos ficou em `runs/eval/gspo1_c4`).
- **Triagem por camadas** (adaptador `adalberto` fundido só em parte, sem treinar; avaliação com thinking; ▼ = IC abaixo de 0):

| Parte mantida | GSM8K | Múltipla escolha | Traço em português |
| --- | --- | --- | --- |
| adaptador completo | 83,3 | 62,3 ▼ | 99,4% |
| camadas 0 a 20 | 83,6 | 62,6 ▼ | 0% (inglês) |
| camadas 21 a 41 | 76,6 ▼ | 18,4 ▼ (fecha `</think>` 22,9%) | 98,9% |
| sem MLP (Mamba e atenção) | 81,7 | 60,0 ▼ | 98,7% |
| **só MLP** (34 módulos) | **83,9** | **69,6** (igual ao Lumi, IC −3,4 a +3,4) | **2,4%** (inglês) |

  Leitura: o adaptador dos MLPs sozinho dá o ganho em matemática (+2,2) sem custo na múltipla escolha, mas o raciocínio continua em inglês; a perda de múltipla escolha vem das partes do Mamba (`in_proj`) e da atenção, mesmo quando o traço sai em inglês (camadas 0 a 20). Isso contraria a ideia de que o português em si é o culpado. As camadas 21 a 41 sozinhas quebram o modelo (as outras se ajustaram em conjunto). É uma triagem aproximada: zerar depois do treino não equivale a treinar só nesses módulos.
- **Próximo teste** (`queue-mlp`): SFT só com LoRA em `up_proj` e `down_proj` (`--targets mlp`, 34 módulos), mesmos dados e hiperparâmetros do `adalberto` (r 16, lr 1e-4, 1 época), para ver se o MLP sozinho aprende o raciocínio em português sem perder a múltipla escolha. `TARGETS` em `peft_setup.py`, `--targets` em `train_sft.py`, 4 testes novos em `tests/test_peft_targets.py`.

### 20.15 Organização dos estudos (08/11/2026, 11:10)

- Criada a pasta `estudos/` com os resultados organizados como estudos de ablação: `README.md` (pergunta, critério, achados, mapa), `01_protocolo.md`, `02_sft_dados_e_professor.md`, `03_ablacao_camadas_e_mlp.md`, `04_rl_grpo.md`, `05_diagnosticos_e_referencias.md`, `06_registro_experimentos.md` (tempos de treino e de avaliação de cada rodada, ≈ 18 h de GPU no total), `07_hipoteses_e_proximos_passos.md` (registro de hipóteses H0 a H25 com veredito, ameaças à validade, próximos experimentos) e `resultados_auto.md`, **gerado** por `scripts/gen_estudos_resultados.py` a partir de `runs/eval/` (nota estrita na validação limpa, diferença contra o Lumi com IC 95%, forma da saída, minutos de avaliação). Regerar depois de cada rodada nova.

### 20.16 `adalberto_mlp`: o treino só com MLP refuta a triagem (08/11/2026, 11:30)

- SFT só com LoRA em `up_proj` e `down_proj` (34 módulos, 8,5 M de parâmetros), mesmos dados e hiperparâmetros do `adalberto`: 23,9 min de treino, 12 min de avaliação (os dois modos).

| | Lumi | `adalberto` | `adalberto_mlp` |
| --- | --- | --- | --- |
| GSM8K com / sem thinking | 81,7 / 77,2 | 83,3 / 79,3 | 83,4 / 80,2 (▲ por 0,0 no limite inferior) |
| Múlt. escolha com / sem thinking | 69,6 / 63,9 | 62,3 ▼ / 58,2 ▼ | **62,3 ▼ / 54,3 ▼** |
| Fecha `</think>` / trunca (MCQ) | 86,2 / 13,8 | 86,8 / 13,8 | 84,7 / 15,8 |
| Traço em português (MCQ) | 0% | 99,4% | 98,5% |

- Leitura: o MLP sozinho aprende português e a matemática, mas perde a múltipla escolha igual ao adaptador completo. A triagem pós-treino (69,6%) enganou, como a ameaça (i) previa. H9a e H9c ficam refutadas; a causa provável são os dados, e o teste que decide é o par em inglês e português (H21) do pacote 3 da Thor. Seis mudanças de adaptador ou de otimização deram 62 a 64%; só o RL passou disso.
- Documentos atualizados: `estudos/03` (E10 completo, grade 2×2 rebaixada), `README`, `06` (tempos), `07` (H9a, H9b, H9c, H10, próximos passos), `resultados_auto.md` regerado.

### 20.17 Estudo de capacidade, tipo de bloco e posição (08/11/2026, 11:40)

- Pedido: testar hiperparâmetros diferentes por tipo de bloco e por camada, e a RSLoRA em rank alto, enquanto a Thor não devolve os dados. "Todas as combinações" são centenas de rodadas (~37 min cada), então virou um plano em etapas de 11 rodadas, uma mudança por vez, com hipóteses e previsões escritas antes (`estudos/03`, E11; `estudos/07`, H26 a H30).
- Código (padrões iguais ao do PRD, 36 testes novos, 322 no total): `peft_setup.py` (`parse_targets`, `parse_layers`, `target_regex`, `resolve_alpha_pattern`; `check_targets` com contagem independente quando há faixa de camadas), `sft.py` (`alpha`, `rslora`, `layers`, `rank_pattern`, `alpha_pattern`, `lr_mult`; grupos de parâmetros do AdamW com multiplicador do lr), `train_sft.py` (flags). `merge_adapter.py` não mudou: o PEFT grava rank e alpha por módulo e a RSLoRA no `adapter_config.json`.
- Fumaça na GPU (3 passos): r=64 com RSLoRA e `--lr-mult` treina (68,2 M de parâmetros, pico de 20,6 GiB reservados) e a fusão passa; Mamba + atenção nas camadas 0 a 20 com rank 4 no `in_proj` treina (1,6 M de parâmetros).
- Fila: `scripts/run_capacity.sh` lê `scripts/capacity_plan.txt` uma linha por vez (dá para reordenar e estender sem editar o script); espera a GPU ficar livre; pula rodadas completas; `touch runs/jobs/cap.stop` para entre rodadas; estado em `runs/jobs/cap.status`; regera `resultados_auto.md` a cada rodada. Sessão tmux `cap-queue`, início 11:35 com `cap_r04_std`.
- Leis de escala: com alpha fixo, `std` (alpha 32) dá escala 32/r e `rs` (alpha 8, RSLoRA) dá 8/raiz(r); as duas passam pelo ponto do `adalberto` (r=16, escala 2). Rank isolado não distingue as leis; r=4 e r=64 entram nas duas.

### 20.18 Primeira entrega da Thor: traduzidos com raciocínio (08/11/2026, 12:00)

- Chegou `traduzidos_com_raciocinio.jsonl` (5.278 registros, 30 MB; o `.gz` tem o mesmo conteúdo, sha conferido) em `~/barros/`, fora do projeto; copiado (não movido) para `data/thor3/out/translated.jsonl`. É o pacote 3 **parcial**: só os 70% com raciocínio (1.746 `gsm8k_train`, 1.059 `gv_math`, 2.473 `mot_science`); faltam os 30% sem raciocínio e as questões novas.
- **Bug encontrado e corrigido** em `scripts/filter_translated.py`: quebrava (`KeyError: 'prompt'`) em registros de matemática, que não têm `prompt`; e não aplicava o filtro de duplicata da validação (assinatura de números + Jaccard) que o resto do pipeline aplica e que o `estudos/05` (D3) descreve. Agora `record_text` e `drop_reason` (testados, 1 teste novo, 323 no total) fazem as duas coisas. Efeito: 5.278 → 5.246 (17 duplicatas da validação, 15 por 8-grama; as versões PT e EN sofrem as mesmas remoções).
- Dados de treino: `data/thor_pt.parquet`, `thor_en.parquet` (os mesmos 4.927 problemas; limite de 2.048 tokens; interseção entre as duas versões) e `thor_pt_math.parquet` (2.753 problemas, só matemática); todos somam as 5.544 linhas sem raciocínio do GigaVerbo. Conferência: 100% em português, formato e gabarito corretos em todos.
- **Achado de estilo:** o Nemotron 30B repete a instrução de formato dentro do raciocínio em 97% (matemática) e 82% (múltipla escolha) das vezes, o Lumi em 84% e 33%, o Nemotron 4B em 77% e 62%, e o `adalberto` (Qwen3-30B) em 0% e 1%. Sustenta que o Qwen3 ensinou um estilo estranho ao Lumi (H19).
- **Confundidor tratado:** o `adalberto` não tinha questões de múltipla escolha no treino; o `thor_pt` tem 2.174. Por isso o controle `thor_pt_math` (mesmo professor, sem múltipla escolha).
- Fila: `thor_pt`, `thor_en` e `thor_pt_math` entram na frente do estudo de capacidade (linhas futuras de `scripts/capacity_plan.txt`, inseridas com troca atômica do arquivo; a rodada em execução, `cap_r04_std`, não foi tocada). Hipóteses, previsões e regra de leitura escritas antes dos resultados em `estudos/02` (E12).

### 20.19 `thor_pt`: o professor da mesma família não erra mais, mas entra em laço (08/11/2026, 12:50)

- `thor_pt` (SFT com 4.927 problemas do Nemotron 30B em português + 5.544 linhas sem raciocínio do GigaVerbo): múltipla escolha com thinking **54,3% ▼** (−15,3 contra o Lumi), GSM8K com thinking 77,8% ▼, **fecha `</think>` só em 66,0%** (múlt. escolha) e 88,1% (GSM8K); trunca 34,0%. Sem thinking: 62,9% (−1,0) na múltipla escolha, 74,6% no GSM8K.
- **Mecanismo:** 97% das truncadas (127/131) são laços de repetição ("Na verdade, … Na verdade, …"). Laços em todas as respostas de múltipla escolha: Lumi 13,2%, `adalberto` 10,4%, `thor_pt` 33,0%. Nos itens que o `thor_pt` e o Lumi terminam, a nota é a mesma (84,0% contra 84,5%); o "acerto quando termina" do resumo engana (viés de seleção: só os fáceis terminam). Dos 131 itens em laço, o Lumi termina 94 e acerta 71%.
- Diferença para o `adalberto`: o Qwen perde por **errar** (72,3% de acerto quando termina), o Nemotron por **laço**. Muda o próximo passo: eliminar laços (RL com recompensa zero para truncada) em vez de procurar dados melhores. Hipótese H33. Aguardam `thor_en` e `thor_pt_math`.
- Documentado em `estudos/02` (E12), `06` e `07` (H19, H20 e H33).

### 20.20 Foco no modo sem thinking (08/11/2026, 13:00)

- Decisão do usuário: melhorar primeiro o modo sem thinking, depois analisar como não piorá-lo ao treinar o thinking. O GSPO a partir do `thor_pt` (H33) e a proteção do modo sem thinking no RL (H22) ficam adiados.
- **Diagnóstico (`estudos/05`, D8):** a múltipla escolha sem thinking perde por **formato**, não por conteúdo. Flexível 62,9 a 63,9% em todas as rodadas (Lumi 63,4%); estrita cai porque a linha `Resposta: X` some (87,8% no `adalberto`, 83,9% no `adalberto_mlp`, 96,1% no `thor_pt`). Os SFTs só de matemática respondem mais longo (400 contra 166 tokens). No GSM8K sem thinking, o estilo passo a passo do Qwen sobe +2 a +4; as respostas curtas do Nemotron não.
- Três SFTs só com exemplos sem thinking na fila (`nt_qwen`, `nt_thor`, `nt_both`; `data/nt_*.parquet`, ~3 milhões de tokens cada), na frente do `thor_pt_math` e do estudo de capacidade. Hipóteses H34 a H37 e previsões em `estudos/02` (E13).
- Pedido à Thor (prompt em `handoff/thor3/PROMPT_THOR_08NOV.md`): entregar os registros sem raciocínio (30% por hash) e gerar as 12.000 questões novas (gabarito oficial) **em blocos** de 3.000, com tradução e pacote a cada bloco.

### 20.21 O foco volta ao modo com thinking; GSPO a partir do `thor_pt` (08/11/2026, 13:05)

- Decisão do usuário (13:05): esquecer o sem thinking por ora; melhorar o **com thinking**. O E13 (`nt_*`) foi adiado: saíram da fila (`# ADIADOS` no fim de `scripts/capacity_plan.txt`), dados em `data/nt_*.parquet`. O diagnóstico D8 (sem thinking perde por formato) continua válido.
- **Fila:** `touch runs/jobs/cap.stop` (a `cap-queue` termina depois do `thor_en`, sem iniciar o `thor_pt_math`); a sessão `rl-thor` espera a `cap-queue` acabar e roda `gspo_thor`: `run_grpo.sh gspo_thor --init runs/train/thor_pt/adapter --iters 8 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 --max-new 2048 --temperature 0.6 --top-p 0.95 --ratio seq --max-noclose 0.85` (a receita do `gspo1`; guarda de 5% para 85% porque o `thor_pt` já começa com ~34% de truncadas). Previsões em `estudos/04`.
- Pendentes depois do RL: retomar `thor_pt_math` e o estudo de capacidade (apagar `cap.stop` e relançar `scripts/run_capacity.sh`; ele pula o que já está completo).

### 20.22 O que pedir à Thor com o foco no thinking: tradução só das questões (08/11/2026, 13:10)

- Argumento: o RL de múltipla escolha usa 3.715 questões de ciência com gabarito fraco (letra majoritária do R1). As 12.000 questões do pacote 3 têm gabarito oficial e cobrem negócios, economia, finanças (837 a 838 cada), matemática, física, química, biologia, psicologia, história e áreas médicas: mais parecidas com a suíte de teste (MMLU-ProX, ENEM, BLUEX, OAB) do que ciência de pós-graduação.
- Problema: o `translate_thor.py` só traduz registros **com traço do professor** (`kept.jsonl`), então só as questões que o professor acertou virariam prompts (as mais fáceis, viés de dificuldade). Solução sem alterar nada do pacote: novo arquivo `handoff/thor3/translate_questions.py` (sha256 `9e66416a…c5bb`), que reaproveita `question_block`, `translate_piece`, `check_piece` e `split_question_block` e traduz **todas** as 12.000, sem raciocínio (~20 a 40 min na Thor). Teste na CPU (`tests/test_thor3_pkg.py`, 29 testes no arquivo): enunciado e alternativas traduzidos, "respondeu em vez de traduzir" recusado, questão sem gabarito ignorada, retomada sem repetir.
- `handoff/thor3/PROMPT_THOR_08NOV.md` reescrito: 1) estado; 2) **traduzir só as questões** (teste de 30, depois todas, `questions_pt.tar.gz`); 3) terminar o passo 2 e entregar também os registros sem raciocínio; 4) traços do pacote 3 em blocos de 3.000, com tradução e pacote a cada bloco. O usuário precisa copiar o novo arquivo para a pasta do pacote (a mesma do `translate_thor.py`).
- Do lado da 4090, falta o conversor de `questions_pt.jsonl` em prompts de RL (`prompt`, `gold`, `n_options`, `kind`), a escrever quando o arquivo chegar.

### 20.23 `thor_en`: o português custa 5 a 6 pontos; qualquer SFT do Nemotron piora a múltipla escolha (08/11/2026, 13:20)

- `thor_en` (mesmos 4.927 problemas e respostas do `thor_pt`, raciocínio original em inglês): com thinking, múltipla escolha **60,3%** ▼ e GSM8K **83,1%**; fecha `</think>` em 75,3%; laços 22,6% (múlt. escolha) e 3,7% (GSM8K). Contra o `thor_pt`: **+6,0** (IC +1,6 a +10,4) e **+5,3** (IC +2,7 a +8,0). H21 apoiada: o português custa 5 a 6 pontos, em parte por laços.
- O inglês também perde para o Lumi (−9,4 ▼): o SFT com traços do Nemotron piora a múltipla escolha em qualquer idioma; o português é parte do problema, não o todo.
- Sem thinking, o `thor_en` cai (GSM8K 65,7% ▼, múlt. escolha 56,1% ▼) por **formato**: as respostas estão em português, mas a linha final some em ~10% (flexível 79,1% no GSM8K).
- O `gspo_thor` (RL a partir do `thor_pt`) começou às 13:14 (ciclo 0, geração); cap-queue parou como previsto (13:14).

### 20.24 E11 encerrado em 1 de 15 rodadas (08/11/2026, 13:25)

- Pergunta do usuário: por que analisar `thor_pt_math` e `thor_en` se o problema é raciocínio em português na múltipla escolha? Resposta honesta: eram diagnósticos. `thor_en` já respondeu (o português custa 5 a 6 pontos; o inglês também perde para o Lumi). `thor_pt_math` era um controle para explicar um ganho que não houve, e só tem matemática: **descartado**.
- Mesma lógica para o resto do E11: as 14 rodadas de capacidade, tipo de bloco, posição e escala variam um eixo que `cap_r04_std` e o E10 já mostraram irrelevante. **Canceladas** a pedido (removidas de `scripts/capacity_plan.txt`; as concluídas ficam nos resultados). Código e testes (`--rank-pattern`, `--alpha-pattern`, `--lr-mult`, `--layers`, `--targets`, `--rslora`) continuam no repositório. `resultados_auto.md` só mostra as rodadas feitas.
- Foco: `gspo_thor` (rodando), RL com as 12.000 questões de gabarito oficial (aguarda a Thor), suíte de teste dos candidatos e variações de uma coisa por vez no RL.


### 20.25 `gspo_thor` terminou: o RL desfaz os laços do `thor_pt`, mas empata com o `gspo1` e não passa o Lumi (08/11/2026, 14:50)

- 8 ciclos a partir do `thor_pt`, 13:14 a 14:50 (1,4 h: geração 47 min, treino 29 min, avaliação final 7 min). Guarda de `</think>` nunca disparou (0% em todos os ciclos).
- Validação limpa, com thinking: múltipla escolha **67,0%** (−2,6 contra o Lumi, IC −7,0 a +1,6), GSM8K **82,5%** (+0,9). Contra o `thor_pt`: múlt. escolha +12,7 (IC +8,6 a +16,9) e GSM8K +4,7; truncação 34% → 12,2%. Contra o `gspo1`: −0,5 (IC −4,7 a +3,6), empate.
- Ainda falta para o critério do PRD: fecha `</think>` em 87,8% (pede 98%); sem thinking, múlt. escolha 60,0% (−3,9 contra o Lumi, IC −8,3 a +0,5; pede perda de no máximo 1 ponto); o GSM8K sem thinking sobe (78,7%, +1,5).
- Detalhe novo: a mediana de tokens na múlt. escolha é 534, contra 1.035 do `gspo1` (490 do Lumi): mesma nota com raciocínio mais curto. É a mediana de todas as respostas, sem controlar pelas que terminam.
- Os ciclos 5 a 8 renderam pouco na validação (65,7% no ciclo 4 → 67,0%), enquanto a recompensa de treino subiu 0,72 → 0,76 e a truncação amostrada caiu 25% → 4%.
- Leitura: dois pontos de partida diferentes (Qwen e Nemotron) chegam ao mesmo nível (~67%) com o mesmo RL e o mesmo gabarito fraco. Isso aponta o gabarito, e não o SFT, como o limite; o gabarito oficial das questões da Thor testa isso.
- `thor_pt_short` (H38) começou às 14:50 na `cap-queue`.

### 20.26 `thor_pt_short` (H38): cortar a cauda longa dos traços ajuda pouco (08/11/2026, 15:15)

- `thor_pt` só com raciocínio de até 600 tokens (3.738 dos 4.927 problemas; ciência 1.255 de 2.174; os mesmos 5.544 sem thinking): treino 13 min (291 passos), avaliação 10 min.
- Com thinking (validação limpa): múltipla escolha **57,4%** (+3,1 sobre o `thor_pt`, IC −1,0 a +7,5; −12,2 ▼ contra o Lumi), fecha `</think>` 73,8%, trunca 26,2% (antes 34%); GSM8K 78,8% (+1,0, IC −1,0 a +3,1). Sem thinking: múlt. escolha 63,1% (+0,3), GSM8K 73,1% (−4,1 ▼ contra o Lumi).
- Leitura: a direção é a prevista (menos laços, nota maior), mas a magnitude é pequena e está dentro do ruído; abaixo da regra de sinal do projeto (+4 com thinking e mesma direção sem). A cauda longa não é a causa principal dos laços. Confusão: o corte também reduziu a ciência (múltipla escolha) em 42%.
- Decisão: sem RL a partir do `thor_pt_short` (o limiar combinado era ~62% e truncação bem abaixo de 34%) e sem pedir traços mais curtos à Thor. A Thor fica para as questões de gabarito oficial.
- A `cap-queue` terminou às 15:15 e a `suite-queue` assume a GPU (Lumi, `gspo_thor`, `gspo1`).

### 20.27 Chegaram as 12.000 questões traduzidas: prompts de RL com gabarito oficial (08/11/2026, 15:30)

- `questions_pt.tar.gz` (sha256 `8c4e333920a66208…ca941`, íntegro, sem caminhos suspeitos): 12.000 questões, 11.455 aceitas (95,5%; 545 recusadas: 111 números divergentes, 208 tamanho, 207 alternativas não voltaram, 19 não terminou). Fontes aceitas: WebInstruct-verified 6.738, MedMCQA 4.717; 125,6 min na Thor. A cópia parcial (9.252 linhas) foi descartada a pedido do usuário.
- Conversor novo `scripts/make_questions_rl_prompts.py` (+9 testes): prompt = o da avaliação (`mcq.build_prompt`), gabarito = o oficial. Filtros: veredito ok, esquema (2 a 10 alternativas, gabarito dentro do intervalo, id único), 8-grama contra as provas e a validação em português (131 saem) e contra o MMLU-ProX em inglês (0 saem; o índice foi conferido: pega um item conhecido), prompt até 1.024 tokens (0 saem). Saída: `data/grpo_prompts_gold.jsonl` com 11.324 de múltipla escolha + 5.612 de matemática (a múltipla escolha antiga, de gabarito fraco, não entra).
- Gabarito: A 3.006, B 2.865, C 2.701, D 2.281, E 424, F 26 e poucos de G a J; 9.010 questões com 4 alternativas, 1.934 com 5, 248 com 3 e 132 com 6 a 10. A recompensa de múltipla escolha foi conferida com 4, 5 e 10 alternativas.
- Qualidade da tradução (2 pares lidos, inglês e português): alternativas na mesma ordem e números iguais; termos técnicos às vezes errados ("bond" → "bonde/bônus"). O gabarito é o oficial, então a recompensa não depende da tradução, mas o enunciado pode ficar estranho.
- `scripts/run_grpo.sh` ganhou `--prompts` (padrão igual ao anterior). Decisão do usuário (15:35): rodar o RL depois do `gspo_thor` da suíte. Cadeia `gold-chain` (`scripts/run_gold_chain.sh`): espera `INICIO suíte gspo_thor`, cria `suite.stop` (a suíte termina o `gspo_thor` e não começa o `gspo1`), roda `gspo_gold` (adalberto, receita do `gspo1`, 8 ciclos, 1.024 questões de múltipla escolha no total: 618 WebInstruct e 406 MedMCQA) e retoma a suíte (`gspo1`). Ponto de partida `adalberto` e não `thor_pt_short`: 62,3% contra 57,4% em múlt. escolha com thinking (−4,9, IC −9,9 a 0,0) e fecha 86,8% contra 73,8%.
- Mudança (15:38, pedido do usuário): a suíte de teste roda o `gspo1` em vez do `gspo_thor`. Ordem: Lumi (já rodando) → suíte do `gspo1` → RL `gspo_gold`. Como o `run_suite.sh` estava rodando, não foi editado: foi criado `suite.stop` (a suíte antiga para depois do Lumi), a `gold-chain` foi relançada (versão 2) e há um script novo, `scripts/run_suite_models.sh`, que recebe a lista `NOME|modelo` por argumento. A suíte do `gspo_thor` pode ser feita depois com esse script.
