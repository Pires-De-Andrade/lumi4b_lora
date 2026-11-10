# Descobertas da fase 0

Nov 6, 2026 · RTX 4090 · registro de evidências, separado do [PRD](PRD.md). O PRD guarda decisões e requisitos; este arquivo guarda o que foi medido, como e com que limites.

Tudo abaixo foi medido nesta máquina, com Transformers 5.22.2, PEFT 0.21.2, PyTorch 2.14.1+cu130, `mamba-ssm` 2.3.2.post1 e `causal-conv1d` 1.7.0, sobre o Lumi v0.2 (commit `27b30f40`), em BF16. Os resultados brutos ficam em `runs/phase0/*.json` e `runs/jobs/*.log`, que não vão para o Git.

## Índice

Seções 1 a 19: fase 0 a fase 2 (ambiente, modelo, equivalência, memória e velocidade, dados, viabilidade, contaminação). A seção 20 é um diário cronológico das rodadas (20.1 a 20.50); as referências nos estudos usam o formato `§20.N`.

- 1. Ambiente e kernels
- 2. Estrutura do modelo
- 3. Código do repositório × implementação nativa
- 4. DoRA e as linhas zeradas da `in_proj`
- 5. Equivalência: adaptador recém-criado, fusão e piso de ruído
- 6. Gradientes
- 7. Parada e `eos`
- 8. Lições de processo
- 9. Extração de resposta (A-05 e A-06)
- 10. Modelo fundido e formato de chaves
- 11. Memória e velocidade (A-14)
- 12. vLLM e outras otimizações
- 13. Fase 1: dados
- 14. Treino: laço, memória e retomada
- 15. Fase 2: viabilidade (gate falhou na nota)
- 16. Contaminação da validação e experimento com o dataset do adalberto
- 17. Pacote da Thor e autodestilação do Lumi
- 18. Pendências
- 19. Como reproduzir
- 20. Resultado do selfdist, levantamento de datasets e ordem seguida

Diário da seção 20, por assunto:

- **§20.1 a 20.9**: SFT, dados e primeiras ideias (autodestilação, levantamento de datasets, escala do adaptador, tradução, GRPO inicial)
- **§20.10 a 20.17**: DoRA, Nemotron 4B, português x inglês, entropia do GRPO, triagem por camadas e o treino só com MLP
- **§20.18 a 20.26**: Dados da Thor (`thor_pt`, `thor_en`, `thor_pt_short`) e `gspo_thor`
- **§20.27 a 20.31**: Questões traduzidas, RL com gabarito oficial, revisões externas do GSPO, registro do treino e auditoria das questões
- **§20.32 a 20.38**: `gspo_gold`, `gspo_gold4k`, benchmark de velocidade, regra de parada e replanejamento
- **§20.39 a 20.44**: Qualidade do raciocínio quando o modelo termina, pacote 4 da Thor, resultado do benchmark, `gspo_gold` com 2.048 (ramo B), entrega 1 do pacote 3 com o SFT `thor_pt_e1` e o resultado dele; auditoria contra o TRL e a fila de testes (§20.45); emenda do critério de fechamento (§20.46); RL `gspo_thor_e1` (§20.47); diagnóstico de log-probabilidades (§20.48); SFT com perda DFT e fila v4 (§20.49); `gspo1_T06` (§20.50)

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
- Os arquivos Python do repositório (`modeling_nemotron_h.py`, `configuration_nemotron_h.py` e `nano_v3_reasoning_parser.py`) foram examinados por padrão de busca: sem rede, `exec`, `eval`, `subprocess`, `pickle` nem `os.system`. Isso não é uma auditoria completa; foram lidos só os trechos de módulos, kernels e geração.

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

A causa do erro com padding à esquerda **não foi isolada**. A hipótese é que o caminho rápido do código do repositório não zera os tokens de pad antes do estado do Mamba: no `forward()`, a multiplicação pela máscara vem depois do desvio para `cuda_kernels_forward`. Esse método não foi lido por inteiro, então ele pode tratar a máscara de outro jeito. É hipótese de leitura de código, não um experimento.

### Geração

- No código do repositório, `generate` falha no Transformers 5.22.2: `TypeError: 'NoneType' object is not subscriptable`, em `prepare_inputs_for_generation` (linha 1531). O `cache_position` chega nulo, porque esse método é uma cópia do Jamba, no estilo antigo. Versões anteriores do Transformers não foram testadas.
- Na nativa, `generate` funciona. Com `eos_token_id=[2, 11]` parou em 3 tokens (`olá<|im_end|>`, último id 11). Só com `eos_token_id=2`, gerou os 128 tokens permitidos. Isso confirma a afirmação do PRD sobre o `eos`.

### Kernels na nativa

O código nativo escolhe, por função, kernels do Hub (se pedidos), depois os pacotes `mamba-ssm` e `causal-conv1d` (se importáveis) e só depois PyTorch puro, com aviso de lentidão. Evidências de que usa os pacotes:

- Nenhum log contém o aviso de fallback.
- O profiler de um forward em modo `eval` mostrou `_state_passing_fwd_kernel` (Triton, do `mamba-ssm`) e `causal_conv1d_channellast_fwd_kernel` (CUDA, do `causal-conv1d`).

A nativa não tem a variável `is_fast_path_available`; o código do repositório tem, e vale `True`.

## 4. DoRA e as linhas zeradas da `in_proj`

- A `in_proj` tem **213 linhas com norma exatamente zero** (média de cerca de 10 por camada; a camada 0 tem 25 e a camada 2 tem 17). Nenhum outro tipo de módulo-alvo tem linhas zeradas.
- Só foram classificadas as posições das camadas 0 e 2: nas duas, os primeiros índices listados caem no trecho `xBC` da saída da `in_proj` (a ordem é `z`, `xBC`, `dt`). As demais camadas não foram classificadas.
- Sintoma sem o patch: com o DoRA, os logits viram `nan` e os 213 parâmetros treináveis ficam sem gradiente válido, já no primeiro passo. O LoRA no mesmo teste passou, com diferença exatamente 0.
- Causa: o DoRA do PEFT inicializa a magnitude com a norma da linha (0) e calcula `magnitude / norma` sem epsilon. É 0/0 nessas linhas. O mesmo cálculo acontece na fusão.
- Verificação da causa: com `lora_B` aleatório (diferente de zero), a norma deixa de ser zero e os logits ficam finitos. Isso é coerente com a explicação.
- Correção (`src/lumi_ft/modeling/peft_patch.py`): onde a norma é exatamente 0, o divisor vale 1. O patch fica desligado durante `update_layer`, para a magnitude inicial continuar sendo a norma verdadeira. Se o patch valesse também ali, as linhas mortas nasceriam com magnitude 1 e virariam canais ativos.
- Efeito: nessas linhas o fator é 0/1 = 0, então elas continuam produzindo zero. O DoRA **não** as ativa, ao contrário do LoRA, que pode. É uma diferença real entre os dois braços.
- Os testes unitários em CPU mostram que: sem o patch há NaN; com o patch a saída de um adaptador novo é igual à base; em modelos sem linhas zeradas a saída é bit a bit igual com e sem o patch; os gradientes ficam finitos; a fusão bate com o adaptador ativo; e as linhas mortas continuam zero após a fusão.
- Limite: o `unmerge` dividiria por um fator 0 nessas linhas. O projeto não o usa.

## 5. Equivalência: adaptador recém-criado, fusão e piso de ruído

### Como o critério foi definido (e mudou)

1. Primeiro, os limites foram escolhidos à mão: diferença máxima de 1% (adaptador novo) e 2% (fusão) da escala dos logits, e top-1 igual em 98%.
2. Esses limites falharam por falhas de desenho: a "guarda" que exigia efeito máximo 10× maior que a diferença da fusão era arbitrária; depois foram trocados por médias, KL e top-1 igual em 99%, número escolhido **depois** de ver o LoRA passar com folga.
3. O DoRA ficou fora desse limite (top-1 igual em 98,87%).
4. Em vez de afrouxar o número, passou-se a medir o **piso de ruído do próprio modelo em BF16**: o mesmo batch processado em batch com padding × cada sequência sozinha (matematicamente equivalentes). O critério passou a ser: nada pior que **2× esse piso** em diferença média, KL e discordância do top-1 (com folga de 1 posição). Os números do critério vêm da medição, e não de uma escolha.

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
- **Revisão dos filtros (D-06):** foram vistas 3 amostras de cada filtro numa prévia de 3.000 linhas (resíduo, último número, descontaminação): faziam o que se espera. Uma revisão maior dos removidos está em `runs/data/removed_samples.json` (15 por filtro) e não foi feita.

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
- Limites desta leitura: uma execução, 1 época, uma taxa; não foi separado quanto da perda vem do idioma do traço, do comprimento menor ou da deriva do LoRA sobre dados diferentes da distribuição do Lumi. Foram lidas ~5 saídas, não 50 (item da fase 2 não cumprido).

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
- **Uma execução, uma frase de prefill**; outras frases não foram testadas. A leitura "texto inteiro" só serve para diagnóstico, não é a nota do PRD.

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

- **`handoff/thor/`:** `PLANO_THOR.md` (roteiro de execução na Thor), `gen_traces.py` (cliente de `/v1/completions`, retomável), `filter_traces.py`, `pack.sh` e 50.908 problemas embaralhados (5.612 do GSM8K-PT `train` com gabarito e 45.296 do GigaVerbo com referência fraca). Professor: Qwen3-30B-A3B-Thinking-2507, com o começo do raciocínio forçado em português. Testado só contra um servidor falso (`tests/test_thor_pkg.py`); o motor real ainda não.
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

**Fonte escolhida: `open-r1/Mixture-of-Thoughts`, subconjunto `science`.** 172.514 linhas, 53.195 questões distintas, todas de múltipla escolha (`A: ...`, `B: ...`), 2 a 4 traços do R1 por questão, resposta em `\boxed{letra}`. Não há gabarito: usou-se a letra majoritária (≥ 75% de ao menos 3 traços; é um indicador, não verdade). Seleção (`scripts/prepare_mot_science.py`, `lumi_ft/data/mot_science.py`): 37.231 questões e 71.422 traços (até 3.000 tokens), ~99 M de tokens, em `data/mot_science_en.jsonl`. No arquivo 0, a mediana é 1.341 tokens: 35% cabem em 1.024, 68% em 2.048 e 88% em 4.096.

### 20.3 Ordem seguida

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

- **PRD:** nova seção "Atualização de escopo" (metas, critério de vitória e limiares **não mudaram**, regra 4); escopo ampliado com traços de professor, tradução do raciocínio e GRPO; Fora do escopo agora lista só APO, QLoRA, o CPT (avaliado e adiado) e o raciocínio em inglês (teste posterior); requisitos novos D-12 a D-16 (duplicatas em outra tradução, aceitação de traços de professor, prompt de treino, tradução, licenças) e A-15 a A-17 (validação limpa, comparação em quatro sistemas, Nemotron 4B de referência); tabela de fontes de professor; tabela dos experimentos executados e previstos; seção do GRPO; fase 2b com checklist e gate; 9 riscos novos; decisão sobre o adalberto fechada e 3 decisões novas em aberto (ponto de partida do GRPO, raciocínio em português como requisito, reabrir o CPT).
- **`comparacao.md`:** quatro sistemas (Nemotron 3 Nano 4B, Lumi, nosso Lumi em português e nosso Lumi em inglês); ganho do treino e custo do português como diferenças entre colunas; avaliação primeiro na validação (15 a 20 min) e depois na suíte; template, prompts e vocabulário do Nemotron 4B conferidos como idênticos aos do Lumi; esclarecido que "nenhuma resposta do Nemotron entra no treino" vale para a referência (4B), não para o professor de 30B; tabela do estado atual.
- **Ferramenta:** `scripts/report_compare.py` gera a tabela de comparação (nota, diferença com IC 95% contra uma referência e forma da saída) para qualquer conjunto de sistemas e provas.

### 20.10 DoRA e Nemotron 4B de referência (07/11/2026, 19:35)

- **DoRA = LoRA.** GSM8K limpo (676) 84,0% / 79,1% (com / sem thinking) contra 83,3% / 79,3% do LoRA; múltipla escolha (385) 62,6% / 57,7% contra 62,3% / 58,2%. A diferença é de 0,3 a 0,7 ponto, dentro do ruído; o GRPO continua partindo do `adalberto` LoRA.
- **Nemotron 3 Nano 4B de referência** (thinking, traço em inglês): GSM8K 85,2% (+3,6 sobre o Lumi, IC +1,0 a +6,1), múltipla escolha 68,3% (−1,3, empate). Sem thinking: 63,5% (−13,8) e 58,7% (−5,2). Acerto estrito e flexível são iguais, então não é formato. O Lumi empata com o pai na múltipla escolha; o nosso treino com traço em português perde 7 pontos.
- **Fontes dos prompts do pacote 2:** Mixture-of-Thoughts `science` (origem: nvidia/Llama-Nemotron-Post-Training-Dataset, fora da lista de SFT do Lumi), GSM8K-PT `train` e GigaVerbo (4.000 de matemática; o GigaVerbo-v2 está no pré-treino do Lumi, como texto cru).
- **Fila da GPU** (19:39, reordenada): diagnóstico português x inglês no Lumi base (`queue-diag`, ~20 min), depois `grpo1` (~3 h) e `adalberto_lr5e5`. O `grpo1` tinha começado às 19:34; a iteração 0 foi interrompida (sem rollouts salvos, o script é retomável) para o diagnóstico rodar antes.

### 20.11 Diagnóstico português x inglês e pacote 3 da Thor (07/11/2026, 20:12)

- **Diagnóstico** (Lumi original, as mesmas 1.200 questões do teste do MMLU-ProX nos dois idiomas, instrução no idioma da questão, marcador `Resposta:` igual; bootstrap pareado, 10.000 reamostras):

| | Português | Inglês | PT − EN (IC 95%) |
| --- | --- | --- | --- |
| Com thinking, estrita | 52,2% | 51,8% | +0,4 (−2,1 a +2,9) |
| Com thinking, flexível | 52,1% | 53,4% | −1,3 (−3,8 a +1,1) |
| Sem thinking, estrita | 48,7% | 44,8% | +3,8 (+0,8 a +6,8) |
| Sem thinking, flexível | 47,1% | 48,5% | −1,4 (−4,2 a +1,4) |

  O idioma da questão não muda o acerto do Lumi (empate na nota flexível nos dois modos; a vantagem do português na estrita sem thinking é de formato: em inglês ele escreve menos a linha `Resposta: X`). O Lumi raciocina em inglês nos dois casos, então isso não mede o custo de **pensar** em português; mede que a perda de 7 pontos dos nossos treinos não vem da questão estar em português, e sim do treino (traços em português, estilo do professor ou os dados). Truncadas com thinking: 27,5% (PT) e 26,1% (EN) em 4.096 tokens (questões de 10 alternativas).
- **Pacote 3 da Thor** (`handoff/thor3/`, `handoff/thor3_pacote.tar.gz`, sha256 `c5b2b18204e5dbf5626d90e16042af48778417d854e867ddffbcde44a87cf380`), para traduzir na Thor com o 30B e ampliar o domínio:
  - `translate_thor.py`: tradução com o próprio Nemotron 30B (`/v1/completions`, `enable_thinking=False`, T=0), em blocos de até 3.500 caracteres. Confere cada pedaço: terminou, sem `<think>`, tamanho 0,8 a 1,7× (0,4 a 2,6× abaixo de 200 caracteres), números (sequências de dígitos) preservados com até 10% de falta e "respondeu em vez de traduzir" (`\boxed`, `Resposta:`, `####`, `Answer:` que não estavam no original). Só os 70% de problemas com `bucket(id) < 7000` têm o raciocínio traduzido (a mesma divisão de `build_teacher_dataset.py`). O registro guarda o original em inglês.
  - Prompts novos (`scripts/make_thor3_prompts.py`, `lumi_ft/data/thor3.py`): 7.000 do WebInstruct-verified (Apache-2.0; das 27.764 de múltipla escolha, 18.360 com alternativas separáveis; nenhuma área acima de 20%) e 5.000 do MedMCQA (Apache-2.0; só `single`, sem imagem e sem o defeito de OCR "aery"; nenhuma área acima de 8%). Fora 925 + 2.007 com 8-grama em comum com o MMLU-ProX (teste e validação) ou o MMLU (teste) em inglês, e 1.160 + 19.489 enunciados repetidos. Nenhuma das duas fontes está na lista de treino do Lumi.
  - Do nosso lado: `scripts/prepare_thor_translated.py` escreve `think.jsonl` (raciocínio traduzido), `nothink.jsonl` (os outros 30%, disjuntos) e `think_en.jsonl` (o par com o raciocínio original em inglês, mesmo prompt e mesma resposta em português: o experimento "raciocínio em inglês" com os mesmos problemas). `filter_translated.py` agora também descontamina `think_en`.
  - Testes: `tests/test_thor3_pkg.py` (27; tradução de ponta a ponta com servidor falso, retomada, conversão, parsers). Não testado: a tradução com o servidor real (é o passo 1 do plano da Thor).
- **GRPO, ciclo 0** (`grpo1`, 256 questões × 8 respostas, 11 min, 2,2 M de tokens, limite de 2.048): recompensa média 0,87 em matemática e 0,54 em múltipla escolha; truncadas 3% e 23%; sinal (grupos com respostas diferentes) 23% e 66%. O treino do ciclo 0 estourou a memória (OOM no `backward`, 22 GiB, com uma só resposta de ~2.350 tokens e sem gradient checkpointing); a fila disparou o `adalberto_lr5e5` no lugar, interrompido no passo ~170 e apagado. Correção: `grpo_train.py --ckpt full` (padrão), pico de 12,3 GiB; o `grpo1` recomeçou às 20:36 aproveitando as respostas do ciclo 0. O limite de 2.048 tokens na geração (contra 4.096 na avaliação) é por tempo e memória do treino e porque a truncada vale 0 (pressiona a terminar); é uma escolha, não um requisito.
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

- Decisão: melhorar primeiro o modo sem thinking, depois analisar como não piorá-lo ao treinar o thinking. O GSPO a partir do `thor_pt` (H33) e a proteção do modo sem thinking no RL (H22) ficam adiados.
- **Diagnóstico (`estudos/05`, D8):** a múltipla escolha sem thinking perde por **formato**, não por conteúdo. Flexível 62,9 a 63,9% em todas as rodadas (Lumi 63,4%); estrita cai porque a linha `Resposta: X` some (87,8% no `adalberto`, 83,9% no `adalberto_mlp`, 96,1% no `thor_pt`). Os SFTs só de matemática respondem mais longo (400 contra 166 tokens). No GSM8K sem thinking, o estilo passo a passo do Qwen sobe +2 a +4; as respostas curtas do Nemotron não.
- Três SFTs só com exemplos sem thinking na fila (`nt_qwen`, `nt_thor`, `nt_both`; `data/nt_*.parquet`, ~3 milhões de tokens cada), na frente do `thor_pt_math` e do estudo de capacidade. Hipóteses H34 a H37 e previsões em `estudos/02` (E13).
- Pedido à Thor (prompt em `handoff/thor3/PROMPT_THOR_08NOV.md`): entregar os registros sem raciocínio (30% por hash) e gerar as 12.000 questões novas (gabarito oficial) **em blocos** de 3.000, com tradução e pacote a cada bloco.

### 20.21 O foco volta ao modo com thinking; GSPO a partir do `thor_pt` (08/11/2026, 13:05)

- Decisão (13:05): esquecer o sem thinking por ora; melhorar o **com thinking**. O E13 (`nt_*`) foi adiado: saíram da fila (`# ADIADOS` no fim de `scripts/capacity_plan.txt`), dados em `data/nt_*.parquet`. O diagnóstico D8 (sem thinking perde por formato) continua válido.
- **Fila:** `touch runs/jobs/cap.stop` (a `cap-queue` termina depois do `thor_en`, sem iniciar o `thor_pt_math`); a sessão `rl-thor` espera a `cap-queue` acabar e roda `gspo_thor`: `run_grpo.sh gspo_thor --init runs/train/thor_pt/adapter --iters 8 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 --max-new 2048 --temperature 0.6 --top-p 0.95 --ratio seq --max-noclose 0.85` (a receita do `gspo1`; guarda de 5% para 85% porque o `thor_pt` já começa com ~34% de truncadas). Previsões em `estudos/04`.
- Pendentes depois do RL: retomar `thor_pt_math` e o estudo de capacidade (apagar `cap.stop` e relançar `scripts/run_capacity.sh`; ele pula o que já está completo).

### 20.22 O que pedir à Thor com o foco no thinking: tradução só das questões (08/11/2026, 13:10)

- Argumento: o RL de múltipla escolha usa 3.715 questões de ciência com gabarito fraco (letra majoritária do R1). As 12.000 questões do pacote 3 têm gabarito oficial e cobrem negócios, economia, finanças (837 a 838 cada), matemática, física, química, biologia, psicologia, história e áreas médicas: mais parecidas com a suíte de teste (MMLU-ProX, ENEM, BLUEX, OAB) do que ciência de pós-graduação.
- Problema: o `translate_thor.py` só traduz registros **com traço do professor** (`kept.jsonl`), então só as questões que o professor acertou virariam prompts (as mais fáceis, viés de dificuldade). Solução sem alterar nada do pacote: novo arquivo `handoff/thor3/translate_questions.py` (sha256 `9e66416a…c5bb`), que reaproveita `question_block`, `translate_piece`, `check_piece` e `split_question_block` e traduz **todas** as 12.000, sem raciocínio (~20 a 40 min na Thor). Teste na CPU (`tests/test_thor3_pkg.py`, 29 testes no arquivo): enunciado e alternativas traduzidos, "respondeu em vez de traduzir" recusado, questão sem gabarito ignorada, retomada sem repetir.
- `handoff/thor3/PROMPT_THOR_08NOV.md` reescrito: 1) estado; 2) **traduzir só as questões** (teste de 30, depois todas, `questions_pt.tar.gz`); 3) terminar o passo 2 e entregar também os registros sem raciocínio; 4) traços do pacote 3 em blocos de 3.000, com tradução e pacote a cada bloco. É preciso copiar o novo arquivo para a pasta do pacote (a mesma do `translate_thor.py`).
- Do lado da 4090, falta o conversor de `questions_pt.jsonl` em prompts de RL (`prompt`, `gold`, `n_options`, `kind`), a escrever quando o arquivo chegar.

### 20.23 `thor_en`: o português custa 5 a 6 pontos; qualquer SFT do Nemotron piora a múltipla escolha (08/11/2026, 13:20)

- `thor_en` (mesmos 4.927 problemas e respostas do `thor_pt`, raciocínio original em inglês): com thinking, múltipla escolha **60,3%** ▼ e GSM8K **83,1%**; fecha `</think>` em 75,3%; laços 22,6% (múlt. escolha) e 3,7% (GSM8K). Contra o `thor_pt`: **+6,0** (IC +1,6 a +10,4) e **+5,3** (IC +2,7 a +8,0). H21 apoiada: o português custa 5 a 6 pontos, em parte por laços.
- O inglês também perde para o Lumi (−9,4 ▼): o SFT com traços do Nemotron piora a múltipla escolha em qualquer idioma; o português é parte do problema, não o todo.
- Sem thinking, o `thor_en` cai (GSM8K 65,7% ▼, múlt. escolha 56,1% ▼) por **formato**: as respostas estão em português, mas a linha final some em ~10% (flexível 79,1% no GSM8K).
- O `gspo_thor` (RL a partir do `thor_pt`) começou às 13:14 (ciclo 0, geração); cap-queue parou como previsto (13:14).

### 20.24 E11 encerrado em 1 de 15 rodadas (08/11/2026, 13:25)

- Questão levantada: por que analisar `thor_pt_math` e `thor_en` se o problema é raciocínio em português na múltipla escolha? Resposta honesta: eram diagnósticos. `thor_en` já respondeu (o português custa 5 a 6 pontos; o inglês também perde para o Lumi). `thor_pt_math` era um controle para explicar um ganho que não houve, e só tem matemática: **descartado**.
- Mesma lógica para o resto do E11: as 14 rodadas de capacidade, tipo de bloco, posição e escala variam um eixo que `cap_r04_std` e o E10 já mostraram irrelevante. **Canceladas** (removidas de `scripts/capacity_plan.txt`; as concluídas ficam nos resultados). Código e testes (`--rank-pattern`, `--alpha-pattern`, `--lr-mult`, `--layers`, `--targets`, `--rslora`) continuam no repositório. `resultados_auto.md` só mostra as rodadas feitas.
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

- `questions_pt.tar.gz` (sha256 `8c4e333920a66208…ca941`, íntegro, sem caminhos suspeitos): 12.000 questões, 11.455 aceitas (95,5%; 545 recusadas: 111 números divergentes, 208 tamanho, 207 alternativas não voltaram, 19 não terminou). Fontes aceitas: WebInstruct-verified 6.738, MedMCQA 4.717; 125,6 min na Thor. A cópia parcial (9.252 linhas) foi descartada.
- Conversor novo `scripts/make_questions_rl_prompts.py` (+9 testes): prompt = o da avaliação (`mcq.build_prompt`), gabarito = o oficial. Filtros: veredito ok, esquema (2 a 10 alternativas, gabarito dentro do intervalo, id único), 8-grama contra as provas e a validação em português (131 saem) e contra o MMLU-ProX em inglês (0 saem; o índice foi conferido: pega um item conhecido), prompt até 1.024 tokens (0 saem). Saída: `data/grpo_prompts_gold.jsonl` com 11.324 de múltipla escolha + 5.612 de matemática (a múltipla escolha antiga, de gabarito fraco, não entra).
- Gabarito: A 3.006, B 2.865, C 2.701, D 2.281, E 424, F 26 e poucos de G a J; 9.010 questões com 4 alternativas, 1.934 com 5, 248 com 3 e 132 com 6 a 10. A recompensa de múltipla escolha foi conferida com 4, 5 e 10 alternativas.
- Qualidade da tradução (2 pares lidos, inglês e português): alternativas na mesma ordem e números iguais; termos técnicos às vezes errados ("bond" → "bonde/bônus"). O gabarito é o oficial, então a recompensa não depende da tradução, mas o enunciado pode ficar estranho.
- `scripts/run_grpo.sh` ganhou `--prompts` (padrão igual ao anterior). Decisão (15:35): rodar o RL depois do `gspo_thor` da suíte. Cadeia `gold-chain` (`scripts/run_gold_chain.sh`): espera `INICIO suíte gspo_thor`, cria `suite.stop` (a suíte termina o `gspo_thor` e não começa o `gspo1`), roda `gspo_gold` (adalberto, receita do `gspo1`, 8 ciclos, 1.024 questões de múltipla escolha no total: 618 WebInstruct e 406 MedMCQA) e retoma a suíte (`gspo1`). Ponto de partida `adalberto` e não `thor_pt_short`: 62,3% contra 57,4% em múlt. escolha com thinking (−4,9, IC −9,9 a 0,0) e fecha 86,8% contra 73,8%.
- Mudança (15:38): a suíte de teste roda o `gspo1` em vez do `gspo_thor`. Ordem: Lumi (já rodando) → suíte do `gspo1` → RL `gspo_gold`. Como o `run_suite.sh` estava rodando, não foi editado: foi criado `suite.stop` (a suíte antiga para depois do Lumi), a `gold-chain` foi relançada (versão 2) e há um script novo, `scripts/run_suite_models.sh`, que recebe a lista `NOME|modelo` por argumento. A suíte do `gspo_thor` pode ser feita depois com esse script.

### 20.28 Revisão externa do RL: o `clip_fraction` do GSPO estava errado como diagnóstico (08/11/2026, 16:15)

- Procede: o registro calcula a razão por token e a compara com os limites da sequência (3e-4 e 4e-4); a perda corta a razão da sequência. A frase "40 a 50% dos tokens cortados, logo o modelo se mexe pouco" (§20.x do `gspo1`, H16) **não é sustentada**; corrigida em `estudos/04` e `07`. A perda e o gradiente estão certos.
- Procede: a perda GSPO é ponderada pelo comprimento (gradiente por token `-A·s/N`), não a média uniforme por sequência; documentado em `estudos/04`. Ablação adiada.
- Procede em parte: a recompensa mistura objetivos; os `whys` são um motivo por amostra em ordem de prioridade. Nos rollouts, grupos sem gradiente: múlt. escolha 26 a 53% todos certos e só 5 a 16% todos zero; matemática 72 a 78% todos certos (só ~20% de grupos úteis). No início do `gspo_thor`, 62% das amostras dos grupos todos-zero de múlt. escolha eram truncadas.
- Idioma: a heurística (97,9% e 96,1% de português) não mostra inflação nos casos limítrofes lidos; revisão humana de ~50 traços fica pendente para o candidato final.

### 20.29 Segunda leitura externa (`lumi_4b_tres_acoes_prioritarias.md`): o que procede (08/11/2026, 16:30)

- **Procede e corrige o que eu disse:** o `gspo_gold` compara conjuntos de treino (questões, domínios, fontes e rótulos mudam juntos); eu tinha escrito que "só o gabarito muda". O rótulo do WebInstruct-verified não é de banca (resposta da web verificada por modelo); conferido: usamos a revisão `3e8a350b` (a mais recente) e o `train` corrigido, não o `train_legacy`.
- **Procede e é mais preciso que o registro anterior:** o corte efetivo do GSPO é A > 0 com s > 1 + eps_alto e A < 0 com s < 1 - eps_baixo, e não "razão fora do intervalo"; registrar por sequência e por token, com os denominadores.
- **Confirmado no código, e ainda não estava em nenhum estudo:** o treino calcula as log-probabilidades **sem dividir pela temperatura** (`forward_lp` em `grpo_train.py`), mas os rollouts do `gspo1` e do `gspo_thor` foram gerados com T 0,6 e top-p 0,95. A razão `new/old` é consistente (as duas na mesma conta), mas as amostras vêm de uma distribuição mais fria do que a que se otimiza.
- **PRD:** a entrega é o adaptador **e a versão fundida em BF16**, então "rotear" (adaptador só com thinking) não cumpre o critério do modo sem thinking para o modelo fundido; serve só como opção de implantação à parte.
- **Teste reservado:** a suíte de teste do `gspo1` só vale como calibração da validação; a escolha da receita final continua na validação, e o teste do candidato final roda uma vez, com receita e checkpoint congelados.

### 20.30 Registro do treino, filtro de qualidade das questões e auditoria (08/11/2026, 16:55)

- Decisão (16:20): fazer o registro do treino; a auditoria sem julgar o conteúdo médico (ninguém do projeto vai julgar).
- Registro (só registro, a perda não muda): corte ativo por sequência e por token com denominadores, |log s| e ruído do primeiro passo; `parts` por amostra e `summary` por ciclo nos rollouts; `grpo_status.py` atualizado; 17 testes novos (corte ativo bate com o gradiente da perda, as partes batem com a recompensa, resumo dos grupos). 350+ testes na CPU. Teste rápido (`smoke_diag`) na `gold-chain` antes do `gspo_gold`, porque o laço de treino não pôde ser rodado com a GPU ocupada.
- Auditoria: **a tradução da Thor tem defeitos que passam nas conferências dela**: 219 enunciados deixados em inglês (1,9%) e ~60 negações perdidas ("not" some e o gabarito inverte). Filtro novo exclui 343 questões; pool de múltipla escolha 11.324 → 10.981. Falsos alarmes da primeira versão dos marcadores (alternativas "repetidas" que só diferiam no sinal ou em LaTeX; "nenhuma das alternativas" sem correspondente) corrigidos.
- Pendente: revisão humana de conteúdo (ninguém), e a revisão de ~50 traços para o idioma do candidato final.

### 20.31 Terceira leitura externa (`lumi_4b_sft_masking_datasets.md`): o que serve (08/11/2026, 17:00)

- **A premissa "traço longo contra resposta final curta" não vale para os nossos dados.** Parcela da loss (por token supervisionado) em exemplos com thinking: `adalberto` raciocínio 34,6% e resposta final 32,0% (a resposta depois do `</think>` tem ~315 tokens: é a solução inteira do Qwen; o raciocínio, ~341); `thor_pt` 61,4% e 16,5% (resposta ~125 tokens, raciocínio ~465); `selfdist` 48,7% e 19,2%. A letra da alternativa é 1 a 3 tokens em ~800 (~0,2%). Ponderar "a resposta final" reforçaria a solução, não a escolha; para reforçar a escolha o peso teria que ser na linha final `Resposta: X`.
- **Prompt sem loss:** em média 135 tokens de prompt contra 400 supervisionados (a razão em que os papers citados mostram pouco efeito; eles ganham com prompt longo e resposta curta).
- **Platô de intensidade do SFT:** lr 5e-5 (63,4% / 57,9%), adaptador x0,5 (63,4% / 64,9%), x0,75 (64,4% / 58,7%), limite 2.048 (64,2% / 55,3%), r=4 e só MLP (62,3%): a múltipla escolha com thinking fica em 62 a 64% qualquer que seja a "força" do SFT; o sem thinking é que se mexe (x0,5 volta ao nível do Lumi, +1,0; uma execução, ruído de ±4 a 5). Isso não favorece "menos pressão de SFT" como remédio da múltipla escolha com thinking; favorece o custo do idioma (5 a 6 pontos) e do formato.
- **Datasets sugeridos (ECQA, SciQ, OpenBookQA, CommonsenseQA, EU-Instruct):** em inglês, de nível fácil (senso comum e ciências do ensino básico), longe do ENEM/OAB/BLUEX/MMLU-ProX; licenças pendentes (SciQ CC BY-NC; ECQA CDLA-Sharing; OpenBookQA sem esclarecer); exigem tradução na Thor e descontaminação. Temos 10.981 questões em português de domínios mais próximos, com rótulo, sem explicação.
- **DFT:** pesa cada token pela probabilidade que o modelo já dá a ele, então reduz justamente o aprendizado dos tokens em português que o Lumi acha improváveis (traço de raciocínio em português: 0% no Lumi). Se for testado, com portão de idioma (português >= 90%).
- **O que serve:** (1) supervisão direta no modo sem thinking com as nossas questões rotuladas (`Resposta: X`, rótulo do dataset, sem explicação), que ataca o critério do PRD (o sem thinking perde 3,9 pontos no `gspo_thor`, e a perda é de formato, D8); (2) registrar a loss por segmento no SFT (diagnóstico barato); (3) cuidado dos denominadores (`n_targets`) se a loss for ponderada.

### 20.32 `gspo_gold`, ciclo 0 (08/11 17:39): o registro novo do corte, lido de verdade

- **Ruído numérico:** no primeiro passo do ciclo (antes de qualquer atualização) o desvio médio |log s| já é 4,5e-4, da ordem da faixa de corte (3e-4 / 4e-4). O teste rápido (8 sequências) tinha dado 4,8e-4. Só isso explica parte dos ~31% de sequências com corte ativo no passo 1.
- **Movimento real domina depois:** |log s| sobe para 1,7e-3 a 3,2e-3 a partir do passo 3 (4 a 7 vezes o ruído) e o corte fica ativo em 45 a 62% das sequências nos passos 3 a 12 (média do ciclo: 50,1% com gradiente zerado, 82,9% fora da faixa, 12 passos, 744 amostras usadas). Então o corte do GSPO com eps tão pequeno age como região de confiança apertada, e o ruído só pesa nos passos 1 e 2. Recalcular a log-prob antiga com o mesmo lote seria correção de baixa prioridade.
- **Dificuldade da base:** recompensa em múltipla escolha 0,357 (`gspo1`: 0,593); WebInstruct 25% de acerto e 52% truncadas (limite de 2.048), MedMCQA 53% e 11,5%; grupos zerados 46/128 (22 todas truncadas, 24 todas erradas), mistos 67/128.
- **Tempo:** geração 697 s, recomputação da log-prob antiga 91 s, treino 264 s: ~18 min por ciclo (8 ciclos + 2 avaliações: fim por volta das 20:00).

### 20.33 `gspo_gold` com 2.048 interrompido; novo RL com 4.096 (08/11 18:05)

- **Por que:** o limite de geração do RL (2.048) nunca foi testado e é a única variável comum às três rodadas de RL (`gspo1`, `gspo_thor`, `gspo_gold`), que pararam todas em ~67% de múltipla escolha com thinking (Lumi 69,6%). No ciclo 0 do `gspo_gold`, 36% das respostas de múltipla escolha truncavam (WebInstruct 52%, MedMCQA 11,5%) e truncada vale 0. No ciclo 1: recompensa 0,365 (0,357 no ciclo 0), WebInstruct 36% de acerto e 42% truncadas, MedMCQA 39% e 24% (50 a 78 questões por fonte e por ciclo: ruído grande).
- **O que foi feito:** o RL foi parado no ciclo 2 (ciclos 0 e 1 prontos; pasta `runs/grpo/gspo_gold`); teste de memória com 4.096 (`smoke_4k`, 32 questões x 4, 21 de 128 truncadas no limite, 10 amostras com 4.000 tokens ou mais entre as usadas no treino): pico de 12,54 GiB alocados, o mesmo do treino a 2.048 (12,46). Novo RL `gspo_gold4k`: receita do `gspo1` (ponto de partida `adalberto`, 8 ciclos, T 0,6, top-p 0,95, `--ratio seq`, lr 2e-5), mudando **duas coisas** em relação a ele: as questões e `--max-new 4096`. Se ganhar, não separa qual causou; separar depois, se valer a pena.
- **Decisão de uso (08/11):** o adaptador só liga com `enable_thinking=True`; sem thinking roda o Lumi original. A queda sem thinking dos candidatos deixa de pesar no uso (o critério do PRD é atendido por construção, e o relatório precisa dizer isso; o BF16 fundido sozinho degrada o sem thinking). Variantes antes descartadas pelo sem thinking (ex.: `adalberto_L2048`, 64,2% com thinking e 55,3% sem) voltam a valer o teste; `nt_*` e H22 perdem prioridade; a suíte dos candidatos pode rodar só com thinking (o Lumi sem thinking continua como referência).
- **Fila:** só a `gold-chain` (o RL). A sessão `suite-gspo1` foi criada às 18:05 e **removida às 18:20**: com o limite maior há outros testes a fazer antes da suíte de teste do `gspo1`, que fica adiada (nada dispara depois do RL; a GPU fica livre para o próximo teste).

### 20.34 `gspo_gold4k`, ciclo 0 (08/11 18:27): mesmo conjunto de 256 questões do ciclo 0 do `gspo_gold` (a seleção é determinística), só muda o limite

| Ciclo 0 | 2.048 | 4.096 |
| --- | --- | --- |
| Recompensa múlt. escolha / matemática | 0,357 / 0,870 | 0,398 / 0,885 |
| Múlt. escolha que termina (não trunca) | 63,9% | 73,1% |
| WebInstruct: truncadas / recompensa | 51,9% / 0,25 | 39,3% / 0,31 |
| MedMCQA: truncadas / recompensa | 11,5% / 0,525 | 7,7% / 0,54 |
| Grupos de múlt. escolha: todos zero (todas truncadas) / mistos | 46 (22) / 67 | 35 (11) / 76 |
| Corte ativo (sequências) / pico de memória | 50% / 12,46 GiB | 39% / 12,69 GiB |
| Geração / treino | 697 s / 264 s | 971 s / 202 s |

- Ruído: uma execução, 128 questões de múlt. escolha por ciclo (a recompensa de 0,357 para 0,398 está dentro do ruído); a queda das truncadas e o aumento dos grupos mistos são o que se esperava do limite maior.
- **Truncadas a 4.096 não são só laços:** nas 276 truncadas de múlt. escolha com ≥1.000 tokens, 25,7% têm 8-gramas muito repetidos nos últimos 1.000 tokens (razão de únicos < 0,5; 12,7% abaixo de 0,2), contra 0% nas que terminam. WebInstruct: 21% repetitivas (79% são raciocínio longo sem laço); MedMCQA: 61% repetitivas (laços, em perguntas curtas). Heurística de n-gramas: não pega laço com paráfrase.
- **4.096 é o limite certo para o treino:** a avaliação e o PRD cortam em 4.096, então a recompensa passa a coincidir com o que a avaliação mede; mais que isso treinaria um comportamento que a nota não vê.

### 20.35 Benchmark de velocidade da geração do RL (enfileirado 08/11 ~18:55)

- **Por quê:** a geração é o gargalo do ciclo (971 s de ~21 min, 77%; 2,2 h por RL de 8 ciclos a 4.096 tokens). Uma revisão externa propôs quatro ajustes sem perda de qualidade (nenhum testado): backend FlashInfer para o Mamba, ajuste do kernel Triton, prefix caching desligado, `gpu_memory_utilization` 0,88 para 0,93.
- **Verificado no código instalado:** o vLLM 0.33.0 tem `MambaBackendEnum` (triton padrão, flashinfer, cpu) e o flashinfer 0.7.2 está instalado; o código diz que o FlashInfer "não suporta prefix caching ainda" (na doc do ReplaySSM; escopo incerto), então o FlashInfer provavelmente exige o prefix caching desligado. Nós ligamos o prefix caching de propósito (`grpo_rollout.py`, linha 46). A capacidade de cache é de 172.373 tokens (concorrência de 33,7x para 5.120 tokens), mas o log do RL não imprime a concorrência real (estatísticas desligadas).
- **O que roda:** `scripts/run_bench_rollout.sh` (sessão `bench-rollout`, espera a `gold-chain` acabar, ~21:15): `scripts/bench_rollout.py` com o `gspo1/merged`, 64 questões x 8 respostas (as mesmas do ciclo 0), 4.096 tokens, T 0,6, top-p 0,95, uma configuração por processo: `base`, `prefix_off`, `prefix_off_mem93`, `flashinfer_prefix_off`, `base_repete` (ruído entre execuções). Liga o log de estatísticas do vLLM (respostas rodando, esperando, uso do cache de KV). Tempo máximo de 25 min por configuração; uma que falhe não derruba as outras. Resultados em `runs/bench_rollout/results.jsonl` e `tabela.md`.
- **Critério (antes de ver os números):** adotar nos RLs seguintes a configuração que ganhar ≥15% em tokens por segundo contra a base E mantiver a recompensa e o tamanho das respostas iguais dentro do ruído (as saídas não são idênticas entre kernels). O `gspo_gold4k` fica como está (homogêneo).
- Só arquivos novos (`bench_rollout.py`, `run_bench_rollout.sh`, `tests/test_bench_rollout.py`; 361 testes passam); o RL em curso não foi tocado.

### 20.36 Fila: triagem do RL a partir do `thor_pt` com 4.096 (enfileirada 08/11 19:20)

- **Comprimento dos traços do Nemotron (5.278 registros traduzidos com raciocínio):** ciência p50/p90/p99 de 585 / 1.836 / 2.846 tokens de raciocínio; GSM8K 269 / 674 / 1.918; matemática do `gv_math` 259 / 562 / 1.484 (o Qwen do `adalberto` tem p90 de ~493). Exemplos (pergunta + raciocínio + resposta) acima de 2.048 tokens: 269 de 5.278 (5,1%; 237 na ciência, 9,6% dela), nenhum acima de 4.096. **Subir o limite do SFT para 4.096 acrescentaria no máximo ~269 exemplos**, então refazer o SFT só por causa do comprimento não vale; o `thor_pt_short` (cortar a cauda) já tinha dado +3,1, sem sinal.
- **Onde o limite novo pode ajudar o Nemotron é no RL:** com 2.048 o RL cortava justamente a cauda longa (o `gspo_thor` terminou com mediana de 534 tokens, contra 1.035 do `gspo1`).
- **Fila:** sessão `thor4k` (`scripts/run_thor4k_chain.sh`): espera a `gold-chain` e o `bench-rollout`, depois roda `gspo_gold4k_thor`: `--init runs/train/thor_pt/adapter`, 4 ciclos, as questões do gold, `--max-new 4096`, mesma receita do `gspo_gold4k`. A avaliação do final equivale ao ciclo 4 do `gspo_gold4k` (comparação direta; só muda o ponto de partida). Continuar até 8 ciclos só se o ciclo 4 for promissor. Motivo extra: o `adalberto` não tem licença declarada. Usa a configuração padrão do vLLM, sem as mudanças do benchmark (para ficar homogêneo com o `gspo_gold4k`). Cancelar: `tmux kill-session -t thor4k` (só enquanto espera).

### 20.37 `gspo_gold4k`, ciclo 4 (08/11 19:40): a regra de parada fixada às 19:30 foi acionada; RL interrompido

Validação limpa, com thinking (GSM8K 676 itens, múltipla escolha 385), IC 95% pareado:

| | Lumi | `gspo1_c4` | `gspo_gold4k` ciclo 4 |
| --- | --- | --- | --- |
| GSM8K | 81,7% | 84,3% (+2,7; +0,0 a +5,3) | 84,6% (+3,0; +0,4 a +5,6) ▲ |
| Múlt. escolha | 69,6% | 67,8% (−1,8; −6,2 a +2,6) | **63,1% (−6,5; −11,2 a −1,8) ▼** |
| Fecha `</think>` e responde (múlt. escolha) | 86,2% | 90,9% | **83,9%** |
| Truncadas (múlt. escolha) | 13,8% | 10,6% | **16,4%** |
| Traço em português (múlt. escolha) | 0% | 95,4% | 98,5% |
| Mediana de tokens (múlt. escolha) | 490 | 1.086 | 1.250 |

- Contra o `gspo1_c4`: múlt. escolha −4,7 (IC −9,4 a +0,0); GSM8K +0,3 (IC −1,9 a +2,5).
- **Regra:** "múlt. escolha < ~62% **ou** fecha < ~87% → parar". Fecha 83,9% (< 87%), então parou, embora a nota (63,1%) esteja acima de 62%. Adaptador `it003` (o avaliado) guardado; retomável com o mesmo comando (os ciclos 0 a 3 estão prontos).
- **Onde foi a perda (385 itens pareados):** só o `gspo1_c4` acerta 51 itens e só o `gspo_gold4k` acerta 33; **30 dos 51 estão truncados** no `gspo_gold4k`. Truncadas: 63 (contra 41); com repetição forte nos últimos 700 palavras (razão de 8-gramas únicos < 0,5): **39 de 63 no `gspo_gold4k` contra 11 de 41 no `gspo1_c4`**. Ou seja, o `gspo_gold4k` entra mais em laço (≈10% dos itens contra ≈3%). Heurística de n-gramas, uma execução.
- **Hipóteses (nenhuma testada):** (a) o limite de 4.096 premia raciocínios longos que terminam entre 2.048 e 4.096, e raciocínio longo aumenta o risco de laço; (b) as questões do gold (WebInstruct longa e difícil, MedMCQA com laços já no ciclo 0: 61% das truncadas) deslocam o modelo para uma distribuição diferente da validação; (c) ruído (385 itens, erro padrão ~2,4). Como mudamos duas coisas ao mesmo tempo, não dá para separar.
- **O que separa:** (1) retomar o `gspo_gold` de 2.048 (pasta `gspo_gold`, ciclos 0 e 1 prontos) até o ciclo 4, com as mesmas questões: ~50 min; (2) o `gspo1` com as questões antigas e 4.096.
- **Fila depois da parada:** o benchmark de geração começou às 19:41 e roda sozinho (~40 min). A sessão `thor4k` (triagem do `thor_pt` com 4.096 e questões do gold) foi **cancelada às 19:45**, porque repetiria a configuração suspeita; reativar se o teste de separação inocentar o limite.

### 20.38 Replanejamento (08/11 19:55)

Fila agora: `bench-rollout` (termina ~20:05) e depois `gold2048` (`scripts/run_gold2048_resume.sh`): retoma o `gspo_gold` (2.048, as mesmas questões) até o ciclo 4 e avalia. Regra de leitura fixada antes do número, em `estudos/08_registro_de_decisoes.md` (X ≥ 67% e fecha ≥ 90%: o culpado é o limite; X ≤ 64% ou fecha < 87%: são as questões). A triagem do `thor_pt` a 4.096 está cancelada. Benchmark de geração, primeira medida (base): 215 s, 2.837 tokens/s, rodando em média 90 (máximo 115, nunca os 128 permitidos), 186 esperando em média e uso de KV de 86% em média (100% no pico): o cache é o limitador, não o `max_num_seqs`.

### 20.39 Qualidade do raciocínio quando o modelo termina: o SFT só de matemática (`adalberto`) é parte do problema (08/11 ~19:55)

Validação, múltipla escolha com thinking, acerto **só nos itens em que todos os modelos comparados fecham o `</think>` e respondem** (condicionar no fechamento escolhe itens mais fáceis: não é uma comparação causal limpa), bootstrap pareado 95%:

| Modelos (itens comuns) | Lumi | `adalberto` (só matemática) | `thor_pt` (matemática + 2.174 problemas de ciência, PT) | `thor_en` (os mesmos dados, em inglês) | `thor_pt_short` (1.255 de ciência) |
| --- | --- | --- | --- | --- | --- |
| 189 itens | 87,8% | **82,5%** (−5,3; IC +1,1 a +10,1) | 87,3% (−0,5; IC −3,2 a +4,2) | 86,8% | 83,6% |

- O português **não** custa acerto quando os dados são os mesmos: `thor_pt` e `thor_en` empatam com o Lumi nos itens comuns. Isso **corrige** o achado anterior de "custo do idioma de 5 a 6 pontos".
- O `adalberto` perde ~5 pontos de qualidade nos itens que fecha; o `thor_pt`, que tem múltipla escolha de ciência, não (`thor_pt` − `adalberto`: +4,8; IC −0,5 a +10,1, no limite). O `thor_pt_short`, com menos ciência (cortado por tamanho), fica no meio (83,6%): coerente com um efeito de dose, **não testado**.
- O problema do `thor_pt` é **fechar**: termina 254 de 385 itens (66%), contra 332 do Lumi, 332 do `adalberto` e 354 do `gspo1`. Por isso o acerto geral dele é 54,3%.
- Depois do RL (299 itens comuns): Lumi − `gspo1` = +6,7 (IC +2,0 a +11,0); Lumi − `gspo_thor` = +4,3 (IC +0,7 a +8,0); `gspo_thor` − `gspo1` = +2,3 (IC −1,7 a +6,4, empate). O RL não fecha a diferença de qualidade.
- **Leitura:** dois problemas separados. (a) Qualidade do raciocínio em múltipla escolha: o SFT só de matemática a prejudica; o SFT com ciência a preserva. (b) Fechamento: o `adalberto` fecha bem e o `thor_pt` não (cauda longa e laços do Nemotron). O que falta é traços de múltipla escolha que fechem limpo, que é o papel do pacote do Qwen. Corrige também a frase "o SFT não é o gargalo" (baseada só no empate depois do RL).
- Teto, por conta de guardanapo (hipótese forte): qualidade igual à do Lumi nos itens que fecham e fechamento de ~93 a 98% daria ~73 a 75% de múltipla escolha, ou +4 a +5 sobre o Lumi.

### 20.40 Pacote 4 da Thor: traços de múltipla escolha em português com o Qwen3-30B-A3B-Thinking (preparado 08/11 ~20:00)

- **Por quê:** §20.39 (o SFT só de matemática prejudica a qualidade em múltipla escolha; o do Nemotron com ciência a preserva mas não fecha). O pacote gera o que falta: traços de múltipla escolha, em português, de um professor que fecha limpo.
- **Conteúdo** (`handoff/thor4/`, empacotado em `handoff/thor4_pacote.tar.gz`, sha256 `6430a38d055908992ee26f426f1ca3c1de4540f31f95f2fb945fbd9a6d6229e3`): 9.957 questões já em português (MedMCQA 4.098, WebInstruct 5.859; gabarito do dataset), tiradas das 10.981 do `grpo_prompts_gold.jsonl` menos as 1.024 que o RL do `gspo_gold` usa em 8 ciclos; embaralhadas e intercaladas entre as fontes (`scripts/make_thor4_prompts.py`; `data/prompts.jsonl` sha256 `3c0195f9…53be`); o `gen_traces.py` do pacote 1 (idêntico ao do 3); `filter_traces_pt_mcq.py` (fecha uma vez, termina sozinho, traço em português, `Resposta: <letra>` na última linha igual ao gabarito; **sem limite de tamanho**: reporta a distribuição de tokens e o acerto do professor por fonte); `pack.sh`; `PLANO_THOR4.md` e `PROMPT_THOR_QWEN.md` (texto de apresentação do pacote).
- **Parâmetros:** `--max-tokens 4096` (o da avaliação; traços maiores não fecham e saem), T 0,6, top-p 0,95, top-k 20, 1 traço por questão, começo forçado em português (3 alternativas testadas no teste de 30 questões), blocos de 2.000 com entrega parcial a cada bloco, servidor do Qwen com `--max-model-len 6144`.
- **Condição:** um servidor por vez na Thor. O servidor do Nemotron só é parado depois da ENTREGA 1 do pacote 3 e com confirmação; os blocos 3 e 4 do Nemotron ficam cancelados.
- **O limite de 1.024 tokens da receita do `adalberto` NÃO é aplicado na Thor:** o corte do exemplo de SFT é escolhido na 4090, olhando a distribuição de tamanhos (ponto de partida: 2.048, como no `thor_pt`).
- **Aceite:** ≥3.000 mantidos (meta ≥5.000), ≥90% dos traços em português, rendimento ≥20% (abaixo disso a Thor para e reporta).
- **Depois que chegar:** SFT misto (os dados do `thor_pt`, o resto do pacote 2, o pacote 3 com limite de ~2.048 tokens e os traços do Qwen), medir qualidade e fechamento no ponto de partida (~40 min de SFT, ~13 min de avaliação) e triagem de RL de 4 ciclos.
- Testes: `tests/test_thor4_filter.py` (4 testes; 361 + 4 passam, sem os de GPU).

### 20.41 Benchmark de velocidade da geração do RL: resultado (08/11 20:01)

`gspo1/merged`, 64 questões x 8 respostas, 4.096 tokens, T 0,6, top-p 0,95 (512 amostras; a cauda pesa mais que nos 2.048 do RL):

| Configuração | Geração | Tokens/s | vs média da base | Truncadas | Recompensa mcq / math | Rodando (média / máx.) | KV médio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `base` | 215 s | 2.837 | | 7,8% | 0,531 / 0,840 | 90 / 115 | 86% |
| `base_repete` | 213 s | 2.995 | **ruído: 5,4% entre duas execuções iguais** | 8,2% | 0,555 / 0,813 | 94 / 115 | 90% |
| `prefix_off` | 195 s | 3.106 | +6,5% | 6,8% | 0,578 / 0,836 | 96 / 115 | 92% |
| `prefix_off_mem93` | 189 s | 3.186 | **+9,3%** | 6,5% | 0,602 / 0,832 | 109 / 126 | 94% |
| `flashinfer_prefix_off` | 219 s | 2.896 | −0,7% | 7,8% | 0,539 / 0,832 | 91 / 115 | 88% |

- **Critério fixado antes (≥15% em tokens por segundo e qualidade igual): nenhuma configuração passa.** A melhor (sem prefix caching e memória em 0,93) dá +9,3% sobre a média das duas bases, ~1,7 vez o ruído entre execuções iguais. Não adotada.
- O cache era o limitador (a base nunca passa de 115 respostas rodando; com 0,93 chega a 126 de 128), mas a mudança dá pouco. O FlashInfer no Mamba não ajuda (−0,7%).
- Ganho possível se adotássemos o melhor: ~80 s por ciclo de ~970 s, ~11 min por RL de 8 ciclos. Sem risco de qualidade visível (recompensa e truncadas iguais dentro do ruído), mas abaixo do critério.

### 20.42 `gspo_gold` com 2.048, ciclo 4 (08/11 20:51): a regra fixada antes aponta o ramo B

- Retomado às 20:02 (ciclos 2 e 3, `scripts/run_gold2048_resume.sh`), avaliado às 20:50. Validação limpa, com thinking: múltipla escolha **63,9%** (−5,7 contra o Lumi, IC −10,1 a −1,3 ▼), GSM8K 85,2% (+3,6 ▲), fecha `</think>` e responde 89,4%, trunca 11,4%, mediana 1.167 tokens, traço em português 95,1%. Sem thinking (o adaptador sozinho, que não será usado assim): GSM8K 79,4% e múltipla escolha 60,3%.
- Comparações pareadas: contra o `gspo1_c4` (mesmo limite, questões antigas) −3,9 (IC −7,8 a +0,0); contra o `gspo_gold4k` c4 (mesmas questões, limite 4.096) o 2.048 fica +0,8 (IC −3,1 a +4,7), com fechamento de 89,4% contra 83,9%.
- Leitura pela regra de `estudos/08` (fixada antes do número): `X ≤ 64%` → ramo B. Disparou por 0,1 ponto; o limiar é estimado. O limite de 4.096 não mexe na nota e piora o fechamento; as questões do gold não ajudam (~4 pontos abaixo, no limite do significativo, uma semente).
- Fila: `scripts/run_noite_chain.sh` (sessão `noite`, criada às 20:46, decide sozinha pela mesma regra; testada antes em `gspo1_c4` → A, `gspo_gold4k_it4` → B, `gspo_thor` → C) leu `B X=0,6390 C=0,8935` às 20:51 e iniciou o `gspo1_4k` (receita do `gspo1`, questões antigas, 4.096, 4 ciclos). Depois: SFT `thor_pt_e1` e a suíte de teste do Lumi que falta. A avaliação foi renomeada de `runs/eval/gspo_gold` para `gspo_gold_it4` (mesmo padrão do `gspo_gold4k_it4`).

### 20.43 Entrega 1 do pacote 3 e SFT `thor_pt_e1` (08/11 20:38 a 20:48)

- **Entrega:** `thor3_out.tar.gz` (sha256 `768372c52d01112e85fee14eaaf5b353ae0fa577a31795b7883a4ac513158e64`, 46 MB), extraída em `data/thor3_e1/` (pasta nova; `data/thor3/` não foi tocado). Pacote 3: 7.908 prompts gerados de 12.000 e 3.527 mantidos (44,6%; 43,6% truncados em 2.501 tokens; mediana de 761 tokens nos mantidos). Tradução: 17.241 registros (13.714 do pacote 2 e 3.527 do pacote 3), 16.131 `ok` (93,6%), ids únicos; os números batem com os relatórios da Thor. Pacote 1 abaixo do mínimo do plano; geração do pacote 2 parou no orçamento de 420 min (17.781 de 21.592).
- **Tradutor:** a Thor usou uma cópia modificada do `translate_thor.py` (prefill na resposta, limpeza de espaços especiais, reserva frase a frase, processamento contínuo). O diff vinha no `NOTAS.md`; aplica sem rejeições no original e gerou `handoff/thor3/translate_thor_v2.py`. A versão que produziu os dados é anterior ao teste A/B das 00:38 (rodar com `T3_SEM_2A_TENTATIVA=1`). Os limites de aceitação são os do original.
- **Preparo:** `prepare_thor_translated.py` e `filter_translated.py` em `data/thor_tr_e1/`: 10.929 exemplos com raciocínio (removidos 78 por 8-grama ou duplicata da validação) e 5.097 sem. Composição com raciocínio: 4.493 de matemática e 6.436 de múltipla escolha (4.259 `mot_science`, 1.186 MedMCQA, 991 WebInstruct). Tokens do raciocínio (tokenizador do Lumi): mediana de 567 a 948 e p90 de ~2.000 a 2.250 na múltipla escolha; 86% a 91% cabem em 2.048. 3,6% das respostas de múltipla escolha repetem `Resposta: X` (3,4% no `thor_pt`; mantido para comparar).
- **SFT `thor_pt_e1`:** `build_teacher_dataset.py --think data/thor_tr_e1/think_clean.jsonl --max-think 20000 --max-len 2048` (mesma receita do `thor_pt`, que usou `think_match`): 9.850 exemplos com raciocínio (1.079 descartados por passar de 2.048 tokens; 5.426 de múltipla escolha e 4.424 de matemática) e as 5.544 linhas sem raciocínio do GigaVerbo; 15.394 linhas, 9,3 M de tokens; `data/thor_pt_e1.parquet` (sha256 `1ae6c8e8a1235fc9e6a7b13a2cc85749a3c133291f478e0c965335d44f57a2e5`), mesmo esquema do `thor_pt.parquet`. Primeira etapa da `noite` (versão 2, 21:00; SFT iniciado às 20:57): LoRA r=16, alpha 32, lr 1e-4, 1 época. Depois dele vem o RL `gspo_thor_e1` (receita do `gspo_thor`, 4 ciclos, a partir do `thor_pt_e1`), o `gspo1_4k` do ramo B e a suíte do Lumi. A primeira versão da fila (20:40) não tinha RL depois do SFT; a falha foi apontada e corrigida às 21:00 (o `gspo1_4k` tinha rodado 6 min e foi reiniciado).

### 20.44 `thor_pt_e1`: mais múltipla escolha do Nemotron não melhora o ponto de partida (08/11 21:42)

- Treino de 481 passos (20:57 a 21:28, ~31 min). A avaliação terminou com segfault (código 139) no processo principal do `eval_vllm.py`, ao desserializar o `tokenizer.json` (Rust, `tokenizers`), **depois** de o motor do vLLM ter subido; os arquivos do tokenizador são idênticos aos do `thor_pt` (sha256 igual) e a repetição às 21:30 funcionou sem mudanças: falha intermitente. A fila (`noite`) ia pular para o `gspo1_4k` sem a avaliação; foi parada, a avaliação refeita (sessão `eval-e1`, até 3 tentativas) e a fila ajustada para esperar essa sessão e pular o SFT já avaliado.
- Resultado (validação limpa, com thinking): múltipla escolha **51,4%** (−18,2 contra o Lumi, IC −22,9 a −13,5 ▼; −2,9 contra o `thor_pt`, IC −7,0 a +1,3), GSM8K 76,9% (−4,7 ▼; o 82,0% do conjunto de 1.000 itens inclui os itens contaminados e não se compara com o 81,7% do Lumi nos 676 limpos), fecha 63,4%, trunca 36,6%, mediana 802 tokens, traço em português 99,2%. Sem thinking: GSM8K 76,5% (1.000 itens) e múltipla escolha 61,8%.
- Comparação das cinco SFTs do Nemotron (múlt. escolha com thinking / fecha): `thor_pt` 54,3% / 66,0%; `thor_pt_short` (≤600 tokens) 57,4% / 73,8%; `thor_pt_e1` 51,4% / 63,4%. Dobrar a quantidade de dados (5.426 contra 2.174 exemplos de múltipla escolha) e manter a cauda até 2.048 tokens não melhora; cortar a cauda (≤600) foi o que mais melhorou o fechamento. Compatível com a cauda longa do professor como causa dos laços, sem isolá-la (muda a quantidade e o comprimento ao mesmo tempo).
- Fila: o RL `gspo_thor_e1` (4 ciclos, a partir do `thor_pt_e1`) começou às 21:43, como programado. O ponto de partida está abaixo do limiar de ~62% (estimado) que se usou para não rodar RL a partir do `thor_pt_short`; mantido como triagem barata.

### 20.45 Auditoria do pipeline contra as práticas do TRL e fila de testes (08/11 22:00 a 22:21)

- **Base:** documentação oficial do TRL (GRPO e SFT) lida em 08/11. Não se usou a parte de treino na nuvem (HF Jobs e envio ao Hub): a regra do projeto é não publicar, e o dataset do `adalberto` não tem licença.
- **Igual ao TRL:** perda só na resposta; LoRA com taxa ≈1e-4; RL em nível de sequência (GSPO, `importance_sampling_level="sequence"`); sem KL (`beta=0`); vantagem normalizada por grupo (`scale_rewards="group"`); métricas equivalentes (`entropy`, `frac_reward_zero_std`, `completions/clipped_ratio`, `clip_ratio`). Melhor que o padrão do TRL: o SFT descarta exemplos longos em vez de truncar (o `max_length` do TRL trunca e mantém o começo, o que cortaria o `</think>` e a resposta).
- **Lacunas:** (1) sem a correção do descompasso vLLM x treinador (`vllm_importance_sampling_correction`, ligada por padrão no TRL) e sem a métrica `sampling_logp_difference`; (2) os logits do treinador não são divididos pela temperatura de amostragem (0,6) (o TRL provavelmente divide: de memória, a página lida não trata disso); (3) o ruído do primeiro passo (|log s| ≈ 4,5 a 5e-4) tem a escala do eps do GSPO (3e-4 e 4e-4), e o TRL cita a cabeça FP32 (`cast_lm_head_to_fp32`, exige embeddings não amarrados; o Lumi tem `tie_word_embeddings=False`) como remédio; (4) a perda `dft` (`loss_type="dft"` do SFTConfig) não foi testada.
- **Riscos do método:** 385 itens de múltipla escolha (erro padrão ~2,4 pontos) e uma semente por configuração; várias decisões do dia estão no limite do distinguível.
- **Implementado (padrões inalterados, 373 testes de CPU passando):** `--loss dft` no SFT (`train_sft.py`, `batch_loss_sum(..., loss_type)`); divisão dos logits por T no treinador do RL (`completion_logprobs(..., temperature)`, `--logit-temp` ou `GRPO_LOGIT_TEMP`); semente do RL por `GRPO_SEED` (rollout e treinador); `scripts/diag_logprob.py` com `lumi_ft/train/logprob_diag.py` (vLLM x treinador, ruído do corte em duas composições de batch, cabeça FP32, adaptador desligado = Lumi). O RL em execução não foi afetado: as mudanças só valem com as variáveis ou os argumentos novos.
- **Fila (`noite3`, 22:21):** RL `gspo_thor_e1` (retomado no ciclo 3), diagnóstico, SFT `adalberto_dft`, RL `gspo1_T06`, RL `gspo1_s1`, RL `gspo1_4k`, suíte do Lumi; ver `estudos/08`. A primeira tentativa de lançar a `noite3` esperou a si mesma (`tmux has-session -t noite` casa por prefixo com `noite3`) e foi relançada com `=noite` em 1 min.
- **Ficou de fora:** validação de múltipla escolha ampliada (precisa checar o tamanho do pool fora dos testes e descontaminar) e a correção TIS/MIS do TRL (precisa guardar as log-probabilidades do vLLM nos rollouts).

### 20.46 Critério de fechamento do `</think>`: de 98% para 90% (08/11 22:20)

- Emenda no PRD: meta mínima de fechamento passa de ≥ 98% para ≥ 90% (meta alvo de ≥ 99,5% mantida). **Decidida depois de conhecidos os resultados**: o `gspo1` (93,2%) passa a cumprir, o que não ocorria com 98%; contra o espírito da regra 4, por isso está declarada como exceção no PRD, em `estudos/08` e nos resumos, e os relatórios devem citar os dois limiares. Na validação de múltipla escolha, com 90% cumprem o fechamento: `gspo1` (93,2% com 8 ciclos; 90,9% com 4), `grpo1` (91,7%), `grpo3` (91,7%) e `grpo4` (90,6%). Não cumprem: o Lumi (86,2%), o `adalberto` (86,8%), o `gspo_thor` (87,8%) e o `gspo_gold` com 2.048 (89,4%). Os demais critérios (múltipla escolha acima do Lumi, queda sem thinking) não mudaram e nenhuma rodada cumpre todos.

### 20.47 `gspo_thor_e1`: o RL recupera 10,7 pontos do `thor_pt_e1`, mas termina abaixo das outras rodadas (08/11 22:42)

- RL de 4 ciclos a partir do `thor_pt_e1` (receita do `gspo_thor`: questões antigas, 2.048 tokens, T 0,6, top-p 0,95, `--ratio seq`, `--max-noclose 0.85`), 21:43 a 22:42 (a sessão `noite` foi trocada pela `noite3` na passagem do ciclo 2 para o 3). Validação limpa, com thinking: múltipla escolha **62,1%** (−7,5 contra o Lumi, IC −11,9 a −3,1 ▼), GSM8K 84,3% (+2,7 ▲), fecha `</think>` e responde 80,8%, trunca 19,2%, mediana 612 tokens, traço em português 99,4%.
- Contra o ponto de partida: +10,7 em múltipla escolha (51,4% → 62,1%), truncação de 36,6% para 19,2%. Contra o `gspo_thor` no ciclo 4 (65,7%, fecha 82,9%): −3,6 (IC −7,5 a +0,0). Contra o `gspo1_c4` (67,8%): −5,7. No ciclo 0, as duas rodadas começaram iguais na amostragem de treino (recompensa 0,629 contra 0,618; truncadas 23% contra 25%), então a diferença final vem dos ciclos seguintes.
- Leitura: o RL reduz os laços a partir de qualquer ponto de partida do Nemotron, mas o `thor_pt_e1` (mais dados, mesmo professor) não leva a um ponto final melhor; nenhuma das SFTs do Nemotron (`thor_pt`, `thor_pt_short`, `thor_pt_e1`) chegou ao nível do `gspo1`. O pacote 4 (Qwen) é o que ainda não foi testado.

### 20.48 Diagnóstico de log-probabilidades: o ruído numérico tem a escala do corte do GSPO (08/11 22:44)

- `scripts/diag_logprob.py`, sobre 96 respostas (76.619 tokens) do ciclo 0 do `gspo_thor_e1` (modelo fundido `thor_pt_e1` no vLLM; adaptador `runs/train/thor_pt_e1/adapter` no treinador nativo, BF16). Tempo: ~1 min por estágio. Relatório: `runs/diag/lp_report.json`.
- **Adaptador desligado = Lumi:** diferença exatamente zero nos 76.619 tokens (mesmos núcleos, mesma composição).
- **Ruído do corte, sem mudar nenhum peso** (mesma sequência em duas composições de batch: a da passada "antiga" e a de um minibatch de treino): |log s| por sequência, média 3,9e-4, mediana 2,3e-4, p90 1,1e-3. **46% das sequências passam de 3e-4 e 38% de 4e-4**, os eps do GSPO (3e-4 e 4e-4). O corte só zera o gradiente quando o ruído empurra a razão para o lado em que o corte age (A > 0 e s acima do limite, ou A < 0 e s abaixo), então ~metade disso, uns 20% das sequências, perde o gradiente por acaso no primeiro passo.
- **Cabeça FP32** (`cast_lm_head_to_fp32` do TRL; o Lumi tem embeddings não amarrados): média 2,3e-4 (−40%), p90 6,4e-4 (−43%); 31% das sequências acima de 3e-4 e 25% acima de 4e-4. Reduz, não elimina: o resto vem do backbone em BF16.
- **Descompasso vLLM x treinador** (temperatura 1): erro médio por token 1,1e-2; por sequência, |média| de 8e-4 (p90 1,8e-3), viés pequeno (−1,1e-4 por sequência). Com a cabeça FP32 fica igual (1,06e-2 e 7,6e-4): não vem da camada final, vem dos núcleos do backbone (vLLM contra HF). Não afeta o corte (a razão usa log-probabilidades antigas recalculadas no treinador), mas é o desvio que a correção TIS/MIS do TRL trataria.
- **Leitura:** nos primeiros passos o corte do GSPO é em parte ruído; depois do passo 3 o movimento real (1,4 a 3e-3 por sequência, já medido) passa do ruído e o corte vira o que se pretendia, uma região de confiança, só que muito estreita: ~metade das sequências tem gradiente zero a partir daí. As opções que o diagnóstico abre: eps maior (o corte atual é menor que o passo real do otimizador), cabeça FP32, ou calcular as log-probabilidades antigas na mesma composição de batch do treino. Nenhuma foi testada.
- **Limites:** uma amostra (96 respostas, um ciclo, um adaptador), temperatura 1 (não mede a divisão por 0,6 da amostragem).

### 20.49 SFT com a perda DFT: sem ganho e sem o português; fila v4 (08/11 23:22)

- `adalberto_dft`: a receita do `adalberto` (`data/train_adalberto.parquet`, lr 1e-4, 1 época, 450 passos) com `--loss dft` (cada token pesa a própria probabilidade, sem gradiente no peso). Validação limpa, com thinking: múltipla escolha **61,3%** (−8,3 contra o Lumi, IC −13,0 a −3,9 ▼; −1,0 contra o `adalberto`, IC −6,0 a +4,2), GSM8K 83,3%, fecha `</think>` 77,7% (86,8% no `adalberto`), trunca 7,8%. **Traço em português: 0,7% na múltipla escolha e 15,6% no GSM8K**, contra 99,4% e 99,7% do `adalberto`.
- Leitura: a perda DFT não reduz o dano e **não ensina o raciocínio em português**. A ponderação por p tira peso justamente dos tokens de baixa probabilidade, e os que viram o idioma do traço são desse tipo. A confusão com uma taxa de aprendizado efetiva menor (gradientes menores) não foi separada; uma taxa maior não foi testada. Sem thinking, a múltipla escolha ficou em 63,1% (o `adalberto`: 58,2%; o Lumi: 63,9%), coerente com um adaptador que mudou menos o modelo. H44 refutada nas condições testadas.
- Fila: a `noite3` foi trocada pela `noite4` (`scripts/run_noite4_chain.sh`) na passagem do SFT para o `gspo1_T06`. Mudanças: entram `gspo1_eps3x` (corte do GSPO 3× maior: `GRPO_EPS_LOW=1e-3`, `GRPO_EPS_HIGH=1.3e-3`) e `gspo1_same` (`GRPO_OLD_SAME_BATCHES=1`: log-probabilidades antigas nos micro-batches que o treino usa), e a suíte do Lumi vai para o fim de tudo, como preenchimento que sai da frente quando o pacote 4 da Thor chega (previsto para ~08:50). Implementação em `scripts/grpo_train.py` e `lumi_ft/train/grpo.py` (`minibatch_indices`), padrões inalterados, 374 testes de CPU passando.
- **Pacote 4 (previsto para ~08:50):** `scripts/run_qwen_mix.sh` (extrair e conferir, `prepare_thor4.py`, filtro de contaminação, `data/qwen_mix.parquet`, SFT `qwen_mix`, RL `gspo_qwen_mix`) está pronto; a `noite4` o chama quando `data/thor4/thor4_out.tar.gz` fica parado por mais de 60 s. Ensaiado com dados sintéticos nas partes de CPU (conversão, filtro, montagem; 375 testes passando) e com relatórios sintéticos na conferência do mínimo. Um erro achado no ensaio: a chave `idioma_dos_mantidos` existe só no relatório do pacote 3, não no do 4; a conferência de idioma virou informativa. A suíte do Lumi preenche a GPU de ~07:15 até o pacote chegar e sai da frente (retomável).

### 20.50 `gspo1_T06`: dividir os logits do treinador por 0,6 não ajudou (09/11 00:43)

- RL de 4 ciclos com a receita do `gspo1` (a partir do `adalberto`, questões antigas, 2.048, T 0,6, top-p 0,95, `--ratio seq`, semente 0) e os logits do treinador divididos por 0,6 (`GRPO_LOGIT_TEMP=0.6`), 23:24 a 00:43. Validação limpa, com thinking: múltipla escolha **63,6%** (−6,0 contra o Lumi, IC −10,4 a −1,6 ▼; −4,2 contra o `gspo1_c4`, IC −8,3 a +0,0), GSM8K 84,0% (−0,3), fecha `</think>` e responde 87,3% (90,9% no `gspo1_c4`), trunca 12,7%, mediana 1.027 tokens, traço em português 98,5%.
- Durante o treino, as duas rodadas ficaram quase iguais na recompensa de múltipla escolha (ciclos 0 a 2: 0,576, 0,594, 0,625 contra 0,593, 0,593, 0,625); a fração de sequências com corte ativo foi menor com a temperatura (37 a 42% contra ~44 a 50%).
- Leitura: H45 sem apoio. É um valor único com IC tocando o zero; a variação entre sementes (`gspo1_s1`, em andamento) decide se −4,2 é efeito ou ruído de execução.

### 20.51 `gspo1_s1`: a mesma receita varia 3 a 4 pontos entre sementes (09/11 02:11)

- RL de 4 ciclos com a receita do `gspo1` (a partir do `adalberto`, questões antigas, 2.048, T 0,6, top-p 0,95, `--ratio seq`) com a semente 1 (`GRPO_SEED=1`). `select_prompts` depende da semente, então a semente 1 também sorteia outros prompts (a variação medida inclui amostragem, ordem dos minibatches e prompts).
- Validação limpa, com thinking: múltipla escolha 64,4% contra 67,8% do `gspo1_c4` (−3,4; IC −7,5 a +0,8; contra o Lumi, −5,2, IC −9,9 a −0,8); GSM8K 82,5% contra 84,3% (−1,8; IC −3,7 a +0,1); fecha o `</think>` em 87,8% contra 90,9%; trunca 13,0% contra 10,6%; mediana 1.222 contra 1.086 tokens.
- Leitura: o ruído de uma rodada de RL é da ordem de 3 a 4 pontos em múltipla escolha e 3 no fechamento. Comparações de uma rodada contra outra só valem fora da faixa 64,4 a 67,8. O critério de fechamento de 90% é cumprido por uma semente da receita e não pela outra; o de 98% por nenhuma.
- Efeito nas leituras anteriores: `gspo1_T06` (63,6%) e `gspo_thor` (67,0%) ficam dentro da faixa; H45 sem efeito demonstrável; H46 refutada.

### 20.52 `qwen_mix_p`: 1.810 traços de MCQ do Qwen sobre o `adalberto` não melhoram o SFT (09/11 14:41)

- SFT com os dados do `adalberto` como estão (14.386 linhas, limite 1.024 tokens) mais 1.810 traços de múltipla escolha da entrega parcial do pacote 4 (1.141 MedMCQA, 669 WebInstruct; até 2.048 tokens; 1 removido pelo filtro de 8-gramas, 380 acima de 2.048); receita do `adalberto` (LoRA r=16, lr 1e-4, 1 época, 507 passos, 1.903 s). A queda da máquina (02:45 a 12:48) veio quando a avaliação começava; a avaliação foi refeita sobre o mesmo modelo mesclado (íntegro: mesmo tamanho do `adalberto`, 263 tensores).
- Validação limpa, com thinking: múltipla escolha 61,8% contra 62,3% do `adalberto` (−0,5; IC −4,7 a +3,6); fecha o `</think>` em 83,1% contra 86,8%; trunca 17,7% contra 13,8%; mediana de 1.044 contra 1.176 tokens; traço em português 100%. GSM8K 84,5% contra 83,3% (+1,2; IC −0,7 a +3,1). Sem thinking: 61,6% contra 58,2%.
- Todos os traços começam com "Vou raciocinar em português." (começo forçado da geração) e o modelo reproduz o começo (100% dos traços em português).
- Letras: previstas na validação A 23,2%, B 22,9%, C 21,3%, D 24,1%, E 2,2% (gabarito 22,3, 22,6, 22,1, 22,3, 2,1); a validação tem só 8 itens com gabarito E, então o efeito da letra E rara nos traços (2,4% no parcial, 3,0% no pacote completo) fica para as provas de 5 alternativas.
- Leitura: sem apoio para H52 com essa dose; a avaliação é de um SFT só (ruído de ±4 pontos). O `qwen_mix` com 5.055 traços e o RL a partir dele decidem.

### 20.53 `qwen_mix`: 5.055 traços de MCQ do Qwen dão o mesmo que 1.810 (09/11 15:39)

- SFT com os dados do `adalberto` como estão (14.386 linhas) mais 5.055 traços de múltipla escolha do pacote 4 completo (2.752 MedMCQA, 2.303 WebInstruct; até 2.048 tokens; 6.260 passaram do filtro de 8-gramas), 19.441 linhas e 13,5 milhões de tokens; receita do `adalberto`, 608 passos.
- Validação limpa, com thinking: múltipla escolha 62,1% contra 62,3% do `adalberto` (−0,3; IC −4,7 a +4,2) e 61,8% do `qwen_mix_p`; fecha o `</think>` em 83,9% contra 86,8%; trunca 16,9% contra 13,8%; mediana de 1.021 contra 1.176 tokens; GSM8K 83,4% (+0,1; IC −1,6 a +1,9). Sem thinking: 62,9% contra 58,2% do `adalberto`.
- Leitura: nenhuma dose de MCQ do Qwen (0, 1.810, 5.055 traços) muda a múltipla escolha com thinking no SFT (62,3%, 61,8%, 62,1%); o sem thinking sobe 3 a 5 pontos e o fechamento cai ~3 pontos. Letras previstas sem viés (E 1,9% contra 2,1% do gabarito).

### 20.54 `gspo_qwen_mix`: o RL a partir do `qwen_mix` fica na faixa do `gspo1` (09/11 16:59)

- RL de 4 ciclos com a receita do `gspo1` a partir do SFT `qwen_mix` (adalberto + 5.055 traços de MCQ do Qwen). Recompensa média de múltipla escolha por ciclo: 0,561, 0,633, 0,592, 0,607 (a do `gspo1_s1`: 0,598, 0,631, 0,679, 0,657); de matemática: 0,880, 0,918, 0,920, 0,932.
- Validação limpa, com thinking: múltipla escolha 64,4% (−3,4 contra o `gspo1_c4`, IC −7,5 a +0,8; −5,2 contra o Lumi, IC −9,9 a −0,8), igual ao `gspo1_s1`; fecha 87,8%; trunca 12,5%; mediana de 982 tokens; traço em português 98,8%; GSM8K 84,0% (−0,3 contra o `gspo1_c4`, +2,4 contra o Lumi, IC −0,1 a +5,0). Sem thinking: 65,7%.
- Leitura: dentro da faixa de 64,4 a 67,8% das duas sementes do `gspo1`. Os traços de MCQ do Qwen (0, 1.810 e 5.055) não mudam a múltipla escolha com thinking no SFT nem depois do RL.

### 20.55 Laços de repetição e levantamento de literatura (09/11 17:05)

- **Medição** (saídas já avaliadas, múltipla escolha com thinking, 385 itens; laço = bloco de ≥ 60 caracteres repetido em sequência ≥ 4 vezes, ou mais de 50% de 8-gramas repetidos nas últimas 1.500 palavras): das respostas que não fecham o `</think>`, têm repetição forte no fim 94% no Lumi (50 de 53), 86% no `adalberto` (44 de 51), 81% no `qwen_mix` (50 de 62), 60% no `gspo1` (21 de 35), 70% no `gspo1_s1` (33 de 47) 98% no `gspo_thor` (46 de 47) e 53% no `gspo_qwen_mix` (25 de 47).
- **Literatura** (`estudos/09`): forçar o idioma do raciocínio custa acerto (XReasoning, Bode Reasoning, M-Thinker no 4B a ~5,5 pontos do inglês); estudantes destilados entram em laço muito mais que os professores (arXiv 2512.12895); o Antidoom, da Liquid AI (blog, não revisado), reporta laço de 22,9% para 1% em um 4B com preferência no token que inicia o laço.
- **Datasets:** o Bode-Reasoning traz ENEM, BLUEX e OAB (as provas do teste) em 52% do conteúdo: não usar as partes de prova. O GigaVerbo-v2 SFT (Tucano 2) fica fora por constar no pré-treino do Lumi.
- **Sopas de pesos:** os adaptadores de RL da receita do `gspo1` ficam a ~2% da média (distância relativa) e a média de A e B separadas aproxima a média dos produtos B·A com erro máximo de 0,17% (mediana 0,07%).


### 20.56 Onde o `gspo1` ganha e perde contra o Lumi: decomposição pareada (09/11 17:40)
- **Método:** `scripts/decompose_paired.py` classifica cada item da validação com thinking (mesmos ids, decodificação gulosa, 4.096 tokens) pelo fechamento do `</think>` em cada modelo e soma a contribuição de cada grupo à diferença total.
- **Múltipla escolha (385 itens), `gspo1` de 8 ciclos contra o Lumi (−2,1 pontos; IC −6,8 a +2,3, empate estatístico):** os dois fecham, 320 itens (83,1%): Lumi 81,9% e `gspo1` 74,4%, **−6,2 pontos**; só o Lumi fecha, 12 itens: −1,6; só o `gspo1` fecha, 39 itens (10,1%): `gspo1` acerta 56,4%, **+5,7**; nenhum fecha, 14 itens (3,6%): 0.
- **GSM8K limpo (676 itens), +2,7 pontos:** os dois fecham, 633 itens: 86,4% contra 86,9% (+0,4); só o `gspo1` fecha, 31 itens: +3,0; só o Lumi fecha, 6 itens: −0,7. Todo o ganho vem de fechar onde o Lumi não fecha.
- **O custo nos itens em comum é estável:** −6,5 (`adalberto`), −6,2 (`gspo1`), −5,5 (`gspo1_c4`), −7,0 (`gspo1_soup3`). O RL e a média de pesos não o movem; o ganho do RL está todo nos itens que o Lumi não fecha. Isso corrige a leitura anterior de que a queda do acerto entre os que fecham (72 a 76% contra 80,7%) vinha só de o modelo fechar também itens difíceis: nos mesmos 320 itens a diferença continua em −7,5 pontos.
- **Por área (nos itens em que os dois fecham, `gspo1`; 50 a 109 itens por área, ruído de ±6 pontos):** STEM e saúde −8,3, sociais −16,0, outras −4,7, humanas −3,7. A queda é difusa, sem área única que a concentre.
- **Teto da fase anti-laço (H57):** os grupos que ela pode recuperar são os 12 itens que só o Lumi fecha (+1,6 se acertasse como o Lumi) e os 14 que nenhum fecha (até +2,0 se acertasse 56%); soma até ~+3,6 pontos (67,5% → ~71%) se **todo** item sem fechamento fosse corrigido; o detector de laço exato só pega parte deles (§20.58), então o teto da fase anti-laço é bem menor.
- **Limite da medida:** a validação de múltipla escolha (global-MMLU dev, 4 alternativas) é o único proxy sem tocar no teste. O Lumi tem 13,8% de respostas que não fecham nela e 27% no MMLU-ProX de teste (10 alternativas), então o ganho de fechamento pode pesar mais na suíte do que aqui; só a suíte final mede.

### 20.57 `base_ptprefill` reavaliado: o custo do idioma sozinho e o do treino (09/11 17:40)
- **Reprodução:** a reavaliação do Lumi com `--prefill "Vou raciocinar em português."` (noite10, 3 min) deu o mesmo que em 07/11: múltipla escolha 61,3% (−8,3 contra o Lumi, IC −12,7 a −4,2), fecha 79,7%, traço em português 74,9%; GSM8K limpo 47,5% (o prefill quebra o formato: em 261 dos 676 itens, 38,6%, o Lumi fecha e o Lumi forçado não).
- **Itens em comum (múltipla escolha, `scripts/decompose_paired.py` e bootstrap pareado):** só o Lumi forçado perde por não fechar (44 itens, −8,8 pontos) e ganha pouco onde o Lumi não fecha (+2,9). Nos 282 itens que o Lumi, o Lumi forçado e o `gspo1` fecham: 81,9%, 78,4% e 75,2%. Forçar o português custa 3,5 pontos (IC −7,1 a +0,4); o `gspo1` contra o Lumi forçado, −3,2 (IC −7,4 a +1,1); o `gspo1` contra o Lumi, −6,7 (IC −11,3 a −2,5). Ordem de grandeza: o idioma explica a metade do custo e o treino (SFT mais RL) a outra, mas a segunda metade não é significativa sozinha.
- **Leitura para H54:** se o `selfdist_min`/`selfdist_300` (traços do próprio Lumi em inglês, em pouca quantidade) recuperarem o acerto entre os que fecham, o custo do treino é de quantidade ou de formato e não de idioma; essa é a etapa da fila que separa as duas partes.

### 20.58 Referência da taxa de laço: o detector pega pouco do que não fecha (09/11 17:50)
- **Referência em `heldout` (`gspo1` de 8 ciclos, 750 prompts, gulosa, 3.072 tokens):** múltipla escolha, 100 de 500 não terminam (20,0%) e 8 têm laço exato (1,6%); matemática, 3 de 250 não terminam (1,2%) e nenhuma com laço. Detector: bloco de ≥ 40 caracteres repetido em sequência ≥ 4 vezes até o fim (`find_loop_start`).
- **Mesma conta na validação (`gspo1`, 385 itens, saídas já avaliadas, 4.096 tokens):** 26 respostas não fecham; o detector exato pega 4 (15%); com ≥ 3 repetições, também 4; 9 têm mais de 50% de 8-gramas repetidos nas últimas 1.500 palavras e 13 mais de 30%. As demais são raciocínio longo que vai e volta sem repetir (por exemplo, a dúvida entre "ciclos de MUL e ADD" retomada sem fim). A medição da §20.55 (60% com repetição forte para o `gspo1` do ciclo 4) inclui o critério de 8-gramas, mais largo que o do detector.
- **Consequência para a H57:** o ganho máximo em múltipla escolha, se todos os 4 itens com laço exato passassem a acertar como os demais (56%), é de ~+0,6 ponto (4 de 385). A regra de leitura escrita antes de ver a referência (não terminar em `heldout` cair à metade, 20% → 10%; fechar ≥ 95%, hoje 93,2%) pede corrigir mais respostas do que existem com laço exato (8 de 100 em `heldout`; 4 de 26 na validação): é inalcançável por construção. Emenda registrada em `estudos/07` (H57) e `estudos/08`, antes de qualquer resultado do tratamento.
- **O que sobra do não fechamento:** raciocínio longo e indeciso, que o Antidoom (preferência no token do laço) não trata. Os testes da fila que miram traços mais curtos são o `qwen_mix_short` (H53, MCQ ≤ 1.024 tokens) e o `selfdist_min`/`selfdist_300` (H54).

### 20.59 O laço exato explica quase todo o não fechamento na família da Thor (09/11 18:25)
- **Medição (validação, múltipla escolha com thinking, 385 itens, saídas já avaliadas, detector exato de `find_loop_start` com bloco ≥ 40 caracteres repetido ≥ 4 vezes até o fim):** respostas que não fecham e delas com laço exato: `thor_pt` 131 e 129; `thor_en` 95 e 82; `gspo_thor` 47 e 44; `gspo_thor_e1` 74 e 68; `gspo1` 26 e 4.
- **Teto grosseiro** (se todo item com laço passasse a acertar como o Lumi acerta nesses itens; o Lumi fecha só parte deles): `thor_pt` 54,3% → 71,7%; `gspo_thor` 67,0% → 70,6%; `gspo_thor_e1` 62,1% → 69,9%; `thor_en` 60,3% → 68,8%; `gspo1` 67,5% → 67,5%.
- **Decomposição pareada contra o Lumi (itens em que os dois fecham):** `thor_pt` 84,0% contra 84,5% (−0,3 ponto de contribuição); `thor_en` 81,9% contra 83,3% (−1,0); `gspo_thor` 77,6% contra 82,1% (−3,6); `gspo_thor_e1` 78,6% contra 82,8% (−3,1); `gspo1` 74,4% contra 81,9% (−6,2). A família da Thor perde pouco quando fecha; perde quando entra em laço (`thor_pt` −17,4 pontos nos 94 itens que só o Lumi fecha).
- **Leitura:** o Antidoom (preferência no token que inicia o laço) é a ferramenta certa onde o não fechamento é laço, isto é, na família da Thor, e quase inútil no `gspo1`. Hipótese H58 e regra de leitura em `estudos/07`; fila `noite11`. Limite: o teto mede o que os laços custam hoje, não o que a fase vai recuperar; com o ruído de uma execução (3 a 4 pontos) só um ganho acima de ~69% distingue o `gspo_thor_antidoom_r2` do `gspo1`.

### 20.60 A fase anti-laço é implementação própria; há repositório oficial (Apache-2.0) e as diferenças (09/11 18:50)
- **Origem:** a implementação (`src/lumi_ft/train/antidoom.py`, `antidoom_mine.py`, `antidoom_train.py`) foi escrita a partir da descrição do [blog da Liquid AI](https://www.liquid.ai/blog/antidoom). A existência do repositório [Liquid4All/antidoom](https://github.com/Liquid4All/antidoom) (Apache-2.0; geração, treino e fusão do LoRA) só foi confirmada depois; nenhum código dele foi usado, e a leitura do README (sem a fórmula exata da perda) é a base da comparação abaixo.
- **Igual ao descrito:** laço = bloco repetido ≥ 4 vezes até o fim; rejeitado = primeiro token da primeira repetição; escolhidos = até 20 alternativas do topo da referência com ≥ 2 caracteres alfanuméricos; perda em espaço de logit sem softmax, com o resto do vocabulário mais preso que escolhidos e rejeitado; parada antecipada em `chosen_win` (0,35 no blog; 0,4 sugerido no README; 0,15 a 0,3 já reduz muito o laço); taxa de aprendizado de 1e-5 a 2e-5; uma época.
- **Diferente na nossa versão:** (1) bloco mínimo de 40 caracteres (o blog usa 60); (2) a preferência é um softplus sem corte (no oficial a perda de preferência desliga quando o escolhido passa o rejeitado por uma margem em logit, `clip_epsilon_logits`); (3) o resto do vocabulário é preso por soma de Δ² ponderada pela probabilidade de referência (no oficial, MSE com peso `lambda_mse`); (4) **não há regularização de frequência dos tokens** (o oficial poda amostras para achatar a frequência do rejeitado, força 0,3, e dos escolhidos, 0,5, para que "Wait", "So", "the" não dominem); (5) **LoRA r=16 nos 71 módulos que já temos**, continuando o adaptador do RL (o oficial usa r=128 a 256 em todas as projeções de atenção e MLP mais `lm_head`); (6) **quantidade de eventos**: o oficial gera 15 a 20 mil pares a partir de pelo menos 15 mil prompts (usa até 12 mil no treino); nós teremos algumas centenas (`gspo_thor`: 5.000 prompts × 11 a 18% de laço).
- **Resultados do blog (dados do fabricante, tabelas só em imagem):** laço de 22,9% para 1% no Qwen3.5-4B (guloso) e de 10,2% para 1,4% no LFM2.5-2.6B; o acerto nas provas "sobe" sem números no texto; o desempenho com temperatura 1,0 cai depois do treino.
- **Consequência:** o teste da fila é uma versão leve do método (poucos eventos, adaptador pequeno, sem a regularização de frequência). Um resultado negativo dele **não refuta** o Antidoom oficial; um positivo mostra que a ideia funciona mesmo em escala pequena. Se os eventos do `gspo_thor` ficarem concentrados em poucos tokens rejeitados (a contagem por token rejeitado será feita nos eventos minerados, `events.jsonl`), a regularização de frequência passa a ser a primeira mudança; o caminho seguinte é o `thor_pt` (33% de laço, mais eventos por prompt) ou rodar o repositório oficial, o que exige copiá-lo para fora deste projeto (pedir permissão antes).

### 20.61 Leitura do código oficial do Antidoom e portagem da receita (09/11 19:15)
- **Fonte:** [Liquid4All/antidoom](https://github.com/Liquid4All/antidoom), commit `bd6a126` (07/07/2026), Apache-2.0, clonado em `../antidoom_oficial` (fora do projeto). Trechos portados em `src/lumi_ft/train/antidoom_oficial.py`, com licença e lista de mudanças em `third_party/antidoom/`.
- **O que a receita oficial faz (configuração padrão):** gera com temperatura 0,1, top-k 50, min-p 0,01 e até 4.000 tokens; o detector procura repetição em caracteres em **qualquer trecho** da resposta (≥ 4 repetições somando ≥ 60 caracteres, período até 1.024), terminada ou não; o rejeitado é o primeiro token legível da primeira repetição, com ajuste por período em tokens e por palavras de recomeço de frase; os escolhidos são as alternativas do topo renormalizadas na temperatura 0,1 com corte min-p 0,01 relativo à maior, o que deixa só as que têm pelo menos ~63% da probabilidade da melhor alternativa (em geral 1 a 3); a regularização de frequência dos rejeitados (0,3) só age acima de 12.000 linhas, e a dos escolhidos (0,5) só em chaves com mais de 50 ocorrências; o treino usa um LoRA novo r=256/alpha 128 com `lm_head`, lote efetivo de 16, lr 1,5e-5 com aquecimento de 10% e queda linear, decaimento de peso 0,01, corte de gradiente 2,5, uma época; a perda é softplus(2 − Δ) ponderada por clamp((2 − Δ)/2, 0, 1) para cada escolhido (Δ = logit do escolhido − logit do rejeitado; desliga quando passa a margem de 2 logits) mais 0,4 × MSE dos logits no resto do vocabulário e 0,05 × MSE com folga de 0,5 nos alvos; o `chosen_win` é a **fração** dos escolhidos acima do rejeitado, e o treino para quando a média de treino a cada 5 passos chega a 0,4.
- **O que a versão própria fazia diferente:** `chosen_win` pelo melhor escolhido (fácil de passar; a parada em 0,35 cortaria o treino cedo demais), escolhidos com ≥ 2 caracteres alfanuméricos sem corte min-p, perda pela média dos escolhidos sem margem, resto do vocabulário ponderado pela probabilidade, detector só no fim de respostas não terminadas e com bloco ≥ 40 caracteres, LoRA r=16 continuando o adaptador. A versão própria fica no código (`--recipe propria`), fora da fila.
- **Adaptações da portagem:** NemotronH em vez de transformador puro (o PEFT recusa LoRA no `out_proj` e no `conv1d` do Mamba: os alvos são `in_proj`, `up_proj`, `down_proj`, `q/k/v/o_proj` e `lm_head`); logits do último token em FP32 com a cabeça manual (base mais termo do LoRA); um evento por passada, sem preenchimento até 6.000 tokens; referência calculada uma vez antes do treino; escolhidos a partir das 20 maiores log-probabilidades do modelo de partida no ponto do rejeitado (no original, do vLLM na geração, do mesmo modelo); palavras de recomeço com equivalentes em português.
- **Braço de conferência:** `scripts/antidoom_oficial_train.py` usa o `FTPOTrainer`, o `FTPODataset`, o `sample_rows`, o `ThresholdStop` e o `select_chosen_tokens` do repositório oficial sem mudança de conta. Ajustes para o ambiente instalado (TRL 1.17, Transformers 5.22; o original travou o TRL 1.2 e o Transformers 5.6): `datasets.Dataset` como entrada, a checagem do TRL que recusa LoRA no `lm_head` contornada na criação do treinador (vale para a perda DPO, não para a do FTPO), fração de aquecimento em `warmup_steps`, lote 1 × 16 com checkpoint de gradiente, coleta sem preenchimento além do maior exemplo do lote, `adamw_torch` (sem bitsandbytes instalado).
- **Testes antes de usar a GPU:** 19 testes unitários, 8 deles comparando com as funções do repositório oficial (detector, rejeitado, escolhidos, amostragem, regularização e a perda contra a conta do `compute_loss`); treino nas duas formas e fusão (`scripts/merge_on_base.py`) com um NemotronH minúsculo na CPU (cabeça manual contra a passada completa: diferença máxima de 0,002 logit; a fusão muda só os 10 módulos-alvo do modelo minúsculo). Os testes acharam dois erros que teriam derrubado a fila: o tronco do NemotronH nativo se chama `model` (não `backbone`) e o `out_proj` do Mamba não aceita LoRA.
- **Mineração do `gspo1` (gulosa, 3.072 tokens, detector exato, 10.000 prompts, 67 min):** múltipla escolha 15,1% sem terminar e 1,7% com laço exato (135); matemática 1,2% e 0,05% (1). São os eventos do H57.

### 20.62 `qwen_mix_short`: traços de múltipla escolha curtos fazem o melhor SFT (09/11 19:40)
- **Resultado (validação limpa, thinking):** múltipla escolha 66,5% (Lumi 69,6%; −3,1, IC −7,5 a +1,3), fecha 89,1%, traço em português 99,1%, mediana de 884 tokens; GSM8K 83,7% (+2,1, IC −0,4 a +4,6), fecha 97,0%. Os SFTs anteriores ficaram em 62,1% a 62,3% (`adalberto`, `qwen_mix`): **+4,2 pontos** só por cortar os traços de múltipla escolha em 1.024 tokens (2.221 traços, 1.517 de MedMCQA e 704 de WebInstruct), com as mesmas receita e dose. Sem thinking (adaptador não usado): 62,3% em múltipla escolha, 83,7% em GSM8K.
- **Decomposição pareada contra o Lumi (múltipla escolha):** os dois fecham, 307 itens: 77,5% contra 82,4% (−3,9 pontos de contribuição; a queda por item é de 4,9, contra 7,5 do `gspo1`); só o Lumi fecha, 25 itens: −3,9; só o `qwen_mix_short` fecha, 36 itens (50% de acerto): +4,7; nenhum fecha: 17. GSM8K: os dois fecham, 628 itens, 87,3% contra 86,3% (+0,9).
- **Não fechamento:** 42 respostas; 17 têm laço exato (detector próprio) e 11 laço pelo detector oficial. No `gspo_thor`: 44 e 45 de 47. O RL, que paga o fechamento por recompensa, é a ferramenta natural aqui, ao contrário da família da Thor.
- **Leitura:** a regra do H53 (fecha ≥ 87% com múltipla escolha ≥ 62%) foi atendida e manda fazer o RL (H60, `noite13`). O ganho tem duas leituras que o RL separa: o curto reduz o dano do SFT (menos laço e menos traço longo para decorar) ou o MedMCQA/WebInstruct curtos trazem mais conteúdo útil. Uma execução só, com ruído de 3 a 4 pontos entre sementes: contra o `adalberto`, o ganho de múltipla escolha é de +4,2 pontos com IC pareado de −0,5 a +8,8 (no limite da significância) e o GSM8K limpo fica igual (+0,4, IC −1,5 a +2,4); o `qwen_mix` (mesmos dados, traços de até 2.048 tokens) fica igual ao `adalberto` (−0,3, IC −4,7 a +4,2).

### 20.63 RL a partir do `qwen_mix_short`: o ganho do RL já estava no SFT; cinco rotas no mesmo patamar (09/11 21:05)
- **Resultado (validação limpa, thinking):** `gspo_qwen_mix_short` (receita do `gspo1`, 4 ciclos, 75 min): múltipla escolha 67,0% (−2,6 contra o Lumi, IC −7,0 a +1,8), fecha 90,1%, traço em português 98,6%, mediana de 905 tokens; GSM8K limpo 83,1% (+1,5, IC −1,0 a +4,1). Contra o SFT de partida: +0,5 em múltipla escolha, +1,0 em fechamento, −0,6 em GSM8K. O `adalberto` ganhou +5,2 do RL e o `thor_pt` +12,7; o `qwen_mix_short` já partia perto do teto.
- **Decomposição pareada contra o Lumi (múltipla escolha):** os dois fecham, 307 itens: 82,1% contra 78,2% (−3,9 por item); só o Lumi fecha, 25 itens (Lumi acerta 64%): −4,2 pontos; só ele fecha, 40 itens (45% de acerto): +4,7; nenhum fecha, 13. GSM8K limpo: os dois fecham, 628 itens, 86,9% contra 86,3% (−0,6); +3,0 dos 29 itens que só ele fecha.
- **Rotas no mesmo patamar:** `qwen_mix_short` 66,5%, `gspo_thor` 67,0%, `gspo_qwen_mix_short` 67,0%, `gspo1` 67,5%, `gspo1_c4` 67,8% (fecham 89,1%, 87,8%, 90,1%, 93,2% e 90,9%), de professores e dados diferentes (Qwen3-30B, Nemotron 30B) e com ou sem RL. Todas dentro de −2 a −3 pontos do Lumi (69,6%), com IC contra o Lumi incluindo zero. Quem tem a maior perda por item nos itens em comum é o `gspo1` (−7,5); a menor, o `qwen_mix_short` e o seu RL (−3,9 a −4,9), mas elas fecham menos.
- **O que sobra:** nos 25 itens que o Lumi fecha (mediana de 712 tokens) e o `gspo_qwen_mix_short` não, 19 das 38 respostas que não fecham têm laço (qualquer detector); o teto grosseiro de uma fase anti-laço nele é 68,8% (`gspo_thor`: 70,6%; `thor_pt`: 71,7%). Esse é o argumento para a fila (Antidoom no `thor_pt` e no `gspo_thor`) continuar como estava.
- **Direção aberta:** o corte de 2.048 para 1.024 tokens melhorou o SFT em +4,2 pontos (H53); ir além (traços de múltipla escolha de até 512 tokens, ou penalidade de comprimento no RL) é a continuação natural, sem teste na fila. Quantidade de traços ≤ 512 tokens no pacote 4: a verificar.

### 20.64 O Antidoom oficial tira o laço e leva o `thor_pt` de 54,3% a 73,2% (10/11 02:30)
- **Mineração (receita oficial: temperatura 0,1, top-k 50, min-p 0,01, 4.000 tokens, detector oficial):** `thor_pt` em 10.000 prompts, 1,7 h: 40,3% dos 8.000 de múltipla escolha e 8,65% dos 2.000 de matemática têm laço oficial, 3.394 eventos; `gspo_thor` em 8.000 de múltipla escolha, 48 min: 13,1% com laço, 1.049 eventos; `gspo1` em 10.000 (detector exato, gulosa, 3.072 tokens): 1,7%, 136 eventos.
- **Treino (LoRA novo r=256/alpha 128 com `lm_head`, 307 milhões de parâmetros, lote 16, lr 1,5e-5, parada em `chosen_win` ≥ 0,4):** `thor_pt` 75 passos (4 min); treinador oficial 90 passos (6 min); `gspo_thor` 66 passos (5 min, o `chosen_win` chegou a 0,36); `gspo1` 9 passos (43 s, `chosen_win` ~1%, maior mudança de peso 6e-5: não treinou). Pico de 15,5 GiB. A maior mudança de peso na fusão em BF16 é de ~6e-4 nos outros três.
- **Resultado (validação limpa, thinking; contra o Lumi 69,6% de múltipla escolha e 81,7% de GSM8K):** `thor_pt` 54,3% / 77,8% → `thor_pt_antidoom` 73,2% (+3,6, IC −0,5 a +7,8) / 84,0% (+2,4) e `thor_pt_antidoom_of` 73,2% (+3,6, IC −0,8 a +8,1) / **85,8% (+4,1, IC +1,6 a +6,7)**; `gspo_thor` 67,0% / 82,5% → `gspo_thor_antidoom` 69,9% (+0,3) / **86,1% (+4,4, IC +1,9 a +7,0)**; `gspo1` 67,5% → `gspo1_antidoom` 65,7% (−3,9, no ruído). Fecha o `</think>` em 93,0%, 95,6%, 93,2% e 91,2%; traço em português 100%, 99,5%, 96,1% e 98,3%; mediana de 788, 720, 609 e 1.066 tokens (a do Lumi é 490).
- **Laço em `heldout` (detector oficial, múltipla escolha, 500 prompts, gulosa, 3.072 tokens):** `thor_pt` 48,6% → 2,2% (nossa portagem) e 0,4% (treinador oficial); não termina 53,0% → 18,8% e 15,2%; matemática 11,2% → 0%. `gspo_thor` 17,6% → 2,0%, não termina 24,0% → 13,8%. `gspo1` 1,6% → 1,4%.
- **Decomposição pareada contra o Lumi (múltipla escolha):** `thor_pt_antidoom_of`: os dois fecham, 325 itens, 79,4% contra 81,5% (−2,1 por item, −1,8 ponto de contribuição); só o Lumi fecha, 7 itens: −0,8; **só ele fecha, 43 itens (55,8% de acerto): +6,2**; nenhum fecha, 10. `thor_pt_antidoom`: 322 itens, 80,7% contra 81,7% (−1,0 por item); só ele fecha 36 itens (61,1%): +5,7. Para comparar com o RL, em conjuntos de tamanho parecido (~323 itens em comum): `gspo_thor_antidoom` perde 4,3 por item (77,4% contra 81,7%) e o `gspo_thor` 4,5. O Antidoom tira o laço sem o custo por item do RL; o RL mais Antidoom (`gspo_thor_antidoom`) herda o custo do RL.
- **Leitura:** o laço era o único dano real do SFT com os traços do Nemotron (H20/H33); a preferência no token que inicia o laço o remove com 4 a 6 minutos de treino, sem tocar no resto do vocabulário, e o raciocínio fica curto e em português. O patamar de 67% das cinco rotas anteriores era o fechamento obtido com RL, que cobra acerto por item; o Antidoom o obtém por outro caminho. Ressalva: o IC da múltipla escolha contra o Lumi ainda inclui zero (385 itens); a GSM8K limpa é significativa nos dois melhores. Só uma execução por braço, e a portagem e o treinador oficial dão o mesmo valor (H61), mas a variação entre sementes de treino está em teste (H65).
- **Suíte de teste:** a regra única (`scripts/pick_suite_candidate.py`: múltipla escolha ≥ 69,6%, fecha ≥ 93%, GSM8K limpo ≥ 83%, português ≥ 90%) escolheu o `thor_pt_antidoom_of`; o `thor_pt_antidoom` ficou a 0,01 ponto do limite de fechamento (92,99%). A suíte rodou a partir de 01:40 de 10/11.

### 20.65 Segunda rodada do Antidoom: o laço quase acabou, o que resta é raciocínio longo (10/11 02:40)
- **Sobra de não fechamento:** `thor_pt_antidoom_of`: 17 das 385 respostas de múltipla escolha da validação não fecham (4,4%); só 1 tem laço (detector exato ou oficial) e 1 repetição de 8-gramas acima de 30%; as outras são raciocínio longo e indeciso, sem repetição ("Na verdade, há uma nuance...", "No entanto, talvez..."), que chega a 4.096 tokens. `thor_pt_antidoom`: 27 não fecham, 3 com laço; `gspo_thor_antidoom`: 26 não fecham, 4 com laço.
- **Consequência para uma segunda mineração:** o laço oficial em `heldout` do `thor_pt_antidoom_of` é 0,4% (2 de 500; 0% na matemática), então 8.000 prompts devem dar ~30 eventos, contra 3.394 na primeira rodada; com 136 eventos o treino oficial (lote 16) já não saiu do lugar. Por isso a segunda rodada entra como **sonda de 3.000 prompts com limiar de 100 eventos** (H64), e não como mineração direta.
- **Outros pontos de partida com laço, para o Antidoom:** pelos mesmos números da validação, nenhum outro modelo existente combina muito laço e boa acurácia por item: `thor_pt_e1` perde 4,7 pontos por item nos 232 itens em comum (contra 0,5 do `thor_pt`, em 238), `thor_pt_short` 3,0, `gspo_thor_e1` 4,2; `adalberto_MLPonly` (69,6%, 36 laços em 47 respostas que não fecham) tem o traço de múltipla escolha em inglês (330 de 338), então não serve ao objetivo.
- **Sementes (H65):** como só há uma execução de cada braço, a fila refaz o treino da primeira rodada com duas sementes novas sobre os mesmos eventos.
