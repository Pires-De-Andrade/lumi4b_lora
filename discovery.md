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

## 9. Pendências da fase 0

1. Testes de extração de resposta (A-05 e A-06).
2. Pico de memória, maior batch viável e tokens por segundo por braço (A-14), e o maior batch de geração com 4.096 tokens novos.
3. Baseline do Lumi nos dois modos (precisa de aprovação, por passar de 1 hora de GPU).
4. Verificar se o modelo fundido salvo pela nativa mantém o formato de chaves esperado por quem carrega o Lumi com o código do repositório (`backbone.` contra `model.`).
5. Isolar a causa do erro de padding à esquerda no código do repositório, se for útil. Não é necessário para seguir com a nativa.

## 10. Como reproduzir

```bash
source env.sh
scripts/build_kernels.sh                       # compila os kernels (não foi executado de ponta a ponta; o build real seguiu os mesmos passos à mão)
LUMI_IMPL=native python -m pytest tests/       # 13 testes
PYTHONPATH=src python scripts/compare_impls.py # remote x nativa, geração
PYTHONPATH=src python scripts/diag_impls.py    # sensibilidade ao padding
```

`LUMI_IMPL=remote` roda a suíte no código do repositório; nesse caso o teste de geração falha, como descrito na seção 3.
