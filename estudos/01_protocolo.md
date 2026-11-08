# 01. Protocolo experimental

Vale para todos os estudos. Mudar algo aqui invalida a comparação com o que já foi rodado.

## 1. Modelos

| | Descrição |
| --- | --- |
| **Lumi-4B v0.2** (`ceia-br/lumi-4b`, commit `27b30f40…`) | Modelo híbrido NemotronH (Mamba2, MLP e atenção), 42 camadas, 4 B de parâmetros. `eos_token_id=[2, 11]`, `<think>`=12, `</think>`=13, `<|im_end|>`=11. Vem do Nemotron 3 Nano 4B com 50 B de tokens de pré-treino contínuo e SFT em português. O cartão diz que o thinking fica desligado nesta versão e que não houve RL |
| **Nemotron 3 Nano 4B** | Referência. Template de chat, prompts renderizados e vocabulário **idênticos** aos do Lumi (conferido). Raciocina em inglês |
| **Professores** (geração de dados, na Thor) | Qwen3-30B-A3B-Thinking, Nemotron 3 Nano 30B-A3B |

Regra de dados: não treinamos com datasets que o cartão do Lumi lista no SFT dele; cada fonte é conferida contra essa lista.

## 2. Máquinas e software

- **RTX 4090** (24 GiB, compartilhada, sem sudo nem Docker): todo treino e toda avaliação. **Jetson AGX Thor** (122 GiB de memória unificada, aarch64): só gera dados.
- Três ambientes: `.venv` (treino, Transformers com a implementação **nativa** do NemotronH), `.venv-vllm` (avaliação e geração, vLLM 0.33.0 com `VLLM_USE_FLASHINFER_SAMPLER=0`) e `.micromamba` (nvcc). `env.sh` mantém todos os caches dentro do projeto.
- Tarefas longas rodam desacopladas (`scripts/detach.sh [--gpu] <nome> <comando>`, tmux), com registro em `runs/jobs/<nome>.log` e código de saída em `<nome>.exit`.

## 3. Treino (SFT)

Laço próprio (`src/lumi_ft/train/sft.py`), sem TRL.

| Item | Valor |
| --- | --- |
| Adaptador | LoRA r=16, alpha=32, sem dropout, em 71 módulos: `in_proj` dos 21 blocos Mamba, `up_proj` e `down_proj` dos 17 MLPs, `q/k/v/o_proj` das 4 atenções. Adaptador em FP32; autocast BF16 |
| Fora do adaptador | `out_proj` do Mamba (o kernel fundido ignora o módulo), `conv1d`, parâmetros do SSM, embeddings, `lm_head`, normalizações |
| Otimização | AdamW (β 0,9 e 0,999, sem weight decay), lr 1e-4 com 3% de aquecimento e cosseno, clip 1,0, batch efetivo de 32 sequências, 1 época |
| Memória | sem gradient checkpointing; micro-batches por orçamento de tokens (2.048 no LoRA, 1.024 no DoRA); pico de 19 GiB no LoRA e 16 GiB no DoRA |
| Perda | só nos tokens da resposta (do fim do prompt ao último `<|im_end|>`); soma dividida pelo total de tokens-alvo do batch efetivo (testado: a acumulação dá o mesmo gradiente de um batch único) |
| Velocidade | LoRA ~5,5 mil tokens/s; DoRA ~2,1 mil tokens/s |

Prompt de treino = prompt de avaliação (instrução `#### N` ou `Resposta: <letra>`). No modo com thinking o exemplo é `prompt + <think>\ntraço\n</think>\nresposta<|im_end|>`; sem thinking é `prompt + <think></think>resposta<|im_end|>`.

## 4. Avaliação

- vLLM, decodificação **gulosa**, até **4.096** tokens novos, parada em `[2, 11]`.
- A resposta é lida **só depois do primeiro `</think>`**. Nota **estrita**: o primeiro `#### N` (GSM8K) ou a última linha `Resposta: <letra>` (múltipla escolha). A flexível (qualquer número ou letra) é registrada, não decide.
- **Forma da saída**: fração que fecha um único `</think>` e escreve a resposta; fração dos traços em português (heurística de palavras funcionais, `lumi_ft/eval/shape.py`, não é um detector de idioma); fração truncada; mediana de tokens.
- **Estatística:** bootstrap pareado por item (10.000 reamostras, IC 95%) da diferença contra o Lumi no mesmo modo (`scripts/compare_runs.py`, `scripts/report_compare.py`).
- **Ruído:** a saída do vLLM não é reprodutível bit a bit (batching contínuo): a nota varia ±0,15 a 1,2 ponto entre execuções do mesmo comando, fora do IC. Cada rodada é **uma execução com uma semente**.

## 5. Conjuntos de dados de avaliação

| Conjunto | Tamanho | Uso |
| --- | --- | --- |
| **GSM8K validação limpa** | 676 itens | Seleção de candidatos. Os 1.000 itens originais, menos os que repetem um problema do treino em **outra tradução** (assinatura de números + sobreposição de palavras ≥ 0,3) |
| **Múltipla escolha validação** | 385 itens | Seleção. Global-MMLU `dev` (285), MMLU-ProX `validation` (70), INCLUDE `validation` (30) |
| **Suíte de teste** (nota do PRD) | GSM8K-PT 1.295; MMLU-ProX PT 11.759; ENEM 1.431; BLUEX 722; OAB 2.210 | Só o resultado final. **Candidatos ainda não rodados** |
| **Diagnóstico português x inglês** | 1.200 itens do teste do MMLU-ProX, nos dois idiomas | Só para o Lumi original (não entra em seleção) |

**Contaminação:** índice de 8-gramas (46.440 itens das provas, 2,96 milhões de 8-gramas) aplicado a todo dado de treino, mais o filtro por assinatura de números para duplicatas em outra tradução (que achou ~325 de 1.000 itens da validação do GSM8K duplicados no treino de matemática do GigaVerbo; o **teste está limpo**, 0 casos). Faltam MMLU PT e IFEval PT na suíte do cartão (ids desconhecidos).

## 6. Testes automatizados

- **286 testes de CPU** (`PYTHONPATH=src python -m pytest tests --ignore tests/test_phase0_model.py`), cobrindo: extração de resposta, carregadores das provas, montagem dos dados, laço de treino (retomada, perda), fusão do adaptador (escala, camadas), GRPO (recompensas, vantagens, entropia, perda GSPO, KL), tradução, pacotes da Thor, conjuntos de alvos do adaptador.
- `tests/test_phase0_model.py` (modelo na GPU: equivalência entre implementações, gradientes, fusão) só roda com a GPU livre.
- Não testados de ponta a ponta antes de usar: `grpo_rollout.py`, `grpo_train.py` e `run_grpo.sh` (o teste de fumaça do RL cobriu a cadeia), e a tradução na Thor com o servidor real.

## 7. Regras de conduta

- **Nunca** mudar métricas, metas ou limiares para um candidato passar (PRD, regra 4).
- Sem publicação no Hugging Face sem pedido explícito.
- Dados de licença desconhecida (o dataset do adalberto) servem só a experimentos.

## 8. Limites que valem para todo o conjunto

- Uma execução por configuração; sem estimativa de variância entre sementes.
- A validação de múltipla escolha tem 385 itens: IC de ±4 a 5 pontos.
- A seleção entre muitas variantes na mesma validação tem risco de "maldição do vencedor"; por isso só os finalistas devem ir à suíte de teste.
- Os traços "em português" são identificados por heurística.
