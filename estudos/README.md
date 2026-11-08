# Estudos de ablação: raciocínio em português no Lumi-4B

Estado em 08/11/2026. Este diretório organiza o projeto como uma série de estudos controlados: para cada pergunta há hipótese, desenho, resultado, tempo gasto, limites e veredito. O que mudou o plano e o que não mudou está no `PRD.md`; os detalhes de engenharia e as decisões do dia a dia estão no `discovery.md`.

## A pergunta

Ajustar o `ceia-br/lumi-4b` v0.2 (commit `27b30f40…`) com LoRA ou DoRA para que, com `enable_thinking=True`, ele **raciocine em português e acerte mais**, sem piorar o modo sem thinking. O Lumi original raciocina em inglês, e o cartão dele diz que o thinking deve ficar desligado nesta versão.

**Critério de vitória:** contra o Lumi, em cada modo, o intervalo de confiança da diferença fica acima de 0; pelo menos 98% das respostas fecham `</think>` e respondem; pelo menos 90% dos traços saem em português; o modo sem thinking não cai mais de 1 ponto. A nota é a média de 5 provas: GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX e OAB (4 das 5 são de múltipla escolha).

## Onde estamos

Melhor candidato até agora: **`gspo1`** (SFT `adalberto` + 8 ciclos de RL com razão por sequência). Validação limpa, nota estrita, diferença contra o Lumi:

| | Lumi | `gspo1` | Critério |
| --- | --- | --- | --- |
| GSM8K com thinking | 81,7% | 84,3% (+2,7; IC +0,1 a +5,2) | IC acima de 0: **cumprido** (por pouco) |
| Múltipla escolha com thinking | 69,6% | 67,5% (−2,1; IC −6,8 a +2,3) | empate: **não cumprido** |
| Múltipla escolha sem thinking | 63,9% | 57,7% (−6,2; IC −11,4 a −1,3) | queda máxima de 1 ponto: **violado** |
| Fecha `</think>` (múlt. escolha) | 86,2% | 93,2% | ≥ 98%: **não cumprido** |
| Traço em português | 0% | 96,1% | ≥ 90%: cumprido |

Nenhum candidato passou o critério, e **nenhum candidato foi rodado ainda na suíte de teste** (só o Lumi e o Nemotron 4B têm GSM8K de teste). Todos os números abaixo são da validação.

## Achados principais

1. **O português se aprende, a múltipla escolha se perde.** Qualquer SFT com traços de professor põe o raciocínio em português (97 a 100% dos traços) e acerta a matemática, mas a múltipla escolha com thinking cai de 5 a 12 pontos ([02](02_sft_dados_e_professor.md)).
2. **Dos fatores testados no SFT, nenhum recuperou a múltipla escolha:** limite de tokens, taxa de aprendizado, DoRA, tamanho do efeito do adaptador, tipo de bloco (só MLP). O professor importa: a autodestilação do Lumi foi o pior caso.
3. **O idioma da pergunta não é o problema.** O Lumi original acerta o mesmo em português e em inglês (diferença −1,3 a +0,4, sem significância) ([05](05_diagnosticos_e_referencias.md)).
4. **O lugar do adaptador não explica o dano.** A triagem pós-treino sugeria que só o Mamba e a atenção causavam a perda (o "só MLP" ficava em 69,6%), mas **treinado do zero** o adaptador só com MLP (34 módulos, metade dos parâmetros) aprende português (98,5%), ganha na matemática (83,4%) e perde igual na múltipla escolha (62,3% ▼; sem thinking 54,3% ▼). A triagem enganou. Cinco mudanças de adaptador ou de otimização deram 62 a 64%; a causa provável são os dados ([03](03_ablacao_camadas_e_mlp.md)).
5. **O RL recupera quase tudo na múltipla escolha com thinking** (62,3% → 66 a 68%, empate com o Lumi), mas só o GSPO ficou estável; os outros estouraram a entropia ou perderam forma. O RL treina só o modo com thinking e **não protege o sem thinking**: em duas rodadas ele caiu a 43,1% ([04](04_rl_grpo.md)).

## Mapa dos documentos

| Arquivo | Conteúdo |
| --- | --- |
| [01_protocolo.md](01_protocolo.md) | Modelos, máquinas, treino, avaliação, conjuntos, estatística, testes, limites |
| [02_sft_dados_e_professor.md](02_sft_dados_e_professor.md) | Estudos de SFT: dados, professor, taxa, limite de tokens, DoRA, escala, tradução |
| [03_ablacao_camadas_e_mlp.md](03_ablacao_camadas_e_mlp.md) | Triagem por camadas e por tipo de bloco; treino só com MLP (que refutou a triagem) |
| [04_rl_grpo.md](04_rl_grpo.md) | RL: GRPO e variantes, entropia, estabilidade, ferramentas |
| [05_diagnosticos_e_referencias.md](05_diagnosticos_e_referencias.md) | Nemotron 4B, português x inglês, contaminação, dados e pacotes da Thor |
| [06_registro_experimentos.md](06_registro_experimentos.md) | Toda rodada: configuração, tempo de treino e de avaliação, pasta, estado |
| [07_hipoteses_e_proximos_passos.md](07_hipoteses_e_proximos_passos.md) | Registro de hipóteses com veredito, ameaças à validade, próximos experimentos |
| [08_registro_de_decisoes.md](08_registro_de_decisoes.md) | Decisões de rumo de 08/11 (RL com 4.096, adaptador só com thinking, regras do ciclo 4), com o que estudar se der errado |
| [resultados_auto.md](resultados_auto.md) | Tabela de todas as rodadas, **gerada** dos arquivos de avaliação |

## Como reproduzir os números

```bash
source env.sh
PYTHONPATH=src python scripts/gen_estudos_resultados.py          # regera resultados_auto.md
python scripts/report_compare.py --ref Lumi --col "Lumi=base:thinking" --col "gspo1=gspo1:thinking" \
    --tasks gsm8k_val,mcq_val --only-ids data/val_gsm8k_clean_ids.json   # comparação com IC 95%
PYTHONPATH=src python scripts/grpo_status.py gspo1               # resumo por ciclo de uma rodada de RL
```

Os dados brutos ficam em `runs/` (fora do Git): `runs/train/<nome>`, `runs/merged/<nome>`, `runs/eval/<nome>/<prova>_<modo>.jsonl` e `_scores.json`, `runs/grpo/<nome>/itNNN`. Os registros das tarefas longas ficam em `runs/jobs/<nome>.log`.

## Convenções

- **Nota** é a estrita: `#### N` no GSM8K e `Resposta: <letra>` na última linha na múltipla escolha, lidos só depois do primeiro `</think>`. A nota flexível é registrada, não decide.
- **▲ e ▼** nas tabelas: o IC 95% da diferença contra o Lumi, por bootstrap pareado, fica acima ou abaixo de 0. Sem marca é empate estatístico. **Diferenças menores que uns 3 a 4 pontos na múltipla escolha são ruído** (385 itens).
- **"Ciclo"** é uma iteração do RL (gerar 256 × 8 respostas, treinar); **"passo"** é um passo do otimizador.
