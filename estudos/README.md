# Estudos de ablação: raciocínio em português no Lumi-4B

Estado em 08/11/2026. Este diretório organiza o projeto como uma série de estudos controlados: para cada pergunta há hipótese, desenho, resultado, tempo gasto, limites e veredito. O que mudou o plano e o que não mudou está no `PRD.md`; os detalhes de engenharia e as decisões do dia a dia estão no `discovery.md`.

## A pergunta

Ajustar o `ceia-br/lumi-4b` v0.2 (commit `27b30f40…`) com LoRA ou DoRA para que, com `enable_thinking=True`, ele **raciocine em português e acerte mais**, sem piorar o modo sem thinking. O Lumi original raciocina em inglês, e o cartão dele diz que o thinking deve ficar desligado nesta versão.

**Critério de vitória:** contra o Lumi, em cada modo, o intervalo de confiança da diferença fica acima de 0; pelo menos 90% das respostas fecham `</think>` e respondem (era 98% até 08/11; emenda no PRD, feita depois de conhecidos os resultados); pelo menos 90% dos traços saem em português; o modo sem thinking não cai mais de 1 ponto. A nota é a média de 5 provas: GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX e OAB (4 das 5 são de múltipla escolha).

## Onde estamos

> **Atualização de 10/11:** o melhor candidato passou a ser o `thor_pt_antidoom_of` (múltipla escolha com thinking 73,2% na validação) e a suíte de teste dele terminou (Nota de raciocínio 69,9% contra 63,2% do Lumi com thinking e 65,5% sem, na mesma máquina); detalhes em [../discovery.md](../discovery.md) (20.66 a 20.68) e hipóteses em [07_hipoteses_e_proximos_passos.md](07_hipoteses_e_proximos_passos.md). O texto abaixo é o de 08/11.

Melhor candidato até agora: **`gspo1`** (SFT `adalberto` + 8 ciclos de RL com razão por sequência). Validação limpa, nota estrita, diferença contra o Lumi:

| | Lumi | `gspo1` | Critério |
| --- | --- | --- | --- |
| GSM8K com thinking | 81,7% | 84,3% (+2,7; IC +0,1 a +5,2) | IC acima de 0: **cumprido** (por pouco) |
| Múltipla escolha com thinking | 69,6% | 67,5% (−2,1; IC −6,8 a +2,3) | empate: **não cumprido** |
| Múltipla escolha sem thinking | 63,9% | 57,7% (−6,2; IC −11,4 a −1,3) | queda máxima de 1 ponto: **violado** |
| Fecha `</think>` (múlt. escolha) | 86,2% | 93,2% | ≥ 90%: **cumprido** (com o limiar original de 98%: não cumprido) |
| Traço em português | 0% | 96,1% | ≥ 90%: cumprido |

Nenhum candidato passou o critério, e **nenhum candidato foi rodado ainda na suíte de teste** (do Lumi há o GSM8K e o MMLU-ProX PT com thinking; do Nemotron 4B, o GSM8K). Todos os números abaixo são da validação.

**O que mudou na tarde e na noite de 08/11** (decisões e o que estudar se der errado: [08](08_registro_de_decisoes.md)):

- **O adaptador só liga com thinking.** Sem thinking roda o Lumi original; a queda sem thinking da tabela acima deixa de pesar no uso. O BF16 fundido sozinho degrada o modo sem thinking, e o relatório final precisa dizer isso.
- **RL com limite de 4.096 tokens e questões de gabarito oficial** (`gspo_gold4k`): no ciclo 4, 63,1% em múltipla escolha com thinking e fecha o `</think>` em 83,9%, contra 67,8% e 90,9% do `gspo1` no mesmo ciclo. A regra combinada antes parou a rodada. Duas coisas mudaram juntas (limite e questões). O `gspo_gold` com 2.048 e as mesmas questões, no ciclo 4, deu 63,9% e fecha 89,4%: o limite não muda a nota (−0,8 contra o 4.096, IC −4,7 a +3,1), mas com 4.096 o fechamento cai 5,5 pontos. Contra o `gspo1` no mesmo limite, as questões do gold ficam −3,9 (IC −7,8 a +0,0). A regra fixada antes ([08](08_registro_de_decisoes.md)) cai no ramo B (as questões não ajudam), por 0,1 ponto abaixo do limiar de 64%; com uma semente e IC que toca o zero, é indício, não prova. O `gspo1` com 4.096 e as questões antigas roda agora.
- **Qualidade do raciocínio, nos itens em que todos os modelos fecham o `</think>`:** o SFT só de matemática (`adalberto`) fica ~5 pontos abaixo do Lumi (82,5% contra 87,8%, 189 itens); o `thor_pt`, com ciência em português, iguala o Lumi (87,3%). Condicionar no fechamento escolhe itens mais fáceis, então isso é indício, não prova ([discovery.md](../discovery.md) §20.39).
- **Mais dados do Nemotron não ajudaram** (entrega 1 do pacote 3, 2,5× mais múltipla escolha): o SFT `thor_pt_e1` ficou em 51,4% (fecha 63,4%) e o RL a partir dele em 62,1% (fecha 80,8%), abaixo do `gspo_thor` (65,7%) e do `gspo1` (67,8%). O RL recupera +10,7 pontos sobre o SFT, mas a família Nemotron não passa o `gspo1`.
- **Próximo dado a chegar:** traços de múltipla escolha em português gerados pelo Qwen na Thor (pacote 4, [handoff/README.md](../handoff/README.md)), para um SFT misto.

## Achados principais

1. **O português se aprende, a múltipla escolha se perde.** Qualquer SFT com traços de professor põe o raciocínio em português (97 a 100% dos traços) e acerta a matemática, mas a múltipla escolha com thinking cai de 5 a 12 pontos ([02](02_sft_dados_e_professor.md)).
2. **Dos fatores testados no SFT, nenhum recuperou a múltipla escolha:** limite de tokens, taxa de aprendizado, DoRA, tamanho do efeito do adaptador, tipo de bloco (só MLP). O professor importa: a autodestilação do Lumi foi o pior caso.
3. **O idioma da pergunta não é o problema.** O Lumi original acerta o mesmo em português e em inglês (diferença −1,3 a +0,4, sem significância) ([05](05_diagnosticos_e_referencias.md)).
4. **O lugar do adaptador não explica o dano.** A triagem pós-treino sugeria que só o Mamba e a atenção causavam a perda (o "só MLP" ficava em 69,6%), mas **treinado do zero** o adaptador só com MLP (34 módulos, metade dos parâmetros) aprende português (98,5%), ganha na matemática (83,4%) e perde igual na múltipla escolha (62,3% ▼; sem thinking 54,3% ▼). A triagem enganou. Cinco mudanças de adaptador ou de otimização deram 62 a 64%; a causa provável são os dados ([03](03_ablacao_camadas_e_mlp.md)).
5. **O RL recupera quase tudo na múltipla escolha com thinking** (62,3% → 66 a 68%, empate com o Lumi), mas só o GSPO ficou estável; os outros estouraram a entropia ou perderam forma. O RL treina só o modo com thinking e **não protege o sem thinking**: em duas rodadas ele caiu a 43,1% ([04](04_rl_grpo.md)).

6. **Só matemática no SFT prejudica o raciocínio de múltipla escolha; ciência em português não** (indício, §20.39 do `discovery.md`). O que impede o `thor_pt` de ganhar é o fechamento do `</think>` (cauda longa e laços), não a qualidade do raciocínio quando fecha.

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
| [09_literatura_datasets_e_taticas.md](09_literatura_datasets_e_taticas.md) | Literatura sobre o custo de pensar em português e sobre laços de repetição, datasets avaliados (e os que não podem ser usados) e táticas propostas |
| [08_registro_de_decisoes.md](08_registro_de_decisoes.md) | Decisões de rumo de 08/11 (RL com 4.096, adaptador só com thinking, regras do ciclo 4), com o que estudar se der errado |
| [resultados_auto.md](resultados_auto.md) | Tabela de todas as rodadas, **gerada** dos arquivos de avaliação |

Fora desta pasta: [../PRD.md](../PRD.md) (requisitos e critério), [../discovery.md](../discovery.md) (diário de engenharia, com índice no topo), [../scripts/README.md](../scripts/README.md) (o que cada script faz), [../handoff/README.md](../handoff/README.md) (pacotes da Thor) e [../README.md](../README.md) (resumo em inglês).

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
