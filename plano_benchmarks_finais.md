# Plano: provas novas e fechamento da primeira etapa

10/11/2026 (terça) · apresentação: quarta, 11/11, à noite · candidato: `thor_pt_antidoom_of`

## Resumo

A suíte oficial já está completa: Nota 69,9% contra 63,2% (Lumi com thinking) e 65,5% (Lumi sem thinking), na 4090. Faltam três coisas para apresentar bem:

1. **Provas que nenhum modelo nosso viu durante a seleção**, para mostrar que o ganho não é sorte da suíte: OAB 2018 a 2025, ENEM 2024 e um recorte do MMLU-PT do CEIA (ENADE, concursos, olimpíadas).
2. **O Nemotron 3 Nano 4B nas mesmas provas**, porque hoje ele só tem a validação e o GSM8K de teste.
3. **As decisões que dependem de pessoa** (limiar de fechamento, revisão dos traços) e os slides.

São cerca de 7,5 h de 4090, em fila, de hoje à noite até a madrugada. **Os experimentos acabam na quarta ao meio-dia**; o que não tiver terminado vira "próximo passo".

## Regras, escritas antes de rodar

- Estas provas são **checagem externa**. Não trocam o candidato, não mudam o critério do PRD e não entram na Nota, que continua sendo a média das cinco provas.
- **Tudo o que rodar entra na apresentação**, favorável ou não.
- Mesmo protocolo da suíte: vLLM, decodificação gulosa, 4.096 tokens, nota estrita, bootstrap pareado contra o Lumi da 4090.
- Em toda prova nova, reportar também o fechamento do `</think>` e a decomposição pareada (`scripts/decompose_paired.py`), porque o ganho da suíte veio todo do fechamento.
- **Previsões** (para poder errar):

| Prova | Contra o Lumi com thinking | Contra o Lumi sem thinking |
| --- | --- | --- |
| OAB 2018–2025 | +15 a +25 | −2 a +3 (empate, como no teste) |
| ENEM 2024 (180 itens, IC de ±6) | 0 a +8; vale só a direção | 0 a +8 |
| MMLU-PT, três fatias | positivo e menor que na OAB | sem previsão |

- **Leitura:** o resultado "generaliza" se, na média das provas novas, o IC da diferença contra o Lumi com thinking ficar acima de 0. Se ficar acima de 0 também contra o Lumi sem thinking, é o resultado forte. Se o IC incluir 0, dizemos isso.

## Colunas de toda tabela

| Coluna | Modelo | Modo |
| --- | --- | --- |
| Nosso | `thor_pt_antidoom_of` | com thinking (o adaptador só liga nesse modo) |
| Lumi | `base` | com thinking e sem thinking |
| Nemotron | `nemotron4b` | com thinking e sem thinking |

## Passo 1: carregadores (terça à tarde, ~2 h, sem GPU)

Três tarefas novas em `src/lumi_ft/eval/mcq.py` (`LOADERS`), com teste de CPU para cada uma.

| Tarefa | Fonte | Itens | Como montar |
| --- | --- | --- | --- |
| `oab_novas` | [`russ7/oab_exams_2011_2025_combined`](https://huggingface.co/datasets/russ7/oab_exams_2011_2025_combined) (MIT) | **1.146** | Mesmo esquema do `eduagarcia/oab_exams`, mas `answerKey` é índice (`'0'` a `'3'`). Manter só `exam_id` que não está no teste e tirar as 39 questões de texto igual ao teste. Sobram 16 edições, de 2018-01 a 2025-01 |
| `enem_2024` | [`maritaca-ai/enem`](https://huggingface.co/datasets/maritaca-ai/enem), config `2024` (Apache-2.0) | 180 | Colunas `question`, `alternatives`, `label`, `description`, `figures`. As imagens vêm descritas em `description`; conferir no cartão como juntar ao enunciado. O teste atual não tem 2024 |
| `mmlu_pt_exatas`, `mmlu_pt_enade`, `mmlu_pt_concursos` | [`Portuguese-MMLU/mmlu-pt-knowledge-annotated`](https://huggingface.co/datasets/Portuguese-MMLU/mmlu-pt-knowledge-annotated) (licença não declarada; 35 MB) | **1.000 cada** (sorteio com semente 0) | Filtrar pela coluna `exam`: exatas = OBI, POSCOMP, IME, AFA (5.638); ENADE (10.785); concursos = BNDES, BACEN, CNU (5.717). `answer` é índice; `choices` tem 4 ou 5 alternativas |

Cuidados ao montar:

- **Não usar o MMLU-PT inteiro.** Ele contém 10.251 questões da OAB e 2.687 do ENEM; 2.429 e 1.434 têm o mesmo texto do nosso teste. Por isso só as três fatias acima.
- **Descontaminar nos dois sentidos**, com o índice de 8-gramas que já existe (`lumi_ft/data/decontaminate.py`):
  1. tirar das provas novas o que repete a suíte de teste ou a validação;
  2. conferir as provas novas contra os dados de treino do candidato (`thor_pt`, eventos do Antidoom, prompts do RL) e **tirar da prova** o que bater. Registrar quantos itens saíram.
- **Imagens:** tirar das fatias do MMLU-PT as questões que dependem de figura, se houver marcação; senão, registrar como limite.
- Gravar os ids finais de cada tarefa em `data/` (como `val_gsm8k_clean_ids.json`), para a prova ser a mesma nos três modelos.
- Teste de fumaça: `--limit 20` em cada tarefa, com o Lumi, antes da fila.

## Passo 2: fila da 4090 (terça à noite, ~7,5 h)

Uma fila só, no molde das `run_noiteNN_chain.sh`, nesta ordem. A ordem põe primeiro o que mais importa, para o caso de a fila não terminar.

| # | O que roda | Modelos e modos | Tempo estimado |
| --- | --- | --- | --- |
| 1 | `oab_novas`, `enem_2024` | nosso (thinking); Lumi e Nemotron (os dois modos) | ~1,5 h |
| 2 | ENEM, BLUEX e OAB da suíte | Nemotron (os dois modos) | ~1,2 h |
| 3 | `mmlu_pt_exatas`, `mmlu_pt_enade`, `mmlu_pt_concursos` | nosso (thinking); Lumi e Nemotron (os dois modos) | ~2,3 h |
| 4 | MMLU-ProX PT da suíte | Nemotron (os dois modos) | ~2,5 h |

As estimativas vêm dos tempos medidos na suíte (57 a 98 itens por minuto com thinking). O item 4 fecha a Nota do Nemotron; se não terminar, a Nota dele fica de fora e as provas individuais entram.

Molde dos comandos (o `eval_vllm.py` pula o que já está no `summary.json`):

```bash
source env.sh
E="VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py"
NOVAS=oab_novas,enem_2024
MMLU=mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos

# 1
$E --name thor_pt_antidoom_of --model runs/merged/thor_pt_antidoom_of --tasks $NOVAS --modes thinking
$E --name base --tasks $NOVAS
$E --name nemotron4b --model <Nemotron, como na rodada D1> --tasks $NOVAS
# 2
$E --name nemotron4b --model <Nemotron> --tasks enem,bluex,oab
# 3
$E --name thor_pt_antidoom_of --model runs/merged/thor_pt_antidoom_of --tasks $MMLU --modes thinking
$E --name base --tasks $MMLU
$E --name nemotron4b --model <Nemotron> --tasks $MMLU
# 4
$E --name nemotron4b --model <Nemotron> --tasks mmlu_prox_test
```

## Passo 3: leitura (quarta de manhã, ~2 h, sem GPU)

1. `report_compare.py` por prova nova: nosso contra o Lumi com thinking, contra o Lumi sem thinking e contra o Nemotron nos dois modos, com IC 95%.
2. `decompose_paired.py` por prova nova: os dois fecham, só o Lumi fecha, só o nosso fecha.
3. Uma tabela única para o slide: as cinco provas da suíte e as provas novas, com as cinco colunas.
4. Registrar em `estudos/` (hipótese nova, com as previsões acima e o veredito) e no `discovery.md`.

## Em paralelo, sem GPU (terça e quarta)

| Tarefa | Quem | Tempo |
| --- | --- | --- |
| **Decidir e datar o limiar de fechamento**: em que conjunto vale o ≥ 90% (por prova, média simples ou ponderada). É decisão posterior aos resultados e precisa ser dita assim | os dois | 15 min |
| **Revisão humana de 50 traços em português** do candidato (25 cada), às cegas: está em português? o raciocínio faz sentido? Substitui a heurística na afirmação "≥ 90% em português" | os dois | ~1 h cada |
| Atualizar os diagramas de `docs/` com o Antidoom (o pipeline hoje mostra SFT → RL; o candidato é SFT → anti-laço) | João | 30 min |
| Slides | os dois | quarta à tarde |

## Se sobrar tempo (só depois dos passos 1 a 3)

- **HealthQA-BR** ([`Larxel/healthqa-br`](https://huggingface.co/datasets/Larxel/healthqa-br), CC BY 4.0): 1.000 questões sorteadas do Revalida e do Enare. Se virar prova, sai da lista de dados de treino.
- **IFEval-PT** ([`Polygl0t/IFEval-PT`](https://huggingface.co/datasets/Polygl0t/IFEval-PT), 300 prompts): mede se o modelo com thinking ainda segue instruções. Precisa do verificador do IFEval adaptado; não começar na quarta.
- **Avaliação com amostragem** (T 0,6, top-p 0,95) do Lumi e do nosso na `oab_novas`: responde se parte da vantagem de fechamento some quando o Lumi não decodifica de forma gulosa. O `eval_vllm.py` ainda não tem as opções de temperatura.

## O que a apresentação afirma, e com que ressalvas

| Afirmação | Evidência | Ressalva |
| --- | --- | --- |
| O modelo raciocina em português | 97,3% a 100% dos traços por prova | heurística; a revisão humana confirma ou não |
| A Nota sobe com thinking | 69,9% contra 63,2% (+6,8; IC +5,9 a +7,6) e 65,5% (+4,4; IC +3,4 a +5,4) | meta mínima do PRD cumprida; meta alvo (+5,0) não |
| O ganho vem de fechar o raciocínio | nos itens que só o nosso fecha: +2,1 a +23,8 pontos por prova | nos itens que os dois fecham, o nosso acerta 0,3 a 2,7 pontos a menos |
| A configuração do adaptador não importa; os dados e o anti-laço sim | LoRA = DoRA = r=4 = só MLP; Antidoom: 54,3% → 73,2% na validação | uma semente por rodada de SFT; três no Antidoom (72,5% a 74,0%) |
| O ganho se repete em provas novas | resultado deste plano | a preencher na quarta |

Limites a dizer em voz alta: o OAB com thinking só empata com o Lumi sem thinking; o custo é de 358 a 1.597 tokens por resposta contra 149 a 216; os ICs não incluem a variância de treino; o cartão do Lumi pede thinking desligado, então a comparação mais justa é contra o Lumi sem thinking.

## Horários

| Quando | O quê |
| --- | --- |
| Terça, até ~18h | Passo 1: carregadores, descontaminação, teste de fumaça |
| Terça, ~19h | Fila do passo 2 no ar |
| Terça à noite | Revisão dos traços; decisão do limiar |
| Quarta, ~8h | Passo 3: tabelas e decomposição |
| **Quarta, 12h** | **Fim dos experimentos** |
| Quarta à tarde | Slides, ensaio, folga |
