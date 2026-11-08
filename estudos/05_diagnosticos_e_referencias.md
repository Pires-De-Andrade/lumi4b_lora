# 05. Diagnósticos, referências, contaminação e dados

Estudos que não treinam o nosso modelo: servem para saber **de onde partimos**, **contra o quê comparamos** e **o que pode invalidar a comparação**. Protocolo: [01_protocolo.md](01_protocolo.md).

## D1. Referência: o Nemotron 3 Nano 4B (pai do Lumi)

**Pergunta:** o que um modelo da mesma família, com raciocínio em inglês, faz nas nossas provas? É o teto realista para o Lumi com thinking.

**Desenho:** Nemotron 3 Nano 4B sem treino nosso, mesmo protocolo (template e vocabulário idênticos aos do Lumi, conferidos), validação limpa, 5 min de avaliação. Pelo cartão, ele foi comprimido do Nemotron-Nano-9B-v2; o cartão **não** cita o Nemotron 30B como fonte de dados dele.

| Validação limpa | Lumi | Nemotron 4B | Diferença |
| --- | --- | --- | --- |
| GSM8K com thinking | 81,7% | 85,2% | +3,6 (IC +1,0 a +6,1) ▲ |
| Múltipla escolha com thinking | 69,6% | 68,3% | −1,3 (IC −5,7 a +3,1) |
| GSM8K sem thinking | 77,2% | 63,5% | −13,8 ▼ |
| Múltipla escolha sem thinking | 63,9% | 58,7% | −5,2 ▼ |

Forma: fecha `</think>` em 89,6% (MCQ) e 100% (GSM8K), mediana de 241 tokens no GSM8K e 503 na múltipla escolha, traço em inglês. As notas estrita e flexível são **iguais**, então a nota baixa sem thinking não é de formato: ele responde em 89 tokens de mediana e erra mais conta. **Leitura:** na múltipla escolha com thinking o Lumi **empata** com o pai (69,6 contra 68,3); a matemática do pai é melhor. Isso fixa a barra: superar o Lumi na múltipla escolha com raciocínio é superar o que a família já faz.

## D2. O idioma da pergunta afeta o Lumi original?

**Hipótese H18:** o Lumi acerta menos quando a pergunta vem em português do que em inglês; isso explicaria a perda na múltipla escolha dos nossos modelos.

**Desenho:** as mesmas **1.200 questões** do teste do MMLU-ProX (sorteadas com semente fixa, ids iguais nos dois idiomas, só questões válidas nos dois), com enunciado, alternativas e instrução em português ou em inglês; mesmo marcador `Resposta: <letra>`; Lumi original, com e sem thinking; bootstrap pareado. 32 min de geração no total (quatro blocos).

| | Português | Inglês | PT − EN (IC 95%) |
| --- | --- | --- | --- |
| Com thinking, estrita | 52,2% | 51,8% | +0,4 (−2,1 a +2,9) |
| Com thinking, flexível | 52,1% | 53,4% | −1,3 (−3,8 a +1,1) |
| Sem thinking, estrita | 48,7% | 44,8% | +3,8 (+0,8 a +6,8) |
| Sem thinking, flexível | 47,1% | 48,5% | −1,4 (−4,2 a +1,4) |

**Veredito:** H18 refutada. A vantagem do português na estrita sem thinking é de formato (em inglês o Lumi escreve menos a linha `Resposta:`); na flexível é empate nos dois modos. Truncadas com thinking: 27,5% (PT) e 26,1% (EN) em 4.096 tokens (questões de até 10 alternativas). **Limite:** o Lumi raciocina em **inglês** nos dois casos (100% dos traços), então isto mede o idioma da *pergunta*, **não** o custo de *raciocinar* em português.

## D3. Auditoria de contaminação

**Achado:** o filtro de 8-gramas só pega a **mesma tradução**. Os dados do GigaVerbo e o `adalberto` trazem problemas do GSM8K `train` em **outras traduções**.

| Comparação | Mesma assinatura de números | Duplicata provável (Jaccard ≥ 0,3) |
| --- | --- | --- |
| Validação do GSM8K (1.000) contra o treino de matemática do GigaVerbo | 599 | ~325 (112 com ≥ 0,5) |
| **Teste** do GSM8K-PT (1.295) contra todo o treino | 473 | **0** |

**Consequência:** o teste está limpo; a validação não estava. O subconjunto limpo (676 itens, `data/val_gsm8k_clean_ids.json`, `scripts/clean_validation.py`) passou a ser o conjunto de seleção. O efeito: a contaminação inflava o candidato `viab` em ~4 pontos (71,5% contra 67,2%) e o Lumi em outro tanto (85,7% contra 81,7%). Os 8-gramas do índice cobrem 46.440 itens das provas; o filtro por assinatura de números e Jaccard cobre as duplicatas por tradução, aplicado também aos dados dos professores (`prepare_adalberto.py`, `prepare_thor.py`, `filter_translated.py`).

## D4. Linha de base do Lumi (suíte de teste) e equivalência do motor

| | Resultado |
| --- | --- |
| GSM8K-PT teste (1.295), com thinking | 88,9% (duas execuções: 88,9 e 88,7); sem thinking 85,2 e 85,3 |
| Efeito do thinking no GSM8K-PT | +3,7 pontos (IC 95% +1,85 a +5,56) |
| vLLM contra Transformers (256 itens) | com raciocínio 90,6% contra 91,0%; sem raciocínio 88,3% contra 87,9%; mesmo comportamento |
| Velocidade | vLLM 8 a 12 vezes mais rápido que o Transformers; o GSM8K-PT inteiro leva 233 s com thinking e 116 s sem |
| Ruído entre execuções do vLLM | ±0,15 a 1,2 ponto (texto idêntico em só 23% dos itens com raciocínio) |

Faltam na suíte de teste: MMLU-ProX PT, ENEM, BLUEX, OAB (e MMLU PT e IFEval PT, cujos ids o PRD não dá), do Lumi e de todos os candidatos.

## D5. Estilo do raciocínio dos professores

**Hipótese H19:** traços de estilo diferente do Lumi ensinam um modo de raciocinar que ele não tem, e isso custa na múltipla escolha. **Não testada diretamente**; o que há é a descrição:

| Fonte | Mediana de palavras (GSM8K) | Abre com | "wait" por 1.000 palavras |
| --- | --- | --- | --- |
| Lumi original | 96 | "We need to…" | 0,2 |
| Qwen3-30B (`adalberto`) | 177 | "Certo, vamos…" | 1,9 |
| DeepSeek-R1 (ciência) | 610 | "Okay, so…" | 7,6 |

O Lumi pensa curto e em forma de plano. O teste que falta é treinar com traços de um professor da **mesma família** (Nemotron 30B, pacotes 2 e 3 da Thor) e ver se a múltipla escolha deixa de cair.

## D6. Fontes de dados avaliadas

| Fonte | Veredito |
| --- | --- |
| `recogna-nlp/Bode-reasoning` | **Fora:** os 3.869 itens de OAB, BLUEX e ENEM têm 100% de 8-grama com as nossas provas |
| `xandykati98/PTBR-Opus-4.6-Reasoning-3000x-filtered` | Pequeno (2.326); traços do Claude Opus, termos do provedor a conferir |
| `cnmoro/reasoning-v1-20m-portuguese` | Perguntas sem gabarito e tradução com erros: arriscado |
| `AxionLab-Co/ThinkSet-PTBR` | Aritmética trivial: descartada |
| `open-r1/Mixture-of-Thoughts` (ciência) | **Usada:** enunciados traduzidos como prompts do RL; origem das questões dos pacotes da Thor; sem gabarito (letra majoritária do R1) |
| `openlifescienceai/medmcqa` | **Pacote 3:** 182.822 questões de treino com gabarito oficial, Apache-2.0 |
| `TIGER-Lab/WebInstruct-verified` | **Pacote 3:** 229 mil questões de várias áreas, Apache-2.0, gabarito checado por modelo |
| ENEM, BLUEX, OAB, Alvorada, LegalScore, MMLU | Avaliação: não servem para treino (contaminariam a suíte) |

## D7. Pacotes da Thor (geração de dados, 122 GiB de memória unificada)

| Pacote | Conteúdo | Estado (08/11, 11:00) |
| --- | --- | --- |
| 1 (`handoff/thor/`) | Qwen3-30B-A3B-Thinking com o começo do raciocínio forçado em português; 50.908 problemas (5.612 do GSM8K-PT `train`, 45.296 do GigaVerbo) | Gerando; nada devolvido |
| 2 (`handoff/thor2/`) | Nemotron 3 Nano 30B-A3B, raciocínio em inglês; 21.612 prompts (9.612 de matemática, 12.000 de múltipla escolha de ciência) | Gerando; nada devolvido |
| 3 (`handoff/thor3/`) | O Nemotron 30B **traduz** para o português os exemplos dos pacotes 2 e 3 (enunciado, alternativas, raciocínio e resposta), com conferências por pedaço (tamanho, números, "respondeu em vez de traduzir"); gera traços para 12.000 questões novas com gabarito (7.000 do WebInstruct-verified, 5.000 do MedMCQA; sem 8-grama com MMLU e MMLU-ProX, 2.932 removidas, e 20.649 enunciados repetidos removidos); guarda o raciocínio original em inglês | Pronto, 27 testes, ainda não enviado |

A tradução na Thor substitui a tradução com o Lumi na 4090, que falhou em escala ([02](02_sft_dados_e_professor.md), E8). Cada exemplo traduzido guarda o par em inglês, o que permite o experimento **"raciocínio em inglês contra português"** com os mesmos problemas.
