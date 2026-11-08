# 02. Estudos de SFT: dados, professor e hiperparâmetros

Pergunta: o SFT com traços de raciocínio em português faz o Lumi raciocinar em português **e** acertar mais? De que dependem os resultados: dados, professor, taxa de aprendizado, limite de tokens, método (LoRA ou DoRA), tamanho do efeito do adaptador?

Convenções e protocolo: [01_protocolo.md](01_protocolo.md). Tabela completa e atualizada: [resultados_auto.md](resultados_auto.md). Todas as rodadas partem do Lumi original, 1 época, LoRA r=16 com lr 1e-4, salvo indicação.

## Resumo

| | Resultado |
| --- | --- |
| Idioma | Todo SFT com traços em português passa o raciocínio para português (93 a 100% dos traços) |
| Matemática | Empata ou melhora um pouco (+1,6 a +2,4 pontos, sem significância); só a autodestilação perde |
| Múltipla escolha | Perde de 5 a 12 pontos com thinking em todas as variantes, e nenhum ajuste testado corrige |
| Professor | O professor mais forte (Qwen3-30B) ganha da autodestilação do Lumi |
| Hiperparâmetros | Taxa de aprendizado, limite de tokens, DoRA e escala do adaptador não resolvem a múltipla escolha |

## E0. Linha de base: o Lumi raciocina em português sem treino?

**Hipótese H0:** uma instrução no prompt ou um começo de traço forçado em português basta para o Lumi original raciocinar em português sem perder nota.

**Desenho:** Lumi original com (a) prefill "Vou raciocinar em português." no começo do traço; (b) a frase "Pense e raciocine em português." acrescentada ao prompt. Validação completa de 1.000 itens do GSM8K e 385 de múltipla escolha, com thinking. Sem treino; ~6 min de avaliação cada.

| | Lumi | + prefill | + instrução |
| --- | --- | --- | --- |
| GSM8K (nota estrita) | 85,7% | 48,1% | 85,2% |
| Múltipla escolha | 69,6% | 61,3% | 70,1% |
| Idioma do traço | inglês (954/954) | português (570/570 dos que fecham) | inglês (956/957) |
| Fecha `</think>` e responde (GSM8K / múlt. escolha) | 95,4% / 86,2% | 57,0% / 79,7% | 95,7% / 87,8% |

**Leitura:** a instrução não muda o idioma do traço. O prefill muda o idioma, mas quebra a estrutura: 361 das 1.000 saídas terminam sem fechar o `</think>`, com a solução e o `#### N` dentro do traço (329 delas certas). Lendo o texto inteiro, o acerto no GSM8K é 81,3% (o Lumi: 85,7%): raciocinar em português preservou quase toda a capacidade, e o que se perde é o formato. **Veredito:** H0 refutada. **Limite:** uma frase de prefill, uma execução; a leitura do "texto inteiro" é só diagnóstico.

## E1. Viabilidade: traços do GigaVerbo

**Hipótese H1:** o SFT com 10.000 traços em português do subset `reasoning` do GigaVerbo, mais dados sem raciocínio, produz o comportamento desejado.

**Dados:** 78.249 linhas do subset `reasoning`; funil de qualidade (um turno e formato válido, duplicatas, resíduo do pipeline, último número do traço fora da resposta, idioma, 8-grama das provas, mais de 1.024 tokens) deixa 57.754. Treino: 15.544 exemplos (10.000 com raciocínio, mais os sem raciocínio até 25% dos tokens), 6,35 M de tokens. 84% do raciocínio é matemática, e 4.007 dos 12.182 exemplos `reasoning` usam prompts de `harmfull-reasoning` (as amostras lidas são recusas).

**Custo:** 486 passos, **21,7 min**, pico de 19,0 GiB; perda de treino 0,53 → 0,40; avaliação 7 min.

| Validação (completa, 1.000 e 385 itens) | Lumi | viab | Diferença |
| --- | --- | --- | --- |
| GSM8K com thinking, estrita | 85,7% | 71,5% | −14,2 |
| GSM8K sem thinking, estrita | 81,4% | 61,2% | −20,2 |
| Múltipla escolha com thinking | 69,6% | 43,4% | −26,2 |
| Múltipla escolha sem thinking | 63,9% | 43,1% | −20,8 |
| Fecha `</think>` e responde | 95,4% / 86,2% | 99,7% / 99,2% | |
| Traço em português | 0% | 99,7% / 100% | |

Na validação limpa do GSM8K (676 itens) o `viab` tem 67,2% e o Lumi 81,7% (a contaminação inflava o candidato em ~4 pontos).

**Leitura:** a forma foi atingida (português, `</think>` fecha, quase não há laços). A nota caiu: parte é formato (o modelo escreve a resposta em prosa em vez de `#### N`; no GSM8K a nota flexível cai 5 pontos contra 14 da estrita), parte é capacidade perdida (na múltipla escolha, mesmo com a leitura mais generosa, a letra em qualquer lugar do texto, são 14,5 pontos a menos: 55,1% contra 69,6%). **Veredito:** H1 refutada para a nota; o gate da fase 2 falhou. **Limites:** uma execução, uma taxa, 1 época; não foram separados idioma do traço, comprimento menor e deriva do LoRA.

## E2. `adalberto`: traços de um professor forte

**Hipótese H2:** traços de um professor forte (Qwen3-30B-A3B-Thinking), verificados por concordância, ensinam português e mantêm ou melhoram a nota.

**Dados** (`adalberto-temp/gsm8k-sft-thinking-pt`, sem licença declarada, só experimento): 56.296 respostas do Qwen3-30B para 7.225 problemas do GSM8K `train`. Filtros: resposta majoritária em ≥ 75% das respostas, um único `</think>`, traço em português, última linha trocada por `#### N`, sem 8-grama nem duplicata por tradução das provas. Treino: 5.842 exemplos com raciocínio (4,6 M de tokens; o limite de 1.024 tokens descartou 34% dos candidatos), 3.000 sem raciocínio e 5.544 linhas sem raciocínio do GigaVerbo = **14.386 exemplos, 7,7 M de tokens**. **Quase só matemática.**

**Custo:** 450 passos, **25,7 min**, 5,5 mil tokens/s, perda 0,42 → 0,21; avaliação 13 min.

| Validação limpa | Lumi | `SFT LORA adalberto` | Diferença (IC 95%) |
| --- | --- | --- | --- |
| GSM8K com thinking | 81,7% | 83,3% | +1,6 (−1,0 a +4,3) |
| GSM8K sem thinking | 77,2% | 79,3% | +2,1 (−0,7 a +4,9) |
| Múltipla escolha com thinking | 69,6% | 62,3% | **−7,3** (−11,7 a −2,9) |
| Múltipla escolha sem thinking | 63,9% | 58,2% | **−5,7** (−10,6 a −0,8) |
| Traço em português | 0% | 99,4% (MCQ) | |
| Fecha `</think>` (GSM8K / MCQ) | 94,5% / 86,2% | 96,6% / 86,8% | |
| Truncadas (MCQ) | 13,8% | 13,8% | |

Contra o Lumi **sem** thinking, o `adalberto` com thinking ganha na matemática: +6,1 (IC +3,4 a +8,7), que é a meta da fase 2. **Leitura:** a matemática funcionou; a múltipla escolha perdeu 7 pontos, e sem thinking a perda estrita é em parte de formato (flexível igual ao Lumi). **Veredito:** H2 aceita para matemática, refutada para múltipla escolha. **Limites:** treino e validação vêm da mesma distribuição (GSM8K `train`); licença desconhecida.

## E3. `selfdist`: o Lumi como professor de si mesmo

**Hipótese H3:** o próprio Lumi, com raciocínio forçado em português (prefill) e filtrado, serve de professor.

**Dados:** geração de ~1 h; de 14.000 problemas com raciocínio, 6.742 aceitos (descartados: 4.125 sem um único `</think>`, 1.938 resposta errada, 1.158 truncados, 37 traço fora do português); 3.000 sem raciocínio, 2.212 aceitos. Treino: 6.365 com raciocínio + 2.204 sem + 5.544 do GigaVerbo = 14.113 exemplos, 5,59 M de tokens.

**Custo:** 442 passos, **19,4 min**; avaliação 10 min.

| Validação limpa | Lumi | `selfdist` | Diferença | vs `adalberto` |
| --- | --- | --- | --- | --- |
| GSM8K com thinking | 81,7% | 78,1% | −3,6 (−6,5 a −0,7) | −5,2 |
| Múltipla escolha com thinking | 69,6% | 57,9% | **−11,7** (−16,4 a −7,0) | −4,4 |
| GSM8K sem thinking | 77,2% | 77,5% | +0,3 | |
| Múltipla escolha sem thinking | 63,9% | 63,4% | −0,5 | |

Forma: fecha `</think>` em 92,2% (GSM8K) e 79,2% (MCQ), trunca 19,7% na MCQ. **Veredito:** H3 refutada: aprende idioma e forma, mas o professor precisa ser mais forte que o aluno. O modo sem thinking ficou igual ao Lumi, porque o treino ali quase não o alterou.

## E4. Limite de tokens do treino (`adalberto_L2048`)

**Hipótese H4:** descartar 34% dos candidatos longos (limite de 1.024 tokens) enviesa o treino para traços curtos e causa a perda na múltipla escolha.

**Desenho:** igual ao `adalberto`, com limite de 2.048 tokens (só 346 candidatos descartados; 8,8 M de tokens). **Custo:** 450 passos, **28,6 min**.

| Validação limpa | `adalberto` | `adalberto_L2048` |
| --- | --- | --- |
| GSM8K com / sem thinking | 83,3 / 79,3 | 82,2 / 81,7 |
| Múltipla escolha com / sem thinking | 62,3 / 58,2 | 64,2 / 55,3 |

**Veredito:** H4 refutada. O limite de tokens não é a alavanca; a múltipla escolha sem thinking piorou (−8,6 contra o Lumi).

## E5. Taxa de aprendizado menor (`adalberto_lr5e5`)

**Hipótese H5:** metade da taxa (5e-5) danifica menos o Lumi e preserva a múltipla escolha. **Custo:** 450 passos, **25,7 min**.

Resultado (validação limpa): GSM8K 83,1 / 80,2 (▲ sem thinking); múltipla escolha 63,4 (▼ −6,2) / 57,9 (▼ −6,0). Parecido com a taxa 1e-4 (62,3 / 58,2). **Veredito:** H5 refutada.

## E6. DoRA em vez de LoRA (`adalberto_dora`)

**Hipótese H6:** o DoRA (que separa magnitude e direção) preserva melhor o modelo. **Custo:** 450 passos, **63,9 min** (2,1 mil tokens/s, micro-batches de 1 sequência), pico de 15,9 GiB; avaliação 12 min.

Resultado: GSM8K 84,0 / 79,1; múltipla escolha 62,6 / 57,7. A diferença contra o LoRA é de 0,3 a 0,7 ponto, dentro do ruído. **Veredito:** H6 refutada: o DoRA custa 2,5 vezes mais tempo e não melhora nada.

## E7. Tamanho do efeito do adaptador (sem treino)

**Hipótese H7:** interpolar entre o Lumi (escala 0) e o `adalberto` (escala 1), multiplicando o efeito do adaptador, mantém o ganho em matemática e recupera a múltipla escolha. Fusão na CPU com `merge_adapter.py --scale`; sem treino.

| Escala do adaptador | GSM8K com / sem | Múltipla escolha com / sem |
| --- | --- | --- |
| 1,0 (`adalberto`) | 83,3 / 79,3 | 62,3 / 58,2 |
| 0,75 | 84,0 / 79,1 | 64,4 / 58,7 |
| 0,5 | 84,0 / 79,6 | 63,4 / **64,9** |

**Veredito:** H7 refutada para o modo com thinking (continua 5 a 6 pontos abaixo do Lumi); a escala 0,5 sem thinking (64,9%) é compatível com o ruído de 385 itens. Mantém o ganho de matemática (+2,4) por inteiro.

## E8. Piloto de tradução (Lumi como tradutor)

**Hipótese H8:** o Lumi traduz bem questões de ciência e traços de raciocínio (inglês → português), a ponto de ser o tradutor de dados de professores em inglês.

**Desenho:** 200 questões de ciência com traços do DeepSeek-R1 (`scripts/translate_traces.py`, vLLM, temperatura 0, modo sem thinking, blocos de parágrafos de até 1.800 caracteres); cada bloco é aceito se terminar, não trouxer marcadores de raciocínio e tiver tamanho compatível. 88 s para 200 registros.

| Medida | Resultado |
| --- | --- |
| Aceitos | 172 de 200 (86%): com raciocínio 125/150, só resposta 47/50 |
| Rejeitados (28) | 16 tamanho incompatível, 10 não terminou, 2 alternativas não voltaram |
| Tamanho PT/EN (caracteres) | mediana 1,04 |
| Números do original presentes | média 94,7%, mediana 100% |
| Inglês que sobrou | média 1,2% |

Defeitos: 120 dos 125 traços abrem com "Okay, então…", terminologia técnica errada em química, erros leves de gênero. Os mesmos scripts traduziram 4.000 enunciados de múltipla escolha para o RL (93,5% aceitos).

**Em lote maior** (3.200 registros de uma fonte que depois saiu do plano), só **35% dos raciocínios** foram aceitos: em 1.101 blocos o Lumi não traduziu e **respondeu a questão** (`Answer: \boxed{D}`). O filtro de tamanho barrou esses casos, e nenhum texto ruim entrou em treino. **Veredito:** H8 aceita para enunciados e piloto, **refutada para traços longos em escala**; a tradução dos traços de professor passou para a Thor, com o Nemotron 30B (`handoff/thor3/`). **Não medido:** a correção de sentido da tradução (3 exemplos lidos) e se treinar com o traço traduzido melhora o modelo.

## Conclusões dos estudos de SFT

1. O SFT resolve **forma e idioma** com pouco custo (20 a 30 min por rodada), mas a perda na múltipla escolha com thinking é consistente (−5 a −12) e **insensível a** limite de tokens, taxa, DoRA e escala.
2. A qualidade e a força do professor importam (E3 contra E2). O estilo do professor é a hipótese principal que **não foi testada**: o Lumi raciocina curto e em forma de plano (mediana de 96 palavras, abre com "We need to…"), o Qwen3-30B é mais longo e falado (177 palavras) e o R1 ainda mais (610). Isso só será testado com os traços do Nemotron 30B (Thor).
3. O dano sem thinking, onde existe, é parte formato e parte deriva do adaptador sobre dados fora da distribuição do Lumi.

Hipóteses abertas e próximos experimentos: [07_hipoteses_e_proximos_passos.md](07_hipoteses_e_proximos_passos.md). A parte arquitetural (qual bloco do adaptador causa o dano): [03_ablacao_camadas_e_mlp.md](03_ablacao_camadas_e_mlp.md).

## E12. SFT com os traços do Nemotron 30B da Thor, em português e em inglês (entrega parcial)

**Perguntas:** (1) traços de um professor **da mesma família** do Lumi (Nemotron 3 Nano 30B) evitam a perda na múltipla escolha que o Qwen3-30B causou (H19 e H20)? (2) o raciocínio em português custa nota em relação ao mesmo raciocínio em inglês (H21)?

**Dados** (`traduzidos_com_raciocinio.jsonl`, pacote 3 da Thor, recebido em 08/11 às 11:47; **parcial**: só os 70% com raciocínio, os 30% sem raciocínio não vieram): o Nemotron 30B gerou o raciocínio em inglês e o traduziu com ele mesmo (T=0, conferências por pedaço: tamanho, números, "respondeu em vez de traduzir"); 5.278 registros, todos aprovados (`verdict: ok`):

| Fonte | Recebidos | Depois do filtro de contaminação | Tipo |
| --- | --- | --- | --- |
| `gsm8k_train` (matemática) | 1.746 | 1.744 | gabarito oficial |
| `gv_math` (matemática) | 1.059 | 1.041 | gabarito do GigaVerbo |
| `mot_science` (ciência, múltipla escolha) | 2.473 | 2.461 | letra majoritária dos traços do R1 |
| **Total** | **5.278** | **5.246** (17 duplicatas da validação por assinatura de números e Jaccard, 15 por 8-grama) | |

Com o limite de 2.048 tokens (o de 1.024 do `adalberto` cortaria o p90 dos raciocínios de ciência, 1.848 tokens) e a interseção entre as versões em português e em inglês, ficam **4.927 problemas** (2.174 de ciência, 1.719 do `gsm8k_train`, 1.034 do `gv_math`), os mesmos nas duas versões. Às duas se somam as 5.544 linhas sem raciocínio do GigaVerbo (`--base`, como no `adalberto`): 10.471 linhas; 5,15 milhões de tokens (português) e 4,75 milhões (inglês); ~328 passos por época.

**Conferência de qualidade** (CPU): 100% dos raciocínios em português (heurística de idioma) e 100% em inglês nos originais (as versões em inglês têm 2 a 3% "mistos"); todas as respostas terminam no formato de avaliação (`Resposta: X` igual ao gabarito; `#### N`); letras do gabarito balanceadas (A 573, B 615, C 632, D 641). Mediana do raciocínio: 587 tokens (ciência) e ~265 (matemática), contra 176 palavras do Qwen3-30B na matemática.

**Achado de estilo (sustenta H19):** a fração de raciocínios que **repetem a instrução de formato do enunciado** ("Garanta que…", "formato `#### N`", "sem unidades") é característica da família Nemotron e do Lumi, e **não** do Qwen:

| Fonte | Matemática | Múltipla escolha |
| --- | --- | --- |
| Qwen3-30B (`adalberto`, saídas do modelo treinado) | 0% | 1% |
| Lumi original | 84% | 33% |
| Nemotron 4B | 77% | 62% |
| **Nemotron 30B (os traços desta entrega, em inglês)** | **97%** | **82%** |

(A mesma expressão regular em português e inglês, aplicada ao raciocínio de cada fonte; saídas de validação para os modelos e traços de treino para o 30B. É uma medida de estilo, não de qualidade.) Ou seja: o Qwen3-30B ensinou um estilo que o Lumi não tem; o Nemotron 30B ensina o estilo que o Lumi já tem.

**Desenho** (três SFTs, mesmos hiperparâmetros do `adalberto`: r=16, alpha 32, lr 1e-4, 1 época; entram na fila `cap-queue` na frente do estudo de capacidade):

| Rodada | Raciocínio | Problemas | O que isola |
| --- | --- | --- | --- |
| `thor_pt` | Nemotron 30B, traduzido para o português | 4.927 (matemática + ciência) | H20: professor da mesma família |
| `thor_en` | os **mesmos** problemas, raciocínio original em inglês (prompt e resposta finais em português) | 4.927 | H21: o idioma do raciocínio, com tudo o mais igual |
| `thor_pt_math` | Nemotron 30B, português, **só a matemática** | 2.753 | separa o estilo do professor do formato da prova (ver abaixo) |

**Confundidor, e por que o terceiro treino existe:** o `adalberto` só tinha traços de matemática. O `thor_pt` traz **2.174 exemplos de múltipla escolha no formato exato da avaliação**. Se a múltipla escolha subir no `thor_pt`, a causa pode ser o estilo do professor **ou** o treino no formato da prova. O `thor_pt_math` tem o estilo do Nemotron sem nenhuma questão de múltipla escolha: se ele já recupera a nota, é estilo; se só o `thor_pt` recupera, é formato ou domínio.

**Previsões** (escritas antes dos resultados; a regra de leitura do E11 vale aqui: sinal = +4 pontos ou mais contra o `adalberto` na múltipla escolha com thinking, no mesmo sentido sem thinking):

| Resultado | Leitura |
| --- | --- |
| `thor_pt` ≥ 66% e `thor_pt_math` ≥ 66%, português ≥ 90% | Estilo da família: H19 e H20 apoiadas; esse é o SFT base do RL |
| `thor_pt` ≥ 66% e `thor_pt_math` ≈ 62% | O ganho vem de ver questões de múltipla escolha no treino, não do estilo |
| `thor_pt` ≈ 62% | O professor e o estilo não são a causa; a perda vem de ajustar o modelo a traços de qualquer fonte |
| `thor_en` − `thor_pt` ≥ +4 | O português custa nota (H21 apoiada) |
| \|`thor_en` − `thor_pt`\| < 4 | O idioma do raciocínio não custa nota: o objetivo de raciocinar em português é "gratuito" |

**Limites:** entrega parcial (4.927 problemas contra 5.842 de raciocínio no `adalberto`; sem os exemplos sem raciocínio do professor, só os do GigaVerbo); o gabarito de ciência é fraco (letra majoritária do R1, não oficial); uma execução por rodada; a tradução foi feita pelo próprio Nemotron 30B (erros de tradução não auditados por pessoa: a revisão D-10 continua pendente).

**Estado:** dados prontos (`data/thor_pt.parquet`, `thor_en.parquet`, `thor_pt_math.parquet`); `thor_pt` e `thor_en` feitos; `thor_pt_math` **descartado** (o controle só faria sentido para explicar um ganho na múltipla escolha, que o `thor_pt` não teve).

### Resultado do `thor_pt` (treino 17,6 min, 328 passos, perda 0,83 → ~0,77; avaliação 12 min)

| Validação limpa | Lumi | `adalberto` | **`thor_pt`** |
| --- | --- | --- | --- |
| Múlt. escolha com thinking | 69,6% | 62,3% ▼ | **54,3%** ▼ (−15,3; −8,1 contra o `adalberto`, IC −12,7 a −3,4) |
| Múlt. escolha sem thinking | 63,9% | 58,2% ▼ | **62,9%** (−1,0; +4,7 contra o `adalberto`, IC −0,3 a +9,9) |
| GSM8K com thinking | 81,7% | 83,3% | **77,8%** ▼ (−3,8; −5,5 contra o `adalberto`, IC −8,1 a −3,0) |
| GSM8K sem thinking | 77,2% | 79,3% | 74,6% (−2,7) |
| Fecha `</think>` (múlt. escolha / GSM8K) | 86,2% / 95,3% | 86,8% / 96,8% | **66,0% / 88,1%** |
| Trunca em 4.096 tokens (múlt. escolha) | 13,8% | 13,8% | **34,0%** |
| Traço em português | 0% | 99,4% | 99,6% |
| Mediana de tokens (múlt. escolha) | 490 | 1.176 | 703 |

**O que aconteceu:** o `thor_pt` **piorou** o raciocínio com thinking, e a causa é **laço de repetição**, não erro de raciocínio.
- **Laços:** 127 das 131 respostas truncadas (97%) terminam repetindo a mesma frase ("Na verdade, o osso hioide se origina do primeiro arco faríngeo (ou seja, …). Na verdade, …") até estourar os 4.096 tokens. Em todas as respostas de múltipla escolha, laços passam de 13,2% (Lumi) e 10,4% (`adalberto`) para **33,0%** (`thor_pt`); no GSM8K, de 4,5% e 2,3% para 11,9%. O Lumi original já entra em laço em 96% das suas truncadas: é uma fragilidade da família, que este treino amplifica.
- **Raciocínio, quando termina, está igual ao do Lumi.** Nos itens que os dois terminam: múltipla escolha 84,0% contra 84,5% (−0,4; IC −4,2 a +3,8; n=238) e GSM8K 90,6% contra 88,9% (+1,8; IC −0,2 a +3,7; n=566). Contra o `adalberto` (n=237): 84,4% contra 81,9% (+2,5; IC −2,1 a +7,2). (O "82,3% de acerto quando termina" do resumo engana: os itens que terminam são os mais fáceis; só vale a comparação nos mesmos itens.) Já o `adalberto` perde por **errar** (72,3% de acerto quando termina, contra 80,7% do Lumi), com laço normal.
- **Dos 131 itens em que o `thor_pt` entra em laço, o Lumi termina 94 e acerta 67 (71%).** Sem laços, o `thor_pt` ficaria perto do Lumi.
- **Sem thinking** é o melhor SFT até agora: múltipla escolha 62,9% (−1,0, dentro da tolerância do critério), mas o GSM8K sem thinking cai 2,7.

**Leitura contra as previsões:** caiu na terceira linha ("`thor_pt` ≈ 62%"), pior ainda. Mas o mecanismo é outro do esperado. **H19 e H20 parcialmente apoiadas:** o professor da mesma família **não** causa erro de raciocínio (quando termina, iguala o Lumi, enquanto o Qwen perdia 8 pontos por erro), mas troca o dano por laços de repetição. Isso muda o próximo passo: não é mais "achar dados melhores", é **eliminar os laços**, que é o que o RL com recompensa zero para truncadas já fez (fechamento 86,8% → 93,2% no `gspo1`). **Hipótese H33:** o RL (GSPO) a partir do `thor_pt` leva a múltipla escolha acima do Lumi, porque tira os laços sem perder a qualidade de raciocínio. A decidir depois de `thor_en` (o laço vem do idioma?) e `thor_pt_math` (vem das questões de múltipla escolha?).

**Limites:** uma execução; treino com 4.927 problemas e 328 passos (não 450); os laços foram contados por um critério simples (6-grama repetido 8 vezes ou mais nas últimas 6.000 letras); a qualidade da tradução não foi auditada por pessoa.

## E13. SFT só com exemplos sem thinking: melhorar o modo sem thinking primeiro (ADIADO, 08/11 13:05)

**Atualização (08/11, 13:05): adiado.** O foco voltou ao modo **com thinking** (RL a partir do `thor_pt`, [04](04_rl_grpo.md)); os três treinos saíram da fila e os dados continuam prontos em `data/nt_*.parquet`. O diagnóstico D8 ([05](05_diagnosticos_e_referencias.md)) vale de qualquer forma. **Decisão original (08/11, 13:00):** a prioridade passaria a ser **melhorar o modo sem thinking**; só depois se analisa como treinar o thinking sem estragá-lo. O RL a partir do `thor_pt` (H33) e a proteção do modo sem thinking no RL (H22) ficam para essa segunda fase.

**Ponto de partida (D8, [05](05_diagnosticos_e_referencias.md)):** o conteúdo da múltipla escolha sem thinking não caiu (flexível 63 a 64%); o que cai é o formato (`Resposta: X`). A matemática sem thinking sobe com respostas passo a passo do Qwen3-30B. Ou seja, parte do que falta se corrige com **dados no formato certo**, e isso se testa barato: treino só com exemplos sem thinking (11 a 15 min) mais a avaliação nos dois modos (12 min).

**Desenho** (as linhas sem thinking do GigaVerbo, 5.544, entram em todas; hiperparâmetros do `adalberto`; nenhum exemplo com thinking):

| Rodada | Exemplos sem thinking de professor | Tokens |
| --- | --- | --- |
| `nt_qwen` | 2.914 respostas de matemática do Qwen3-30B (passo a passo, as do `adalberto`) | 3,07 M |
| `nt_thor` | 4.927 respostas do Nemotron 30B (2.174 de múltipla escolha no formato da prova, 2.753 de matemática curtas) | 2,85 M |
| `nt_both` | 2.174 de múltipla escolha do Nemotron + 2.914 de matemática do Qwen | 3,61 M |

**Previsões** (escritas antes dos resultados; sinal = +4 pontos contra o Lumi ou contra a rodada de comparação):

| ID | Hipótese | Previsão |
| --- | --- | --- |
| H34 | A perda estrita da múltipla escolha sem thinking é de formato | `nt_thor` e `nt_both` com estrita ≥ 62% e linha final ≥ 95%; `nt_qwen` (sem exemplo de múltipla escolha) fica abaixo, como o `adalberto` (≤ 60%) |
| H35 | O estilo passo a passo do Qwen é o que sobe o GSM8K sem thinking | GSM8K sem thinking: `nt_qwen` e `nt_both` ≥ 79% na validação limpa (Lumi 77,2%); `nt_thor` ≈ 75% |
| H36 | Treinar só o modo sem thinking deixa o thinking como está | Com thinking: múltipla escolha e GSM8K dentro de ±3 do Lumi (69,6% e 81,7%), traço em português ≈ 0% |
| H37 | Só formato não passa o Lumi: a múltipla escolha sem thinking estaciona em ~64% | `nt_both` com estrita entre 62 e 66% |

**O que cada resultado decide:** se H34 e H35 se confirmam, `nt_both` vira a base do modo sem thinking (múltipla escolha empatada com o Lumi e matemática acima). Se H37 também se confirma, passar do Lumi na múltipla escolha sem thinking exige mais que formato: **RL no modo sem thinking** (recompensa pela nota estrita, respostas curtas) com gabarito oficial das 12.000 questões novas da Thor, e/ou conhecimento novo. H36 diz se os dois modos podem ser treinados em separado (uma resposta antecipada para a segunda fase).

**Limites:** uma execução por rodada; sem exemplos com thinking, o adaptador não ensina português no raciocínio (esse objetivo fica para a fase seguinte); as respostas do Nemotron são o texto final **depois** do raciocínio, não geração sem thinking do próprio professor.



### Resultado do `thor_en` (os mesmos 4.927 problemas, raciocínio original em inglês; treino 16 min, avaliação 10 min)

| Validação limpa | Lumi | `thor_pt` (português) | **`thor_en`** (inglês) | `thor_en` − `thor_pt` (IC 95%) |
| --- | --- | --- | --- | --- |
| Múlt. escolha com thinking | 69,6% | 54,3% ▼ | **60,3%** ▼ (−9,4) | **+6,0** (+1,6 a +10,4) |
| GSM8K com thinking | 81,7% | 77,8% ▼ | **83,1%** (+1,5) | **+5,3** (+2,7 a +8,0) |
| Fecha `</think>` (múlt. escolha) | 86,2% | 66,0% | 75,3% | |
| Laços (múlt. escolha / GSM8K) | 13,2% / 4,5% | 33,0% / 11,9% | **22,6% / 3,7%** | |
| Múlt. escolha sem thinking | 63,9% | 62,9% | 56,1% ▼ (−7,8) | −6,8 (−11,4 a −2,1) |
| GSM8K sem thinking | 77,2% | 74,6% | 65,7% ▼ (−11,5) | −8,9 (−12,0 a −5,8) |

**Leitura (H21):** com os mesmos problemas, as mesmas respostas e o mesmo professor, raciocinar em **inglês** rende **+5 a +6 pontos** com thinking contra raciocinar em português. **H21 apoiada: o português tem custo real.** Parte do custo é laço (22,6% contra 33,0% na múltipla escolha; 3,7% contra 11,9% no GSM8K), então o idioma torna o modelo mais propenso a repetir. Mas o inglês **também** perde para o Lumi na múltipla escolha (60,3%, −9,4 ▼, e laços de 22,6% contra 13,2%): **qualquer SFT com os traços do Nemotron 30B piora a múltipla escolha com thinking**, em qualquer idioma. O português é parte do problema; o SFT com estes traços é o resto.

**Sem thinking, o `thor_en` cai por formato:** as respostas estão em português (99% a 100%), mas a linha final some em ~10% (`#### N` no GSM8K, 90% têm; `Resposta: X` na múltipla escolha, 88% têm). Exemplo: "A resposta é: 6" no lugar de "#### 6". O acerto flexível é 79,1% no GSM8K (contra 80,9% do `thor_pt`), então o conteúdo quase não mudou. Treinar o raciocínio em inglês com resposta em português parece enfraquecer a obediência ao formato do modo sem thinking.

**Consequência para o objetivo:** o projeto exige raciocínio em português, e esse custo de ~5 a 6 pontos precisa ser recuperado depois do SFT. O `gspo_thor` (RL a partir do `thor_pt`, [04](04_rl_grpo.md)) é o teste: se ele levar o `thor_pt` acima do Lumi, o custo do português foi pago pelo RL.
