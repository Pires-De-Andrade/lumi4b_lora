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

**Leitura:** a forma foi atingida (português, `</think>` fecha, quase não há laços). A nota caiu: parte é formato (o modelo escreve a resposta em prosa em vez de `#### N`; no GSM8K a nota flexível cai 5 pontos contra 14 da estrita), parte é capacidade perdida (na múltipla escolha, mesmo com a leitura mais generosa, a letra em qualquer lugar do texto, são 14,5 pontos a menos: 55,1% contra 69,6%). **Veredito:** H1 refutada para a nota; o gate da fase 2 falhou. **Limites:** uma execução, uma taxa, 1 época; não separei idioma do traço, comprimento menor e deriva do LoRA.

## E2. `adalberto`: traços de um professor forte

**Hipótese H2:** traços de um professor forte (Qwen3-30B-A3B-Thinking), verificados por concordância, ensinam português e mantêm ou melhoram a nota.

**Dados** (`adalberto-temp/gsm8k-sft-thinking-pt`, sem licença declarada, só experimento): 56.296 respostas do Qwen3-30B para 7.225 problemas do GSM8K `train`. Filtros: resposta majoritária em ≥ 75% das respostas, um único `</think>`, traço em português, última linha trocada por `#### N`, sem 8-grama nem duplicata por tradução das provas. Treino: 5.842 exemplos com raciocínio (4,6 M de tokens; o limite de 1.024 tokens descartou 34% dos candidatos), 3.000 sem raciocínio e 5.544 linhas sem raciocínio do GigaVerbo = **14.386 exemplos, 7,7 M de tokens**. **Quase só matemática.**

**Custo:** 450 passos, **25,7 min**, 5,5 mil tokens/s, perda 0,42 → 0,21; avaliação 13 min.

| Validação limpa | Lumi | `adalberto` | Diferença (IC 95%) |
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
