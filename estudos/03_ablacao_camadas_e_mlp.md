# 03. Ablação por camadas e por tipo de bloco

Pergunta: **qual parte do adaptador causa a perda na múltipla escolha, e qual carrega o raciocínio em português?** Os estudos de SFT ([02](02_sft_dados_e_professor.md)) mostraram a perda e não mostraram a causa. Este estudo abre o adaptador por camada e por tipo de bloco.

## A arquitetura

O Lumi tem 42 camadas numeradas de 0 a 41, e cada uma é **um** bloco, de um de três tipos, num padrão fixo (`M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-`, em que `M` é Mamba2, `-` é MLP e `*` é atenção):

| Tipo de bloco | Camadas | Módulos com LoRA no adaptador completo | Observação |
| --- | --- | --- | --- |
| Mamba2 | 21 | `in_proj` (21) | o `out_proj` fica fora (o kernel fundido o ignora) |
| MLP | 17 | `up_proj` e `down_proj` (34) | |
| Atenção | 4 (camadas 12, 17, 24 e 32) | `q/k/v/o_proj` (16) | |

Total: 71 módulos, 17,05 milhões de parâmetros. **Camadas 0 a 20** (primeira metade): 10 Mamba, 9 MLP, 2 atenção (36 módulos). **Camadas 21 a 41:** 11 Mamba, 8 MLP, 2 atenção (35 módulos).

## E9. Triagem pós-treino (sem treinar)

**Hipóteses:**
- **H9a:** a perda na múltipla escolha vem de algumas partes do adaptador, não do conjunto.
- **H9b:** o idioma do raciocínio é decidido pelas camadas finais.
- **H9c:** os MLPs concentram ganhos de capacidade, e o Mamba e a atenção concentram o dano.

**Desenho:** o adaptador `adalberto` (treinado com os 71 módulos juntos) é fundido ao Lumi **só em parte**, com o efeito dos demais módulos zerado (`scripts/merge_adapter_layers.py`, na CPU; 7 a 9 min de avaliação por variante, só com thinking). Cinco variantes: completo, camadas 0 a 20, camadas 21 a 41, sem MLP (Mamba e atenção) e só MLP.

| Parte mantida | Módulos | GSM8K | Múltipla escolha | Fecha `</think>` (MCQ) | Traço em português (MCQ) |
| --- | --- | --- | --- | --- | --- |
| Lumi original | 0 | 81,7% | 69,6% | 86,2% | 0% |
| adaptador completo | 71 | 83,3% (+1,6) | 62,3% ▼ (−7,3) | 86,8% | 99,4% |
| camadas 0 a 20 | 36 | 83,6% (+1,9) | 62,6% ▼ (−7,0) | 81,3% | **0%** |
| camadas 21 a 41 | 35 | 76,6% ▼ (−5,0) | **18,4%** ▼ (−51,2) | 22,9% | 98,9% |
| sem MLP | 37 | 81,7% (0,0) | 60,0% ▼ (−9,6) | 81,3% | 98,7% |
| **só MLP** | 34 | **83,9%** (+2,2) | **69,6%** (0,0) | 87,8% | **2,4%** |

(▼: o IC 95% da diferença contra o Lumi fica abaixo de 0. Intervalo da linha "só MLP" na múltipla escolha: −3,4 a +3,4.)

**Leituras:**
1. **Só os MLPs do adaptador** mantêm a múltipla escolha do Lumi e dão o ganho de matemática (+2,2; IC −0,3 a +4,7, **não** significativo sozinho), mas o raciocínio continua em inglês (2,4% em português).
2. **O dano na múltipla escolha vem do Mamba e da atenção**, e **não do idioma do traço**: nas camadas 0 a 20 o traço sai em inglês (0% em português) e a múltipla escolha perde os mesmos 7 pontos do adaptador completo. Isso enfraquece a explicação "o português em si causa a perda".
3. **O idioma do raciocínio parece ser decidido no fim da pilha**: só as camadas 21 a 41 produzem português (98,9%). Mas esse adaptador parcial **quebra o modelo** (fecha `</think>` em 22,9%), porque as camadas finais se ajustaram ao que as iniciais entregavam. Logo, a leitura "o fim carrega a tradução" é uma hipótese, não um resultado.

**Veredito provisório:** H9a apoiada; H9c apoiada **na triagem**, a verificar com treino; H9b apoiada em parte e confundida pela quebra do modelo. **Atualização:** o treino só com MLP (E10, abaixo) **não reproduziu** a triagem: H9a e H9c ficaram refutadas.

**Ameaças à validade:** (i) zerar módulos depois de treinar tudo junto **não equivale** a treinar só neles; o resultado "só MLP" pode refletir um adaptador de baixo efeito, que quase não muda o comportamento do Lumi (o raciocínio nem muda de idioma); (ii) o ganho de +2,2 na matemática tem IC que inclui zero; (iii) uma execução por variante, 385 itens (±4 a 5 pontos); (iv) o tamanho do efeito (e do dano) pode depender da interação entre blocos.

## E10. SFT só com MLP, treinado do zero (`adalberto_mlp`)

**Pergunta:** um adaptador com LoRA apenas nos 34 módulos de MLP, treinado desde o início, aprende o raciocínio em português **e** mantém a múltipla escolha do Lumi?

**Desenho:** mesmos dados (`adalberto`, 14.386 exemplos) e mesmos hiperparâmetros (r=16, alpha=32, lr 1e-4, 1 época, 450 passos); só muda `--targets mlp`, que reduz o adaptador de 71 para 34 módulos (**8,5 milhões** de parâmetros, metade do completo). Implementação: `TARGETS` em `peft_setup.py`, `--targets` em `train_sft.py`, 4 testes em `tests/test_peft_targets.py`. Treino: 450 passos, **23,9 min**, ~6,1 mil tokens/s, pico de 18,3 GiB; perda 0,47 → 0,21 (a do adaptador completo: 0,44 → 0,21). Adaptador de 8.529.920 parâmetros em 34 módulos.

**Previsões** (para poder refutá-las):

| Resultado | O que indicaria |
| --- | --- |
| Traço em português ≥ 90% **e** múltipla escolha ≥ 66% | O MLP sozinho aprende o idioma sem o dano: o Mamba e a atenção não são necessários para o português, e a hipótese H9c ganha força |
| Traço em inglês, múltipla escolha ≈ Lumi | O MLP não carrega o idioma: o português exige o Mamba ou a atenção (com o dano) |
| Traço em português, múltipla escolha ≈ 62% | O dano não era só do Mamba e da atenção: vem do português ou dos dados |

**Resultado** (validação limpa, nota estrita; ▲ ▼ = IC 95% da diferença contra o Lumi; avaliação de 12 min nos dois modos):

| | Lumi | `adalberto` (71 módulos) | **`adalberto_mlp`** (34 módulos) |
| --- | --- | --- | --- |
| GSM8K com thinking | 81,7% | 83,3% (+1,6) | **83,4%** (+1,8; IC −0,9 a +4,3) |
| GSM8K sem thinking | 77,2% | 79,3% (+2,1) | **80,2%** (+3,0; IC 0,0 a +5,9) |
| Múltipla escolha com thinking | 69,6% | 62,3% ▼ (−7,3) | **62,3%** ▼ (−7,3; IC −11,9 a −2,6) |
| Múltipla escolha sem thinking | 63,9% | 58,2% ▼ (−5,7) | **54,3%** ▼ (−9,6; IC −14,8 a −4,4) |
| Fecha `</think>` / trunca (múlt. escolha) | 86,2% / 13,8% | 86,8% / 13,8% | 84,7% / 15,8% |
| Traço em português (múlt. escolha) | 0% | 99,4% | **98,5%** |
| Mediana de tokens (múlt. escolha) | 490 | 1.176 | 1.147 |

Contra o `adalberto`, o `adalberto_mlp` empata em tudo, com ou sem thinking. Em múltipla escolha com thinking a diferença é 0,0 (IC −4,2 a +4,2); sem thinking, −3,9 (IC −8,6 a +0,8).

**Leitura:** caiu na terceira linha da tabela de previsões ("português, múltipla escolha ≈ 62%").
1. **Só o MLP aprende o português.** Com metade dos parâmetros (8,5 milhões contra 17), 98,5% dos traços saem em português, e a matemática ganha tanto quanto no adaptador completo. O Mamba e a atenção **não** são necessários para o idioma nem para o ganho em matemática.
2. **O dano na múltipla escolha continua.** 62,3%, o mesmo número do adaptador completo, e sem thinking até pior (54,3%, dentro do ruído do 58,2%). A hipótese H9c ("o dano está no Mamba e na atenção") **não se confirmou por treino**.
3. **A triagem pós-treino enganou.** Zerar os módulos do adaptador `adalberto` deixava o "só MLP" em 69,6% porque aquele adaptador quase não mudava o comportamento do Lumi (o traço seguia em inglês). Treinado sozinho, o MLP faz o trabalho todo e paga o mesmo preço. Era a ameaça (i) da triagem, e se confirmou.
4. **O que sobra como causa:** os dados (estilo e conteúdo dos traços do Qwen3-30B) ou o ato de ajustar o modelo a eles, **não** o lugar do adaptador nem o seu tamanho. Cinco mudanças de adaptador ou de otimização (lr, DoRA, escala, limite de tokens, tipo de bloco) deram 62 a 64%. Só o RL, que muda os dados, passou disso (66 a 68%).

**Veredito:** H9c **refutada** por treino; H9a (o dano vem de partes do adaptador) **refutada** na forma que a triagem sugeria. H10 (o português causa a perda) segue **aberta**: a evidência "contra" da triagem (camadas 0 a 20 com traço em inglês perdendo 7 pontos) perde força, porque a própria triagem se mostrou enganosa. O teste que decide é o par em inglês e português dos mesmos problemas (H21, pacote 3 da Thor).

**Ameaças à validade:** uma execução, 385 itens (±4 a 5 pontos); o 54,3% sem thinking não é distinguível do 58,2% do adaptador completo; o ganho de +3,0 sem thinking na matemática tem IC de 0,0 a +5,9, no limite.

## E11. Estudo de capacidade, tipo de bloco e posição (ENCERRADO em 08/11 13:25: 1 de 15 rodadas feita, 14 canceladas)

**Encerramento:** por decisão do usuário, as 14 rodadas restantes foram canceladas. O que já saiu (`cap_r04_std` e o E10) mostra que o tamanho e o tipo de bloco do adaptador não mudam a nota (62,3% nos três: r=16 completo, r=4 e só MLP); as rodadas restantes variariam o mesmo eixo e **não atacam o problema do projeto** (raciocínio em português na múltipla escolha). Fica abaixo o desenho original, como registro do que não foi feito. Código e testes continuam no repositório (`--rank-pattern`, `--alpha-pattern`, `--lr-mult`, `--layers`, `--targets`, `--rslora`).

**Pergunta:** depois do E10, o que no adaptador controla o preço pago na múltipla escolha: o **tamanho** (rank e escala), o **tipo de bloco** que recebe o adaptador, ou a **posição** (camadas)? Todas as rodadas usam os mesmos dados, `lr` e passos do `adalberto` (`adalberto`: r=16, alpha=32, escala 2, 71 módulos, 62,3%); muda **uma coisa por rodada**. Plano e comandos: `scripts/capacity_plan.txt` e `scripts/run_capacity.sh` (uma rodada leva ~35 a 40 min: ~25 de treino, ~12 de avaliação nos dois modos).

**Implementação** (tudo com padrão igual ao do PRD; 36 testes novos, 322 no total): `--r`, `--alpha`, `--rslora`, `--targets` (lista de tipos `mamba`, `mlp`, `attn`), `--layers A-B`, `--rank-pattern` e `--alpha-pattern` (rank e alpha por tipo de módulo, com a escala alpha/r mantida por módulo), `--lr-mult` (multiplicador do lr por grupo de parâmetros). Teste de fumaça: r=64 treina com pico de 20,6 GiB (68,2 M de parâmetros) e a fusão aceita o adaptador.

**Desenho** (14 rodadas; as 3 de escala isolada entram no fim da fila; ordem das primeiras 11: a ordem põe primeiro o que mais separa as hipóteses):

| Eixo | Rodada | Mudança contra o `adalberto` | Escala (alpha/r ou alpha/raiz r) | Módulos |
| --- | --- | --- | --- | --- |
| Capacidade | `cap_r04_std` | r=4, alpha 32 | 8 | 71 |
| Tipo | `type_nomlp` | só Mamba + atenção (o complemento do E10) | 2 | 37 |
| Capacidade | `cap_r64_rs` | r=64, RSLoRA, alpha 8 | 1 | 71 |
| Capacidade | `cap_r64_std` | r=64, alpha 32 | 0,5 | 71 |
| Capacidade | `cap_r04_rs` | r=4, RSLoRA, alpha 8 | 4 | 71 |
| Tipo | `type_attn`, `type_mamba` | só atenção; só Mamba | 2 | 16; 21 |
| Posição | `pos_L00_20`, `pos_L21_41` | só camadas 0 a 20; só 21 a 41 | 2 | 36; 35 |
| Tipo | `type_mlp_attn`, `type_mlp_mamba` | MLP + atenção; MLP + Mamba | 2 | 50; 55 |
| Capacidade | `cap_r32_a16` | r=32, alpha 16 (r > alpha); com `sc_a08` e `cap_r64_std` dá o rank 16, 32 e 64 na mesma escala 0,5 | 0,5 | 71 |
| Escala (r=16 fixo) | `sc_a08`, `sc_a16`, `sc_a64` | só o alpha: 8 (r > alpha), 16 (r = alpha, padrão do PEFT), 64 | 0,5; 1; 4 | 71 |

As duas "leis de escala" passam pelo ponto do `adalberto` em r=16 (escala 2). Com alpha fixo, a lei padrão (`std`) dá escala alpha/r e mantém o tamanho total da atualização constante quando o rank muda, por isso "mais rank" nela quase não ajuda; a RSLoRA (`rs`) usa alpha/raiz(r), que o método defende como a escala estável. **Um rank isolado não distingue as duas** (basta trocar o alpha); a diferença só aparece comparando ranks, por isso r=4 e r=64 entram nas duas leis.

**Hipóteses e previsões** (escritas antes dos resultados):

| ID | Hipótese | Se verdadeira | Se falsa (nula) |
| --- | --- | --- | --- |
| H26 | O dano na múltipla escolha cresce com a capacidade do adaptador ("aprende mais, esquece mais", Biderman et al. 2024) | r=4 recupera a múlt. escolha (≥ 66%) mantendo ≥ 90% de português; r=64 fica ≤ 62% | Todos os ranks em 62 ± 3%: a capacidade é irrelevante e o dano vem dos dados |
| H27 | A lei de escala importa em rank alto: no r=64 a RSLoRA aprende mais que a padrão | `cap_r64_rs` com perda de treino menor e mais próxima do base que `cap_r64_std` (escala 0,5 pode nem chegar a 90% de português) | As duas empatam em perda e nota |
| H32 | O preço na múltipla escolha depende da **escala** alpha/r, não do rank | `sc_a08`, `sc_a16`, `sc_a64` (r=16 fixo): se a nota segue a escala e os ranks 4 e 64 ficam perto do r=16 de mesma escala, a escala manda | Escala menor (0,5) ≈ 64%+ mantendo ≥ 90% de português; escala maior (4) pior que 62% | As três ficam em 62 ± 3%: nem escala nem rank mudam a nota |
| H28 | O dano não depende do tipo de bloco (corolário do E10) | `type_nomlp`, `type_attn` e `type_mamba` ficam em ~62% quando aprendem português | Algum tipo sozinho mantém ≥ 66% com ≥ 90% de português: esse tipo é o caminho |
| H29 | O adaptador mínimo (atenção só, 16 módulos) já ensina o idioma | `type_attn` com ≥ 90% de português | `type_attn` fica em inglês: o idioma exige o Mamba ou o MLP |
| H30 | O idioma é decidido nas camadas finais (H9b, agora com treino) | `pos_L21_41` ≥ 90% de português e `pos_L00_20` bem abaixo | Os dois metades ensinam português, ou nenhuma |

**Regra de leitura** (contra a maldição do vencedor: 11 rodadas, 385 itens, ±4 a 5 pontos, uma semente): só conto como **sinal** uma diferença de pelo menos +4 pontos contra o `adalberto` na múltipla escolha com thinking **e** no mesmo sentido sem thinking; qualquer rodada que a cumprir é repetida com outra semente antes de entrar em conclusão. Diferenças menores são tratadas como empate.

**Próxima camada (a definir depois das rodadas acima):** hiperparâmetros **por tipo de bloco**: rank por tipo (`--rank-pattern`, ex.: MLP r=32 e Mamba e atenção r=4) e lr por tipo (`--lr-mult`, ex.: Mamba x0,2). Só faz sentido escolhê-los depois de ver quais tipos importam; as opções já estão implementadas e testadas.

**Resultados:** na tabela gerada ([resultados_auto.md](resultados_auto.md), grupos "Capacidade", "Tipo de bloco" e "Posição"), que a fila regera a cada rodada; a leitura entra aqui quando o conjunto fechar.

**Resultados parciais (atualizados a cada rodada):**
- `cap_r04_std` (r=4, alpha 32, escala 8; 4,3 milhões de parâmetros, um quarto do `adalberto`): perda final 0,206 (a do `adalberto`: 0,21); português 98,8%; GSM8K 82,5% com e 81,2% sem thinking (+4,0 ▲); múltipla escolha **62,3% ▼ com thinking** (idêntica ao r=16; diferença contra o `adalberto` 0,0, IC −3,9 a +3,9) e 60,0% sem thinking (−3,9 contra o Lumi, sem significância). Contra H26: **um quarto da capacidade paga o mesmo preço**, o que aponta para os dados, não para o tamanho do adaptador. Falta r=64 para fechar o outro lado.

