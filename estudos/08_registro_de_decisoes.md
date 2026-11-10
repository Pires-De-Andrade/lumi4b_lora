# Registro de decisões (08/11/2026)

Cada decisão de rumo tomada com o RL em andamento, a evidência na hora, o que ficou de fora e **o que estudar se der errado**. Serve para voltar a elas depois: nenhuma foi tomada com resultado de teste, e várias usam limiares que foram palpites. Os números são de validação limpa, com thinking, salvo indicação.

## Regras de decisão fixadas antes do resultado (fixadas às 19:30)

Valem para a avaliação do ciclo 4 do `gspo_gold4k` (esperada ~19:45), comparada com o `gspo1` no ciclo 4 (**67,8%** de múltipla escolha com thinking; Lumi 69,6%).

| Resultado no ciclo 4 | O que fazer |
| --- | --- |
| Múlt. escolha com thinking **≥ ~72%** | Priorizar a segunda semente do melhor e a suíte de teste (com thinking) do melhor candidato |
| **~67%** (sem passar o `gspo1`) | O teto não está no ponto de partida nem no limite: priorizar os ajustes do GSPO (passos por lote, temperatura no treinador, média uniforme) e o SFT misto com o Qwen |
| **< ~62%** ou fecha o `</think>` em **< ~87%** | Parar o RL e replanejar (guardar o último adaptador, o log e as estatísticas por ciclo) |

**De onde vêm os limiares (palpites, sem calibração prévia):** 72 = 67,8 do `gspo1` mais os 4 pontos da regra de sinal; 62 = o ponto de partida `adalberto` (62,3); 87 = o fechamento do Lumi (86,2). O ruído de uma execução na múltipla escolha é de ±4 a 5 pontos (385 itens, uma semente; erro padrão ~2,4). **Riscos se der errado:** (1) uma parada por um valor ruidoso do ciclo 4 pode matar uma rodada boa (o `gspo_thor` foi de 65,7% no ciclo 4 para 67,0% no 8); (2) o limiar de 87% de fechamento vem de outra distribuição (o Lumi não passou por RL). **O que estudar:** a curva por ciclo do `gspo_gold4k` (`scripts/grpo_status.py`), o `gspo1_c4` e o `gspo_thor_it4` contra os finais, e o IC pareado do ciclo 4 contra o `gspo1_c4`.

**Tempo:** triagens de 4 ciclos no lugar de 8 (o ciclo 4 foi quase igual ao final: `gspo1` 67,8 para 67,5; `gspo_thor` 65,7 para 67,0), e os itens "outro ponto de partida de SFT" e "revisão humana" são os mais cortáveis. Tudo com 8 ciclos somaria ~17 h de 4090; com triagens, ~11 h (estimativas).

## Decisões do dia

| Hora | Decisão | Por quê (evidência na hora) | Ficou de fora | Se der errado, estudar |
| --- | --- | --- | --- | --- |
| 16:55 | O RL `gspo_gold` passa para antes da suíte do `gspo1` | A suíte do Lumi já tinha tomado 1h40 e travava o RL | Suíte do `gspo1` primeiro (plano original) | Nada a estudar: ordem de execução |
| 17:05 | Parar a suíte do Lumi logo depois do MMLU-ProX de teste com thinking | A 4090 só tinha o Lumi rodando; o RL é o experimento que decide a direção; perder só o começo do sem thinking | Matar na hora (perdia 1h45); deixar terminar (RL só às 19:00) | O Lumi ainda precisa de MMLU-ProX sem thinking, ENEM, BLUEX e OAB; rodar na mesma máquina (4090) para a comparação ser válida |
| 18:02 | Interromper o `gspo_gold` (limite 2.048) no ciclo 2 e refazer como `gspo_gold4k` (limite 4.096) | O limite de 2.048 nunca foi testado e é comum às três rodadas de RL (todas em ~67%); no ciclo 0, 36% das respostas de múlt. escolha truncavam (52% na WebInstruct) e truncada vale 0 | Seguir até o ciclo 4 e decidir com a validação | O `gspo_gold` tem os ciclos 0 e 1 com **as mesmas questões** do `gspo_gold4k`: comparar recompensa, truncadas e grupos por fonte. A mudança soma duas diferenças contra o `gspo1` (questões e limite); separar com o `gspo1` a 4.096 |
| 18:05 | Manter o `adalberto` como ponto de partida do `gspo_gold4k` | Só muda questões e limite; o `adalberto` fecha mais depois do RL (93,2% contra 87,8% do `thor_pt`) | Partir do `thor_pt` | A triagem do `thor_pt` (abaixo) |
| 18:20 | Adiar a suíte de teste do `gspo1`; outros testes antes | Com o limite maior há testes mais informativos; o `gspo1` não é finalista | Suíte automática depois do RL | A validação pode não representar ENEM, BLUEX e OAB; por isso a suíte do melhor candidato (só com thinking) entra cedo, só como checagem, nunca para escolher candidato (regra do projeto: teste fora da seleção) |
| ~18:25 | O adaptador só liga com `enable_thinking=True`; sem thinking roda o Lumi original | Qualquer ganho com thinking é bem-vindo; a queda sem thinking passa a não pesar no uso | Um adaptador único nos dois modos | O BF16 fundido sozinho degrada o sem thinking: o relatório precisa dizer isso. Isto muda **como** o critério da queda de 1 ponto é atendido (por construção), não o critério. Conferir que adaptador desligado dá exatamente o Lumi |
| 18:50 | Benchmark de velocidade da geração, depois do RL; adotar nos RLs seguintes só o que ganhar ≥15% em tokens por segundo com recompensa e tamanho iguais dentro do ruído | A geração é 77% do ciclo; ajustes propostos por revisão externa, nenhum testado | Mexer no vLLM durante o RL | Os kernels diferentes não dão saídas idênticas: se a nota mudar depois, comparar com a base do benchmark (`results.jsonl`) |
| ~19:05 | Tirar o `thor_pt_math` da lista | O problema agora é a múltipla escolha; o GSM8K já passa o Lumi (82 a 84% contra 81,7%) | SFT só com a matemática do Nemotron | O que ele mediria é a múlt. escolha: diz se a perda do `thor_pt` vem do professor ou da ciência misturada. Os dados ficam prontos |
| 19:20 | Triagem do RL a partir do `thor_pt` com 4.096, 4 ciclos (`gspo_gold4k_thor`) | Com 4.096 a cauda longa do Nemotron cabe no RL; compara o ponto de partida; o `adalberto` não tem licença declarada | Refazer o SFT com limite de 4.096 (só 269 dos 5.278 exemplos passam de 2.048 tokens) | Comparar com o `gspo_gold4k` no ciclo 4 e com o `gspo_thor` antigo; se piorar, a causa pode ser o ponto de partida (laços, 34% de truncadas) e não o limite |
| 19:30 | Regras de decisão acima | Evitar mudar a meta depois de ver o resultado | Decidir caso a caso | Ver a seção acima |
| 22:20 | Meta mínima de fechamento do `</think>` passa de ≥ 98% para ≥ 90% (emenda no PRD) | A meta de 98% estava acima do que o próprio Lumi fecha (86,2%) e do melhor RL (93,2%); 90% mede o comportamento sem exigir fechamento quase perfeito | Manter 98% | **Foi decidida depois de conhecidos os resultados** (o `gspo1`, 93,2%, passa a cumprir), contra o espírito da regra 4: citar sempre os dois limiares nos relatórios e dizer quem passa em cada um. A meta alvo (≥ 99,5%) não mudou |

## Resultado das regras (19:40): acionada a de parada

Ciclo 4 do `gspo_gold4k`: múlt. escolha com thinking **63,1%** (−4,7 contra o `gspo1_c4`, IC −9,4 a +0,0; −6,5 contra o Lumi, IC −11,2 a −1,8), fecha o `</think>` em **83,9%** (< 87%), truncadas 16,4% (`gspo1_c4`: 10,6%). Parou pela regra, como combinado, às 19:41; o adaptador do ciclo 3 (o avaliado) ficou guardado e a rodada é retomável. Detalhes e hipóteses: `discovery.md` §20.37. Pontos para estudar, como pedido:

- **A regra disparou pelo fechamento, não pela nota** (63,1% > 62%). Os dois limiares eram palpites; o do fechamento (87%) foi tomado do Lumi, que não passou por RL, e o `gspo1_c4` tinha 90,9%.
- **A causa provável é laço:** 39 dos 63 itens truncados mostram repetição forte, contra 11 dos 41 do `gspo1_c4`; 30 dos 51 itens que só o `gspo1_c4` acerta estão truncados no `gspo_gold4k`.
- **Duas mudanças ao mesmo tempo** (questões e limite): a decisão de 18:02 de refazer com 4.096 já tinha esse custo anotado. O teste de separação mais barato é retomar o `gspo_gold` de 2.048 até o ciclo 4 (~50 min).
- **A triagem do `thor_pt` com 4.096 (19:20) foi cancelada às 19:45** (waiter), por repetir a configuração suspeita; reativar conforme o teste de separação.

## Replanejamento (19:55) e regras do próximo passo

Depois da parada do ciclo 4 do `gspo_gold4k`: o benchmark de geração segue (termina ~20:05); em seguida a sessão `gold2048` retoma o `gspo_gold` (limite 2.048, as mesmas questões do gold, ciclos 0 e 1 prontos) até o ciclo 4 e avalia (~50 min: 2 ciclos de ~18 min e a avaliação de ~13 min). A pasta `gspo_gold_c2048_abortado` voltou a se chamar `gspo_gold`. A triagem do `thor_pt` a 4.096 continua cancelada.

**Regra para ler o resultado, fixada antes de ver o número (limiares estimados, sem calibração prévia):** `X` = múlt. escolha com thinking na validação e `C` = fecha `</think>` e responde, do `gspo_gold` no ciclo 4.

| `gspo_gold` (2.048) no ciclo 4 | Leitura | Próximo passo |
| --- | --- | --- |
| `X ≥ 67%` e `C ≥ 90%` | O limite de 4.096 é o culpado (mais laços) | Abandonar o 4.096; retomar o `gspo_gold` até o ciclo 8 (~1,2 h); a triagem do `thor_pt` roda a 2.048 com as questões do gold |
| `X ≤ 64%` ou `C < 87%` | As questões do gold são o problema (ou ao menos não ajudam) | `gspo1` com 4.096 e as questões antigas (1,5 h) para ver se o limite importa nelas; subir a prioridade do SFT misto com o Qwen e investigar os laços da MedMCQA; o gold não vira receita |
| entre os dois | Ambíguo | Mais ciclos do `gspo_gold` até o 8 e uma segunda semente antes de concluir |

Riscos se der errado: os limiares são palpites (o mesmo erro padrão de ~2,4 pontos vale); a comparação entre `gspo_gold` e `gspo_gold4k` isola o limite só se o ruído de uma execução não for grande (385 itens, uma semente); o ciclo 4 não é o final (no `gspo_thor` foi de 65,7% para 67,0% no ciclo 8).


## Resultado do `gspo_gold` (2.048) no ciclo 4 (20:51): ramo B

Validação limpa, com thinking; modelo que entra na iteração 4 (4 atualizações), avaliação em `runs/eval/gspo_gold_it4`.

| | GSM8K | Múlt. escolha | Fecha `</think>` e responde | Trunca | Mediana de tokens |
| --- | --- | --- | --- | --- | --- |
| Lumi | 81,7% | 69,6% | 86,2% | 13,8% | 490 |
| `gspo1_c4` (questões antigas, 2.048) | 84,3% | 67,8% | 90,9% | 10,6% | 1.086 |
| `gspo_gold` c4 (questões do gold, 2.048) | 85,2% ▲ | **63,9%** ▼ (−5,7; IC −10,1 a −1,3) | 89,4% | 11,4% | 1.167 |
| `gspo_gold4k` c4 (questões do gold, 4.096) | 84,6% ▲ | 63,1% ▼ (−6,5) | 83,9% | 16,4% | 1.250 |

- **Regra:** `X = 63,9% ≤ 64%` → ramo B (as questões do gold são o problema, ou ao menos não ajudam). `C = 89,4%` está entre 87% e 90%.
- **Comparações pareadas (múlt. escolha):** `gspo_gold` contra `gspo1_c4` (mesmo limite, só as questões): −3,9 (IC −7,8 a +0,0); `gspo_gold4k` contra `gspo_gold` (mesmas questões, só o limite): −0,8 (IC −4,7 a +3,1). O limite de 4.096 não muda a nota, mas o fechamento cai 5,5 pontos (mais laços). As questões do gold ficam ~4 pontos abaixo das antigas, no limite do significativo.
- **Cuidados:** a regra disparou por 0,1 ponto abaixo do limiar de 64% (limiar estimado, sem calibração); uma semente; as questões do gold diferem das antigas em domínio, fonte e rótulo (WebInstruct é verificado por modelo), então "as questões" não isola o rótulo. **O que estudar se o `gspo1_4k` também cair:** aí o limite de 4.096 sozinho é suspeito, e a comparação com `gspo_gold4k` deixa de ser sobre as questões.
- **Próximo passo, como a regra prescreve:** `gspo1_4k` (em andamento); subir a prioridade do SFT misto; investigar os laços da MedMCQA; o gold não vira receita. A triagem do `thor_pt` a 2.048 com as questões do gold, do ramo A, não será feita.

## Fila da noite (versão 4, 23:23)

Mudanças em relação à versão 3: (1) entram mais dois testes do diagnóstico de log-probabilidades (§20.48 do `discovery.md`), `gspo1_eps3x` e `gspo1_same`; (2) a **suíte de teste do Lumi fica por último de todos**, também depois do trabalho com o pacote 4 da Thor (traços de múltipla escolha do Qwen), que só chega por volta de 08:50. Para a GPU não ficar parada até lá, a suíte roda como preenchimento e **sai da frente quando o pacote aparece** (`data/thor4/thor4_out.tar.gz`, arquivo parado há mais de 60 s); ela é retomável por (prova, modo), então volta ao fim. `scripts/run_noite4_chain.sh` (sessão `noite4`) substituiu a `noite3` na troca do SFT `adalberto_dft` para o `gspo1_T06`, sem perda de trabalho.

| Ordem | Etapa | O que mede | Tempo estimado |
| --- | --- | --- | --- |
| 1 | SFT `adalberto_dft` | **feito**: 61,3%, sem ganho, e o traço sai em inglês (ver `estudos/07`, H44) | – |
| 2 | RL `gspo1_T06` (logits / 0,6) | efeito da temperatura nos logits do treinador | ~1,5 h |
| 3 | RL `gspo1_s1` (semente 1) | variação entre rodadas | ~1,5 h |
| 4 | RL `gspo1_eps3x` (corte do GSPO 3×) | se o corte atual, menor que o passo real, limita o RL | ~1,5 h |
| 5 | RL `gspo1_same` (log-probabilidades antigas na composição do treino) | se o ruído do primeiro passo atrapalha | ~1,5 h |
| 6 | RL `gspo1_4k` | limite 4.096 nas questões antigas (ramo B) | ~1,6 h |
| 7 | Suíte do Lumi como preenchimento, até o pacote 4 chegar | referência | até ~08:50 |
| 8 | Trabalho com o pacote 4 (`scripts/run_qwen_mix.sh`, a escrever) | SFT misto com o Qwen e triagem de RL | depende |
| 9 | Suíte do Lumi, até o fim | referência | o que faltar |

Todas as rodadas de RL das etapas 2 a 6 usam a receita e os prompts do `gspo1_c4` (67,8%) e mudam uma coisa de cada vez (no `gspo1_s1` a semente também sorteia outros prompts); `gspo1_s1` dá o tamanho do ruído de uma execução para ler as outras. **Etapa 8, o trabalho com o pacote 4 (`scripts/run_qwen_mix.sh`, escolhas padrão a revisar com o resultado do pacote):** extrai o pacote em `data/thor4/e1`, confere o relatório da Thor (mínimo de 2.000 mantidos), converte (`prepare_thor4.py`), passa o filtro de contaminação e monta `data/qwen_mix.parquet`: 5.800 problemas de matemática do Qwen (`data/adalberto_think.jsonl`, um traço por problema, semente 0) + os traços de múltipla escolha do pacote (até 2.048 tokens por exemplo) + as mesmas 5.544 linhas sem raciocínio do GigaVerbo. Só professor Qwen: os dados do Nemotron ficam de fora porque os três SFTs com eles não passaram do `adalberto`. Depois: SFT `qwen_mix` (receita do `adalberto`, avaliação com até 3 tentativas) e RL `gspo_qwen_mix` (receita do `gspo1`, 4 ciclos). ~2,5 h no total. **Se der errado, estudar:** a proporção matemática para múltipla escolha (5.800 contra o que o pacote render) é palpite; o rendimento do Qwen em questões de cálculo longo da WebInstruct é desconhecido (viés para as fáceis); as questões do pacote têm rótulo de MedMCQA (banca) e WebInstruct (verificado por modelo).

**Se der errado, estudar (etapas 2 a 5):** o `gspo1_same` só muda algo se o ruído vier da composição do batch e não do modo treino contra avaliação (o |log s| do primeiro passo, registrado no `train_log.jsonl`, diz isso nos primeiros minutos). A etapa 8 só roda com o arquivo `data/thor4/thor4_out.tar.gz` parado há mais de 60 s.


## Fila da noite (versão 5, 09/11 01:22)

**Mudança:** a fila da versão 4 foi congelada (SIGSTOP no script, sem tocar no treino) às 01:22, com o `gspo1_s1` no ciclo 2 de 4. A versão 5 (`scripts/run_noite5_chain.sh`) espera o `gspo1_s1` terminar, encerra a versão 4 e roda **antes dos diagnósticos de RL** o teste parcial `qwen_mix_p` (SFT mais RL com a entrega parcial do pacote 4). A suíte do Lumi continua por último de todas.

**Por quê:** a entrega parcial do pacote 4 (3.424 questões geradas, 2.191 mantidas, rendimento 64,0%, sha256 `45e91642…`) repete a primeira entrega (63,4%, mediana de 1.100 tokens, 99,8% dos traços em português, 14% truncados, contra 44,6% e 43,6% do pacote 3). Passa o mínimo de mantidos e antecipa em ~5 h a leitura do teste que mais importa (múltipla escolha do mesmo professor do `adalberto`). A entrega completa, prevista para ~09:00, entra numa segunda rodada (`qwen_mix`).

| Ordem | Etapa | O que mede |
| --- | --- | --- |
| 0 | `gspo1_s1` (em curso, fim ~02:15) | variação entre rodadas |
| 1 | SFT `qwen_mix_p` e RL `gspo_qwen_mix_p` | efeito de acrescentar traços de MCQ do Qwen ao `adalberto` |
| 2 | RL `gspo1_eps3x`, `gspo1_same`, `gspo1_4k` | diagnósticos de RL (o pacote final entra na frente quando chegar) |
| 3 | Pacote 4 final (`qwen_mix`, `gspo_qwen_mix`) e, por último, suíte do Lumi | resultado final do pacote e referência |

**Mistura do `qwen_mix_p`** (`scripts/run_qwen_mix.sh`, `scripts/concat_parquets.py`): `data/train_adalberto.parquet` como está (5.842 traços de matemática de até 1.024 tokens, 3.000 + 5.544 linhas sem raciocínio) mais 1.810 traços de MCQ do pacote (1.141 MedMCQA, 669 WebInstruct; até 2.048 tokens; 1 removido pelo filtro de 8-gramas; 380 acima de 2.048), 16.196 linhas. O desenho anterior (matemática de até 2.048 tokens, sem as 3.000 linhas do `adalberto`) diferia do `adalberto` em três pontos além da MCQ; foi trocado para que a comparação isole o efeito dos traços de MCQ.

**O que ler com cuidado:** (1) a letra E aparece em 2,4% dos traços (A 27,5%, B 27,0%, C 24,9%, D 17,8%), contra ~20% nas provas de 5 alternativas; conferir a distribuição das letras previstas na validação; (2) o gabarito da WebInstruct é verificado por modelo; (3) os mantidos são as questões que o professor acertou, o que favorece itens fáceis; (4) todo traço começa com "Vou raciocinar em português." (começo forçado da geração), o que tende a ajudar a taxa de traços em português.


## Fila da versão 6 (09/11 13:56), depois da queda da máquina

**O que aconteceu:** a máquina ficou fora do ar de ~02:45 a 12:48 (uptime reiniciado às 12:48; sem acesso aos logs do kernel, causa não identificada). A queda veio quando a avaliação do SFT `qwen_mix_p` começava, logo depois do SFT (507 passos, 1.903 s) e da mesclagem; o modelo mesclado está íntegro (mesmo tamanho do `adalberto`, 263 tensores). Perda de ~11 h de GPU. A 4090 é compartilhada e, na volta, outro usuário a ocupou (uso de 97–99%); as etapas da fila esperam a GPU ficar com menos de 1 GiB em uso.

**Mudança:** a versão 5 foi encerrada (só esperava a GPU, sem treino em curso) e a versão 6 (`scripts/run_noite6_chain.sh`) a substitui. O pacote 4 completo chegou (9.927 questões geradas, 6.261 mantidas, rendimento 63,1%, sha256 `72ca5a4b…`), então **o RL do teste parcial foi cancelado**: o `qwen_mix_p` só é avaliado (SFT, sem RL) e o trabalho passa direto ao `qwen_mix` completo.

| Ordem | Etapa | O que mede |
| --- | --- | --- |
| 1 | Avaliação do SFT `qwen_mix_p` (sem RL) | efeito de 1.810 traços de MCQ do Qwen sobre o `adalberto` (62,3%) |
| 2 | SFT `qwen_mix` (completo), avaliação e RL `gspo_qwen_mix` | efeito de 5.055 traços de MCQ, e o RL a partir dele |
| 3 | RL `gspo1_eps3x`, `gspo1_same`, `gspo1_4k` | diagnósticos de RL, lidos contra a faixa de 64,4 a 67,8% das duas sementes do `gspo1` |
| 4 | Suíte do Lumi, até o fim | referência, por último de todas |

**Resultado da etapa 1, SFT `qwen_mix_p` (14:41), validação limpa, com thinking:** múltipla escolha **61,8%** (−0,5 contra o `adalberto`, IC −4,7 a +3,6; −7,8 contra o Lumi), fecha 83,1% (86,8% no `adalberto`), trunca 17,7%, GSM8K 84,5% (+1,2), múltipla escolha sem thinking 61,6% (58,2%). **Leitura:** 1.810 traços de MCQ do Qwen sobre o `adalberto` não mudam a múltipla escolha com thinking (dentro do ruído de ±4 pontos) e o fechamento fica 3,7 pontos pior (também dentro do ruído). As letras previstas seguem o gabarito (E 2,2% contra 2,1%). **Se der errado, estudar:** o `qwen_mix` (5.055 traços) e o RL a partir dele seguem na fila; se ele também ficar em torno de 62%, os traços de MCQ do Qwen não melhoram o SFT e o ganho, se vier, vem do RL; a letra E rara nos traços só se mede nas provas de 5 alternativas (suíte de teste).

**Resultado da etapa 2, SFT `qwen_mix` (15:39), validação limpa, com thinking:** múltipla escolha **62,1%** (−0,3 contra o `adalberto`, IC −4,7 a +4,2; −7,5 contra o Lumi), fecha 83,9% (86,8%), trunca 16,9%, GSM8K 83,4% (+0,1), múltipla escolha sem thinking 62,9% (58,2%); letras previstas sem viés (E 1,9% contra 2,1% do gabarito). **Leitura:** com 5.055 traços de MCQ do Qwen o resultado é o mesmo do `qwen_mix_p` (1.810): 61,8%, 62,1% e 62,3% do `adalberto` não se distinguem; não há efeito de dose no SFT, e o fechamento do `</think>` fica ~3 pontos abaixo do `adalberto`. O RL `gspo_qwen_mix` (início 15:39, ~1,5 h) lê-se contra a faixa de 64,4 a 67,8% das duas sementes do `gspo1`. **Se der errado, estudar:** se o RL ficar na faixa do `gspo1`, os traços de MCQ do Qwen não acrescentam nada nem no SFT nem depois do RL, e o teto da receita (SFT mais RL com este gabarito) está abaixo do Lumi (69,6%).

**Resultado do RL `gspo_qwen_mix` (16:59), validação limpa, com thinking:** múltipla escolha **64,4%** (−3,4 contra o `gspo1_c4`, IC −7,5 a +0,8; −5,2 contra o Lumi, IC −9,9 a −0,8), igual ao `gspo1_s1`; fecha 87,8% (90,9% no `gspo1_c4`), trunca 12,5%, GSM8K 84,0% (+2,4 contra o Lumi, IC −0,1 a +5,0), múltipla escolha sem thinking 65,7%. **Leitura:** o RL a partir do `qwen_mix` fica dentro da faixa de 64,4 a 67,8% do `gspo1`; os traços de MCQ do Qwen não acrescentam nada nem no SFT (62,1%) nem depois do RL. **Se der errado, estudar:** a faixa vale como teto da receita "SFT em português mais RL com este gabarito"; as saídas que não fecham são em 53 a 98% laços de repetição (`estudos/09`, §2), e o diagnóstico `base_ptprefill` (H56) separa o custo do idioma do custo do treino.

**Mistura do `qwen_mix`:** `data/train_adalberto.parquet` como está (14.386 linhas) mais 5.055 traços de MCQ do pacote (2.752 MedMCQA, 2.303 WebInstruct; até 2.048 tokens; 6.260 passaram do filtro de 8-gramas, 1 removido), 19.441 linhas e 13,5 milhões de tokens. Distribuição das letras nos mantidos: A 28,1%, B 25,3%, C 24,5%, D 18,8%, E 3,0%, outras 0,3%; a letra E continua rara (as provas de 5 alternativas têm ~20%), então a distribuição das letras previstas na validação deve ser conferida.


## Fila da versão 7 (09/11 15:46)

**Mudança:** a versão 6 foi congelada (SIGSTOP no script, sem tocar no RL) com o `gspo_qwen_mix` no ciclo 0; a versão 7 (`scripts/run_noite7_chain.sh`) espera o RL terminar (processo `run_qwen_mix.sh`) e encerra a versão 6. Os RLs `gspo1_eps3x`, `gspo1_same` e `gspo1_4k` saem da fila (adiados): são rodadas únicas e o ruído de 3 a 4 pontos entre sementes (§20.51) só deixa ler efeitos grandes.

**Por quê:** os SFTs de seis variantes de dado, dose e lr ficam em 61 a 63% de múltipla escolha, e o acerto entre os que fecham fica em 72 a 76% contra 80,7% do Lumi (`discovery.md` §20.53). O RL conserta o fechamento (87 a 91%) mas não esse acerto. Duas hipóteses de custo baixo separam as causas.

| Ordem | Etapa | O que mede |
| --- | --- | --- |
| 0 | RL `gspo_qwen_mix` (em curso, fim ~17:00) | RL a partir do `qwen_mix` contra a faixa de 64,4 a 67,8% do `gspo1` |
| 1 | Dados dos dois SFTs (CPU) | `scripts/prepare_sft_curtos.sh` |
| 2 | SFT `qwen_mix_short` (adalberto + MCQ de até 1.024 tokens) | H53: traços curtos reduzem os laços |
| 3 | SFT `selfdist_min` (3.000 traços do próprio Lumi, ~94 passos) | H54: quantidade de SFT contra idioma |
| 4 | Suíte do Lumi, como preenchimento interrompível, por último | referência; os RLs a partir dos SFTs de teste entram interrompendo a suíte |

**Dados:** `qwen_mix_short` = `train_adalberto.parquet` (14.386 linhas) + 2.221 traços de MCQ de até 1.024 tokens (1.517 MedMCQA, 704 WebInstruct; 4.039 dos 6.260 passam de 1.024 tokens), 16.607 linhas, 9,6 milhões de tokens. `selfdist_min` = 3.000 traços do `selfdist` (um por problema, até 1.024 tokens), 1,6 milhão de tokens, sem linhas sem raciocínio. **Ordem de leitura:** o teste 2 primeiro, depois o 1.

**Regra de decisão para os RLs seguintes** (escrita antes de ver os resultados): RL a partir do `qwen_mix_short` se fechar ≥ 87% com múltipla escolha ≥ 62%; RL a partir do `selfdist_min` se o acerto entre os que fecham for ≥ 78% e o traço em português ≥ 90%.


## Fila da versão 8 (09/11 16:58)

**Mudança:** a versão 7 (que só esperava o RL terminar) foi encerrada e a versão 8 (`scripts/run_noite8_chain.sh`) a substitui; o RL `gspo_qwen_mix` terminou às 16:59 e a versão 8 assumiu às 17:00. Entram as sopas de pesos e o diagnóstico do Lumi com começo forçado em português, antes dos SFTs de teste.

| Ordem | Etapa | O que mede |
| --- | --- | --- |
| 1 | Avaliação das sopas `gspo1_soup2` e `gspo1_soup3` (montadas na CPU por `scripts/prepare_soups.sh`) | H55: a média dos pesos das rodadas de RL contra a faixa de 64,4 a 67,8% |
| 2 | `base_ptprefill` (Lumi, `--prefill "Vou raciocinar em português."`) | H56: custo de pensar em português sem treino |
| 3 | SFT `qwen_mix_short` | H53: traços curtos reduzem os laços |
| 4 | SFT `selfdist_min` e `selfdist_300` | H54: quantidade de SFT contra idioma (3.000 e 300 traços) |
| 5 | Suíte do Lumi, como preenchimento interrompível, por último | referência |

**Por quê:** levantamento de literatura em `estudos/09`. O custo de pensar em português é conhecido (XReasoning, Bode Reasoning, M-Thinker no 4B deixa ~5,5 pontos abaixo do inglês); os laços de repetição explicam 53 a 98% das respostas que não fecham, inclusive no Lumi (94%).

**Regra de decisão para os RLs seguintes** (a mesma da versão 7): RL a partir do `qwen_mix_short` se fechar ≥ 87% com múltipla escolha ≥ 62%; RL a partir do `selfdist_min` ou do `selfdist_300` se o acerto entre os que fecham for ≥ 78% e o traço em português ≥ 90%. **Proposta ainda sem decisão:** fase anti-laço (H57) sobre o melhor candidato.


## Fila da versão 15 (10/11 02:55, emendada às 10/11 03:03): o Lumi na 4090 nas provas que faltam, depois da `noite14`

**Contexto:** o primeiro resultado da suíte de teste do `thor_pt_antidoom_of` (GSM8K-PT, 01:45 de 10/11) deu 90,35%. Contra o Lumi sem thinking, referência do PRD para o GSM8K-PT (meta mínima +3,0 com IC acima de 0), a diferença é **+5,0 (IC +3,3 a +6,7)**: meta cumprida. Contra o Lumi com thinking, que entra na Nota de raciocínio, é +1,39 (IC −0,23 a +3,01) com o Lumi rodado na Thor e +1,62 (IC 0,00 a +3,24) com o Lumi rodado na 4090. O Lumi rodado nas duas máquinas tem a mesma média, mas discorda em 57 dos 1.295 itens (4,4%); o IC pareado não inclui esse ruído de numérica. A suíte do Lumi disponível para ENEM, BLUEX e OAB é só a da Thor (`lumi_suite_base_thor.tar.gz`, 09/11, em `runs/eval/base_thor/`); na 4090 o Lumi tem só o GSM8K (os dois modos) e o MMLU-ProX de teste (com thinking).

**Desenho:** a versão 15 (`scripts/run_noite15_chain.sh`) espera a sessão `noite14` acabar e roda o Lumi original (`base`) na 4090 em MMLU-ProX PT (sem thinking), ENEM, BLUEX e OAB (os dois modos). Custo: ~9 milhões de tokens com thinking e ~6 milhões sem (contagem da Thor) a ~2.900 tok/s, ~1,5 a 2 h. Se a suíte do `thor_pt_antidoom_of` estiver completa, gera `runs/jobs/noite15_comparacao.md` com duas tabelas de `report_compare.py --mean`, contra o Lumi com thinking e contra o Lumi sem thinking, ambos na 4090 e com o da Thor ao lado, com IC pareado por prova e na nota média das cinco (a opção `--mean` e `paired_bootstrap_mean` em `compare_runs.py` são novas, com teste).

**Emenda de 10/11 03:03 (antes de qualquer resultado da `noite15`):** a primeira versão da fila rodava só o Lumi com thinking, por uma leitura incompleta do PRD, que pede a Nota de raciocínio contra o Lumi **nos dois modos** e o GSM8K-PT contra o Lumi **sem** thinking. A sessão `noite15` ainda estava em espera (nenhum job tinha começado), foi encerrada e relançada com os dois modos; o custo subiu ~35 min.

**Regras escritas antes dos resultados:** H66 em `estudos/07`. A comparação principal do relatório é a da mesma máquina (4090 contra 4090), qualquer que seja o lado favorável, e a da Thor é secundária; as máquinas contam como equivalentes se a nota do Lumi diferir ≤ 1,0 ponto em cada prova e modo novos; o critério do PRD é lido como está no PRD (Nota contra os dois modos do Lumi; GSM8K contra o sem thinking).

**Não faz:** não avalia modelo nosso (é a referência, não uma segunda suíte de candidato) e não muda a fila da `noite14`.

## Fila da versão 14 (10/11 02:40): segunda rodada do Antidoom, depois da suíte

**Contexto:** a versão 13 rodou toda a fase anti-laço entre 21:00 de 09/11 e 01:40 de 10/11 (resultados em [06](06_registro_experimentos.md) e [07](07_hipoteses_e_proximos_passos.md)): o Antidoom oficial levou o `thor_pt` de 54,3% a 73,2% de múltipla escolha com thinking em 4 a 6 min de treino, e a regra única escolheu o `thor_pt_antidoom_of` para a suíte de teste, que roda desde 01:40.

**Pedido e o que os dados mudaram:** a segunda rodada de mineração sobre o modelo novo foi planejada com a ideia de que consertar um laço expõe outros. Depois da primeira rodada, porém, o laço oficial em `heldout` é 0,4% (2 de 500) e, na validação, só 1 das 17 respostas que não fecham tem laço; as outras são raciocínio longo e indeciso, sem repetição. A conta dá ~30 eventos em 8.000 prompts (a primeira rodada teve 3.394), e com 136 eventos a receita oficial nem treinou (`gspo1_antidoom`: 9 passos). Uma mineração de 8.000 prompts custaria ~1 h de GPU para um resultado esperado de "sem material".

**Desenho:** a versão 14 (`scripts/run_noite14_chain.sh`) espera a sessão `noite13` acabar (a suíte é o último passo dela), **não roda nenhuma suíte de teste** e faz: (1) **sonda** de 3.000 prompts de múltipla escolha no `thor_pt_antidoom_of`, com as configurações oficiais (~25 min), que escala para a segunda rodada completa só se achar ≥ 100 eventos (H64); (2) **duas sementes** (1 e 2) do treino da primeira rodada sobre os mesmos 3.394 eventos, com fusão, validação e laço depois (~25 min cada), para medir a variação do ganho (H65). Duração prevista: ~1,3 h depois do fim da suíte.

**Regras escritas antes dos resultados:** H64 e H65 em `estudos/07`. A troca do candidato da suíte exige, na validação, múltipla escolha ≥ 75,2%, fecha ≥ 95,6%, GSM8K limpo ≥ 84,8% e português ≥ 90%; a decisão usa só a validação, a suíte de um segundo modelo só roda com autorização e as duas suítes entram no relatório.

**Alternativas não enfileiradas:** H62 (RL com prompts difíceis; a partir do modelo novo exige adaptar `grpo_train.py` e `run_grpo.sh` para partir de um modelo fundido, sem código) e H63 (SFT com traços ≤ 512 tokens). O RL a partir do `thor_pt_antidoom_of`, que a regra da H59 mandava, não entra: o RL sobre o `thor_pt` custou 4,5 pontos por item (`gspo_thor`) contra 1,0 do Antidoom, e o ganho do RL vinha do fechamento, que o Antidoom já deu.

## Fila da versão 13 (09/11 19:40): RL do `qwen_mix_short` primeiro

**Gatilho:** o SFT `qwen_mix_short` terminou com múltipla escolha 66,5% com thinking (empate estatístico com o Lumi: −3,1, IC −7,5 a +1,3), fecha 89,1%, traço em português 99,1%, GSM8K limpo 83,7%; nos itens em que o Lumi e ele fecham, perde 4,9 pontos (o `gspo1` perde 7,5). A regra do H53, escrita antes do resultado, manda fazer o RL a partir dele quando fecha ≥ 87% com múltipla escolha ≥ 62%.

**Por que na frente do Antidoom:** (1) a regra já estava escrita e foi atendida; (2) o RL é a ferramenta que mais subiu todos os SFTs; (3) das 42 respostas do `qwen_mix_short` que não fecham só 11 a 17 são laço (o Antidoom trata laço, não raciocínio longo); (4) custa ~1,5 h mais a avaliação, contra ~3 h de cada etapa do Antidoom; (5) é a única linha cujo ponto de partida já está no nível do `gspo1`.

**Mudança:** a versão 12 foi congelada (SIGSTOP) com o código da fase anti-laço pronto e o teste de fumaça por começar; a versão 13 (`scripts/run_noite13_chain.sh`) assumiu às 19:40. Ordem: (1) RL `gspo_qwen_mix_short` (H60); (2) teste de fumaça do treino da receita oficial; (3) Antidoom oficial no `thor_pt` (H59); (4) no `gspo_thor` (H58); (5) braço do treinador oficial; (6) `selfdist_min` e `selfdist_300` (H54); (7) Antidoom no `gspo1` (H57); (8) suíte de um candidato, por regra (padrão `gspo1`; `gspo_qwen_mix_short` entra na lista da regra).

**Regra de leitura da H60:** ver `estudos/07`. Resumo: apoia se a múltipla escolha com thinking ficar ≥ 69,5%, fechar ≥ 93% e o português ≥ 90%, sem queda do GSM8K limpo abaixo de 83%.

**Resultado da fila toda (10/11 02:30):** ver a versão 14 acima e `estudos/07`: o Antidoom oficial no `thor_pt` deu 73,2% (H59 apoiada), no `gspo_thor` 69,9% (H58 apoiada), a portagem equivale ao treinador oficial (H61 apoiada), o `gspo1_antidoom` não treinou (H57), os `selfdist` não ajudam (H54 inconclusiva) e a suíte rodou para o `thor_pt_antidoom_of` (escolhido pela regra única; a nossa portagem ficou a 0,01 ponto do limite de fechamento, 92,99% contra 93%).

**Resultado (09/11 20:57):** o RL terminou em 75 min com múltipla escolha 67,0% (fecha 90,1%, português 98,6%, GSM8K limpo 83,1%), abaixo do 69,5% da regra: sem apoio, um segundo `gspo1`. A fila não muda: o próximo passo (teste de fumaça e Antidoom no `thor_pt`) segue como estava, porque os tetos grosseiros por modelo (`thor_pt` 71,7%, `gspo_thor` 70,6%, `gspo_qwen_mix_short` 68,8%) mantêm a ordem. Para a suíte, `gspo_qwen_mix_short` não passa a regra única (múltipla escolha < 69,6% e fecha < 93%); o candidato continua o `gspo1`.

## Fila da versão 12 (09/11 18:45): receita oficial do Antidoom, por ganho esperado

**Decisões de 09/11:** (1) usar a receita do repositório oficial do Antidoom ([Liquid4All/antidoom](https://github.com/Liquid4All/antidoom), Apache-2.0, clonado em `../antidoom_oficial`, commit `bd6a126`) no lugar da versão própria, que tinha diferenças de método, entre elas o `chosen_win` medido pelo melhor escolhido (o oficial mede a fração dos escolhidos acima do rejeitado), o que pararia o treino cedo demais ([discovery §20.61](../discovery.md)); (2) testar a fase também no `thor_pt`, o SFT com mais laço; (3) rodar um braço com o treinador do próprio repositório, para separar efeito do método de efeito da nossa portagem; (4) ordenar a fila pelo ganho esperado; (5) a suíte de teste roda um único candidato, escolhido por regra escrita antes dos resultados (`scripts/pick_suite_candidate.py`), com o `gspo1` como padrão.

**Mudança:** a versão 11 foi encerrada antes de começar (só esperava a mineração do `gspo1`); a versão 12 (`scripts/run_noite12_chain.sh`) assumiu às 18:52, quando a mineração do `gspo1` acabou (136 eventos em 10.000 prompts). Enquanto o código da receita oficial não fica pronto (arquivo `runs/jobs/antidoom_v2.pronto`), a GPU roda os SFTs de teste.

| Ordem | Etapa | Por que nessa posição |
| --- | --- | --- |
| 1 | SFTs `qwen_mix_short`, `selfdist_min`, `selfdist_300`, um por vez, até o código ficar pronto | ocupam a GPU sem depender do código novo |
| 2 | Teste de fumaça do treino na receita oficial | o treino novo ainda não rodou na GPU |
| 3 | `thor_pt_antidoom` (H59): referência de laço, mineração em 10.000 prompts, treino, fusão, avaliação, laço depois | maior teto (54,3% → até ~71,7%) e o regime do método (33% de laço) |
| 4 | `gspo_thor_antidoom` (H58), mineração em 8.000 prompts de múltipla escolha | candidato direto (67,0% → até ~70,6%) |
| 5 | `thor_pt_antidoom_of`: os mesmos eventos com o treinador do repositório oficial (só se `runs/jobs/antidoom_trainer_oficial.pronto` existir) | conferência da portagem |
| 6 | SFTs de teste que sobrarem | diagnóstico (H53, H54) |
| 7 | `gspo1_antidoom` (H57) | teto de ~+0,6 ponto |
| 8 | Suíte de teste de um candidato (regra; padrão `gspo1`) | a última, uma vez |

**Regras escritas antes dos resultados:** H59 e H58 em `estudos/07`; a regra da suíte em `scripts/pick_suite_candidate.py` (múltipla escolha ≥ 69,6%, fecha ≥ 93%, GSM8K limpo ≥ 83% e traço em português ≥ 90% na validação; vence a maior múltipla escolha; senão o `gspo1`).

**Testes antes do portão (CPU, sem tocar na GPU):** 19 testes unitários (8 de equivalência contra o código oficial); treino da receita oficial e fusão com um NemotronH minúsculo na CPU (cabeça manual em FP32 contra o modelo completo: diferença máxima de 0,002 logit; só os módulos-alvo mudam na fusão). Esses testes acharam dois problemas que teriam derrubado a fila: o tronco do NemotronH nativo do Transformers 5 se chama `model`, não `backbone`, e o PEFT recusa LoRA no `out_proj` do Mamba.

## Fila da versão 11 (09/11 18:28): fase anti-laço primeiro na família da Thor

**Motivo:** a pergunta do usuário (aplicar o Antidoom nas variantes da Thor) foi testada nos dados antes de gastar GPU. Na validação de múltipla escolha com thinking, o detector de laço exato (o mesmo da mineração) pega 129 das 131 respostas que não fecham no `thor_pt`, 82 das 95 no `thor_en`, 44 das 47 no `gspo_thor` e 68 das 74 no `gspo_thor_e1`, contra 4 das 26 no `gspo1` ([discovery §20.59](../discovery.md)). No `gspo1` o não fechamento é raciocínio longo e indeciso, que a fase não trata; na Thor é laço. Além disso, o `thor_pt` iguala o Lumi nos itens em que os dois fecham.

**Mudança:** a versão 10 foi congelada (SIGSTOP) com a mineração do `gspo1` em curso (ela termina e fornece os eventos do teste de fumaça); a versão 11 (`scripts/run_noite11_chain.sh`) espera os processos dela acabarem e segue: (1) teste de fumaça do treino; (2) fase anti-laço no `gspo_thor` (H58): referência de laço em 750 prompts de `heldout`, mineração em 5.000 prompts de múltipla escolha (a matemática não tem laço), treino com parada antecipada, avaliação e taxa de laço depois; (3) SFTs `qwen_mix_short`, `selfdist_min` e `selfdist_300`; (4) treino anti-laço no `gspo1` com os eventos já minerados (teto de ~+0,6 ponto: fica depois, por ser só teste de mecanismo); (5) suíte do `gspo1` por último.

**Regra de leitura da H58 (escrita antes de qualquer resultado):** ver `estudos/07`. Resumo: o mecanismo funciona se o laço exato em `heldout` cair pelo menos à metade com a validação sem perda (múltipla escolha ≥ 67,0%, fecha ≥ 92%, português ≥ 90%); o modelo só passa a ser candidato à suíte no lugar do `gspo1` com múltipla escolha ≥ 69,6%, fecha ≥ 93% e GSM8K limpo ≥ 83%. Se ele não passar, o candidato continua o `gspo1`.

## Fila da versão 10 (09/11 17:22): suíte do `gspo1` por último

**Decisão de 09/11:** a suíte de teste do Lumi passa a rodar na Jetson Thor; na 4090 a última etapa é a suíte do `gspo1` (8 ciclos, `runs/grpo/gspo1/merged`, adaptador `runs/grpo/gspo1/it007/adapter`, uma única execução). Os resultados do Lumi vêm da Thor por arquivo, e a comparação final registra o motor de cada lado (`_engine` do `summary.json`). Conferência entre as máquinas (MMLU-ProX PT de teste com thinking, Lumi, 11.759 itens, gulosa, 4.096 tokens): Thor 53,54% estrito, 53,37% flexível, 27,1% truncadas, 72,5% fecham e respondem, mediana de 996 tokens (6,1 h, 944 tokens/s); 4090 (08/11) 53,57% estrito, 53,52% flexível, 27,1% truncadas, 72,4%, mediana de 1.002 (2,0 h, 2.855 tokens/s). Diferença de 0,03 ponto no acerto estrito: as duas máquinas reproduzem a medida do Lumi.

**Mudança:** a versão 9 foi congelada (SIGSTOP) com a avaliação da `gspo1_soup3` em curso; a versão 10 (`scripts/run_noite10_chain.sh`) é a mesma fila, com duas diferenças: espera só os processos vivos da versão 9 (zumbis de um pai parado não contam) e termina com `scripts/run_suite_models.sh "gspo1|runs/grpo/gspo1/merged"` no lugar da suíte do Lumi. Todas as etapas anteriores são retomáveis e puladas se já prontas. Se a fase anti-laço (H57) ou um SFT de teste superar o `gspo1` pela regra pré-registrada, o candidato da suíte final será trocado antes de a etapa começar (a suíte é a última e só roda uma vez por candidato).

## Fila da versão 9 (09/11 17:14): fase anti-laço e contingência

**Decisões de 09/11:** (1) testar a fase anti-laço (H57) sobre o `gspo1`; (2) **contingência:** se nada superar o `gspo1`, o candidato entregue é o `gspo1` (SFT `adalberto` + 8 ciclos de RL), relatando o que se conseguiu: raciocínio em português, melhora em matemática e piora em múltipla escolha (64,4 a 67,8% contra 69,6% do Lumi, na faixa das duas sementes). O relatório cita os dois limiares de fechamento (90% e 98%) e que o adaptador só liga com thinking.

**Mudança:** a versão 8 foi congelada (SIGSTOP, sem tocar na avaliação em curso) e a versão 9 (`scripts/run_noite9_chain.sh`) espera os processos dela acabarem, depois faz as sopas, o `base_ptprefill`, a fase anti-laço e só então os SFTs de teste e a suíte do Lumi.

| Ordem | Etapa | O que mede |
| --- | --- | --- |
| 1 | Sopas `gspo1_soup2` e `gspo1_soup3`; `base_ptprefill` | H55 e H56, como na versão 8 |
| 2 | Taxa de laço de referência do `gspo1` de 8 ciclos em 750 prompts de `heldout` (gulosa, 3.072 tokens) | base de comparação |
| 3 | Mineração de laços em 10.000 prompts de treino (8.000 de múltipla escolha, 2.000 de matemática) | eventos (prefixo até o início da repetição, token rejeitado) |
| 4 | Teste de fumaça (6 eventos, 2 passos) e treino anti-laço, com parada antecipada na validação dos eventos | adaptador `gspo1_antidoom_r1` |
| 5 | Avaliação da validação do `gspo1_antidoom_r1` e taxa de laço nos mesmos 750 prompts de `heldout` | H57 |
| 6 | SFTs `qwen_mix_short`, `selfdist_min`, `selfdist_300` e suíte do Lumi por último | H53, H54, referência |

**Regra de leitura da H57** (escrita antes de ver os resultados): apoia se, em `heldout`, o não terminar cair pelo menos à metade e, na validação, a múltipla escolha com thinking ficar ≥ 65,5% (o `gspo1` tem 67,5%; queda de até 2 pontos tolerada), fechar ≥ 95% e o traço em português ficar ≥ 90%. Se apoiar, uma segunda rodada de mineração sobre o modelo novo (o Antidoom relata que consertar um laço pode expor outros).

**Emenda à regra de leitura da H57 (09/11 17:50, depois da referência e antes de qualquer resultado do tratamento):** a referência do `gspo1` de 8 ciclos em `heldout` deu 100 de 500 prompts de múltipla escolha sem terminar (20,0%), dos quais só 8 com laço exato (1,6%); na validação, 4 das 26 respostas sem fechamento. A regra original (não terminar cair à metade e fechar ≥ 95%) é inalcançável por construção. Regra emendada: apoia se a taxa de laço exato em `heldout` cair pelo menos à metade (8 → ≤ 4), o não terminar em `heldout` não subir e, na validação, a múltipla escolha com thinking ficar ≥ 65,5%, fechar ≥ 93,2% e o traço em português ≥ 90%. O ganho máximo possível em nota é ~+0,6 ponto; o teste fica como diagnóstico do mecanismo. A fila segue como está: a mineração já corre e o treino é curto.

**Desenho da perda** (`src/lumi_ft/train/antidoom.py`; versão própria a partir da descrição do blog da Liquid AI): por evento, logits do último token do prefixo; sobe a média dos até 20 escolhidos (alternativas de maior logit da referência com ≥ 2 caracteres alfanuméricos) e desce o rejeitado, em logit, sem softmax sobre o vocabulário; o resto do vocabulário é preso à referência, ponderado pela probabilidade de referência; escolhidos e rejeitado podem se mover até 3 logits sem custo. Continua treinando o adaptador do `gspo1` (LoRA r=16, lr 1e-5, lote de 8 eventos), não um LoRA de rank 128 a 256 como no blog. **Se der errado, estudar:** a perda e os hiperparâmetros (beta, lam_rest, tau) não foram ajustados; o treino não foi rodado em GPU antes do teste de fumaça; com poucos eventos (a taxa de laço do `gspo1` é de uns 5 a 8% em múltipla escolha) a parada antecipada pode chegar cedo demais.

## Fila da noite (versão 3, 22:21; substituída pela versão 4)

Depois de uma auditoria do pipeline contra as práticas do TRL (`discovery.md` §20.45), os testes que ela sugeriu entram **antes** da suíte do Lumi. A versão 3 (`scripts/run_noite3_chain.sh`, sessão `noite3`) substitui a 2 e retoma o `gspo_thor_e1` do ciclo 3 (a `noite` foi parada na troca do ciclo 2 para o 3, sem perda de trabalho). Um erro da primeira tentativa: `tmux has-session -t noite` casa por prefixo com `noite3`, e a fila esperava a si mesma; corrigido com nome exato (`=noite`).

| Ordem | Etapa | O que mede | Tempo estimado |
| --- | --- | --- | --- |
| 1 | RL `gspo_thor_e1` (a partir do `thor_pt_e1`, 4 ciclos) | triagem do ponto de partida novo; compara com o `gspo_thor` (65,7% no ciclo 4) | termina ~23:20 |
| 2 | Diagnóstico de log-probabilidades (`scripts/diag_logprob.py`) | descompasso vLLM x treinador, ruído numérico do corte do GSPO (eps 3e-4 e 4e-4), cabeça FP32, adaptador desligado = Lumi | ~15 min |
| 3 | SFT `adalberto_dft` (receita do `adalberto`, `--loss dft`) | se a perda DFT reduz o dano do SFT na múltipla escolha (`adalberto`: 62,3%) | ~40 min |
| 4 | RL `gspo1_T06` (receita do `gspo1`, logits do treinador divididos por 0,6, 4 ciclos) | efeito da temperatura nos logits do treinador, com os mesmos prompts do `gspo1_c4` (67,8%) | ~1,5 h |
| 5 | RL `gspo1_s1` (receita do `gspo1`, semente 1, 4 ciclos) | variação entre rodadas (ruído de uma execução) | ~1,5 h |
| 6 | RL `gspo1_4k` (ramo B: receita do `gspo1`, limite 4.096) | se o limite importa nas questões antigas | ~1,6 h |
| 7 | Suíte de teste do Lumi | referência; interrompível com `touch runs/jobs/suite.stop` | ~3 a 3,5 h |

**Resultado da etapa 1, `gspo_thor_e1` (22:42), validação limpa, com thinking:** múltipla escolha **62,1%** ▼ (−7,5 contra o Lumi; −3,6 contra o `gspo_thor` no ciclo 4, IC −7,5 a +0,0; +10,7 sobre o `thor_pt_e1`), GSM8K 84,3% ▲, fecha 80,8% (abaixo dos 87% que pararam o `gspo_gold4k`), trunca 19,2%. **Leitura:** o RL desfaz parte dos laços (trunca 36,6% → 19,2%), mas o ponto final fica abaixo do que as outras rodadas do mesmo tipo atingiram. Pelo desenho (sem regra própria), não se continua até o ciclo 8 e a família Nemotron deixa de ser a aposta para passar o `gspo1`. **Se der errado, estudar:** se o diagnóstico de logprob mostrar que o corte do GSPO é dominado por ruído, as diferenças de 3 a 4 pontos entre pontos de partida não são interpretáveis.

**Resultado do `gspo1_T06` (00:43), validação limpa, com thinking:** múltipla escolha **63,6%** (−4,2 contra o `gspo1_c4`, IC −8,3 a +0,0; −6,0 contra o Lumi ▼), GSM8K 84,0% (−0,3), fecha 87,3% (contra 90,9%), trunca 12,7%. **Leitura:** dividir os logits do treinador por 0,6 não ajudou; é um valor único e o IC toca o zero, então só se pode dizer "sem ganho" até a semente 1 (`gspo1_s1`, em andamento) mostrar quanto uma mesma receita varia de uma rodada para outra. **Se der errado, estudar:** se o `gspo1_s1` também ficar ~4 pontos abaixo do `gspo1_c4`, o `gspo1_c4` foi uma execução favorável e todas as comparações com ele (inclusive esta) precisam ser relidas.

**Resultado do `gspo1_s1` (02:11), validação limpa, com thinking:** múltipla escolha **64,4%** (−3,4 contra o `gspo1_c4`, IC −7,5 a +0,8; −5,2 contra o Lumi, IC −9,9 a −0,8 ▼), GSM8K 82,5% (−1,8; +0,9 contra o Lumi), fecha **87,8%** (90,9% no `gspo1_c4`), trunca 13,0%. **Leitura:** a mesma receita varia 3,4 pontos em múltipla escolha e 3,1 no fechamento de uma semente para a outra. A condição escrita antes ("se o `gspo1_s1` também ficar ~4 pontos abaixo, o `gspo1_c4` foi uma execução favorável") se cumpre em parte (−3,4): as comparações com o `gspo1_c4` devem ser lidas contra a faixa 64,4 a 67,8 (média 66,1), não contra 67,8. Consequências: (1) o `gspo1_T06` (63,6%) está dentro da faixa e a H45 não mostra efeito nem dano; (2) o `gspo_thor` (67,0%) empata com o `gspo1`; (3) **o critério de fechamento de 90% é cumprido por uma das duas sementes da receita (90,9% e 87,8%, média 89,4%) e o de 98% por nenhuma**; (4) nenhuma das duas sementes passa o Lumi em múltipla escolha (−1,8 e −5,2). **Se der errado, estudar:** com uma rodada por braço só se detecta o que sair dessa faixa; para decidir o candidato final, rodar uma segunda semente do melhor braço.

**Resultado da etapa 2, diagnóstico de log-probabilidades (22:44):** ver `discovery.md` §20.48. Em resumo, o ruído numérico do primeiro passo tem a escala do eps do GSPO (46% das sequências acima de 3e-4 sem nenhuma mudança de pesos), a cabeça FP32 o reduz ~40% e o descompasso vLLM x treinador não depende dela. **Se der errado, estudar:** o relatório vem de uma só amostra (96 respostas de um ciclo, um adaptador) e a T = 1; ele não mede o efeito da temperatura 0,6 da amostragem (isso é o `gspo1_T06`).

**Como ler:** `gspo1_c4` (semente 0, T 1,0), `gspo1_s1` (semente 1, T 1,0) e `gspo1_T06` (semente 0, T 0,6). A diferença entre as duas primeiras é o ruído de uma execução; a terceira só é um efeito se sair do intervalo dessa variação. O `adalberto_dft` compara com o `adalberto` e o ruído do SFT é menor que o do RL (mesmo conjunto de treino), mas há uma confusão: a perda DFT dá gradientes menores (cada token pesa p ≤ 1), então o efeito pode vir da perda ou da taxa de aprendizado efetiva menor. **Se der errado, estudar:** se o diagnóstico mostrar ruído de |log s| ≥ eps na maior parte das sequências, as comparações de RL anteriores tinham o corte dominado por ruído, e o `gspo1_T06` não é interpretável sem antes reduzir o ruído (cabeça FP32 ou minibatches iguais).

**Variáveis de ambiente das rodadas novas** (para não mexer no `run_grpo.sh` em execução, os scripts em Python leem `GRPO_LOGIT_TEMP` e `GRPO_SEED`; sem elas o comportamento é o de antes; ficam em `runs/grpo/<nome>/ENV.txt`).

## Fila da noite (versão 2, 21:00; substituída pela versão 3)

A fila estava vazia depois do `gold2048` (os experimentos seguintes dependiam do resultado dele e nada tinha sido enfileirado). `scripts/run_noite_chain.sh` (sessão `noite`) roda, em ordem: o SFT com a entrega 1 do pacote 3, o RL a partir dele, o RL do ramo escolhido pela regra acima e a suíte de teste do Lumi. A primeira versão (20:40) tinha o SFT no fim e **nenhum RL depois dele**; a segunda (21:00) corrige isso e põe o SFT e o RL dele antes do `gspo1_4k`, porque o candidato mais provável é SFT + RL, e o `gspo1_4k` virou diagnóstico (o limite de 4.096 não mexeu na nota nas questões do gold). O `gspo1_4k` tinha rodado 6 min (só a fusão) e foi reiniciado.

| Ordem | Etapa | Receita | Tempo estimado |
| --- | --- | --- | --- |
| 1 | SFT `thor_pt_e1` | receita do `thor_pt` com a entrega 1 do pacote 3 (abaixo); avaliação na validação | ~30 min + ~13 min |
| 2 | RL `gspo_thor_e1` | receita do `gspo_thor` (questões antigas, 2.048, T 0,6, top-p 0,95, `--ratio seq`, `--max-noclose 0.85`), **4 ciclos**, a partir do `thor_pt_e1`; roda sempre que o SFT tiver avaliação | ~1,5 h |
| 3 | Ramo B: `gspo1_4k` | receita do `gspo1`, questões antigas, limite 4.096, 4 ciclos (leitura `B X=63,9% C=89,4%` em `runs/jobs/noite.ramo`) | ~1,6 h |
| 4 | Suíte de teste do Lumi | MMLU-ProX PT sem thinking, ENEM, BLUEX, OAB, nos dois modos; só referência, interrompível com `touch runs/jobs/suite.stop` | ~3 a 3,5 h |

Os ramos A e C da regra continuam no script (retomar o `gspo_gold` até o ciclo 8; no A, também o `gspo_gold_thor`), mas não rodam, porque o ramo já está fixado em B.

**SFT `thor_pt_e1`:** mesma receita do `thor_pt` (LoRA r=16, alpha 32, lr 1e-4, 1 época, limite de 2.048 tokens por exemplo, 5.544 linhas sem raciocínio do GigaVerbo), trocando só o raciocínio: 9.850 exemplos da entrega 1 do pacote 3 (5.426 de múltipla escolha: 3.654 `mot_science`, 1.024 MedMCQA, 748 WebInstruct; 4.424 de matemática) contra 4.927 do `thor_pt` (2.174 de múltipla escolha). Dados: `data/thor_pt_e1.parquet` (sha256 `1ae6c8e8…2e5`), 15.394 linhas, 9,3 M de tokens. Pergunta: mais dados de múltipla escolha, com raciocínio de até 2.048 tokens, resolvem o fechamento do `</think>` (o `thor_pt` fechava em 66%)? Não responde sozinho: muda ao mesmo tempo a quantidade de dados e a de múltipla escolha. **Se der errado, estudar:** se o fechamento continuar baixo, a cauda longa vem do estilo do professor, não da quantidade; as 3,6% de respostas com `Resposta: X` repetido (3,4% no `thor_pt`) ficaram como estão para manter a comparação.

**Resultado do `thor_pt_e1` (21:42), validação limpa, com thinking:** múltipla escolha **51,4%** ▼ (−18,2 contra o Lumi; −2,9 contra o `thor_pt`, IC −7,0 a +1,3), fecha 63,4% (`thor_pt` 66,0%; `thor_pt_short` 73,8%), trunca 36,6%, GSM8K 76,9% ▼ (−4,7). **Leitura:** mais dados de múltipla escolha do mesmo professor, com exemplos de até 2.048 tokens, não melhoram o ponto de partida (H43 refutada); a quantidade de dados não explica os laços. O RL a partir dele segue na fila como triagem, embora o ponto de partida esteja abaixo do limiar de ~62% usado para não rodar RL a partir do `thor_pt_short` (a fila foi montada para rodar sempre; a decisão sobre o 8º ciclo e a suíte fica para a leitura do resultado).

**RL `gspo_thor_e1`:** compara com o `gspo_thor` (a partir do `thor_pt`, 8 ciclos, 67,0% no ciclo 8 e 65,7% no ciclo 4) e com o `gspo1_c4` (67,8%). Sem regra de parada própria: a decisão sobre o 8º ciclo e a suíte de teste fica para a leitura do resultado. **Se der errado, estudar:** o `thor_pt_e1` partir de um ponto pior que o `thor_pt` (fechamento, laços), e o `--max-noclose 0.85` permissivo demais para deixar passar uma degeneração.

Fora da fila, por dependerem de dados que ainda não chegaram: o SFT misto com o Qwen (pacote 4) e a suíte de teste do melhor candidato.

## Em aberto (sem decisão)

- **Pacote do Qwen para a Thor (decidido às ~19:55, depois de §20.39):** preparado às 20:00 (`handoff/thor4_pacote.tar.gz`; `discovery.md` §20.40). Decisão: o SFT só de matemática prejudica a qualidade em múltipla escolha (§20.39), então geramos traços de múltipla escolha em português com o Qwen. Ficou de fora: Nemotron blocos 3 e 4 (cancelados), limite de 1.024 tokens (aplicado só na 4090, depois). **Se der errado, estudar:** a qualidade medida nos itens em que todos fecham foi uma comparação condicionada ao fechamento (itens mais fáceis) com 189 itens; o efeito da "quantidade de ciência" não foi testado (só o `thor_pt_short`, confundido com o corte por tamanho); o rendimento do Qwen em questões de cálculo longo da WebInstruct é desconhecido (viés para as fáceis). Falta ainda decidir quando a Thor para o Nemotron (condicionado à ENTREGA 1 e a uma confirmação).
