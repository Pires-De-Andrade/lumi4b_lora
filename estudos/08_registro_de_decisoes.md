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


## Em aberto (sem decisão)

- **Pacote do Qwen para a Thor (decidido às ~19:55, depois de §20.39):** preparado às 20:00 (`handoff/thor4_pacote.tar.gz`; `discovery.md` §20.40). Decisão: o SFT só de matemática prejudica a qualidade em múltipla escolha (§20.39), então geramos traços de múltipla escolha em português com o Qwen. Ficou de fora: Nemotron blocos 3 e 4 (cancelados), limite de 1.024 tokens (aplicado só na 4090, depois). **Se der errado, estudar:** a qualidade medida nos itens em que todos fecham foi uma comparação condicionada ao fechamento (itens mais fáceis) com 189 itens; o efeito da "quantidade de ciência" não foi testado (só o `thor_pt_short`, confundido com o corte por tamanho); o rendimento do Qwen em questões de cálculo longo da WebInstruct é desconhecido (viés para as fáceis). Falta ainda decidir quando a Thor para o Nemotron (condicionado à ENTREGA 1 e a uma confirmação).
