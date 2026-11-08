# 07. Hipóteses, ameaças à validade e próximos experimentos

Registro único das hipóteses do projeto, com o veredito atual e onde está a evidência. Convenção: **apoiada**, **refutada**, **parcial** (evidência a favor e contra) e **aberta** (não testada ou inconclusiva).

## 1. Registro de hipóteses

| ID | Hipótese | Veredito | Evidência | Estudo |
| --- | --- | --- | --- | --- |
| H0 | O Lumi raciocina em português sem treino, só com instrução ou prefill | **Refutada** | A instrução não muda o idioma; o prefill muda o idioma e quebra a estrutura (GSM8K 48,1%) | [02](02_sft_dados_e_professor.md) E0 |
| H1 | 10 mil traços do GigaVerbo bastam para o comportamento desejado | **Refutada** | Forma ok, nota −14 (GSM8K) e −26 (múlt. escolha) | 02, E1 |
| H2 | Traços de um professor forte (Qwen3-30B) ensinam português e mantêm ou melhoram a nota | **Parcial** | Matemática +1,6 (IC inclui 0); múltipla escolha −7,3 ▼ | 02, E2 |
| H3 | A autodestilação (o Lumi como professor) funciona | **Refutada** | GSM8K −3,6 ▼, múlt. escolha −11,7 ▼ | 02, E3 |
| H4 | O limite de 1.024 tokens causa a perda na múltipla escolha | **Refutada** | Com 2.048: 64,2% (+1,8 sem significância), sem thinking pior | 02, E4 |
| H5 | Taxa de aprendizado menor preserva a múltipla escolha | **Refutada** | 63,4% contra 62,3% | 02, E5 |
| H6 | O DoRA preserva melhor que o LoRA | **Refutada** | Diferença de 0,3 a 0,7 ponto; custa 2,5× o tempo | 02, E6 |
| H7 | Reduzir o efeito do adaptador (x0,5, x0,75) recupera a múltipla escolha | **Refutada** com thinking | 63,4% e 64,4% | 02, E7 |
| H8 | O Lumi traduz bem os traços de professor (inglês → português) | **Refutada em escala** | Piloto 86% aceito; em lote maior, 35% dos raciocínios (o tradutor respondia a questão) | 02, E8 |
| H9a | A perda na múltipla escolha vem de algumas partes do adaptador | **Apoiada** (triagem) | Só os MLPs: 69,6% (igual ao Lumi); sem MLP: 60,0% ▼ | [03](03_ablacao_camadas_e_mlp.md) E9 |
| H9b | O idioma do raciocínio é decidido pelas camadas finais | **Parcial** | Só 21 a 41: 98,9% em português; só 0 a 20: 0%; mas o adaptador parcial quebra o modelo | 03, E9 |
| H9c | Os MLPs concentram ganhos de capacidade; o Mamba e a atenção, o dano | **Parcial**, a confirmar por treino | Triagem a favor; o ganho de matemática do "só MLP" tem IC que inclui 0 | 03, E9 e E10 |
| H10 | O português do raciocínio, por si, causa a perda na múltipla escolha | **Aberta**, com evidência contra | Camadas 0 a 20 (traço em inglês) perdem os mesmos 7 pontos; o `adalberto_mlp` e a grade 2×2 testam | 03 |
| H11 | O RL a partir do SFT recupera a múltipla escolha com thinking | **Parcial** | 62,3% → 66 a 68%, empate com o Lumi (IC inclui 0) | [04](04_rl_grpo.md) |
| H12 | A explosão de entropia vem de gradientes negativos longos sem KL e T alta | **Aberta** | T 0,6 + filtro evita; KL 0,04 não segura; mecanismo não isolado | 04 |
| H13 | O filtro de truncadas tira a pressão de parar | **Apoiada** | `grpo2`: trunca 16,1%, fecha 84,2% | 04 |
| H14 | Treinar só com respostas certas (RAFT) é estável e estaciona mais baixo | **Apoiada** (4 ciclos) | `raft1`: entropia 0,40 → 0,18, múlt. escolha 61,6% ▼ | 04 |
| H15 | O RL só com thinking prejudica o modo sem thinking | **Apoiada** | 43,1% em duas rodadas; `gspo1` 8 ciclos 57,7% ▼ | 04 |
| H16 | A razão por sequência com corte minúsculo (GSPO) estabiliza | **Apoiada**, mecanismo ambíguo | Entropia estável em 8 ciclos, mas 40 a 50% dos tokens cortados | 04 |
| H17 | Mais ciclos de RL melhoram a nota | **Refutada** (nota), apoiada (forma) | `gspo1` 4 → 8: 67,8 → 67,5 | 04 |
| H18 | A pergunta em português reduz o acerto do Lumi original | **Refutada** | PT − EN: −1,3 a +0,4 (flexível), sem significância | [05](05_diagnosticos_e_referencias.md) D2 |
| H19 | O estilo do professor (longo e falado contra curto em forma de plano) custa na múltipla escolha | **Aberta** | Não testada: precisa dos traços do Nemotron 30B | 05, D5 |

## 2. Hipóteses propostas, ainda sem teste

| ID | Hipótese | Teste | Custo |
| --- | --- | --- | --- |
| H20 | Traços de um professor da mesma família (Nemotron 30B), traduzidos na Thor, evitam a perda na múltipla escolha | SFT com os pacotes 2 e 3 traduzidos; comparar com o `adalberto` | Depende da Thor; SFT ~30 a 40 min mais 13 min de avaliação |
| H21 | O raciocínio em inglês **não** perde na múltipla escolha, o português perde | O par em inglês e português dos mesmos problemas (`think_en` e `think`) | Dois SFTs de ~40 min |
| H22 | Misturar uma perda de SFT no modo sem thinking ao RL protege esse modo | Nova opção de `grpo_train`, 4 ciclos | Implementar ~1 h; rodada 1,3 h |
| H23 | Um KL mais forte (coeficiente 0,2 ou mais) segura a entropia sem o filtro de truncadas | `grpo4` com coeficiente maior | 1,4 h |
| H24 | O RL a partir do SFT só com MLP ou dos dados da Thor passa o critério | Rodar o GSPO a partir do melhor SFT | 1,3 a 2,4 h |
| H25 | O ranking entre as rodadas se repete com outras sementes | Duas sementes do `gspo1` e do `grpo1` | 2 × 1,3 h cada |

## 3. Próximos experimentos, em ordem

1. **`adalberto_mlp`** (treinado; avaliação em curso): decide H9c e dá sinal sobre H10. Regra: se ≥ 90% dos traços saírem em português **e** a múltipla escolha com thinking ficar ≥ 66%, adotar o adaptador só com MLP como base do RL; se o traço ficar em inglês, abandonar a ideia de MLP isolado; se português com ~62%, a perda não é do Mamba nem da atenção.
2. **Suíte de teste dos finalistas** (`gspo1` de 8 ciclos, `grpo1`, `adalberto`, e o Lumi): as 5 provas da nota de raciocínio (17.417 itens) levam ~1,3 h por modelo nos dois modos. Critério de triagem para ir à suíte de teste: IC da múltipla escolha contra o Lumi **não** abaixo de 0 nos dois modos (isso não muda o critério de vitória do PRD). Hoje só o `gspo1` e o `grpo1` ficam acima, e só com thinking.
3. **Dados da Thor** (pacotes 2 e 3, quando voltarem): SFT em português e o par em inglês (H20 e H21); depois o RL a partir do melhor SFT (H24).
4. **Proteger o modo sem thinking no RL** (H22), porque o critério do PRD (queda de no máximo 1 ponto) é o que hoje falha em toda rodada de RL.
5. **KL mais forte** (H23) e a grade 2×2 de camadas ([03](03_ablacao_camadas_e_mlp.md)), se sobrar tempo.
6. **Sementes** (H25): sem isso, as diferenças de 1 a 2 pontos entre rodadas não são interpretáveis.

## 4. Ameaças à validade (do conjunto)

| Ameaça | Efeito | Mitigação |
| --- | --- | --- |
| **Uma execução por configuração** | Não sabemos a variância entre sementes; diferenças pequenas podem ser sorte | H25; interpretar só diferenças grandes |
| **385 itens de múltipla escolha** | IC de ±4 a 5 pontos; a "maldição do vencedor" ao escolher o melhor entre muitos | Triagem pela estabilidade; só os finalistas vão à suíte de teste |
| **Nenhum candidato na suíte de teste** | Todos os resultados são de validação | Item 2 da lista acima |
| **Gabarito fraco no RL** (letra majoritária do R1) | O modelo pode ser premiado por concordar com o R1 | A validação usa gabaritos reais |
| **Várias mudanças por rodada de RL** | A causa de cada diferença não está isolada | Variar uma coisa por vez a partir de uma base fixa |
| **Triagem por camadas é pós-treino** | Zerar módulos depois do treino conjunto não equivale a treinar só neles | `adalberto_mlp` e a grade 2×2 |
| **Heurística de idioma** | O "traço em português" é uma contagem de palavras funcionais | Revisão por amostra (feita em ~5 saídas por rodada, não em 50) |
| **Dados do adalberto sem licença** | Só para experimento | Não publicar adaptador treinado neles |
| **vLLM não reprodutível** | Ruído de ±0,15 a 1,2 ponto por execução | Fora do IC; mesma ferramenta para todas as rodadas |
| **Revisão humana dos traços (D-10) não feita** | Qualidade dos dados de SFT não auditada por pessoa | `reports/review_d10.md` aguarda revisão |

## 5. O que mudaria as conclusões

- Uma segunda semente do `gspo1` que **não** repita a múltipla escolha de ~67%: o ganho do RL seria ruído.
- A suíte de teste mostrando o `gspo1` abaixo do Lumi nas provas brasileiras (ENEM, BLUEX, OAB): a validação (Global-MMLU, MMLU-ProX, INCLUDE) não representaria as provas do PRD.
- O `adalberto_mlp` aprendendo português com a nota do Lumi: o caminho passa a ser MLP isolado, não o RL.
- Traços do Nemotron 30B que reduzam a perda na múltipla escolha: o estilo do professor era a causa principal.
