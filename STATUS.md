# Estado do projeto e próximos passos

**Atualizado em 10/11/2026, 07:50.** Ponto de entrada para quem chega sem contexto: o que se quer, onde o projeto está, o que está rodando e o que fazer depois. Este arquivo é **reescrito a cada marco** (não acumula histórico); o histórico fica em [discovery.md](discovery.md) e em [estudos/](estudos/README.md). Idioma dos documentos: português do Brasil, texto impessoal, números em vez de adjetivos.

## 1. Em uma tela

- **Objetivo:** ajustar o `ceia-br/lumi-4b` v0.2 com LoRA para que `enable_thinking=True` raciocine em português e acerte mais que o Lumi original. Entrega: adaptador vencedor, versão fundida em BF16 e relatório reproduzível ([PRD.md](PRD.md)). O adaptador só é usado **com thinking**; sem thinking roda o Lumi original (decisão de 08/11).
- **Candidato atual: `thor_pt_antidoom_of`.** SFT `thor_pt` (raciocínio do Nemotron 30B traduzido para o português) + uma fase de preferência no token que inicia o laço (receita oficial do Antidoom/FTPO, portada). Na validação, múltipla escolha com thinking 73,2% contra 69,6% do Lumi; fecha o `</think>` em 95,6%; traço em português 99,5%. **Contingência:** `gspo1`.
- **Onde estamos (10/11, 07:50): todas as filas terminaram e a suíte de teste está completa.** Nenhum processo nosso roda na GPU. Com thinking, contra o Lumi **na mesma máquina (4090)**, bootstrap pareado por item ([discovery.md](discovery.md) §20.68):
  - **Nota de raciocínio (média das 5): 69,9%.** Contra o Lumi com thinking (63,2%): **+6,8 (IC +5,9 a +7,6)**. Contra o Lumi sem thinking (65,5%): **+4,4 (IC +3,4 a +5,4)**. Meta mínima do PRD (IC acima de 0 contra os dois modos): **cumprida**. Meta alvo (Lumi sem thinking + 5,0): **não cumprida** (+4,4; o IC inclui 5,0).
  - **GSM8K-PT: 90,3%, +5,0 (IC +3,3 a +6,7) contra o Lumi sem thinking:** meta mínima (+3,0 com IC acima de 0) cumprida, pela estimativa e pelo limite inferior; meta alvo (+10,0) não.
  - **OAB: com thinking só empata com o Lumi sem thinking** (44,0% contra 43,5%; +0,5, IC −1,8 a +2,6). O ganho de +22,8 contra o Lumi com thinking é só fechar o `</think>` (o Lumi trunca 60,9% do OAB). Nas outras quatro provas o ganho contra o Lumi sem thinking é de +5,0 a +5,7, com IC acima de 0.
  - **Traço em português: 97,3% a 100% em todas as provas** (meta ≥ 90%: cumprida).
  - **Fechamento do `</think>`: decisão pendente de uma pessoa.** O PRD não diz em que conjunto o limiar vale. Validação de múltipla escolha 95,6%; por prova de teste: GSM8K 99,8%, ENEM 97,1%, BLUEX 95,7%, **OAB 89,4% e MMLU-ProX 82,6%** (abaixo de 90%); média simples 92,9%; média ponderada pelos itens 86,5%. Com ≥ 90% passam GSM8K, ENEM, BLUEX e a média simples; com o original ≥ 98%, só o GSM8K. Como os resultados já são conhecidos, qualquer escolha é posterior a eles: datar e declarar como tal, e mostrar todas as leituras no relatório.
  - **De onde vem o ganho:** todo de fechamento. Nos itens que só o modelo fecha, +2,1 a +23,8 pontos por prova; nos itens que os dois fecham, o modelo acerta 0,3 a 2,7 pontos a menos por item que o Lumi.
  - **Segunda rodada do Antidoom (H64): sem material** (4 eventos em 3.000 prompts, limiar 100). **Sementes (H65): reproduzem** (72,5% e 74,0% de múltipla escolha na validação; 73,2% e 73,2% nas outras duas execuções).
  - **Lumi na Thor × 4090 (H66):** a nota do Lumi difere ≤ 0,83 ponto em 9 das 10 combinações prova e modo; exceção: BLUEX sem thinking, 1,52 (efeito de máquina, não muda conclusão).
- **Próximo marco:** as decisões da seção 5, depois o relatório.
- **Nada neste estado foi publicado** (sem Hugging Face), e nenhum outro candidato foi rodado na suíte de teste.

## 2. Critério de vitória (PRD, "Metas")

Referências medidas no mesmo harness; a Nota de raciocínio é a média simples de GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX e OAB.

| Métrica | Referência | Meta mínima |
| --- | --- | --- |
| Nota de raciocínio, com thinking | Lumi nos **dois** modos | IC 95% da diferença acima de 0 contra **os dois** |
| GSM8K-PT, com thinking | Lumi **sem** thinking | +3,0 com IC 95% acima de 0 |
| Respostas que fecham `</think>` e trazem texto depois | Lumi com thinking | ≥ 90% (original ≥ 98%; emenda de 08/11 feita **depois** de conhecidos resultados: citar **os dois limiares** e quais candidatos passam em cada) |
| Traços em português | Lumi com thinking | ≥ 90% |
| Overall sem thinking | Lumi | queda de até 1 ponto: cumprida se o adaptador só for ligado com thinking (decisão de 08/11). O BF16 fundido sozinho degrada o modo sem thinking (GSM8K de teste: 84,48% contra 85,33%, não significativo; a múlt. escolha cai mais) e o relatório precisa dizer isso |

Lumi na Thor (suíte de teste, 09/11, 4.096 tokens, gulosa): Nota **63,10% com thinking e 66,06% sem**. O Lumi sem thinking tem nota maior porque no OAB o thinking trunca em 62,5% das respostas (OAB 20,4% com thinking, 44,3% sem). Por isso o critério mais duro da Nota é o contra o Lumi **sem** thinking, e o OAB foi onde se decidiu: com thinking, o modelo empata com o Lumi sem thinking nessa prova (seção 3).

## 3. Números

**Validação** (GSM8K limpo n=676 + múltipla escolha n=385; thinking; ruído de uma rodada ±3 a 4 pontos; a validação escolheu o candidato entre dezenas, então o ganho tende a encolher no teste):

| Modelo | Múlt. escolha | Fecha `</think>` | GSM8K limpo | Observação |
| --- | --- | --- | --- | --- |
| Lumi | 69,6% | 86,2% | 81,7% | referência |
| `thor_pt_antidoom_of` | **73,2%** (IC contra o Lumi −0,8 a +8,1) | 95,6% | 85,8% (+4,1; IC +1,6 a +6,7) | candidato; mediana 720 tokens |
| `thor_pt_antidoom` (nossa portagem) | 73,2% | 93,0% | 84,0% | fecha 0,01 ponto abaixo do limite da regra da suíte |
| `gspo_thor_antidoom` | 69,9% | 93,2% | 86,1% | RL e depois Antidoom |
| `gspo1` (contingência) | 67,5% | 93,2% | 84,3% (+2,7; IC +0,1 a +5,2) | SFT `adalberto` + 8 ciclos de RL; nunca rodou na suíte de teste |
| `thor_pt_antidoom_s1` / `_s2` | 72,5% / 74,0% | 91,95% / 95,3% | 85,4% / 85,7% | sementes 1 e 2 da portagem (H65); laço em `heldout` 3,6% / 0,8% |
| `thor_pt` (SFT, sem Antidoom) | 54,3% | – | 77,8% | o dano era laço: 40% dos prompts |

**Teste (suíte completa, `thor_pt_antidoom_of`, com thinking; comparação principal na 4090). Cada coluna do Lumi traz a nota dele e, entre parênteses, a diferença do modelo (IC 95% pareado):**

| Prova (n) | Nosso | Lumi com thinking | Lumi sem thinking | Fecha `</think>` (Lumi com thinking) | Português |
| --- | --- | --- | --- | --- | --- |
| GSM8K-PT (1.295) | 90,3% | 88,7% (+1,6; 0,0 a +3,2) | 85,3% (+5,0; +3,3 a +6,7) | 99,8% (96,8%) | 99,8% |
| MMLU-ProX PT (11.759) | 54,5% | 53,6% (+0,9; +0,1 a +1,7) | 48,8% (+5,6; +4,8 a +6,5) | 82,6% (73,4%) | 99,0% |
| ENEM (1.431) | 83,9% | 80,2% (+3,7; +1,8 a +5,7) | 78,8% (+5,1; +3,0 a +7,2) | 97,1% (91,3%) | 99,8% |
| BLUEX (722) | 76,9% | 72,0% (+4,8; +1,8 a +7,8) | 71,2% (+5,7; +2,2 a +9,0) | 95,7% (83,5%) | 97,3% |
| OAB (2.210) | 44,0% | 21,2% (+22,8; +20,6 a +24,9) | 43,5% (+0,5; −1,8 a +2,6) | 89,4% (39,2%) | 100% |
| **Nota (média das 5)** | **69,9%** | 63,2% (**+6,8; +5,9 a +7,6**) | 65,5% (**+4,4; +3,4 a +5,4**) | simples 92,9% (76,9%); ponderada 86,5% (72,7%) | – |

Contra o Lumi rodado na Thor (comparação secundária): Nota +6,8 (+5,9 a +7,7) com thinking e +3,9 (+2,9 a +4,8) sem thinking; GSM8K +1,4 (−0,2 a +3,0) e +5,0 (+3,3 a +6,7). Lumi na Thor: Nota 63,1% com thinking e 66,1% sem. Os ICs não incluem a variância de treino: o candidato é uma execução (as sementes da portagem variam 1,5 ponto de múltipla escolha na validação).

**Leitura honesta:** o ganho vem de **fechar o `</think>` dentro de 4.096 tokens** e de raciocinar em português, não de raciocinar melhor: nos itens que o Lumi também fecha, o modelo acerta 0,3 a 2,7 pontos a menos por item (MMLU-ProX 70,9% contra 73,6%). O limite de 4.096 tokens do protocolo favorece quem fecha cedo (o Lumi trunca 60,9% do OAB); com thinking o modelo usa mediana de 358 a 1.597 tokens contra 149 a 216 do Lumi sem thinking, para +4,4 pontos de Nota.

## 4. Filas (4090, sessões `tmux`): todas terminaram

| Sessão | O que fez | Terminou |
| --- | --- | --- |
| `noite13` | Fase anti-laço e, por último, a suíte de teste do `thor_pt_antidoom_of` (`scripts/run_suite_models.sh`) | 04:51 de 10/11 |
| `noite14` | Sonda da 2ª rodada do Antidoom (H64, sem material) e sementes 1 e 2 do treino anti-laço (H65, reproduzem). Não rodou suíte de teste | 05:53 de 10/11 |
| `noite15` | Lumi original na 4090 em MMLU-ProX PT (sem thinking), ENEM, BLUEX e OAB (os dois modos) e as tabelas de comparação (`runs/jobs/noite15_comparacao.md`, H66). Não avaliou modelo nosso | 07:29 de 10/11 |

Estado em `runs/jobs/<nome>.status`, `.log` e `.exit`; `tmux ls` deve estar vazio. A GPU é compartilhada com outros usuários: não mexer nos processos deles.

## 5. Próximos passos, em ordem

1. **Decidir o limiar de fechamento (pessoa).** Ver seção 1: o conjunto em que o ≥ 90% (original ≥ 98%) vale não está no PRD, e os resultados já são conhecidos. A escolha deve ser datada e declarada como posterior aos resultados, e o relatório mostra todas as leituras.
2. **Decidir a entrega (pessoa).** Pelo PRD, o `thor_pt_antidoom_of` cumpre a meta mínima da Nota e do GSM8K-PT e o português; o fechamento depende do item 1. O `gspo1` (contingência) nunca rodou na suíte de teste: rodá-la seria um segundo candidato no teste, só com autorização e com as duas suítes no relatório, e o `gspo1` tem 67,5% na validação contra 73,2%, então só faria sentido se o item 1 reprovasse o candidato e se esperasse que o `gspo1` passasse.
3. **Formato da entrega:** o modelo fundido em BF16 é `runs/merged/thor_pt_antidoom_of` (7,5 GB). O adaptador é uma **pilha de dois**: o LoRA r=16 do `thor_pt` (`runs/train/thor_pt/adapter`) e o LoRA r=256 com `lm_head` do Antidoom (`runs/antidoom/thor_pt_of/adapter`, 2,3 GB em disco), fundido por `scripts/merge_on_base.py` sobre o `thor_pt` já fundido. Decidir se o adaptador é entregue como pilha ou se só o fundido é o artefato principal.
4. **Relatório reproduzível:** tabelas por prova com IC pareado (4090 principal, Thor secundária), decomposição por quadrante, as leituras do fechamento, o efeito de máquina do BLUEX sem thinking, o limite de 4.096 tokens, a variância de treino (H65) e o custo em tokens. `scripts/gen_estudos_resultados.py` regenera [estudos/resultados_auto.md](estudos/resultados_auto.md); o README em inglês ainda descreve o estado de 08/11 (com uma nota no topo apontando para este arquivo) e precisa ser reescrito com os resultados finais.
5. **Troca de candidato** só se um modelo novo tiver, na validação, múltipla escolha ≥ 75,2%, fecha ≥ 95,6%, GSM8K limpo ≥ 84,8% e português ≥ 90%. Nesse caso a suíte dele **só roda com autorização** e as duas suítes entram no relatório. Rodar a suíte de vários candidatos e escolher o melhor seria seleção no teste.
6. **Propostas fora da fila** (só para perseguir a meta alvo da Nota ou o fechamento por prova; ver [estudos/07](estudos/07_hipoteses_e_proximos_passos.md)): H62, RL só com prompts difíceis partindo do modelo novo (exige adaptar `grpo_train.py` e `run_grpo.sh` para partir de um modelo fundido; o RL sobre o `thor_pt` custou 4,5 pontos por item contra 1,0 do Antidoom, e a ordem Antidoom e depois RL nunca foi testada); H63, SFT com traços de múltipla escolha ≤ 512 tokens. O laço já não é o gargalo (H64); o que resta do não fechamento é raciocínio longo.
7. **Commits:** só quando uma pessoa pedir (identidade `gustavobarrosss`, sem linha de coautoria). O último commit é `60b2142`; STATUS.md, discovery.md, estudos/ e README.md têm mudanças locais posteriores.

## 6. Regras do protocolo

- **Conjuntos de teste nunca entram em treino nem em seleção.** A suíte de teste roda **uma vez por candidato**, no fim; nunca em vários para escolher o melhor.
- **Regras de leitura são escritas antes dos resultados.** Emendas posteriores são datadas e sinalizadas como emendas (modelo: a do limiar de fechamento no PRD e a da H66 em [estudos/07](estudos/07_hipoteses_e_proximos_passos.md)).
- **Citar sempre os dois limiares de fechamento** (≥ 98% original e ≥ 90% emendado).
- Não treinar com os datasets do cartão do Lumi, não publicar no Hugging Face, não alterar o `ceia-br/lumi-4b` original.
- A Thor (Jetson AGX Thor) **só gera dados** (pacotes em [handoff/](handoff/README.md)) e rodou a suíte do Lumi; treino e avaliação dos modelos ficam na 4090 (RTX 4090 de 24 GiB, compartilhada, sem sudo nem docker).

## 7. Como operar

- Sempre `source env.sh` antes de qualquer comando Python/HF (nada de cache fora da pasta do projeto). Treino no `.venv`; avaliação e geração no `.venv-vllm` (`VLLM_USE_FLASHINFER_SAMPLER=0`).
- Jobs longos desacoplados: `scripts/detach.sh [--gpu] <nome> <comando>` (sessão `tmux` própria; `--gpu` recusa lançar com a GPU ocupada).
- **Não editar um `.sh` em execução** (o bash lê aos poucos). **Não matar a sessão `tmux` de que outra depende:** o dependente dispara na hora (exemplo: a `noite14` esperava a `noite13`, e a `noite15`, a `noite14`).
- **Sem testes de GPU com a fila rodando:** `python -m pytest tests --ignore=tests/test_phase0_model.py` (410 testes de CPU passam).
- Avaliação: greedy, 4.096 tokens, `eval_vllm.py`; retomável (pula o que já está no `summary.json`). Bootstrap pareado por item, 10.000 reamostras: `scripts/compare_runs.py`, `scripts/report_compare.py`, decomposição por quadrante de fechamento em `scripts/decompose_paired.py`.
- Regra única de candidato para a suíte: `scripts/pick_suite_candidate.py` (múlt. escolha ≥ 69,6%, fecha ≥ 93%, GSM8K limpo ≥ 83%, português ≥ 90%; vence a maior múlt. escolha; padrão `gspo1`).

## 8. Armadilhas conhecidas

- O NemotronH nativo do Transformers não tem `.backbone` (o decodificador é `model.model`); o PEFT recusa LoRA no `out_proj` e no `conv1d` do Mamba; o TRL 1.17 recusa LoRA no `lm_head` (contornado só no braço do treinador oficial).
- O vLLM dá segfault intermitente: os scripts de fila repetem até 3 vezes.
- Uma rodada tem ruído de ±3 a 4 pontos na validação (385 itens de múlt. escolha); diferenças menores que isso, sem IC, não decidem nada. A estimativa de "teto" por itens em laço é só um limite inferior aproximado.
- Na Thor, `nvidia-smi` não informa a memória (`[N/A]`): os scripts de espera de GPU são para a 4090.
- Números citados à mão já saíram errados antes (recompensa, itens por ponto): conferir contra os `summary.json` e `resultados_auto.md` antes de escrever.

## 9. Mapa dos documentos

| Para saber | Onde |
| --- | --- |
| Critério, plano e emenda | [PRD.md](PRD.md) |
| Estudos, achados e hipóteses (H64 a H66 resolvidas; H62 e H63 propostas) | [estudos/README.md](estudos/README.md), [estudos/07](estudos/07_hipoteses_e_proximos_passos.md) |
| Por que cada decisão e cada fila mudou | [estudos/08](estudos/08_registro_de_decisoes.md) |
| Cada experimento, com números | [estudos/06](estudos/06_registro_experimentos.md), [estudos/resultados_auto.md](estudos/resultados_auto.md) |
| Cronologia técnica numerada (§20.56 a §20.68: o Antidoom, a suíte de teste e as sementes) | [discovery.md](discovery.md) |
| Scripts, filas (`run_noite*_chain.sh`) e testes | [scripts/README.md](scripts/README.md) |
| Pacotes de dados da Thor | [handoff/README.md](handoff/README.md) |
| Receita oficial do Antidoom usada e sua licença | `third_party/antidoom/`, `src/lumi_ft/train/antidoom_oficial.py` |

Fora do Git de propósito: `runs/`, `data/`, `comparacao.md`, `CLAUDE.md` e `tests/test_thor4_filter.py`.
