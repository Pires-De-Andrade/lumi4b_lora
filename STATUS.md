# Estado do projeto e próximos passos

**Atualizado em 10/11/2026, 03:05.** Ponto de entrada para quem chega sem contexto: o que se quer, onde o projeto está, o que está rodando e o que fazer depois. Este arquivo é **reescrito a cada marco** (não acumula histórico); o histórico fica em [discovery.md](discovery.md) e em [estudos/](estudos/README.md). Idioma dos documentos: português do Brasil, texto impessoal, números em vez de adjetivos.

## 1. Em uma tela

- **Objetivo:** ajustar o `ceia-br/lumi-4b` v0.2 com LoRA para que `enable_thinking=True` raciocine em português e acerte mais que o Lumi original. Entrega: adaptador vencedor, versão fundida em BF16 e relatório reproduzível ([PRD.md](PRD.md)). O adaptador só é usado **com thinking**; sem thinking roda o Lumi original (decisão de 08/11).
- **Candidato atual: `thor_pt_antidoom_of`.** SFT `thor_pt` (raciocínio do Nemotron 30B traduzido para o português) + uma fase de preferência no token que inicia o laço (receita oficial do Antidoom/FTPO, portada). Na validação, múltipla escolha com thinking 73,2% contra 69,6% do Lumi; fecha o `</think>` em 95,6%; traço em português 99,5%. **Contingência:** `gspo1`.
- **Onde estamos:** a suíte de teste (a Nota de raciocínio) do candidato **está rodando na 4090 desde 01:40 de 10/11**. Só o GSM8K-PT terminou: 90,35%, **+5,0 (IC +3,3 a +6,7) contra o Lumi sem thinking**, que é a referência do PRD para essa prova (meta mínima +3,0 com IC acima de 0: cumprida), e +1,6 (IC 0,0 a +3,2) contra o Lumi com thinking. Faltam MMLU-ProX PT, ENEM, BLUEX e OAB.
- **Próximo marco:** ao fim da suíte (previsão 05:00 a 06:00), aplicar o critério do PRD à Nota de raciocínio (seção 2) e decidir a entrega (seção 5).
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

Lumi na Thor (suíte de teste, 09/11, 4.096 tokens, gulosa): Nota **63,10% com thinking e 66,06% sem**. O Lumi sem thinking tem nota maior porque no OAB o thinking trunca em 62,5% das respostas (OAB 20,4% com thinking, 44,3% sem). Por isso o critério mais duro da Nota é o contra o Lumi **sem** thinking, e o OAB é onde provavelmente será decidido (hipótese, ainda sem resultado).

## 3. Números

**Validação** (GSM8K limpo n=676 + múltipla escolha n=385; thinking; ruído de uma rodada ±3 a 4 pontos; a validação escolheu o candidato entre dezenas, então o ganho tende a encolher no teste):

| Modelo | Múlt. escolha | Fecha `</think>` | GSM8K limpo | Observação |
| --- | --- | --- | --- | --- |
| Lumi | 69,6% | 86,2% | 81,7% | referência |
| `thor_pt_antidoom_of` | **73,2%** (IC contra o Lumi −0,8 a +8,1) | 95,6% | 85,8% (+4,1; IC +1,6 a +6,7) | candidato; mediana 720 tokens |
| `thor_pt_antidoom` (nossa portagem) | 73,2% | 93,0% | 84,0% | fecha 0,01 ponto abaixo do limite da regra da suíte |
| `gspo_thor_antidoom` | 69,9% | 93,2% | 86,1% | RL e depois Antidoom |
| `gspo1` (contingência) | 67,5% | 93,2% | 84,3% (+2,7; IC +0,1 a +5,2) | SFT `adalberto` + 8 ciclos de RL |
| `thor_pt` (SFT, sem Antidoom) | 54,3% | – | 77,8% | o dano era laço: 40% dos prompts |

**Teste, até agora:** só o GSM8K-PT, em [discovery.md](discovery.md) §20.66. Resumo: 90,35% (Lumi 88,73% na 4090, 88,96% na Thor; 85,33% sem thinking); fecha 99,8% (Lumi 97,0%); traço em português em 1.290 de 1.295 (Lumi: inglês em 1.254). O ganho contra o Lumi com thinking vem dos 39 itens que o Lumi trunca (acertamos 26); nos 1.256 que ele fecha, 91,08% contra 91,72% (−8 itens).

**Leitura honesta:** o ganho vem principalmente de **fechar o `</think>` dentro de 4.096 tokens** e de raciocinar em português, não de raciocinar melhor nos itens que o Lumi já fecha. O Lumi rodado na Thor e na 4090 tem a mesma média, mas discorda em 57 de 1.295 itens (ruído de numérica que o IC pareado não inclui): a `noite15` mede isso.

## 4. O que está rodando (4090, sessões `tmux`, uma encadeada na outra)

| Sessão | O que faz | Estado | Previsão |
| --- | --- | --- | --- |
| `noite13` | Última etapa: suíte de teste do `thor_pt_antidoom_of` (`scripts/run_suite_models.sh`) | rodando desde 01:40 de 10/11 | acaba 05:00 a 06:00 |
| `noite14` | Espera a `noite13`. Sonda de 3.000 prompts no `thor_pt_antidoom_of` (H64: só escala se achar ≥ 100 eventos de laço) e duas sementes do treino anti-laço (H65). **Não roda suíte de teste.** | em espera | ~1,3 h depois da suíte |
| `noite15` | Espera a `noite14`. Lumi original na 4090 em MMLU-ProX PT (sem thinking), ENEM, BLUEX e OAB (os dois modos), e as tabelas de comparação com IC pareado e nota média (H66). **Não avalia modelo nosso.** | em espera | ~1,5 a 2 h depois da `noite14` |

Como acompanhar: `tmux ls`; `tail runs/jobs/suite.status runs/jobs/noite14.status runs/jobs/noite15.status`; `nvidia-smi`. Cada sessão grava `runs/jobs/<nome>.status`, `.log` e `.exit`. Previsão de tudo pronto: ~08:00 a 09:30 de 10/11. A GPU é compartilhada com outros usuários: não mexer nos processos deles.

## 5. Próximos passos, em ordem

1. **Quando a suíte do `thor_pt_antidoom_of` acabar** (`runs/eval/thor_pt_antidoom_of/summary.json` com as cinco provas nos dois modos):
   - comparar item a item com o Lumi: Thor em `runs/eval/base_thor/` (já extraído de `lumi_suite_base_thor.tar.gz`) e, depois da `noite15`, 4090 em `runs/eval/base/`. Ferramenta: `python scripts/report_compare.py --ref "<rótulo>" --col "<rótulo>=<pasta>:<modo>" ... --tasks gsm8k_test,mmlu_prox_test,enem,bluex,oab --mean` (a `noite15` já grava as duas tabelas em `runs/jobs/noite15_comparacao.md`);
   - a comparação principal do relatório é a da **mesma máquina** (4090 contra 4090); a da Thor entra como secundária (regra H66, escrita antes do resultado).
2. **Ler a `noite14`** (regras escritas antes dos resultados em [estudos/07](estudos/07_hipoteses_e_proximos_passos.md)): H64 (sonda; se < 100 eventos, o laço está esgotado e o que resta do não fechamento, 4,4% da validação, é raciocínio longo) e H65 (as duas sementes reproduzem o ganho se tiverem ≥ 70,0% de múltipla escolha, laço oficial em `heldout` ≤ 5% e fecham ≥ 90%). Registrar em `estudos/06` a `08` e em `discovery.md` (§20.67 em diante).
3. **Troca de candidato** só se um modelo novo tiver, na validação, múltipla escolha ≥ 75,2%, fecha ≥ 95,6%, GSM8K limpo ≥ 84,8% e português ≥ 90%. Nesse caso a suíte dele **só roda com autorização** e as duas suítes entram no relatório. Rodar a suíte de vários candidatos e escolher o melhor seria seleção no teste.
4. **Decisão de entrega** (depende de uma pessoa): se a Nota de raciocínio tiver IC acima de 0 contra os dois modos do Lumi, o candidato vence o PRD. Se as provas grandes tiverem IC incluindo zero ou o candidato perder contra o Lumi sem thinking, a contingência é entregar o `gspo1` e relatar o custo (GSM8K +2,7 significativo na validação; múltipla escolha −2,1, IC −6,8 a +2,3).
5. **Entrega:** o modelo fundido em BF16 é `runs/merged/thor_pt_antidoom_of` (7,5 GB). O adaptador é uma **pilha de dois**: o LoRA r=16 do `thor_pt` (`runs/train/thor_pt/adapter`) e o LoRA r=256 com `lm_head` do Antidoom (`runs/antidoom/thor_pt_of/adapter`, 2,3 GB em disco), fundido por `scripts/merge_on_base.py` sobre o `thor_pt` já fundido. Decidir se o adaptador é entregue como pilha ou se só o fundido é o artefato principal.
6. **Relatório reproduzível:** `scripts/gen_estudos_resultados.py` regenera [estudos/resultados_auto.md](estudos/resultados_auto.md); o README em inglês ainda descreve o estado de 08/11 e precisa ser reescrito com os resultados finais.
7. **Propostas fora da fila** (decisão pendente; ver [estudos/07](estudos/07_hipoteses_e_proximos_passos.md)): H62, RL só com prompts difíceis partindo do modelo novo (exige adaptar `grpo_train.py` e `run_grpo.sh` para partir de um modelo fundido; o RL sobre o `thor_pt` custou 4,5 pontos por item contra 1,0 do Antidoom, mas a ordem Antidoom e depois RL nunca foi testada); H63, SFT com traços de múltipla escolha ≤ 512 tokens.

## 6. Regras do protocolo

- **Conjuntos de teste nunca entram em treino nem em seleção.** A suíte de teste roda **uma vez por candidato**, no fim; nunca em vários para escolher o melhor.
- **Regras de leitura são escritas antes dos resultados.** Emendas posteriores são datadas e sinalizadas como emendas (modelo: a do limiar de fechamento no PRD e a da H66 em [estudos/07](estudos/07_hipoteses_e_proximos_passos.md)).
- **Citar sempre os dois limiares de fechamento** (≥ 98% original e ≥ 90% emendado).
- Não treinar com os datasets do cartão do Lumi, não publicar no Hugging Face, não alterar o `ceia-br/lumi-4b` original.
- A Thor (Jetson AGX Thor) **só gera dados** (pacotes em [handoff/](handoff/README.md)) e rodou a suíte do Lumi; treino e avaliação dos modelos ficam na 4090 (RTX 4090 de 24 GiB, compartilhada, sem sudo nem docker).

## 7. Como operar

- Sempre `source env.sh` antes de qualquer comando Python/HF (nada de cache fora da pasta do projeto). Treino no `.venv`; avaliação e geração no `.venv-vllm` (`VLLM_USE_FLASHINFER_SAMPLER=0`).
- Jobs longos desacoplados: `scripts/detach.sh [--gpu] <nome> <comando>` (sessão `tmux` própria; `--gpu` recusa lançar com a GPU ocupada).
- **Não editar um `.sh` em execução** (o bash lê aos poucos). **Não matar a sessão `tmux` de que outra depende:** o dependente dispara na hora (`noite14` espera a `noite13`; `noite15` espera a `noite14`).
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
| Estudos, achados e hipóteses abertas (H64 a H66) | [estudos/README.md](estudos/README.md), [estudos/07](estudos/07_hipoteses_e_proximos_passos.md) |
| Por que cada decisão e cada fila mudou | [estudos/08](estudos/08_registro_de_decisoes.md) |
| Cada experimento, com números | [estudos/06](estudos/06_registro_experimentos.md), [estudos/resultados_auto.md](estudos/resultados_auto.md) |
| Cronologia técnica numerada (§20.56 a §20.66: o Antidoom e o primeiro resultado do teste) | [discovery.md](discovery.md) |
| Scripts, filas (`run_noite*_chain.sh`) e testes | [scripts/README.md](scripts/README.md) |
| Pacotes de dados da Thor | [handoff/README.md](handoff/README.md) |
| Receita oficial do Antidoom usada e sua licença | `third_party/antidoom/`, `src/lumi_ft/train/antidoom_oficial.py` |

Fora do Git de propósito: `runs/`, `data/`, `comparacao.md`, `CLAUDE.md` e `tests/test_thor4_filter.py`.
