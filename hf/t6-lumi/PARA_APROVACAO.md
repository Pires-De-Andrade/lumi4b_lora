# T6-Lumi no Hugging Face: o que aprovar antes de publicar

10/11/2026. **Nada foi publicado.** O card é o [README.md](README.md) desta pasta (é exatamente o texto que vai para o HF; está em inglês, como o do Lumi-4B). Este arquivo é só para você e não vai para o HF.

## O que o card diz (em uma frase por seção)

| Seção | Conteúdo |
| --- | --- |
| Abertura e aviso | T6-Lumi é o Lumi-4B-v0.2 ajustado para o modo thinking (SFT com traços do Nemotron 30B traduzidos + fase anti-laço Antidoom/FTPO). **Usar com thinking ligado** (o card do Lumi manda desligar). Sem RL, sem identidade, sem tool use |
| Highlights | 97,3–100% do raciocínio em português; fecha o `</think>` em 82,6–99,8% (Lumi 39,2–96,8%); Nota 69,9% contra 63,2% (+6,8) e 65,5% (+4,4) |
| Desempenho | Protocolo, tabela das cinco provas de teste com IC e as figuras 1 a 4 |
| "How to read these results" | O ganho vem de fechar o raciocínio no limite de 4.096 tokens; por item o T6-Lumi acerta 0,3 a 2,7 pontos a menos; OAB só empata com o Lumi sem thinking; custo em tokens; seleção e variância |
| Como foi feito | Figura 5 e tabela das quatro execuções do Antidoom (72,5% a 74,0%) |
| Uso | vLLM offline (o protocolo avaliado) e servidor OpenAI-compatível; limites de uso |
| Treino e dados | Hiperparâmetros das duas etapas, dados, descontaminação |
| Avaliação em andamento | Provas novas, 8.192 tokens, amostragem, outras execuções, revisão humana: **sem números** |
| Riscos, licença, citação | Ver pendências abaixo |

Escolhi mostrar as limitações no mesmo peso que os ganhos: o registro do projeto (`discovery.md` §20.68, H73) já diz que todo o ganho é de fechamento, e um card que dissesse só o +6,8 contradiria o repositório.

## Decisões suas (estão marcadas como `DECISION` ou `TODO` no card; o `publicar.py` recusa enviar enquanto houver alguma)

1. **Onde publicar e o nome.** Seu token do HF é de escopo fino e escreve na organização `CEIA-Ultimatum`; `ceia-br` (a organização do Lumi) não aparece na sua conta, então eu não consigo nem você subir lá sem que um admin te dê acesso. Sugestão de id: `CEIA-Ultimatum/t6-lumi-4b` (o `<repo-id>` do card é trocado na montagem).
2. **Licença.** O Lumi-4B-v0.2 não declara licença; o Nemotron 3 Nano 4B (pai) usa a NVIDIA Nemotron Open Model License, e os traços do professor vêm de um modelo sob a mesma licença. Eu não escolhi: o card traz um `TODO` e a frase com esses fatos. O ideal é combinar com quem mantém o Lumi.
3. **Autores, contato, agradecimentos e data.** Pus os dois GitHubs (`@gustavobarrosss`, `@Pires-De-Andrade`) e não pus e-mail nem nomes completos (o BibTeX tem `TODO`). Falta a data de lançamento.
4. **Logo da CEIA.** O card do Lumi usa o logo; não incluí (não é necessário para rodar e usa a marca da instituição). Se quiser, é uma linha.

## O que conferi e o que não

- **Números:** cada número vem do `discovery.md` (§20.64, 20.66, 20.68) e dos estudos; `tests/test_card_figures.py` confere médias, diferenças pareadas e quadrantes entre si (passam). O GSM8K do T6-Lumi aparece como 90,3 (1.170 de 1.295 = 90,3475; o registro arredondava para 90,35).
- **Dados de treino:** o T6-Lumi **não usa** o dataset `adalberto` (sem licença) nem as provas novas. Usa GSM8K-PT (MIT), GigaVerbo-v2 SFT (Apache-2.0), Mixture-of-Thoughts (**sem licença nos metadados do HF**; vale conferir antes de tornar público), WebInstruct-verified e MedMCQA (Apache-2.0).
- **Trechos de código do card não foram executados** (esta máquina não tem GPU nem vLLM). O de vLLM offline segue linha a linha o `scripts/eval_vllm.py`; o do servidor é modelo, como no card do Lumi, e diz isso.
- **Medianas de tokens do Lumi** em GSM8K, ENEM, BLUEX e OAB vêm da rodada da Thor (a 4090 só tem o MMLU-ProX nesse número); está na nota da tabela.
- **Provas novas (H74), Nemotron (H72), 8.192 tokens (H67) e amostragem (H68)** ainda estão abertas no registro. Se algum resultado chegar antes de publicar, eu acrescento a seção; o card já diz que esses números não existem ainda.

## O que vai para o HF (e só isso)

`model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `configuration_nemotron_h.py`, `modeling_nemotron_h.py`, `nano_v3_reasoning_parser.py`, `__init__.py` (tudo isso, exceto os pesos, copiado do Lumi na revisão `27b30f40`), `README.md` e as cinco figuras SVG. Nada de `handoff/`, scripts, estudos, dados ou este arquivo: `publicar.py` tem a lista fechada e recusa o resto (`tests/test_publish_t6_lumi.py`).

## Como publicar depois de aprovado

Os pesos (`runs/merged/thor_pt_antidoom_of/model.safetensors`, sha256 `1ea6bea5…44e5`) estão só na máquina da 4090 (o `t6_delta.bin` também não está aqui). Lá:

```bash
python hf/t6-lumi/publicar.py --weights runs/merged/thor_pt_antidoom_of/model.safetensors --repo-id ORG/NOME          # monta e confere, não envia
python hf/t6-lumi/publicar.py --weights runs/merged/thor_pt_antidoom_of/model.safetensors --repo-id ORG/NOME --push   # cria PRIVADO e envia
```

O repositório sai **privado**; você olha a página (figuras, tabelas, links) e só então o torna público. Precisa de `huggingface_hub` e de um token com escrita na organização escolhida.
