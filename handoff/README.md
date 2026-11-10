# handoff/

Pacotes de geração de dados para a Jetson Thor. A Thor **só gera**; treino e avaliação ficam na 4090. Cada pasta traz o plano (`PLANO_*.md`), o cliente de geração (`gen_traces.py`), o filtro e o `pack.sh`. O que a Thor devolve (`*_out.tar.gz`) vai para `data/` na 4090, que não é versionada.

| Pacote | Pasta | O que gera | Professor | Estado | No Git |
| --- | --- | --- | --- | --- | --- |
| 1 | `thor/` | Traços de matemática em português | Qwen3-30B-A3B-Thinking | 50.908 problemas (5.612 do GSM8K-PT e 45.296 do GigaVerbo); **sem devolução registrada** nos estudos (D7, 08/11 11:00: gerando) | sim |
| 2 | `thor2/` | Matemática (português) e ciência (inglês) | Nemotron 3 Nano 30B-A3B | 21.612 prompts; o que virou treino (`thor_pt`, `thor_en`) chegou **traduzido pelo pacote 3** (5.278 registros, 08/11 11:47) | sim |
| 3 | `thor3/` | Tradução das questões e dos traços, mais questões de outros domínios | Nemotron 30B | 12.000 questões traduzidas entregues; blocos 3 e 4 de traços **cancelados** em 08/11 | sim |
| 4 | `thor4/` | Traços de múltipla escolha **em português** (9.957 questões com gabarito) | Qwen3-30B-A3B-Thinking, começo do raciocínio forçado em português | pacote pronto e validado; **aguarda a Thor** | **não** (decisão de 08/11) |
| 5 | `thor5/` | **Avaliação, não geração:** Lumi e Nemotron 3 Nano 4B nas provas novas do plano de 10/11 (OAB 2018–2025, ENEM 2024, fatias do MMLU-PT) e o Nemotron na suíte do PRD | – | **rodando na Thor desde 09:01 de 10/11** (`thor5_pacote.tar.gz`, não versionado); a devolução entra com `scripts/import_thor_eval.py` | só o README |

**Pacote 3, entrega 1 (08/11, 20:38):** `thor3_out.tar.gz`, sha256 `768372c52d01112e85fee14eaaf5b353ae0fa577a31795b7883a4ac513158e64`, extraído em `data/thor3_e1/` (não versionado). A Thor traduziu com uma **cópia modificada** do `translate_thor.py` (prefill na resposta, limpeza de espaços especiais, reserva frase a frase, processamento contínuo); o diff está em `NOTAS.md` da entrega e o arquivo reconstruído, em `handoff/thor3/translate_thor_v2.py` (o diff aplica sem rejeições no original). A versão que gerou os dados é a anterior ao teste A/B das 00:38: para reproduzi-la, rode com `T3_SEM_2A_TENTATIVA=1`. Os limites de aceitação (`check_piece`) são os do original.

Regras dos pacotes: a Thor não altera scripts nem parâmetros de geração, não publica nada no Hugging Face, roda tudo em `tmux` e anota divergências em `out/NOTAS.md`. Um servidor de modelo por vez (Qwen e Nemotron não cabem juntos nos ~122 GiB).

Arquivos pesados ficam ignorados pelo Git (`*/data/`, `*/out/`, `*/.venv-gen/`, `*.tar.gz`). O pacote 4 inteiro e o `thor4_pacote.sha256` também. Como conferir um pacote recebido: `sha256sum -c` no arquivo `.sha256` ao lado do `.tar.gz`.

Por que cada pacote existe e o que mudou o plano: [estudos/05](../estudos/05_diagnosticos_e_referencias.md) (D7), [discovery.md](../discovery.md) §17, §20.11, §20.18, §20.22, §20.40 e [estudos/08](../estudos/08_registro_de_decisoes.md).
