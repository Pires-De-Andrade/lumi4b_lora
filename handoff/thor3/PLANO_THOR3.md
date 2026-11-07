# Plano da Thor (3): traduzir com o Nemotron 30B e gerar questões de outros domínios

Este arquivo é para o Claude Code da Jetson Thor. Leia tudo e execute os passos 0 a 5 em ordem. Pare e reporte ao usuário em cada **PARE E REPORTE**.

## Contexto

- O pacote 2 (`thor2/`) fez o Nemotron 3 Nano 30B-A3B (`teacher`) gerar traços de raciocínio em inglês. Este pacote usa **o mesmo servidor** para duas coisas:
  1. **Traduzir para o português do Brasil** os exemplos mantidos do pacote 2 (e depois os deste pacote): enunciado e alternativas, raciocínio e texto final. Antes, a tradução era feita na RTX 4090 pelo Lumi-4B, que muitas vezes respondia a questão em vez de traduzir (só 35% dos raciocínios aproveitados). O 30B deve traduzir melhor.
  2. **Gerar traços para 12.000 questões novas de múltipla escolha com gabarito**, de domínios que faltavam: 7.000 do WebInstruct-verified (negócios, economia, finanças, matemática, física, química, biologia, psicologia, história, filosofia, direito) e 5.000 do MedMCQA (21 áreas médicas). Prompt em inglês terminando em `Resposta: <letter>`, como as de ciência do pacote 2.
- A tradução usa o modo **sem raciocínio** (`enable_thinking=False`) e temperatura 0. Cada pedaço traduzido é conferido (terminou, tamanho compatível, números preservados, não respondeu a questão). Um pedaço ruim descarta o exemplo inteiro, com o motivo em `verdict`.
- Só 70% dos problemas têm o raciocínio traduzido: os outros 30% entram no treino sem raciocínio (a divisão é por hash do id, igual à do treino). O original em inglês fica no registro.
- A Thor só **gera e traduz**. Treino e avaliação ficam na RTX 4090. Não treine nada aqui.
- Entrega: `thor3_out.tar.gz` (gerado pelo `pack.sh`, pode ser parcial).
- **Prioridade**, se faltar tempo: (1) traduzir o pacote 2; (2) gerar o pacote 3; (3) traduzir o pacote 3.

## Regras

1. Não altere `translate_thor.py`, `gen_traces.py`, `filter_traces_en.py` nem `data/prompts.jsonl`, e não mude os limites de aceitação. Se achar um defeito, anote em `out/NOTAS.md` e avise o usuário.
2. Instale pacotes só em venv ou contêiner (reaproveite o `.venv-gen` do pacote 2). Não mexa em outros projetos da máquina. Não publique nada no Hugging Face. Não grave tokens ou senhas em nenhum arquivo do pacote.
3. Anote em `out/NOTAS.md` (crie a pasta `out/`) tudo que divergir do plano: comando, versão, erro e alternativa escolhida.
4. Rode em `tmux` tudo que dure mais de 1 minuto.
5. **Memória:** um servidor de cada vez. Se o servidor do Qwen (pacote 1) ainda estiver de pé, pergunte ao usuário antes de subir o Nemotron.
6. Pode traduzir enquanto a geração do pacote 2 ainda roda (o servidor divide a vazão entre os dois clientes). Se a geração do pacote 2 ainda não terminou, **não a interrompa**.

## Arquivos desta pasta

| Arquivo | Para quê |
| --- | --- |
| `translate_thor.py` | Tradução com o professor (`/v1/completions`, sem raciocínio, T=0), retomável. Saída: `out/translated.jsonl` e `out/translated.jsonl.report.json` |
| `data/prompts.jsonl` | 12.000 questões novas, embaralhadas (`id`, `source` = `webinstruct` ou `medmcqa`, `kind` = `mcq`, `category`, `question`, `options`, `prompt`, `ref` = gabarito). Confira com `cd data && sha256sum -c prompts.sha256` |
| `gen_traces.py`, `filter_traces_en.py` | Os mesmos arquivos do pacote 2 (cliente de geração e filtro) |
| `pack.sh` | Empacota tudo o que já existir em `thor3_out.tar.gz` (a qualquer hora) |

## Passo 0: situação e ambiente (10 min)

```bash
mkdir -p out
export THOR2=<pasta do pacote 2>   # a que tem gen_traces.py, data/ e out/traces.jsonl do Nemotron
export TOK=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16   # ou o id que o pacote 2 usou (veja $THOR2/out/NOTAS.md)
{ date; free -g; df -h .; nvidia-smi; curl -s localhost:8000/v1/models | head -c 300; echo; } 2>&1 | tee out/env_info.txt
cd data && sha256sum -c prompts.sha256 && cd ..
. $THOR2/.venv-gen/bin/activate 2>/dev/null || { python3 -m venv .venv-gen && . .venv-gen/bin/activate && pip install -q transformers; }
python3 $THOR2/filter_traces_en.py --traces $THOR2/out/traces.jsonl --prompts $THOR2/data/prompts.jsonl --outdir $THOR2/out | head -20
wc -l $THOR2/out/traces.jsonl $THOR2/out/kept.jsonl
python3 translate_thor.py --show-template --kept x --tokenizer $TOK --model-name teacher
```

Esperado: o servidor `teacher` respondendo em `/v1/models`, o hash `OK`, um `kept.jsonl` do pacote 2 com alguns milhares de linhas e o template terminando em `<think></think>` (sem raciocínio). Anote se a geração do pacote 2 ainda está rodando e quantos exemplos já tem.

Se o servidor não estiver de pé, suba-o com **o mesmo comando do pacote 2** (anotado em `$THOR2/out/NOTAS.md` ou no relatório 1 dele).

## Passo 1: teste de fumaça da tradução, 30 exemplos (5 a 10 min)

```bash
python3 translate_thor.py --kept $THOR2/out/kept.jsonl --out out/smoke_tr.jsonl --limit 30 --block 30 \
  --tokenizer $TOK --model-name teacher --concurrency 64
python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open("out/smoke_tr.jsonl")]
ok = [r for r in rows if r["verdict"] == "ok"]
for r in [x for x in ok if x["with_thought"]][:2]:
    print("=" * 80, r["id"], r["kind"])
    print("EN:", r["question_en"][:300]); print("PT:", r["question_pt"][:300])
    print("EN:", r["thought_en"][:700]); print("PT:", r["thought_pt"][:700])
    print("RESPOSTA PT:", r["answer_pt"][-300:])
for r in [x for x in rows if x["verdict"] != "ok"][:3]:
    print("-" * 80, r["id"], r["verdict"], json.dumps(r.get("pieces", {}), ensure_ascii=False)[:500])
EOF
```

Leia os 2 exemplos e confira: português natural, números e fórmulas iguais aos do inglês, alternativas na mesma ordem, a resposta termina em `Resposta: X` (ou `#### N`), e o tradutor **não** resolveu a questão no lugar de traduzir.

**PARE E REPORTE 1:** o `out/smoke_tr.jsonl.report.json` (aproveitamento e vereditos), os 2 exemplos lado a lado (resumidos) e os motivos das recusas. Regra: com aproveitamento menor que 60%, pare aqui e mostre 3 recusas. Com 60% ou mais, siga.

## Passo 2: traduzir o pacote 2 inteiro (em tmux; ~1 a 3 h, depende da vazão)

```bash
tmux new -s trad
python3 translate_thor.py --kept $THOR2/out/kept.jsonl --out out/translated.jsonl \
  --tokenizer $TOK --model-name teacher --concurrency 96
```

- Retomável: se cair, rode o mesmo comando. Se a geração do pacote 2 ainda roda, refaça o `filter_traces_en.py` do passo 0 de tempos em tempos e rode a tradução de novo: ela só traduz os ids novos.
- Velocidade: a linha de progresso mostra `tokens/s` e exemplos feitos. Se houver muitos `erro_de_requisicao`, reduza `--concurrency`.
- **Assim que terminar** (ou a cada ~1 h), rode `bash pack.sh` e avise o usuário: ele já pode levar a entrega parcial para a 4090 e treinar.

**PARE E REPORTE 2:** o `out/translated.jsonl.report.json`, a velocidade (exemplos/min e tokens/s), e o `sha256` do `thor3_out.tar.gz`.

## Passo 3: gerar os traços das 12.000 questões novas (em tmux; até ~3 h)

Depois que a geração do pacote 2 terminar. Use **os mesmos parâmetros de geração do pacote 2** (sem `--prefill`; temperatura, top-p, top-k e max-tokens iguais aos do relatório do pacote 2).

```bash
python3 gen_traces.py --tokenizer $TOK --model-name teacher --prompts data/prompts.jsonl --out out/smoke.jsonl --limit 20 --concurrency 20 \
  --prefill "" --temperature 1.0 --top-p 1.0 --top-k 0 --max-tokens 6144
python3 filter_traces_en.py --traces out/smoke.jsonl --prompts data/prompts.jsonl --outdir out/smoke && cat out/smoke/report.json
# se o rendimento for de 30% ou mais:
python3 gen_traces.py --tokenizer $TOK --model-name teacher --prompts data/prompts.jsonl --out out/traces.jsonl --start 20 --limit 12000 \
  --concurrency 96 --time-budget-min 180 --prefill "" --temperature 1.0 --top-p 1.0 --top-k 0 --max-tokens 6144
python3 filter_traces_en.py --traces out/traces.jsonl --prompts data/prompts.jsonl --outdir out
```

Aqui a referência é o **gabarito oficial** (não a maioria de outro modelo, como no pacote 2), então o rendimento pode ser menor nas áreas difíceis. Não mude o filtro.

## Passo 4: traduzir o pacote 3

```bash
python3 translate_thor.py --kept $THOR2/out/kept.jsonl --kept out/kept.jsonl --out out/translated.jsonl \
  --tokenizer $TOK --model-name teacher --concurrency 96
```

(O mesmo arquivo de saída: os exemplos já traduzidos do pacote 2 são pulados.)

## Passo 5: empacotar e entregar

```bash
bash pack.sh
```

O usuário copia para a outra máquina: `scp thor3_out.tar.gz <usuario>@<host-da-4090>:~/barros/peft_lora_study/data/thor3/`.

**PARE E REPORTE 3 (final)**, com números reais:

```
Servidor / flags: ... (o mesmo do pacote 2? ...)
Tradução: exemplos ... | ok ... (com raciocínio ...) | aproveitamento ...% | vereditos: ...
Velocidade da tradução: ... exemplos/min, ... tokens/s | tempo: ... h
Pacote 3: gerados ... | mantidos ... (webinstruct ..., medmcqa ...) | rendimento ...% | funil: ...
Divergências do plano (de out/NOTAS.md): ...
Arquivo: thor3_out.tar.gz, sha256 ...
```

## Problemas comuns

| Sintoma | O que fazer |
| --- | --- |
| O template do `--show-template` não termina em `<think></think>` | O script já fecha o raciocínio sozinho; confira no fumaça se as traduções vêm sem raciocínio. Se vierem com `<think>`, anote e reporte |
| Muitos `respondeu_em_vez_de_traduzir` ou `tamanho_incompativel` | O tradutor está resolvendo a questão. Mostre 3 exemplos (`pieces`) e reporte; não mude o prompt nem os limites |
| Muitos `numeros_divergentes` | Mostre 3 exemplos e reporte. Não mude os limites |
| Muitos `q:alternativas_nao_voltaram` | O tradutor mudou o formato `A) ...`. Mostre 2 exemplos e reporte |
| `erro_de_requisicao` alto | Servidor sobrecarregado: reduza `--concurrency` |
| Erro de memória no servidor | Mesmo procedimento do pacote 2 (FP8, `--max-num-seqs` menor) |

## Critérios de aceite

- Fumaça com aproveitamento de pelo menos 60%, e exemplos lidos que fazem sentido.
- Pelo menos 70% dos exemplos do pacote 2 traduzidos com `ok`.
- Pacote 3: pelo menos 3.000 exemplos mantidos (meta: 6.000).
