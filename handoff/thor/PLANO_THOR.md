# Plano da Thor: gerar traços de raciocínio em português com um professor grande

Este arquivo é para o Claude Code da Jetson Thor. Leia tudo e execute os passos 0 a 5 em ordem. Pare e reporte ao usuário em cada **PARE E REPORTE**.

## Contexto

- O projeto treina o Lumi-4B (modelo pequeno, português) para raciocinar em português. Falta ensiná-lo com exemplos "problema → raciocínio em português → resposta", feitos por um modelo maior e conferidos.
- A Thor só **gera os dados**. O treino e a avaliação ficam na outra máquina (RTX 4090). Não treine nada aqui.
- Professor: `Qwen/Qwen3-30B-A3B-Thinking-2507` (MoE: 30B no total, 3B ativos; Apache-2.0; ~61 GB em BF16). Cabe nos ~122 GiB compartilhados e é rápido porque só 3B são ativos.
- O professor pensa em inglês por padrão. O truque é **forçar o começo do raciocínio em português** (`--prefill`), que já funcionou no Lumi. O script faz isso; você só precisa conferir o resultado.
- Entrega: `thor_out.tar.gz` (gerado pelo `pack.sh`).

## Regras

1. Não altere `gen_traces.py`, `filter_traces.py` nem `data/prompts.jsonl`. Se achar um defeito, anote em `out/NOTAS.md` e avise o usuário.
2. Instale pacotes só em venv ou contêiner. Não mexa em outros projetos da máquina. Não publique nada no Hugging Face. Não grave tokens ou senhas em nenhum arquivo do pacote.
3. Anote em `out/NOTAS.md` (crie a pasta `out/`) tudo que divergir do plano: comando usado, versão, erro e alternativa escolhida.
4. Rode tudo que dure mais de 1 minuto em `tmux`, para sobreviver à queda do SSH.
5. Pare e reporte se: o servidor não subir em 45 min, o rendimento do piloto for menor que 25%, ou faltar disco/memória.

## Arquivos desta pasta

| Arquivo | Para quê |
| --- | --- |
| `data/prompts.jsonl` | 50.908 problemas embaralhados (`id`, `source`, `question`, `prompt`, `ref`). Confira com `cd data && sha256sum -c prompts.sha256` |
| `gen_traces.py` | Cliente: envia os problemas a um servidor `/v1/completions` e grava `out/traces.jsonl`. Retomável |
| `filter_traces.py` | Mantém só o que fecha `</think>`, em português, com `#### N` igual à referência |
| `pack.sh` | Filtra e empacota tudo em `thor_out.tar.gz` (pode rodar a qualquer hora, com a geração em andamento) |

## Passo 0: diagnóstico e ambiente (10 min)

```bash
mkdir -p out && {
  date; uname -a; nvidia-smi; free -g; df -h .; nproc; python3 --version
  docker --version 2>&1; docker info 2>&1 | grep -i -E "runtimes|server version"
} 2>&1 | tee out/env_info.txt
cd data && sha256sum -c prompts.sha256 && cd ..
python3 -m venv .venv-gen && . .venv-gen/bin/activate && pip install -q transformers
python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen3-30B-A3B-Thinking-2507'); print(t.apply_chat_template([{'role':'user','content':'oi'}], tokenize=False, add_generation_prompt=True))"
```

Esperado: `aarch64`, CUDA 13.x, ~122 GiB, o hash `OK`, e o template terminando em `<think>`. Se o disco livre for menor que 80 GB, pare e reporte.

## Passo 1: subir o servidor do professor (até 45 min)

Objetivo: um servidor em `http://localhost:8000/v1/completions` com o modelo, chamado `teacher`. Tente na ordem abaixo e use o primeiro que funcionar. Consulte a documentação atual (NGC, vLLM, JetPack 7) com busca na web, porque as versões mudam.

**A. Contêiner NVIDIA com vLLM para Thor** (preferido, se `docker` funcionar). Procure no catálogo NGC (`nvcr.io/nvidia/vllm`, tag recente com suporte a Jetson Thor/`sm_110`). Modelo do comando:

```bash
docker run -d --name teacher --runtime nvidia --ipc=host --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  <IMAGEM_NGC_COM_VLLM> \
  vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --served-model-name teacher \
  --max-model-len 10240 --max-num-seqs 96 --gpu-memory-utilization 0.80 --dtype bfloat16 --port 8000
```

**B. vLLM por pip** em outro venv (`python3 -m venv .venv-vllm`; `pip install vllm`). Pode faltar kernel para a GPU da Thor; teste. Mesmo comando `vllm serve ...` de A. Se aparecer erro de compilação do FlashInfer (por exemplo `curand.h`), tente `VLLM_USE_FLASHINFER_SAMPLER=0`.

**C. Outro motor com `/v1/completions`**: SGLang, TensorRT-LLM ou `llama-server` do llama.cpp com um GGUF Q8_0 do mesmo modelo. O nome servido deve ser `teacher` (ou ajuste `--model-name`). Se o motor não aceitar `top_k`, passe `--top-k 0` ao `gen_traces.py`.

**D. Se a memória não bastar** (OOM): baixe `--max-model-len` para 8192 e `--max-num-seqs` para 64. Se ainda falhar, use `Qwen/Qwen3-8B` como professor (explique no relatório; é um professor mais fraco).

Teste (deve responder em menos de 30 s depois do aquecimento):

```bash
curl -s localhost:8000/v1/models | head -c 300; echo
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"teacher","prompt":"<|im_start|>user\nQuanto é 17 vezes 23?<|im_end|>\n<|im_start|>assistant\n<think>\nVou raciocinar em português.","max_tokens":300,"temperature":0.6}'
```

**PARE E REPORTE 1:** motor e versão, tempo de subida, memória usada (`free -g`/`nvidia-smi`), e a resposta do `curl` (o texto deve estar em português).

## Passo 2: teste de fumaça, 20 problemas (10 min)

```bash
. .venv-gen/bin/activate
python gen_traces.py --tokenizer Qwen/Qwen3-30B-A3B-Thinking-2507 --model-name teacher \
  --prompts data/prompts.jsonl --out out/smoke.jsonl --limit 20 --concurrency 20
python filter_traces.py --traces out/smoke.jsonl --prompts data/prompts.jsonl --outdir out/smoke
cat out/smoke/report.json; sed -n 1,60p out/smoke/samples.md
```

Leia pelo menos 3 exemplos de `out/smoke/samples.md` e confira:

- o traço está em português (campo `idioma_do_traço_entre_os_que_fecham`: a maioria `pt`);
- `finish_reason` quase todo `stop`;
- aparece `#### N` na resposta (em `funil`, `sem_####_na_resposta` deve ser pequeno);
- o rendimento (`mantidos` ÷ `gerados`) é de pelo menos 25%.

Se o traço **não** sair em português (mais de 30% `en` ou `mixed`), repita o smoke trocando o começo forçado, na ordem, e fique com o melhor (maior fração `pt`, depois maior rendimento):

1. `--prefill "Vou raciocinar em português do Brasil, passo a passo."`
2. `--prefill "Certo, vou resolver este problema em português. Primeiro,"`
3. `--prefill "Preciso resolver isso. Vou pensar sempre em português."`

Use `--out out/smoke_v1.jsonl` (e `_v2`, `_v3`) para não misturar. Registre a escolha em `out/NOTAS.md` e use o mesmo `--prefill` nos passos 3 e 4.

## Passo 3: piloto, 300 problemas (15 a 30 min)

```bash
tmux new -s teacher
python gen_traces.py --tokenizer Qwen/Qwen3-30B-A3B-Thinking-2507 --model-name teacher \
  --prompts data/prompts.jsonl --out out/traces.jsonl --start 20 --limit 300 --concurrency 96
python filter_traces.py --traces out/traces.jsonl --prompts data/prompts.jsonl --outdir out
cat out/report.json
```

Calcule: velocidade (`problemas/min` da última linha de progresso) e o tempo para 20.000 problemas: `20000 / problemas_por_min / 60` horas.

**PARE E REPORTE 2:** `out/report.json`, a velocidade, a estimativa para 20.000 problemas, o `--prefill` escolhido e 2 exemplos de `out/samples.md`. Regra: se o rendimento for menor que 25%, pare aqui. Se for maior ou igual, siga.

## Passo 4: geração completa (até ~5 h, em tmux)

```bash
python gen_traces.py --tokenizer Qwen/Qwen3-30B-A3B-Thinking-2507 --model-name teacher \
  --prompts data/prompts.jsonl --out out/traces.jsonl --start 20 --limit 20000 \
  --concurrency 96 --time-budget-min 300
```

- É retomável: se cair, rode o mesmo comando. Os 300 do piloto não são refeitos.
- Se quiser mais velocidade e a memória sobrar, aumente `--concurrency` (e `--max-num-seqs` do servidor). Se houver muitos `erro_de_requisição` no relatório, diminua.
- A cada ~1 h, rode `bash pack.sh` (passo 5) para gerar uma **entrega parcial**: o usuário pode começar a treinar com ela enquanto a geração continua.

## Passo 5: empacotar e entregar

```bash
bash pack.sh
ls -la thor_out.tar.gz && sha256sum thor_out.tar.gz
```

O usuário copia para a outra máquina, por exemplo: `scp thor_out.tar.gz <usuario>@<host-da-4090>:~/barros/peft_lora_study/data/thor/`.

**PARE E REPORTE 3 (final)**, com este modelo (preencha com números reais do `out/report.json`):

```
Professor / motor: ...
Prefill usado: ...
Problemas gerados: ... | mantidos: ... | rendimento: ...%
Idioma do traço entre os que fecham: pt ...% (en ..., mixed ..., unk ...)
Tokens por exemplo mantido (mediana): ...
Velocidade: ... problemas/min, ... tokens/s | tempo total: ... h
Funil: truncado ..., sem </think> ..., sem ####..., resposta diferente da referência ..., longo demais ...
Divergências do plano (de out/NOTAS.md): ...
Arquivo: thor_out.tar.gz, sha256 ...
```

## Problemas comuns

| Sintoma | O que fazer |
| --- | --- |
| `curl` em `/v1/completions` dá 404 | O servidor só tem `/v1/chat/completions`. Use outro motor (C) ou suba o servidor com a rota de completions. Não mude o cliente |
| Muitos `truncado` | O professor está em loop. Confira `--max-tokens` (6144) e a temperatura (0,6). Reporte a fração |
| `sem_um_unico_</think>` alto | O template do servidor pode estar escrevendo `<think>` também na saída. Mostre 2 exemplos de `out/traces.jsonl` no relatório |
| `sem_####_na_resposta` alto | O professor usa `\boxed{}` em vez de `####`. Mostre 2 exemplos e reporte; não altere o filtro |
| Erro de memória no servidor | Item D do passo 1 |
| `erro_de_requisição` alto | Servidor sobrecarregado: reduza `--concurrency`. Veja o log do servidor |

## Critérios de aceite

- Pelo menos 8.000 exemplos mantidos (meta: 15.000 ou mais).
- Pelo menos 90% dos traços mantidos em português.
- Os exemplos lidos em `samples.md` fazem sentido (raciocínio coerente, termina na resposta certa).
