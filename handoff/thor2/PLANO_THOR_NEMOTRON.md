# Plano da Thor (2): gerar traços de raciocínio com o Nemotron 3 Nano 30B-A3B

Roteiro de execução na Jetson Thor. Leia tudo e execute os passos 0 a 5 em ordem. Pare e reporte em cada **PARE E REPORTE**.

## Contexto

- O projeto treina o Lumi-4B para raciocinar em português. O Lumi vem do **Nemotron 3 Nano 4B**, e o raciocínio dele tem um estilo próprio (curto, em forma de plano: "We need to compute…"), bem diferente do Qwen e do DeepSeek-R1 (mais longos, cheios de "wait", "hmm"). Treinar com traços do Qwen ensinou um estilo que o Lumi não tem e piorou a múltipla escolha. A ideia agora é usar um professor **da mesma família**, o Nemotron 3 Nano 30B-A3B, para gerar traços no estilo nativo.
- O Nemotron 3 Nano **não suporta português oficialmente** (inglês, espanhol, francês, alemão, japonês e italiano). Por isso o professor pensa **em inglês, sem forçar o idioma**. A tradução para português é feita na outra máquina (RTX 4090). **Não use `--prefill`** e não tente forçar português.
- A Thor só **gera os dados**. Treino, tradução e avaliação ficam na outra máquina. Não treine nada aqui.
- Os problemas: 9.612 de matemática (prompt em português, resposta `#### N`) e 12.000 questões de ciência de múltipla escolha (prompt em inglês, resposta `Resposta: <letra>`). A referência de matemática é o gabarito (GSM8K) ou concordância com outro modelo (GigaVerbo); a de múltipla escolha é a letra majoritária de outros traços, não um gabarito garantido.
- Entrega: `thor2_out.tar.gz` (gerado pelo `pack.sh`).

## Regras

1. Não altere `gen_traces.py`, `filter_traces_en.py` nem `data/prompts.jsonl`. Se achar um defeito, anote em `out/NOTAS.md` e avise o usuário.
2. Instale pacotes só em venv ou contêiner. Não mexa em outros projetos da máquina. Não publique nada no Hugging Face. Não grave tokens ou senhas em nenhum arquivo do pacote.
3. Anote em `out/NOTAS.md` (crie a pasta `out/`) tudo que divergir do plano: comando usado, versão, erro e alternativa escolhida.
4. Rode tudo que dure mais de 1 minuto em `tmux`, para sobreviver à queda do SSH.
5. Pare e reporte se: o servidor não subir em 45 min, o rendimento do piloto for menor que 30%, ou faltar disco/memória.
6. **Memória:** a Thor tem ~122 GiB compartilhados. Se o servidor do Qwen (pacote 1) ainda estiver rodando, **não suba o Nemotron ao mesmo tempo**. Pergunte ao usuário se deve esperar o Qwen terminar ou interrompê-lo (o pacote 1 só precisava de pelo menos 8.000 exemplos mantidos; use o `pack.sh` dele antes de parar, para não perder nada).

## Arquivos desta pasta

| Arquivo | Para quê |
| --- | --- |
| `data/prompts.jsonl` | 21.612 problemas embaralhados (`id`, `source`, `kind` = `math` ou `mcq`, `question`, `options`, `prompt`, `ref`). Confira com `cd data && sha256sum -c prompts.sha256` |
| `gen_traces.py` | Cliente: envia os problemas a um servidor `/v1/completions` e grava `out/traces.jsonl`. Retomável. É o mesmo arquivo do pacote 1 |
| `filter_traces_en.py` | Mantém só o que fecha `</think>`, termina sozinho e acerta a referência. O idioma do traço é só registrado |
| `pack.sh` | Filtra e empacota tudo em `thor2_out.tar.gz` (pode rodar a qualquer hora, com a geração em andamento) |

## Passo 0: diagnóstico e ambiente (10 min)

```bash
mkdir -p out && {
  date; uname -a; nvidia-smi; free -g; df -h .; nproc; python3 --version
  docker --version 2>&1; docker ps 2>&1 | head -5
} 2>&1 | tee out/env_info.txt
cd data && sha256sum -c prompts.sha256 && cd ..
python3 -m venv .venv-gen && . .venv-gen/bin/activate && pip install -q transformers
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', trust_remote_code=True)
for th in (True, False):
    try: print(th, repr(t.apply_chat_template([{'role':'user','content':'oi'}], tokenize=False, add_generation_prompt=True, enable_thinking=th)))
    except Exception as e: print(th, 'ERRO', e)
"
```

Esperado: `aarch64`, CUDA 13.x, o hash `OK`, e, com `enable_thinking=True`, um prompt terminando em `<think>\n`. Se o disco livre for menor que 80 GB (BF16, ~60 GB), use a versão FP8 (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, ~32 GB), anote em `out/NOTAS.md` e siga.

## Passo 1: subir o servidor do professor (até 45 min)

Objetivo: um servidor em `http://localhost:8000/v1/completions` com o modelo, chamado `teacher`. **Leia as receitas oficiais antes**, porque as flags do Nemotron 3 (modelo híbrido Mamba-Transformer com MoE) mudam por versão do vLLM:

- Receita do vLLM: https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- Blog do vLLM: https://blog.vllm.ai/2025/12/15/run-nvidia-nemotron-3-nano.html
- Receita para Jetson/Thor: https://www.jetson-ai-lab.com/models/nemotron-3-nano-30b-a3b
- Cartão do modelo: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (confira os parâmetros de amostragem recomendados para o modo com raciocínio)

Tente na ordem abaixo e use o primeiro que funcionar. Use as flags das receitas (por exemplo `--trust-remote-code` e as de cache do Mamba, se constarem), **mas sem** o `--reasoning-parser` e sem o `--tool-call-parser`: o `gen_traces.py` usa `/v1/completions` com texto cru e precisa ver o `</think>` na saída.

**A. Contêiner NVIDIA com vLLM para Thor** (preferido, se `docker` funcionar):

```bash
docker run -d --name teacher --runtime nvidia --ipc=host --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  <IMAGEM_NGC_COM_VLLM> \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --served-model-name teacher \
  --max-model-len 10240 --max-num-seqs 96 --gpu-memory-utilization 0.80 --dtype bfloat16 --trust-remote-code --port 8000
```

**B. vLLM por pip** em outro venv (`python3 -m venv .venv-vllm`; `pip install vllm`). Mesmo comando `vllm serve ...`. Se aparecer erro de compilação do FlashInfer (por exemplo `curand.h`), tente `VLLM_USE_FLASHINFER_SAMPLER=0`.

**C. Outro motor com `/v1/completions`**: SGLang ou TensorRT-LLM. O nome servido deve ser `teacher`. Se o motor não aceitar `top_k`, o `gen_traces.py` já o desliga com `--top-k 0` (use sempre `--top-k 0` neste pacote).

**D. Se a memória não bastar** (OOM): use a versão FP8, baixe `--max-model-len` para 8192 e `--max-num-seqs` para 64. Se ainda falhar, **pare e reporte** (o plano B seria o `NVIDIA-Nemotron-Nano-9B-v2`, mesma família e menor, mas só com a aprovação do usuário).

Teste (deve responder em menos de 30 s depois do aquecimento):

```bash
curl -s localhost:8000/v1/models | head -c 300; echo
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"teacher","prompt":"<|im_start|>user\nQuanto é 17 vezes 23?<|im_end|>\n<|im_start|>assistant\n<think>\n","max_tokens":400,"temperature":1.0,"top_p":1.0}'
```

Ajuste o prompt do `curl` ao template que o Passo 0 imprimiu, se for diferente.

**PARE E REPORTE 1:** motor e versão, as flags usadas, tempo de subida, memória usada (`free -g`/`nvidia-smi`), e a resposta do `curl` (o raciocínio deve aparecer, provavelmente em inglês, e depois `</think>`).

## Passo 2: teste de fumaça, 20 problemas (10 min)

Use sempre estes parâmetros de geração (se o cartão do modelo recomendar outros para o modo com raciocínio, use os do cartão e anote em `out/NOTAS.md`): `--prefill "" --temperature 1.0 --top-p 1.0 --top-k 0`.

```bash
. .venv-gen/bin/activate
python gen_traces.py --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --model-name teacher \
  --prompts data/prompts.jsonl --out out/smoke.jsonl --limit 20 --concurrency 20 \
  --prefill "" --temperature 1.0 --top-p 1.0 --top-k 0 --max-tokens 6144
python filter_traces_en.py --traces out/smoke.jsonl --prompts data/prompts.jsonl --outdir out/smoke
cat out/smoke/report.json; sed -n 1,80p out/smoke/samples.md
```

Leia pelo menos 3 exemplos de `out/smoke/samples.md` (um de matemática e dois de múltipla escolha) e confira:

- `finish_reason` quase todo `stop`;
- matemática com `#### N` e múltipla escolha com `Resposta: <letra>` na última linha (em `funil`, `sem_####_na_resposta` e `sem_Resposta_letra_na_ultima_linha` devem ser pequenos);
- o traço fecha `</think>` uma vez e o raciocínio está em inglês (campo `idioma_do_traço_entre_os_que_fecham`; português ou misto também serve, só anote);
- o rendimento (`mantidos` ÷ `gerados`) é de pelo menos 30%.

Se o modelo ignorar o formato final (por exemplo usar `\boxed{}`), **não mude o filtro**: mostre 2 exemplos no relatório.

## Passo 3: piloto, 300 problemas (15 a 30 min)

```bash
tmux new -s teacher2
python gen_traces.py --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --model-name teacher \
  --prompts data/prompts.jsonl --out out/traces.jsonl --start 20 --limit 300 --concurrency 96 \
  --prefill "" --temperature 1.0 --top-p 1.0 --top-k 0 --max-tokens 6144
python filter_traces_en.py --traces out/traces.jsonl --prompts data/prompts.jsonl --outdir out
cat out/report.json
```

Calcule a velocidade (`problemas/min` da última linha de progresso) e o tempo para 21.612 problemas: `21612 / problemas_por_min / 60` horas.

**PARE E REPORTE 2:** `out/report.json`, a velocidade, a estimativa para 21.612 problemas e 2 exemplos de `out/samples.md`. Regra: se o rendimento for menor que 30%, pare aqui. Se for maior ou igual, siga.

## Passo 4: geração completa (até ~5 h, em tmux)

```bash
python gen_traces.py --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --model-name teacher \
  --prompts data/prompts.jsonl --out out/traces.jsonl --start 20 --limit 21612 \
  --concurrency 96 --time-budget-min 300 \
  --prefill "" --temperature 1.0 --top-p 1.0 --top-k 0 --max-tokens 6144
```

- É retomável: se cair, rode o mesmo comando. Os 300 do piloto não são refeitos.
- Se quiser mais velocidade e a memória sobrar, aumente `--concurrency` (e `--max-num-seqs` do servidor). Se houver muitos `erro_de_requisição`, diminua.
- A cada ~1 h, rode `bash pack.sh` (passo 5) para gerar uma **entrega parcial**: o usuário pode começar a traduzir e treinar com ela enquanto a geração continua.

## Passo 5: empacotar e entregar

```bash
bash pack.sh
ls -la thor2_out.tar.gz && sha256sum thor2_out.tar.gz
```

O usuário copia para a outra máquina, por exemplo: `scp thor2_out.tar.gz <usuario>@<host-da-4090>:~/barros/peft_lora_study/data/thor2/`.

**PARE E REPORTE 3 (final)**, com este modelo (preencha com números reais do `out/report.json`):

```
Professor / motor / flags: ...
Parâmetros de geração: ...
Problemas gerados: ... | mantidos: ... (matemática ..., múltipla escolha ...) | rendimento: ...%
Idioma do traço entre os que fecham: en ...% (pt ..., mixed ..., unk ...)
Tokens por exemplo mantido (mediana): ...
Velocidade: ... problemas/min, ... tokens/s | tempo total: ... h
Funil: truncado ..., sem </think> ..., sem formato final ..., resposta diferente da referência ..., longo demais ...
Divergências do plano (de out/NOTAS.md): ...
Arquivo: thor2_out.tar.gz, sha256 ...
```

## Problemas comuns

| Sintoma | O que fazer |
| --- | --- |
| `curl` em `/v1/completions` dá 404 | O servidor só tem `/v1/chat/completions`. Use outro motor (C) ou suba o servidor com a rota de completions. Não mude o cliente |
| O modelo responde sem `<think>` ou sem `</think>` | Confira no Passo 0 se o prompt termina em `<think>\n` com `enable_thinking=True`. Mostre 2 exemplos de `out/traces.jsonl` no relatório |
| Muitos `truncado` | O professor está em loop ou pensa demais. Confira `--max-tokens` (6144) e a temperatura. Reporte a fração |
| `sem_####_na_resposta` ou `sem_Resposta_letra...` alto | O modelo usa outro formato. Mostre 2 exemplos e reporte; não altere o filtro |
| Erro de memória no servidor | Item D do passo 1 |
| `erro_de_requisição` alto | Servidor sobrecarregado: reduza `--concurrency`. Veja o log do servidor |

## Critérios de aceite

- Pelo menos 6.000 exemplos mantidos, sendo ao menos 2.000 de matemática e 3.000 de múltipla escolha (meta: 12.000 ou mais).
- Pelo menos 85% das respostas fecham `</think>`.
- Os exemplos lidos em `samples.md` fazem sentido (raciocínio coerente, termina na resposta certa).
