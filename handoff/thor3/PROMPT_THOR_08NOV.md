# Prompt para o Claude Code da Thor (08/11): questões com gabarito oficial primeiro, entregas em blocos

Antes de colar o texto abaixo no Claude Code da Thor, copie **um arquivo novo** para a pasta do pacote (a mesma onde está o `translate_thor.py`, porque ele o importa):

```bash
scp handoff/thor3/translate_questions.py <usuario>@<thor>:<pasta do pacote 3>/
# sha256 esperado: 9e66416a8079ae539322b664b250734eddba3c20ecbd097f588fc745f84bc5bb
```

Ele é só um arquivo a mais (não altera nenhum dos existentes): traduz só o enunciado e as alternativas das 12.000 questões, com as mesmas conferências do `translate_thor.py`. O resto continua o `PLANO_THOR3.md` (as regras 1 a 6 valem), só muda a ordem e o ritmo das entregas.

---

Atualização do projeto (08/11). A 4090 recebeu a primeira parte da tradução do pacote 2: 5.278 registros com raciocínio (`with_thought: true`). Obrigado, já está em uso. O foco do projeto agora é melhorar o modo **com thinking** (raciocínio em português), e o ponto mais fraco do nosso aprendizado por reforço é o gabarito das questões de múltipla escolha (hoje é a letra majoritária de outro modelo, só em ciência). As 12.000 questões novas do pacote 3 (WebInstruct-verified e MedMCQA) têm **gabarito oficial** e cobrem negócios, economia, finanças, matemática, física, química, biologia, psicologia, história e medicina, que é bem mais parecido com as nossas provas de teste. Por isso elas ficaram mais importantes do que eram.

As regras do `PLANO_THOR3.md` continuam valendo: não altere `translate_thor.py`, `gen_traces.py`, `filter_traces_en.py` nem `data/prompts.jsonl`, não mude limites de aceitação nem parâmetros de geração, rode tudo em `tmux`, não publique nada no Hugging Face, anote divergências em `out/NOTAS.md`.

Faça agora, em ordem:

1. **Diga em uma linha em que passo você está** (por exemplo: "passo 2, tradução do pacote 2, X de Y feitos, aproveitamento Z%") e se o servidor `teacher` está de pé. Se `translate_questions.py` ainda não estiver na pasta, avise o usuário e siga para o item 3.

2. **Traduza só as questões do pacote 3** (novo, leve: ~12.000 questões, só enunciado e alternativas, sem raciocínio; estimativa de 20 a 40 min). Pode rodar junto com a tradução do pacote 2 (o servidor divide a vazão):

   ```bash
   # teste primeiro, com 30 questões
   python3 translate_questions.py --prompts data/prompts.jsonl --out out/q_smoke.jsonl --limit 30 --block 30 \
     --tokenizer $TOK --model-name teacher --concurrency 64
   # se o aproveitamento for de 60% ou mais, todas, em tmux:
   python3 translate_questions.py --prompts data/prompts.jsonl --out out/questions_pt.jsonl \
     --tokenizer $TOK --model-name teacher --concurrency 96
   tar -czf questions_pt.tar.gz out/questions_pt.jsonl out/questions_pt.jsonl.report.json && sha256sum questions_pt.tar.gz
   ```

   Leia 2 exemplos do teste (inglês e português lado a lado): alternativas na mesma ordem, números iguais, a questão **não** foi respondida. Com aproveitamento menor que 60%, pare e mostre 3 recusas (`piece` e `verdict`). **Avise o usuário** com o `sha256` do `questions_pt.tar.gz` (a 4090 usa esse arquivo como prompts de RL, **independente** de o professor ter acertado a questão).

3. **Termine o passo 2** (tradução do pacote 2) e rode `bash pack.sh`. A entrega que o usuário levou só tinha os registros com raciocínio. **Precisamos também dos registros sem raciocínio** (`with_thought: false`, os 30% por hash do id): eles também estão em `out/translated.jsonl` e entram no treino para o modo sem thinking não se desmontar. Avise o usuário para levar o `thor3_out.tar.gz` inteiro, ou o `out/translated.jsonl` completo (não só a parte com raciocínio).

4. **Passo 3 (traços das 12.000 questões) em blocos, não de uma vez.** Em vez de uma geração de até 3 horas, faça blocos de 3.000 prompts, com os mesmos parâmetros do pacote 2 (sem `--prefill`, mesma temperatura, top-p, top-k e max-tokens):

   ```bash
   # bloco 1 (o smoke de 20 do plano continua valendo antes dele)
   python3 gen_traces.py ... --prompts data/prompts.jsonl --out out/traces.jsonl --start 20 --limit 3000 --time-budget-min 60 ...
   # blocos seguintes: --start 3020, 6020, 9020 (o arquivo de saída é o mesmo; os ids já gravados são pulados)
   ```

   Depois de **cada** bloco: rode `filter_traces_en.py`, traduza só os ids novos (passo 4 do plano, o mesmo `out/translated.jsonl`), rode `bash pack.sh` e **avise o usuário** com o `sha256` do `thor3_out.tar.gz`. Não reordene os prompts (já vêm embaralhados e balanceados entre `webinstruct` e `medmcqa`).

5. Se o rendimento do primeiro bloco do item 4 ficar abaixo de 30%, **pare e reporte** (regra do plano). Não mude o filtro.

6. Em cada aviso, informe em uma tabela curta: questões ou exemplos por fonte (`webinstruct`, `medmcqa`) e por categoria, quantos têm `with_thought` verdadeiro e falso, aproveitamento da tradução e velocidade (itens por minuto).

**Não faça agora:** gerar mais de um traço por questão, mudar o estilo ou o tamanho dos traços, ou gerar respostas sem thinking. Só mude a ordem e o ritmo, como acima.
