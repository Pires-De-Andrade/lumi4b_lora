#!/usr/bin/env bash
# Passa a cadeia do RL (H75, `scripts/run_rl_t6_chain.sh`) na frente da H68 (amostragem T 0,6 na `oab_novas`) na fila `fila_final2` (`scripts/run_final_chain.sh`),
# sem editar o script da fila (o bash lê o .sh aos poucos; editar com ele rodando quebra). A fila faz a H69 (três execuções da receita) e depois começa a H68.
# Fluxo linear numa só sessão, sem sessões esperando umas pelas outras (o travamento de 11/11 01:09 veio de duas sessões que se esperavam):
#   1. vigia o status da fila; quando ele registra o INÍCIO da primeira avaliação da H68 (`base_t06`), encerra a sessão `fila_final2` e a árvore de processos dela
#      (o vLLM da H68 ainda está subindo; a avaliação é retomável). A H69 já terminou nesse ponto: a H68 só começa depois dela.
#   2. escreve a tabela da H69 em runs/jobs/h69_comparacao.md (CPU), porque a geração de tabelas da fila morre junto com a sessão
#   3. roda a cadeia do RL, em primeiro plano (a espera dela por `noite18` não bloqueia: não existe sessão com esse prefixo)
#   4. retoma `scripts/run_final_chain.sh`, que pula o que já está completo e roda a H68 e as tabelas finais
# Só mata a árvore de processos da própria sessão tmux da fila; nada de outros usuários. Os nomes de sessão do tmux casam por prefixo: aqui vão sempre com `=`.
# Variáveis (para teste): SESSAO (padrão fila_final2), STATUS (padrão runs/jobs/noite18.status), PADRAO (padrão: início da H68), SO_VIGIA=1 (para depois do passo 1).
# Uso: scripts/detach.sh h75_antes_h68 bash scripts/reordenar_h75_antes_h68.sh   (NÃO editar com ele rodando)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SESSAO="${SESSAO:-fila_final2}"
STATUS="${STATUS:-runs/jobs/noite18.status}"
PADRAO="${PADRAO:-INICIO avaliação base_t06 (}"   # texto literal (grep -F): a primeira avaliação da H68
NOVAS=oab_novas,enem_2024,mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$STATUS"; }
conta() { grep -cF -- "$PADRAO" "$STATUS" 2> /dev/null || true; }
arvore() { local p; for p in $(ps -o pid= --ppid "$1" 2> /dev/null); do echo "$p"; arvore "$p"; done; }

base=$(conta)
say "vigia ligado: espero o início da H68 em $STATUS (ocorrências agora: $base)"
while [ "$(conta)" -le "$base" ]; do
  if ! tmux has-session -t "=$SESSAO" 2> /dev/null; then
    say "a sessão $SESSAO acabou sem iniciar a H68; sigo para o RL"
    break
  fi
  sleep 2
done
if tmux has-session -t "=$SESSAO" 2> /dev/null; then
  raiz=$(tmux list-panes -t "=$SESSAO:" -F '#{pane_pid}' 2> /dev/null | head -1)
  pids="$raiz $(arvore "$raiz")"
  say "a H68 começou: encerrando a sessão $SESSAO (processos: $pids)"
  tmux kill-session -t "=$SESSAO"
  sleep 2
  kill -TERM $pids 2> /dev/null
  sleep 15
  kill -KILL $pids 2> /dev/null
  say "reordenação: a fila $SESSAO foi encerrada ao começar a H68; a cadeia do RL (H75) roda agora e a H68 volta depois dela (scripts/run_final_chain.sh)"
fi
[ "${SO_VIGIA:-0}" = 1 ] && exit 0

# 2. tabela da H69 (CPU, em segundo plano): contra o Lumi com e sem thinking, que são as duas condições da regra de leitura
(
  R=runs/jobs/h69_comparacao.md; : > "$R"
  echo "# H69: as quatro execuções da receita nas provas novas (escrita em $(date +%d/%m\ %H:%M)); 4.096 tokens, gulosa, 4090. Provas novas = $NOVAS" >> "$R"
  for ref in "Lumi com thinking" "Lumi sem thinking"; do
    echo "## Referência: $ref" >> "$R"
    python scripts/report_compare.py --ref "$ref" --col "Lumi com thinking=base:thinking" --col "Lumi sem thinking=base:nothinking" \
      --col "T6-Lumi=thor_pt_antidoom_of:thinking" --col "portagem, semente 0=thor_pt_antidoom:thinking" \
      --col "portagem, semente 1=thor_pt_antidoom_s1:thinking" --col "portagem, semente 2=thor_pt_antidoom_s2:thinking" \
      --tasks "$NOVAS" --mean >> "$R" 2>&1
    echo >> "$R"
  done
) &

# 3. RL (H75); o código de saída não impede a retomada da fila
for _ in $(seq 1 12); do
  usado=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ')
  [ "${usado:-0}" -le 1024 ] && break
  sleep 5
done
say "INICIO da cadeia do RL (H75) antes da H68 (GPU em uso: ${usado:-?} MiB)"
bash scripts/run_rl_t6_chain.sh
say "FIM da cadeia do RL (código $?); retomando a fila final (H68 e tabelas)"

# 4. retomada: pula o que já está completo
exec bash scripts/run_final_chain.sh
