#!/usr/bin/env bash
# Spark, um comando só. Na pasta onde está o spark1_pacote.tar.gz (o único arquivo que precisa subir):
#   tar xzf spark1_pacote.tar.gz && bash handoff/spark1/rodar.sh
# Acha (ou instala) o vLLM, baixa o Lumi, reconstrói e confere os pesos do T6-Lumi (a partir da diferença que vem no pacote), confere o ambiente e lança
# a fila em segundo plano (sobrevive a fechar o navegador). Se algo falhar, a mensagem diz o quê; o registro fica em out/rodar.log.
# Rodar de novo é seguro: o que já está pronto é pulado.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p out runs/jobs
exec > >(tee -a out/rodar.log) 2>&1
parar() { echo; echo "PAROU: $*"; echo "(registro completo em out/rodar.log; mande esse arquivo para quem montou o pacote)"; exit 1; }
echo "== $(date '+%d/%m %H:%M') spark1: preparação em $(pwd)"
# a Spark pode não ter pgrep, ps nem curl: a fila é procurada em /proc e o Hugging Face é testado pelo python
fila_rodando() {
  local f
  for f in /proc/[0-9]*/cmdline; do
    tr '\0' ' ' < "$f" 2> /dev/null | grep -q "[h]andoff/spark1/run_spark1.sh" && return 0
  done
  return 1
}

if fila_rodando; then
  echo "A fila já está rodando. Acompanhar: bash handoff/spark1/status.sh"; exit 0
fi
disco=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "${disco:-0}" -ge 40 ] || parar "pouco disco livre (${disco} GB; precisa de ~40 GB para o Lumi, os pesos do T6-Lumi e o vLLM)"

# 1. python com vLLM
PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in .venv-eval/bin/python python3 python; do
    command -v "$c" > /dev/null 2>&1 && "$c" -c "import vllm, torch" > /dev/null 2>&1 && { PY="$c"; break; }
  done
fi
if [ -z "$PY" ]; then
  echo "vLLM não encontrado: instalando em .venv-eval (10 a 30 min; registro em out/instalacao.log)..."
  { python3 -m venv .venv-eval && .venv-eval/bin/pip install -U pip && .venv-eval/bin/pip install vllm datasets; } > out/instalacao.log 2>&1 \
    || parar "a instalação do vLLM falhou (ver out/instalacao.log); sem docker/pip adequado a instalação precisa de outra via"
  .venv-eval/bin/python -c "import vllm, torch" > /dev/null 2>&1 || parar "o vLLM instalou mas não importa (ver out/instalacao.log)"
  PY=.venv-eval/bin/python
fi
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || parar "o torch de $PY não enxerga a GPU (instalação sem CUDA para esta máquina)"
echo "python do vLLM: $PY"

# 1b. compilador C (o Triton do vLLM precisa; a Spark não tem): usa o do sistema ou instala o zig cc pelo pip, sem root
PY="$PY" bash handoff/spark1/compilador.sh || parar "sem compilador C para o Triton (ver o erro acima e out/rodar.log)"
. .cc/env.sh

# 2. pesos do T6-Lumi = Lumi (baixado do Hugging Face, revisão fixada) + a diferença do pacote; o sha256 confere o resultado
T6=runs/merged/thor_pt_antidoom_of
TAM=7947142672
REV=$(grep -oP 'REVISION = "\K[0-9a-f]+' src/lumi_ft/modeling/load.py)
[ -s handoff/spark1/t6_delta.bin ] || parar "falta handoff/spark1/t6_delta.bin (o pacote veio incompleto: subir o spark1_pacote.tar.gz de novo)"
if [ "$(stat -c %s "$T6/model.safetensors" 2>/dev/null || echo 0)" != "$TAM" ]; then
  echo "reconstruindo os pesos do T6-Lumi (baixa o Lumi, ~8 GB, na primeira vez)..."
  "$PY" handoff/spark1/delta.py unpack --delta handoff/spark1/t6_delta.bin --out "$T6/model.safetensors" --repo ceia-br/lumi-4b --revision "$REV" \
    || parar "falha ao reconstruir os pesos (sem acesso ao Hugging Face? ver o erro acima)"
fi
echo "conferindo o sha256 dos pesos (~1 min)..."
sha256sum -c handoff/spark1/PESOS.sha256 || { rm -f "$T6/model.safetensors"; parar "os pesos reconstruídos não conferem: o pacote veio corrompido (subir o spark1_pacote.tar.gz de novo)"; }

# 3. ambiente (fica em out/env_info.txt, devolvido junto) e Hugging Face
PY="$PY" bash handoff/spark1/ambiente.sh > /dev/null 2>&1
grep -E "^(vllm|torch|transformers) |Hugging Face" out/env_info.txt
grep -q "Hugging Face (esperado 200): 200" out/env_info.txt || parar "sem acesso ao Hugging Face (o Lumi precisa ser baixado na primeira vez)"

# 4. a fila, em segundo plano
if command -v setsid > /dev/null 2>&1; then
  PY="$PY" setsid nohup bash handoff/spark1/run_spark1.sh > runs/jobs/spark1.log 2>&1 < /dev/null &
else
  PY="$PY" nohup bash handoff/spark1/run_spark1.sh > runs/jobs/spark1.log 2>&1 < /dev/null &
fi
sleep 3
echo
echo "FILA LANÇADA (pid $!). Previsão: ~5 h; a primeira linha de status leva alguns minutos."
echo "  acompanhar:  bash handoff/spark1/status.sh"
echo "  baixar:      spark1_saida.tar.gz  (aparece sozinho a cada par pronto; pelo Jupyter, botão direito > Download)"
echo "  fim:         a linha 'FIM da fila spark1' no status"
