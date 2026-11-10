#!/usr/bin/env bash
# O Triton (que o vLLM usa) precisa de um compilador C, e a Spark não tem: a primeira fila parou no teste de fumaça com "Failed to find C compiler".
# Este script usa o compilador do sistema, se houver; senão instala o `zig cc` pelo pip (sem root, ~100 MB), confere que o Triton compila com ele e grava
# .cc/env.sh (CC, CXX, PATH), que o `rodar.sh` carrega antes de lançar a fila. Rodar de novo é seguro. CC_SO_ZIG=1 ignora o compilador do sistema.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-.venv-eval/bin/python}"
mkdir -p .cc/bin out
falha() { echo "COMPILADOR: $*"; exit 1; }

CCX=""; CXXX=""; CAMINHO=""
if [ "${CC_SO_ZIG:-0}" != 1 ]; then
  for c in gcc cc clang; do command -v "$c" > /dev/null 2>&1 && { CCX="$(command -v "$c")"; break; }; done
  for c in g++ c++ clang++; do command -v "$c" > /dev/null 2>&1 && { CXXX="$(command -v "$c")"; break; }; done
fi
if [ -n "$CCX" ]; then
  echo "compilador do sistema: $CCX"
else
  echo "sem compilador C no sistema: usando o zig cc (pip, sem root)"
  "$PY" -c "import ziglang" > /dev/null 2>&1 || "$PY" -m pip install -q ziglang || falha "pip install ziglang falhou"
  ZIG="$("$PY" -c 'import os, ziglang; print(os.path.join(os.path.dirname(ziglang.__file__), "zig"))')"
  [ -x "$ZIG" ] || falha "não achei o binário do zig em $ZIG"
  for n in cc gcc; do printf '#!/bin/sh\nexec "%s" cc "$@"\n' "$ZIG" > ".cc/bin/$n"; done
  for n in c++ g++; do printf '#!/bin/sh\nexec "%s" c++ "$@"\n' "$ZIG" > ".cc/bin/$n"; done
  chmod +x .cc/bin/*
  CCX="$PWD/.cc/bin/gcc"; CXXX="$PWD/.cc/bin/g++"; CAMINHO="$PWD/.cc/bin"
fi
{
  echo "export CC=\"$CCX\""
  [ -n "$CXXX" ] && echo "export CXX=\"$CXXX\""
  [ -n "$CAMINHO" ] && echo "export PATH=\"$CAMINHO:\$PATH\""
} > .cc/env.sh

# confere com o que o Triton compila de fato (o módulo de utilidades de CUDA), em um cache novo para forçar a compilação
rm -rf .cache/triton_teste
if ( . .cc/env.sh; TRITON_CACHE_DIR="$PWD/.cache/triton_teste" "$PY" -c "
from triton.backends.nvidia.driver import CudaUtils
CudaUtils()
print('o Triton compilou e carregou o módulo de CUDA')
" 2>&1 | tail -6 ) | tee .cc/teste.log | grep -q "o Triton compilou"; then
  echo "COMPILADOR: ok (CC=$CCX)"
else
  cat .cc/teste.log
  falha "o Triton não compilou com $CCX (veja o erro acima; se for Python.h, falta o pacote python3-dev)"
fi
