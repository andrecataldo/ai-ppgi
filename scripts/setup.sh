#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# AI PPGI - Environment Setup
# PPGI/UNIRIO - Inteligência Artificial
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"

# Workaround para compatibilidade do Experta com Python moderno.
#
# experta==1.9.4 declara frozendict==1.2, mas essa versão antiga
# apresenta problemas em versões modernas do Python.
#
# O ambiente validado deste projeto utiliza frozendict==2.4.7.
FROZENDICT_WORKAROUND_VERSION="2.4.7"

echo
echo "============================================================"
echo " AI PPGI - Setup"
echo "============================================================"
echo
echo "Projeto: ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

# ------------------------------------------------------------
# 1. Verifica Python
# ------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERRO: python3 não encontrado."
    exit 1
fi

echo
echo "[1/6] Python encontrado:"
python3 --version

# ------------------------------------------------------------
# 2. Cria ambiente virtual
# ------------------------------------------------------------

if [ ! -d "${VENV_DIR}" ]; then
    echo
    echo "[2/6] Criando ambiente virtual em .venv..."
    python3 -m venv "${VENV_DIR}"
else
    echo
    echo "[2/6] Ambiente virtual .venv já existe."
fi

PYTHON="${VENV_DIR}/bin/python"

# ------------------------------------------------------------
# 3. Atualiza pip
# ------------------------------------------------------------

echo
echo "[3/6] Atualizando pip..."
"${PYTHON}" -m pip install --upgrade pip

# ------------------------------------------------------------
# 4. Instala dependências principais
# ------------------------------------------------------------

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "ERRO: requirements.txt não encontrado."
    exit 1
fi

echo
echo "[4/6] Instalando requirements.txt..."
"${PYTHON}" -m pip install -r "${REQUIREMENTS}"

# ------------------------------------------------------------
# 5. Aplica workaround do frozendict
# ------------------------------------------------------------

echo
echo "[5/6] Aplicando workaround do Experta..."
echo "      frozendict==${FROZENDICT_WORKAROUND_VERSION}"

"${PYTHON}" -m pip install \
    --upgrade \
    --no-deps \
    "frozendict==${FROZENDICT_WORKAROUND_VERSION}"

# ------------------------------------------------------------
# 6. Validação
# ------------------------------------------------------------

echo
echo "[6/6] Validando ambiente..."

"${PYTHON}" - <<'PY'
import importlib.metadata as metadata

import experta
import frozendict
import streamlit

print()
print("Dependências carregadas com sucesso.")
print()
print(f"Python      : {__import__('sys').version.split()[0]}")
print(f"Experta     : {metadata.version('experta')}")
print(f"frozendict  : {metadata.version('frozendict')}")
print(f"Streamlit   : {metadata.version('streamlit')}")
PY

echo
echo "============================================================"
echo " Setup concluído."
echo "============================================================"
echo
echo "Para ativar o ambiente:"
echo
echo "  source .venv/bin/activate"
echo
echo "Para executar o Sommelier Digital:"
echo
echo "  streamlit run apps/sommelier-digital/sommelier_app.py"
echo
echo "Para executar o laboratório de busca:"
echo
echo "  python labs/aula-03-busca/compare_search.py"
echo
echo "OBSERVAÇÃO:"
echo "  O Experta 1.9.4 declara frozendict==1.2 nos metadados."
echo "  Este projeto utiliza deliberadamente frozendict==2.4.7"
echo "  como workaround validado para Python moderno."
echo
