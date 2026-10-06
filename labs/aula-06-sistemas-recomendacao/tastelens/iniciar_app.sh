#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

echo "Iniciando o TasteLens..."
echo
echo "A aplicação será disponibilizada na porta 8501."
echo "Para acesso pelo celular, utilize o QR Code exibido no painel."
echo

python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
