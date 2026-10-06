@echo off
cd /d "%~dp0"

echo.
echo Iniciando o TasteLens...
echo.
echo A aplicacao sera disponibilizada na porta 8501.
echo Para acesso pelo celular, utilize o QR Code exibido no painel.
echo.

python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

pause
