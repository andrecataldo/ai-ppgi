# Manual de uso - LAB 11: Decision Tree, Random Forest e XGBoost

**Disciplina:** Inteligência Artificial - PPGI/UNIRIO  
**Aula:** 11 - Árvores de Decisão, Random Forest e XGBoost  
**Versão:** 1.0   
**Diretório no repositório:** `labs/aula-11-random-forest/`  
**Repositório:** https://github.com/andrecataldo/ai-ppgi

---

## 1. Objetivo deste manual

Orientar a instalação, a execução, a validação e a exploração dos resultados do LAB 11. O laboratório compara **Decision Tree**, **Random Forest** e **XGBoost** na previsão de desvios em estimativas humanas em um experimento de interação humano–IA.

Este documento é destinado tanto à avaliação acadêmica quanto a quem deseja reproduzir a execução em outro computador. 

### O que será executado?

- Leitura e auditoria do dataset `Exp2.csv`.
- Engenharia de atributos de contexto e de histórico temporal.
- Treinamento e avaliação de três algoritmos de regressão, além de modelos de referência (*baselines*).
- Separação por participantes: **90 para treinamento e 30 para teste**, com semente 42.
- Validação cruzada agrupada por participante, com quatro partições na execução completa.
- Cálculo de MAE, RMSE, R², desempenho por condição e intervalos de confiança para o ganho com histórico.
- Geração de tabelas, gráficos, predições e relatório em `results/`.

> **Importante:** o alvo é `signed_error = response - evidence`, isto é, um desvio numérico. O laboratório **não classifica pessoas como cognitivamente enviesadas** nem comprova que os conselhos de IA causaram as respostas observadas.

## 2. Estrutura e localização dos arquivos

A partir da raiz do repositório `ai-ppgi`:

```text
labs/aula-11-random-forest/
├── app.py                            # Painel opcional Streamlit
├── data/
│   ├── Exp2.csv                      # Dados originais do experimento
│   └── SOURCE.md                     # Fonte e identificação do dataset
├── docs/
│   ├── ANALISE_ACADEMICA.md          # Discussão dos resultados e limitações
│   ├── MANUAL_DE_USO.md              # Este manual
│   └── PROTOCOLO_EXPERIMENTAL.md     # Definições metodológicas
├── notebooks/
│   ├── LAB11_Human_AI_Bias.ipynb     # Notebook para execução
│   └── LAB11_Human_AI_Bias_executado.ipynb
├── results/                          # Resultados produzidos pelo script
├── src/                              # Preparação de dados e modelos
├── tests/test_lab.py                 # Testes automatizados
├── requirements.txt                  # Dependências principais
├── requirements-ui.txt               # Dependências adicionais da interface
└── run_lab.py                        # Execução principal
```

> **Ordem recomendada:** instalar → executar testes → executar `run_lab.py` → abrir notebook → abrir Streamlit.

## 3. Pré-requisitos

- Ubuntu/Linux (ambiente utilizado no desenvolvimento; outros sistemas podem funcionar).
- Python **3.11 recomendado** e módulo `venv` disponível.
- Git, se o projeto for obtido do GitHub.
- Acesso à internet para instalar as dependências pela primeira vez.
- VS Code com extensões **Python** e **Jupyter** (opcional, para trabalhar com o notebook).

Verifique a instalação:

```bash
python3 --version
git --version
```

Se o Ubuntu informar que `venv` não está disponível, instale o pacote correspondente à sua versão de Python. Exemplo:

```bash
sudo apt update
sudo apt install python3-venv
```

## 4. Instalação - primeira execução

### 4.1 Obter ou acessar o projeto

Caso ainda não tenha clonado o repositório:

```bash
git clone https://github.com/andrecataldo/ai-ppgi.git
cd ai-ppgi/labs/aula-11-random-forest
```

Caso já tenha o projeto local (cenário do autor do laboratório):

```bash
cd ~/projects/ai-ppgi/labs/aula-11-random-forest
```

Confira se o CSV está disponível:

```bash
ls -lh data/Exp2.csv
```

O arquivo deve ter **14.400 registros de 120 participantes**. A integridade da cópia utilizada nesta versão pode ser verificada por:

```bash
sha256sum data/Exp2.csv
```

SHA-256 esperado:

```text
ec2207efc6fa936031107376857d48310990874d200de39c2880945ace574277
```

### 4.2 Criar e ativar o ambiente virtual

Na pasta `aula-11-random-forest`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O terminal deverá mostrar `(.venv)` antes do prompt. O ambiente `.venv/` é local e **não deve ser versionado**.

**Nas próximas sessões**, basta entrar na pasta do laboratório e ativar o ambiente:

```bash
cd ~/projects/ai-ppgi/labs/aula-11-random-forest
source .venv/bin/activate
```

## 5. Testar o projeto

Antes de executar os experimentos, rode:

```bash
python -m pytest -v
```

**Referência:** `12 passed` (12 testes aprovados). Os testes verificam, entre outros pontos, o contrato do dataset, a separação dos participantes, a ausência de informação da tentativa corrente nos atributos históricos e o funcionamento dos três estimadores.

Se algum teste falhar, consulte a seção **12 - Solução de problemas** antes de prosseguir.

## 6. Executar o experimento completo (principal)

Ainda dentro da pasta `aula-11-random-forest`, com `.venv` ativo:

```bash
python run_lab.py
```

A execução completa:

1. Audita os dados de entrada.
2. Constrói os atributos dos cenários `contexto` e `historico`.
3. Separa participantes sem sobreposição entre treino e teste.
4. Treina e avalia **Decision Tree**, **Random Forest** e **XGBoost** nos dois cenários.
5. Compara os modelos com três *baselines*.
6. Executa validação cruzada agrupada por participante dentro do conjunto de treino.
7. Calcula intervalos de confiança por reamostragem de participantes.
8. Escreve os artefatos em `results/`.

### 6.1 O que deve aparecer no terminal?

O resumo deverá começar aproximadamente assim:

```text
LAB 11 | auditoria: 14400 linhas / 120 participantes
Treino: 90 participantes; teste: 30
```

Em seguida, aparecerá uma tabela com `modelo`, `cenario`, `MAE`, `RMSE`, `R2`, `cv_mae_mean` e `late_block_MAE`, além da tabela de ganho de MAE com histórico.

**Valores de referência da execução local** (semente 42):

| Algoritmo | MAE: contexto | MAE: histórico | R²: histórico |
|---|---:|---:|---:|
| Decision Tree | 13,604 | 13,026 | 0,192 |
| Random Forest | 13,600 | **12,608** | 0,224 |
| XGBoost | 13,535 | 12,674 | **0,229** |

É possível observar pequenas diferenças numéricas conforme as versões das bibliotecas instaladas. Para comparar os resultados, mantenha a mesma semente, o mesmo dataset e, preferencialmente, o mesmo ambiente de dependências.

### 6.2 Modos adicionais de execução

**Ver as opções disponíveis:**

```bash
python run_lab.py --help
```

**Execução rápida para testar o fluxo, sem substituir a avaliação final:**

```bash
python run_lab.py --quick --out results/execucao_rapida
```

O modo `--quick` usa menos árvores e menos partições de validação cruzada. **Os resultados não são diretamente equivalentes aos da execução completa**.

**Executar com outro diretório de saída:**

```bash
python run_lab.py --out results/experimento_local
```

**Executar com semente alternativa (apenas investigação):**

```bash
python run_lab.py --seed 123 --out results/semente_123
```

> Para a **entrega**, utilize `python run_lab.py`, sem opções, para gerar os resultados em `results/` com semente 42. A opção `--no-cv` existe, mas não deve ser usada na avaliação final, pois desabilita a validação cruzada.

## 7. Abrir e executar o notebook no VS Code

O notebook serve para **acompanhar didaticamente** a auditoria, os atributos, o particionamento dos participantes, a comparação e os gráficos.

1. No VS Code, abra a pasta `labs/aula-11-random-forest`.
2. No Explorer, abra `notebooks/LAB11_Human_AI_Bias.ipynb`.
3. No canto superior direito, clique em **Select Kernel** / **Selecionar Kernel**.
4. Escolha **Python Environments** e selecione `.venv/bin/python` deste laboratório.
5. Clique em **Run All** / **Executar Tudo**.
6. Verifique se todas as células terminam sem erro.
7. Se desejar preservar a saída, salve uma cópia como `notebooks/LAB11_Human_AI_Bias_executado.ipynb`.

Se o ambiente não aparecer como kernel, ative `.venv` no terminal e instale:

```bash
python -m pip install ipykernel
```

Depois, reabra o seletor de kernel.

**Atenção:** o notebook lê alguns arquivos gerados pelo script principal. Portanto, execute **`python run_lab.py` antes de executar todas as células**.

### 7.1 Executar o notebook pelo terminal (opcional)

Na raiz do laboratório, com `.venv` ativo:

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/LAB11_Human_AI_Bias.ipynb \
  --output LAB11_Human_AI_Bias_executado.ipynb \
  --ExecutePreprocessor.timeout=600
```

A versão executada será gravada em `notebooks/LAB11_Human_AI_Bias_executado.ipynb`. O comando pode substituir esse arquivo se já existir. Caso o `nbconvert` reclame que não encontrou um kernel, instale `ipykernel` no `.venv` ou execute o notebook pelo VS Code.

## 8. Utilizar a interface Streamlit (opcional)

O Streamlit é um painel **somente para exploração**: não treina automaticamente os modelos e não substitui a execução principal.

### 8.1 Instalar e iniciar

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

O Streamlit normalmente informa um endereço local semelhante a `http://localhost:8501`. Abra o endereço mostrado no terminal.

### 8.2 O que explorar na interface?

- **Indicadores:** número de registros, participantes e condições.
- **Condições:** selecione `Sem IA`, `IA precisa`, `IA enviesada` ou `IA ruidosa`; observe as médias e os gráficos de desvios.
- **Comparação de modelos:** consulte a tabela e o gráfico de MAE dos algoritmos e cenários.
- **Ganho com histórico:** examine a diferença de MAE e seu intervalo de confiança.
- **Previsões individuais:** selecione um participante do conjunto de teste e compare o valor-alvo com uma das predições disponíveis.

Para encerrar a aplicação, volte ao terminal e pressione `Ctrl+C`.

**Observação:** a interface exibe os arquivos de `results/` gerados pela execução-padrão. Se forem produzidos resultados em outro diretório com `--out`, eles não aparecerão automaticamente no painel.

## 9. Onde encontrar os resultados?

| Arquivo em `results/` | Como utilizar |
|---|---|
| `AUDITORIA.json` | Conferir forma, participantes, condições e verificações do CSV. |
| `SPLIT_PARTICIPANTES.json` | Ver os IDs de treino e teste e a semente empregada. |
| `ESTATISTICA_DESCRITIVA.csv` | Examinar desvios e erros descritivos por condição. |
| `METRICAS_MODELOS.csv` | Comparar baselines, modelos, cenários e métricas. |
| `METRICAS_POR_CONDICAO.csv` | Comparar desempenho dos modelos por condição experimental. |
| `INCERTEZA_HISTORICO.csv` | Consultar ganho de MAE e IC de 95% por modelo. |
| `PREDICOES_TESTE.csv` | Consultar predições por participante e tentativa. |
| `01_vies_por_condicao.png` | Visualização descritiva de desvios por condição. |
| `02_mae_modelos.png` | Comparação dos erros dos modelos. |
| `03_tendencia_intra_bloco.png` | Evolução descritiva dentro dos blocos de tentativas. |
| `04_real_vs_previsto.png` | Valores reais versus preditos. |
| `RELATORIO_RESULTADOS.md` | Síntese automática dos resultados. |

**Para uma avaliação rápida:** leia `README.md`, consulte `docs/ANALISE_ACADEMICA.md`, abra o notebook executado e verifique `results/RELATORIO_RESULTADOS.md`.

## 10. Como interpretar as métricas

| Métrica | Interpretação prática |
|---|---|
| **MAE** | Média do erro absoluto entre desvio previsto e observado. **Menor é melhor**. |
| **RMSE** | Erro quadrático médio com raiz; penaliza mais os erros grandes. **Menor é melhor**. |
| **R²** | Fração da variabilidade explicada em relação à referência da média do teste; pode ser negativa. **Maior é melhor**, dentro do mesmo conjunto de avaliação. |
| **`cv_mae_mean`** | MAE médio dos *folds* agrupados por participante, calculados somente no conjunto de treinamento. |
| **`late_block_MAE`** | MAE em tentativas posteriores dos blocos de interação. |
| **`delta_mae`** | `MAE(contexto) − MAE(historico)`. **Positivo favorece o histórico**. |

Na execução local, o **Random Forest com histórico** atingiu o menor MAE (12,608), enquanto o **XGBoost com histórico** obteve o maior R² (0,229). Isso não estabelece que um algoritmo é sempre superior ao outro: depende da métrica utilizada e do contexto da avaliação.

### Cuidado com a interpretação científica

- `contexto` usa o estímulo e o desenho do experimento, sem observações anteriores.
- `historico` combina **comportamento humano anterior** e **informações de IA anteriormente reveladas**.
- Os atributos históricos são construídos com informações de tentativas **anteriores à tentativa prevista**.
- Participantes de teste não são usados no treinamento, mas o cenário `historico` pode usar suas **tentativas anteriores já observadas** na previsão das próximas; portanto, não representa uma avaliação *cold start*.
- Uma melhoria preditiva **não é prova de influência causal da IA** sobre o julgamento.

## 11. Ordem sugerida de apresentação para a professora

1. **Problema:** prever o desvio entre a estimativa humana e o valor do estímulo.
2. **Dados:** apresentar `Exp2.csv`, as quatro condições e a auditoria de 120 participantes/14.400 registros.
3. **Metodologia:** explicar alvo de regressão, cenários, separação por participante e proteção contra vazamento temporal.
4. **Algoritmos:** resumir árvore individual, Random Forest e XGBoost.
5. **Resultados:** comparar baselines, MAE/RMSE/R² e ganhos com histórico.
6. **Limitações:** explicitar que a comparação não identifica a contribuição causal da IA.
7. **Demonstração:** executar `python -m pytest -v`, `python run_lab.py` e, opcionalmente, o Streamlit.

## 12. Solução de problemas

| Sintoma | Causa provável | Como resolver |
|---|---|---|
| `python: command not found` | Python não disponível no PATH | Verifique `python3 --version`, ative `.venv` e use o `python` do ambiente. |
| Erro ao criar `.venv` | Módulo `venv` ausente | Instale `python3-venv` correspondente à sua versão de Python. |
| `ModuleNotFoundError` | Dependências ausentes ou kernel incorreto | Ative `.venv` e execute `python -m pip install -r requirements.txt`. |
| `data/Exp2.csv` não encontrado | Arquivo ausente ou local incorreto | Confirme `ls data/Exp2.csv` na raiz do laboratório. |
| `test_source_hash` falha | CSV alterado ou versão diferente | Compare o SHA-256 com o valor da seção 4; não edite ou reordene o arquivo original. |
| Notebook não encontra kernel | Ambiente não registrado para notebooks | Instale `ipykernel` no `.venv` e selecione `.venv/bin/python` no VS Code. |
| Notebook não encontra CSV/arquivos de resultado | Notebook aberto em local inesperado ou script não executado | Execute o notebook dentro da pasta do projeto e rode `python run_lab.py` antes. |
| Streamlit informa que faltam resultados | `results/` ainda não foi gerado | Rode `python run_lab.py` sem `--out` antes de abrir a interface. |
| Streamlit não inicia | Dependência opcional não instalada | Execute `python -m pip install -r requirements-ui.txt`. |
| Resultados numéricos diferentes | Modo `--quick`, semente ou versões diferentes | Repita `python run_lab.py` sem flags, com CSV íntegro e semente padrão 42. |

Para identificar qual Python está ativo:

```bash
which python
python --version
python -m pip --version
```

## 13. Referências e documentos relacionados

- **Artigo-base:** Glickman, M.; Sharot, T. *How human–AI feedback loops alter human perceptual, emotional and social judgements*. *Nature Human Behaviour* 9, 345–359 (2025). DOI: https://doi.org/10.1038/s41562-024-02077-2.
- **Repositório dos autores:** https://github.com/affective-brain-lab/BiasedHumanAI.
- **Proveniência dos dados:** `data/SOURCE.md`.
- **Protocolo científico:** `docs/PROTOCOLO_EXPERIMENTAL.md`.
- **Análise e limitações:** `docs/ANALISE_ACADEMICA.md`.
- **Introdução ao projeto:** `README.md`.

---

**Resumo operacional:** `source .venv/bin/activate` → `python -m pytest -v` → `python run_lab.py` → abrir notebook e, opcionalmente, `streamlit run app.py`.
