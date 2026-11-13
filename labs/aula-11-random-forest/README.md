# LAB 11 - Human-AI Judgment Bias
## Árvores de Decisão, Random Forest e XGBoost em julgamentos humanos

**UNIRIO - PPGI | Disciplina: Inteligência Artificial | Aula 11 (03/11/2026)**

> **Pergunta de pesquisa:** o histórico de exposição a recomendações algorítmicas melhora a capacidade de modelos de árvores de prever desvios em estimativas humanas independentes?

Este laboratório usa o **Experimento 2 (RDK)** de Glickman & Sharot: decisões de 120 participantes em quatro condições (baseline sem IA; algoritmo preciso, enviesado e ruidoso), 30 tentativas por condição e 14.400 observações. **Não é um conjunto de dados sobre LLMs.**

> 📘 **Manual de Uso - LAB 11**
>
> Para instalar o ambiente, executar os testes e experimentos,
> utilizar o notebook Jupyter, explorar o painel Streamlit e
> interpretar os resultados, consulte o
> **[Manual de Uso do Laboratório](docs/MANUAL_DE_USO.md)**.
>
> O manual apresenta um passo a passo para reproduzir
> os experimentos e facilitar a avaliação acadêmica.


## 1. O que o laboratório ensina

- Como transformar uma pergunta científica em previsão supervisionada com **regressão**.
- Diferenças entre **árvore individual, bagging (Random Forest) e boosting (XGBoost)**.
- Como o modelo se beneficia (ou não) de histórico comportamental.
- Por que é necessário fazer **validação agrupada por participante**.
- Como construir atributos de histórico com `shift(1)`, evitando **vazamento temporal**.
- Como interpretar **MAE, RMSE, R²**, baselines e incerteza por participante.
- Limites entre **previsão, associação, efeitos de tratamento e interpretação causal**.

## 2. Origem e validação dos dados

Arquivo: `data/Exp2.csv` (incluído neste ZIP para execução local; excluído do Git por padrão).

- Fonte: https://github.com/affective-brain-lab/BiasedHumanAI/tree/main/Exp2/Data
- Artigo: Glickman & Sharot, *Nature Human Behaviour* 9, 345-359 (2025); online em 18/12/2024.
- DOI: https://doi.org/10.1038/s41562-024-02077-2
- SHA-256: `ec2207efc6fa936031107376857d48310990874d200de39c2880945ace574277`.
- O hash Git blob coincide com o repositório dos autores.

Os arquivos `Exp2-Accurate.csv` e `Exp2-Biased.csv` são experimentos complementares e **não devem ser concatenados** ao `Exp2.csv`.

## 3. Desenho experimental

Cada participante completa, na ordem efetiva das linhas:

```
30 tentativas sem IA (condition=0)
    → 30 com um algoritmo
    → 30 com outro algoritmo
    → 30 com o terceiro algoritmo
```

As três ordens de algoritmos são contrabalanceadas (40 participantes por ordem).

**Sequência dentro de uma tentativa com IA**:

```
Estímulo apresentado (evidence)
    ↓
Estimativa humana INDEPENDENTE (response)
    ↓
Confiança declarada (confidence)
    ↓
Peso atribuído à colaboração (collaboration)
    ↓
Resposta da IA revelada (responseAI)
    ↓
Tentativa seguinte: o conselho anterior já faz parte do histórico
```

**Instante da previsão do laboratório:** depois de conhecido o estímulo da tentativa `t`, **antes de o participante registrar `response_t`**. A variável `evidence_t` é conhecida ao pesquisador, não necessariamente estimada corretamente pelo participante.

**Alvo de regressão:** `signed_error_t = response_t - evidence_t`. É um desvio numérico, **não um rótulo diagnóstico de viés cognitivo**.

## 4. Hipótese e comparação

Dois cenários, mesmos participantes e mesmas divisões:

| Cenário | Variáveis disponíveis | Por quê? |
|---|---|---|
| `contexto` | `evidence`, `condition`, `order`, `trial_index`, `trial_in_block` | Informações do experimento, sem respostas anteriores |
| `historico` | Tudo de `contexto` **+** lags e médias históricas de erros, confiança, conselho anteriormente revelado e ponderação anterior | Testa valor preditivo incremental da história |

As variáveis históricas usam **sempre** `shift(1)` por participante. `responseAI_t`, `response_t`, `confidence_t` e `collaboration_t` atuais **não** entram no modelo. Respostas de IA presentes no baseline não são tratadas como conselho observado.

### Atenção a dois tipos de generalização

**(a) Para novos participantes:** 90 participantes no treino e 30 participantes completamente novos no teste (`GroupShuffleSplit`). Há também `GroupKFold` apenas entre participantes do treino, para estimar MAE fora de cada fold.

**(b) Para decisões futuras de uma pessoa:** no cenário histórico, o modelo pode usar **tentativas anteriores observadas da pessoa testada**, sem usar o evento corrente ou eventos futuros. Trata-se de previsão *online com histórico*, **não** de predição sem qualquer contato prévio (*cold start*). O relatório separa desempenho nas tentativas finais dos blocos.

**Limitação:** uma divisão participante-agrupada não garante independência de itens/estímulos repetidos entre pessoas, nem generalização para tarefas ou distribuições diferentes. Essa é uma pergunta distinta.

## 5. Estrutura

```
labs/aula-11-random-forest/  # no repositório: labs/aula-11-random-forest/
├── data/
│   ├── Exp2.csv                 # recebido do aluno, ignorado no Git
│   └── SOURCE.md                # proveniência e hash
├── docs/
│   ├── PROTOCOLO_EXPERIMENTAL.md
│   └── ANALISE_ACADEMICA.md
├── notebooks/
│   ├── LAB11_Human_AI_Bias.ipynb
│   └── LAB11_Human_AI_Bias_executado.ipynb
├── results/
│   ├── AUDITORIA.json
│   ├── SPLIT_PARTICIPANTES.json
│   ├── ESTATISTICA_DESCRITIVA.csv
│   ├── METRICAS_MODELOS.csv
│   ├── METRICAS_POR_CONDICAO.csv
│   ├── INCERTEZA_HISTORICO.csv
│   ├── PREDICOES_TESTE.csv
│   ├── 01_vies_por_condicao.png
│   ├── 02_mae_modelos.png
│   ├── 03_tendencia_intra_bloco.png
│   ├── 04_real_vs_previsto.png
│   └── RELATORIO_RESULTADOS.md
├── src/
│   ├── data.py
│   ├── models.py
│   └── experiment.py
├── tests/test_lab.py
├── app.py                      # visualização Streamlit opcional
├── run_lab.py                  # entrada de execução
├── requirements.txt
├── requirements-ui.txt
└── .gitignore
```

## 6. Execução no Ubuntu

```bash
cd ~/projects/ai-ppgi
cd labs/aula-11-random-forest

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# Execução completa: quatro folds de CV agrupado e três algoritmos em dois cenários
python run_lab.py

# Testes (12 cenários)
python -m pytest -v

# Notebook didático
jupyter lab notebooks/LAB11_Human_AI_Bias.ipynb

# Executar mais rapidamente, sem alterar o desenho metodológico
python run_lab.py --quick
```

Para UI (opcional):

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

**Importante:** `run_lab.py` com `--quick` reduz árvores e folds; isso muda os resultados numéricos. Use a execução completa para relatar as métricas documentadas.

## 7. Como ler os resultados

- `METRICAS_MODELOS.csv`: baselines e os três algoritmos, com/sem histórico; teste com participantes novos.
- `METRICAS_POR_CONDICAO.csv`: desempenho nos mesmos participantes separados por condição.
- `INCERTEZA_HISTORICO.csv`: diferença pareada no MAE `contexto - historico`, com bootstrap dos 30 participantes do teste. Se positivo, favorece histórico.
- `SPLIT_PARTICIPANTES.json`: IDs usados no treino e teste para auditoria.
- `PREDICOES_TESTE.csv`: alvo, predições e ordem original dentro de participante para refazer os gráficos.
- `RELATORIO_RESULTADOS.md`: produzido automaticamente pelo código.

## 8. Resultados da execução validada (semente 42)

| Modelo | MAE contexto | MAE histórico | Ganho percentual |
|---|---:|---:|---:|
| Decision Tree | 13,604 | 13,026 | 4,2% |
| Random Forest | 13,600 | **12,605** | **7,3%** |
| XGBoost | 13,535 | 12,674 | 6,4% |

Baseline sem desvio: **MAE 14,176**. Melhor R² histórico: **0,229**, XGBoost; RF histórico: **0,224**. As diferenças dependem da divisão, hiperparâmetros e definição de alvo.

**Interpretação permitida:** histórico de decisões e conselhos anteriores contém sinal preditivo para esta tarefa.

**Interpretação não permitida:** estes modelos provam que determinada resposta da IA *causou* uma mudança individual; a medida de importância explica causalmente o viés; um ML já classifica participantes como "enviesados".

## 9. Referências

- Glickman, M.; Sharot, T. *How human-AI feedback loops alter human perceptual, emotional and social judgements*. **Nature Human Behaviour** 9, 345-359 (2025). DOI: 10.1038/s41562-024-02077-2.
- Quinlan, J. R. (1986). *Induction of decision trees*. Machine Learning 1, 81-106.
- Loh, W.-Y. (2011). *Classification and Regression Trees*. WIREs Data Mining and Knowledge Discovery.
- Loh, W.-Y. (2014). *Fifty years of classification and regression trees*. International Statistical Review.
- Strobl, C. et al. (2007). *Bias in random forest variable importance measures*. BMC Bioinformatics 8, 25.
- Chen, T.; Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD.
