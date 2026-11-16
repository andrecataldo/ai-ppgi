# LAB 12 - PLN e Representações Distribuídas

**PPGI/UNIRIO · Inteligência Artificial · Laboratório didático**

**Tema:** modelo neural probabilístico de linguagem inspirado em Bengio et al. (2003).

## 1. Pergunta do laboratório

> Como um modelo probabilístico de linguagem com embeddings aprendidos pode prever a próxima palavra e em que difere de um modelo tradicional de n-gramas?

O objetivo é **demonstrar os conceitos**, não reproduzir a arquitetura em escala ou os resultados numéricos do artigo original.

## 2. O que será demonstrado

1. Um corpus pequeno, sintético e controlado de **288 frases** em português.
2. Construção de um vocabulário e de pares *(três palavras de contexto → próxima palavra)*.
3. Modelo de **trigramas de contexto** (*4-grama*, 3 palavras anteriores) com suavização aditiva.
4. Modelo neural com **matriz de embeddings**, camada oculta `tanh`, conexões diretas e saída `softmax`, conforme a equação de Bengio.
5. Comparação da **perplexidade no teste**, geração de previsões e inspeção de vizinhos vetoriais por similaridade de cosseno.

**Não inclui:** LLM, Transformer, prompting, ajuste fino, RAG, download de corpus ou interface web. São temas diferentes.

## 3. Estrutura

```text
labs/aula-12-pln-representacoes-distribuidas/
├── .gitignore
├── README.md
├── RELATORIO_LAB.md
├── requirements.txt
├── run_lab.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── experiment.py
├── tests/
│   └── test_lab.py
└── results/
    ├── metricas.json               # gerado, ignorado no Git
    └── RELATORIO_RESULTADOS.md     # gerado e pronto para versionar
```

## 4. Pré-requisitos

- Python 3.11+; PyTorch (CPU basta)
- Sem GPU e sem download de dados

## 5. Execução no Ubuntu

```bash
cd ~/projects/ai-ppgi/labs/aula-12-pln-representacoes-distribuidas
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# Execute o experimento completo
python run_lab.py

# Execute os testes (unittest da biblioteca padrão)
python -m unittest discover -s tests -v
```

Para uma execução de depuração mais rápida:

```bash
python run_lab.py --epochs 3
```

**Atenção:** reduzir épocas altera os resultados. Para entrega, mantenha as configurações padrão (semente 42, 25 épocas).

## 6. Como ler os resultados

O terminal exibe as duas perplexidades, quatro vizinhos vetoriais de `gato` e cinco candidatas para completar `o gato dorme no`.

Em `results/` são gerados:

- `metricas.json`: saída estruturada para auditoria.
- `RELATORIO_RESULTADOS.md`: resultados numéricos e limites da interpretação.

A perplexidade deve ser comparada **somente neste conjunto de teste**, sob o mesmo vocabulário e tokenização. Menor perplexidade indica maior probabilidade das palavras observadas, e não necessariamente compreensão semântica.

## 7. O que investigar

- A loss de treinamento diminuiu?
- Qual abordagem obteve menor perplexidade de teste nesta execução? Por quê isso não prova superioridade geral?
- Quais palavras ficaram próximas de `gato`? São semanticamente próximas? Se não, o que isso revela?
- O modelo prevê uma palavra plausível para `o gato dorme no`?
- O que mudaria se fossem usadas cinco palavras de contexto, como em uma das configurações do artigo?

## 8. Limitações metodológicas

O corpus é **sintético, repetitivo, de domínio fechado**, gerado por combinações previamente definidas. O teste utiliza frases completas não incluídas no treino, porém ambas as partições compartilham o mesmo padrão gramatical e vocabulário. Portanto, **não** é avaliação de generalização para novos domínios.

O n-grama usa suavização aditiva, enquanto Bengio compara com baselines mais fortes, incluindo Kneser-Ney e modelos de classes. Os resultados numéricos **não são comparáveis** à publicação de 2003.

A inspeção de embeddings não comprova aquisição de semântica: relações vetoriais dependem dos dados, da semente e da função de treinamento. O treinamento atual não realiza validação de hiperparâmetros ou múltiplas repetições com intervalos de confiança.

## 9. Referência

Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A neural probabilistic language model*. Journal of Machine Learning Research, 3, 1137–1155.
