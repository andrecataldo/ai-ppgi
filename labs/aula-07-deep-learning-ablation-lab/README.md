# Deep Learning Ablation Lab - V4 Final

Laboratório desenvolvido para a disciplina de **Inteligência Artificial - PPGI/UNIRIO**.

## Pergunta-guia

> **O desempenho em Deep Learning emerge de uma técnica isolada ou da interação entre arquitetura, ativação, regularização, dados e otimização?**

O laboratório usa **CIFAR-10, PyTorch e Streamlit** como um pequeno estudo de ablação.  
O objetivo não é reproduzir numericamente AlexNet/ImageNet, mas transformar decisões discutidas nos artigos da aula em hipóteses experimentais observáveis.

## Regime experimental principal

- 10.000 imagens de treinamento;
- 2.000 imagens de teste;
- 10 épocas;
- batch size 128;
- seed 0;
- CPU.

```bash
python lab.py --quick --epochs 10
```

## Ablações

| Bloco | Comparação | Pergunta |
|---|---|---|
| Arquitetura | `mlp_relu` × `cnn_relu` | O que muda ao introduzir localidade e compartilhamento de pesos? |
| Ativação | `cnn_relu` × `cnn_tanh` | Como ReLU e tanh se comportam sob a mesma arquitetura? |
| Regularização | `reg_base`, `reg_aug`, `reg_dropout`, `reg_all` | Como augmentation e dropout afetam ajuste e generalização? |
| Otimização | `reg_all` × `opt_adam` | Como SGD+momentum e Adam se comportam sob a mesma regularização? |
| Filtros | inicialização × pós-treino | Como backpropagation modifica os kernels da primeira convolução? |
| Representações | features × pixels | O espaço aprendido organiza melhor as imagens do que pixels brutos? |

## Resultados de classificação

| Experimento | Melhor acurácia de teste | Acurácia final de teste | Acurácia final de treino | Gap final |
|---|---:|---:|---:|---:|
| MLP + ReLU | 49,4% | 49,2% | 74,2% | +25,1 p.p. |
| CNN + ReLU | 45,7% | 45,1% | 45,7% | +0,6 p.p. |
| CNN + tanh | **55,9%** | 55,5% | 58,0% | +2,6 p.p. |
| Baseline + weight decay | 44,6% | 44,6% | 44,4% | -0,2 p.p. |
| + Data augmentation | 38,1% | 38,1% | 37,3% | -0,8 p.p. |
| + Dropout | 35,2% | 35,2% | 34,5% | -0,8 p.p. |
| + Augmentation + dropout | 30,3% | 29,9% | 29,3% | -0,6 p.p. |
| Adam + regularização | 50,6% | 50,6% | 51,5% | +0,9 p.p. |

## Resultados exploratórios de representação

Avaliação com **30 imagens de teste, 1.000 referências de treino e k=5**, usando a mesma amostra para os três modelos:

| Modelo | Melhor acurácia | Purity@5 · representação | Purity@5 · pixels | Ganho sobre pixels |
|---|---:|---:|---:|---:|
| CNN + ReLU | 45,7% | 38,0% | 24,7% | +13,3 p.p. |
| CNN + tanh | **55,9%** | 26,7% | 24,7% | +2,0 p.p. |
| Adam + regularização | 50,6% | **42,7%** | 24,7% | **+18,0 p.p.** |

### Accuracy × qualidade da representação

A ordenação por acurácia foi:

1. CNN + tanh;
2. Adam + regularização;
3. CNN + ReLU.

A ordenação por Purity@5 foi:

1. Adam + regularização;
2. CNN + ReLU;
3. CNN + tanh.

Portanto, **o modelo com maior acurácia não foi o modelo com maior organização local do espaço interno segundo Purity@5**. Isso não implica oposição entre accuracy e qualidade da representação; apenas mostra que as duas métricas capturam propriedades diferentes.

## Leitura das ablações

### Arquitetura

A MLP atingiu acurácia de teste ligeiramente superior à CNN+ReLU dentro do orçamento de 10 épocas, mas apresentou gap de generalização de aproximadamente 25 p.p. A CNN+ReLU permaneceu com gap próximo de zero. Ajuste do treino e generalização não são equivalentes.

### Ativação

A CNN+tanh superou a CNN+ReLU neste regime. O resultado não implica superioridade universal da tanh: arquitetura, dataset, learning rate e demais hiperparâmetros são diferentes dos experimentos históricos da AlexNet.

### Regularização

A baseline não apresentou overfitting relevante. Nesse cenário, augmentation e dropout reduziram a acurácia dentro de 10 épocas, sugerindo que regularização adicional pode retardar o ajuste quando o modelo ainda está subajustado.

### Otimização

Com arquitetura e regularização semelhantes, Adam atingiu 50,6%, enquanto SGD+momentum ficou em 30,3% na configuração `reg_all`. A conclusão é sobre convergência dentro deste orçamento experimental, não superioridade universal do Adam.

### Representação

Purity@5 mostrou que a organização do espaço interno não reproduziu a mesma ordenação da acurácia. Isso reforça que “melhor modelo” depende da propriedade avaliada.

## Mensagem central

> **As técnicas não atuaram de forma independente. Arquitetura, ativação, regularização e otimização interagiram fortemente; além disso, diferentes métricas revelaram propriedades diferentes dos modelos. Atribuir o desempenho ou a qualidade de uma rede a um único componente seria uma simplificação excessiva.**

## Interface Streamlit

```bash
streamlit run app.py
```

Abas:

1. **Ablations** - curvas, placar e síntese automática;
2. **Classificação** - Top-1, softmax, ranking e margem Top-1/Top-2;
3. **Filtros** - pesos antes/depois e mudança relativa da Conv1;
4. **Representações** - vizinhos, Purity@k e avaliação amostral.

Na aba **Ablations**, o botão **Calcular síntese de Purity@5** recalcula a comparação para os três modelos usando os checkpoints atuais.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Linux sem NVIDIA:

```bash
python -m pip install torch torchvision   --index-url https://download.pytorch.org/whl/cpu
```

Demais dependências:

```bash
python -m pip install -r requirements.txt
```

## Limitações

- CIFAR-10 é muito menor e mais simples que ImageNet.
- A CNN é uma MiniCNN, não a AlexNet original.
- Apenas uma seed foi utilizada.
- O regime principal usa 10 épocas e 10 mil imagens de treino.
- As ablações não constituem isolamento causal perfeito.
- Purity@5 foi calculada em uma amostra exploratória de 30 queries e 1.000 referências.
- Top-5 no CIFAR-10 é apenas didático.
- Softmax não representa “certeza” humana.
- Purity@k mede organização local de vizinhança, não compreensão semântica.

## Execução

Smoke test:

```bash
python lab.py --quick
```

Experimento principal:

```bash
python lab.py --quick --epochs 10
```
