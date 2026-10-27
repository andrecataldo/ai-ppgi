# Relatório - Deep Learning Ablation Lab

## 1. Objetivo

O laboratório investiga, em escala reduzida, como diferentes decisões de projeto afetam o comportamento de modelos de Deep Learning. A pergunta central é se o desempenho observado pode ser atribuído a uma técnica isolada ou se emerge da interação entre arquitetura, função de ativação, regularização e otimização.

O experimento foi inspirado nos três textos estudados na aula: AlexNet como caso experimental, LeCun, Bengio e Hinton como consolidação conceitual do aprendizado de representações, e Mienye e Swart como panorama da evolução das arquiteturas e técnicas de treinamento.

## 2. Método

Foi utilizado o CIFAR-10, com imagens RGB 32×32 e dez classes. Para manter a execução viável em CPU, o regime principal utilizou 10.000 imagens de treinamento, 2.000 imagens de teste, 10 épocas, batch size 128 e seed 0.

Foram organizadas quatro ablações: arquitetura (MLP versus CNN), ativação (ReLU versus tanh), regularização (baseline, augmentation, dropout e combinação) e otimização (SGD+momentum versus Adam).

Além das métricas de classificação, foram examinados os filtros da primeira camada convolucional e a organização do espaço interno de representação. Para esta última análise foi utilizada Purity@5, calculada com 30 imagens de teste e 1.000 referências de treino.

## 3. Resultados de classificação

A maior acurácia de teste observada foi 55,9%, obtida pela CNN+tanh. A MLP+ReLU atingiu 49,4%, mas apresentou gap final de generalização de 25,1 pontos percentuais, contra 0,6 p.p. na CNN+ReLU.

Na ablação de regularização, a baseline com weight decay atingiu 44,6%. Augmentation, dropout e a combinação reduziram o desempenho dentro do orçamento de 10 épocas. Entretanto, quando a configuração com augmentation+dropout foi treinada com Adam, sua melhor acurácia subiu de 30,3% para 50,6%.

## 4. Qualidade da representação

A avaliação exploratória de Purity@5 produziu os seguintes resultados:

| Modelo | Melhor acurácia | Purity@5 aprendido | Purity@5 pixels |
|---|---:|---:|---:|
| CNN + ReLU | 45,7% | 38,0% | 24,7% |
| CNN + tanh | 55,9% | 26,7% | 24,7% |
| Adam + regularização | 50,6% | 42,7% | 24,7% |

O ranking por acurácia foi CNN+tanh, Adam+regularização e CNN+ReLU. Já o ranking por Purity@5 foi Adam+regularização, CNN+ReLU e CNN+tanh.

Assim, o modelo com maior acurácia não foi o modelo com maior organização local segundo Purity@5. Isso não demonstra oposição entre as métricas; mostra que elas avaliam propriedades diferentes. Acurácia mede desempenho da decisão final de classificação, enquanto Purity@5 mede a composição local das vizinhanças no espaço interno segundo distância Euclidiana.

## 5. Discussão

Os resultados mostram que as técnicas não se comportam de forma independente. A MLP ajustou o treino rapidamente, mas generalizou pior. A tanh superou ReLU neste cenário, embora isso não reproduza o comportamento histórico reportado pela AlexNet. A regularização adicional não trouxe benefício quando a CNN baseline ainda não apresentava overfitting relevante. Já a troca de SGD+momentum por Adam alterou fortemente o desempenho da mesma configuração regularizada.

A análise da representação acrescenta outra dimensão: um modelo pode atingir maior acurácia sem necessariamente produzir as vizinhanças mais puras na camada oculta analisada. Portanto, “melhor modelo” depende da propriedade considerada.

## 6. Relação com os artigos

A AlexNet mostrou empiricamente que uma combinação de CNN profunda, GPU, ReLU, data augmentation, dropout e grande volume de dados podia superar fortemente o estado da arte. O laboratório não tenta reproduzir seus números, mas transforma parte dessas decisões em variáveis experimentais.

A discussão de LeCun, Bengio e Hinton sobre representation learning aparece nas abas de filtros e representações: os kernels começam aleatórios, são ajustados por backpropagation e o espaço interno resultante pode ser avaliado independentemente da accuracy final.

A revisão de Mienye e Swart ajuda a contextualizar a ampliação das técnicas de treinamento e a necessidade de avaliar modelos por múltiplas propriedades, não apenas por uma única métrica.

## 7. Limitações

O experimento usa CIFAR-10 reduzido, uma seed, 10 épocas e uma arquitetura pequena. As ablações são didáticas e não constituem isolamento causal completo. A avaliação de Purity@5 usa somente 30 queries e 1.000 referências, devendo ser interpretada como exploratória.

## 8. Conclusão

O laboratório sugere que o desempenho de Deep Learning não decorre de uma técnica isolada. Arquitetura, ativação, regularização e otimização interagem. Além disso, acurácia e organização local da representação não produziram o mesmo ranking entre os modelos avaliados, reforçando que qualidade de modelo é multidimensional.

A principal contribuição didática do experimento é tornar essas interações observáveis e mostrar por que resultados como os da AlexNet devem ser interpretados como consequência de uma configuração integrada, e não de um único componente.
