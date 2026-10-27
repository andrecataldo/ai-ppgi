# Manual técnico - Deep Learning Ablation Lab

## Fluxo de treinamento

Em cada mini-batch:

1. normaliza as imagens;
2. aplica augmentation quando configurado;
3. executa o forward pass;
4. calcula cross-entropy;
5. executa `loss.backward()`;
6. atualiza parâmetros com `optimizer.step()`.

Esse fluxo separa conceitualmente **backpropagation** (cálculo dos gradientes) de **otimização** (atualização dos pesos).

## Modelos

### MLPBaseline

Recebe os pixels 32×32×3 achatados e usa duas camadas ocultas totalmente conectadas. Serve como baseline sem o viés espacial da convolução. Possui aproximadamente 3,68 milhões de parâmetros.

### MiniCNN

Possui cinco convoluções, ReLU ou tanh, max-pooling sobreposto e duas camadas ocultas de 512 unidades. Possui aproximadamente 2,08 milhões de parâmetros. Não pretende reproduzir as dimensões da AlexNet; apenas sua lógica geral em escala compatível com CIFAR-10.

## Métricas

Ao fim de cada época, o modelo entra em modo `eval` e as métricas de **treino e teste** são recalculadas sem augmentation e com dropout desligado. Isso evita comparar um treino artificialmente mais difícil com um teste limpo.

O laboratório registra por época:

- loss de treino;
- loss de teste;
- erro de treino;
- erro de teste;
- tempo.

Na interface, a acurácia é exibida como `1 - erro` e o **gap de generalização** como `acurácia_treino - acurácia_teste`.

## Interpretação correta

Não trate uma execução como evidência definitiva. Para uma análise experimental mais rigorosa, execute várias sementes e reporte média e desvio-padrão. O laboratório foi desenhado para discussão didática e para mostrar como uma ablação pode transformar afirmações de artigos em hipóteses testáveis.
