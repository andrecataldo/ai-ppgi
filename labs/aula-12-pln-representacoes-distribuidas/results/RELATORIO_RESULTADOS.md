# Relatório de resultados - PLN e representações distribuídas

**Experimento didático com corpus sintético.** Não reproduz os números de Bengio et al. (2003).

- Semente: 42; épocas: 25; frases treino/teste: 230/58.
- Vocabulário: 37 tokens (incluindo símbolos especiais); contexto: 3 palavras.
- Perplexidade no teste - n-grama aditivo: **5.303**.
- Perplexidade no teste - rede neural: **2.508**.
- Loss média no treino (primeira/última época): **1.755 / 0.852**.

## Vizinhos no espaço de embeddings

- **gato**: cachorro (cos=0.5635), pousa (cos=0.3289), no (cos=0.2805), quintal (cos=0.2605)
- **cachorro**: brinca (cos=0.6107), gato (cos=0.5635), pousa (cos=0.4108), descansa (cos=0.2988)
- **menino**: menina (cos=0.929), tapete (cos=0.2999), quintal (cos=0.2641), biblioteca (cos=0.2381)
- **pássaro**: canário (cos=0.5573), voa (cos=0.2742), corre (cos=0.2694), dorme (cos=0.2256)

## Próxima palavra - entrada: `o gato dorme no`

- sofá: 0.5855
- tapete: 0.4052
- sala: 0.0055
- quintal: 0.0006
- jardim: 0.0005

## Interpretação e limites

A diferença entre perplexidades não permite afirmar compreensão semântica.
O corpus é artificial, pequeno e repetitivo; os modelos recebem os mesmos exemplos de treino e teste,
mas arquiteturas, regularização e esforço computacional não são equiparados.
A ordem e conteúdo do conjunto de teste dependem da semente. Os vizinhos vetoriais são indicativos,
não uma avaliação semântica independente. Não há alegação de superioridade geral.
