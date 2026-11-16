# Relatório acadêmico conciso - Lab 12

**Tema:** PLN e representações distribuídas.

## Objetivo

Implementar um modelo de linguagem que aprende embeddings de palavras para prever o próximo token, comparar com um modelo estatístico simples de n-gramas e analisar a perplexidade obtida.

## Referência conceitual

Bengio et al. (2003) propõem aprender conjuntamente representações distribuídas das palavras e a função de probabilidade condicional. A arquitetura deste laboratório ilustra a equação `y = b + Wx + U tanh(d + Hx)`, seguida de softmax.

## Metodologia

O corpus de 288 frases sintéticas contém pequenas variações de sujeitos, ações e locais, construídas sem fontes externas. As frases são divididas em 80% treino e 20% teste por sorteio com semente 42. Os exemplos são montados com três tokens anteriores. O baseline utiliza suavização aditiva com α=0,5. O modelo neural usa embeddings de 16 dimensões, 32 unidades ocultas, Adam e 25 épocas. Os dois são avaliados nos mesmos exemplos de teste, utilizando perplexidade.

## Resultados

Os valores medidos e os exemplos de vizinhos vetoriais são gerados em [`results/RELATORIO_RESULTADOS.md`](results/RELATORIO_RESULTADOS.md). A versão desse arquivo no pacote reflete uma execução local com parâmetros padrão.

## Discussão crítica

Uma redução de perplexidade, se observada, indica melhor previsão nesse corpus e não prova compreensão semântica. A combinação de pequenos grupos de palavras e estruturas repetitivas favorece a aprendizagem de regularidades locais. Os vetores aprendidos são específicos da tarefa; próximos vizinhos não substituem uma avaliação semântica independente. O baseline didático é mais simples que os modelos de Kneser-Ney estudados no artigo.

## Conclusão

O experimento demonstra como embeddings participam do treinamento conjunto de um modelo de linguagem, como os scores são normalizados e como comparar modelos por perplexidade. Os limites de escala e de validade externa impedem generalização dos resultados para linguagem natural aberta.

## Declaração de uso de IA

O código, a documentação e a estrutura inicial foram produzidos com auxílio de um assistente de IA. A execução, revisão das hipóteses, inspeção dos resultados e responsabilidade pela entrega devem ser assumidas pelo estudante. Conferir resultados no ambiente local antes da submissão.
