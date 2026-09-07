# Aula 03 - Busca e Resolução de Problemas

Laboratório comparativo dos principais algoritmos de busca estudados
na disciplina de Inteligência Artificial do PPGI/UNIRIO.

## Objetivo

Comparar, sobre exatamente o mesmo espaço de estados:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform Cost Search (UCS)
- Greedy Best-First Search
- A*

O experimento registra:

- frontier a cada iteração;
- nó selecionado;
- caminho encontrado;
- custo da solução;
- número de nós expandidos;
- optimalidade da solução encontrada.

## Grafo

```text
                         S
                    /    |    \
                 1 /   4 |   2 \
                  /      |      \
                 A       B       C
                 |       |       |
               9 |     2 |     2 |
                 |       |       |
                 G       G       D
                                 |
                               3 |
                                 |
                                 G
```

Caminhos possíveis:

```text
S -> A -> G    custo = 10
S -> B -> G    custo = 6
S -> C -> D -> G    custo = 7
```

A solução ótima é:

```text
S -> B -> G
```

com custo:

```text
6
```

## Heurística

Para Greedy e A*:

| Estado | h(n) |
|---|---:|
| S | 2 |
| A | 1 |
| B | 2 |
| C | 4 |
| D | 3 |
| G | 0 |

A heurística utilizada neste experimento é admissível.

## Critério de cada algoritmo

| Algoritmo | Critério para escolher o próximo nó |
|---|---|
| BFS | menor profundidade |
| DFS | maior profundidade / caminho atual |
| UCS | menor g(n) |
| Greedy | menor h(n) |
| A* | menor f(n) = g(n) + h(n) |

## Executar

A partir da raiz do projeto:

```bash
source .venv/bin/activate
python labs/aula-03-busca/compare_search.py
```

## Resultado esperado

| Algoritmo | Caminho | Custo | Nós expandidos | Ótima? |
|---|---|---:|---:|---|
| BFS | S -> A -> G | 10 | 4 | Não |
| DFS | S -> A -> G | 10 | 2 | Não |
| UCS | S -> B -> G | 6 | 5 | Sim |
| Greedy | S -> A -> G | 10 | 2 | Não |
| A* | S -> B -> G | 6 | 3 | Sim |

## Interpretação

### BFS

Prioriza profundidade, não custo.

Como os custos das arestas são diferentes, a primeira solução encontrada
não é necessariamente ótima.

### DFS

Explora um caminho até o fim antes de tentar alternativas.

Neste exemplo encontra rapidamente uma solução, mas ela não é ótima.

### UCS

Expande sempre o estado cujo caminho possui menor custo acumulado:

```text
g(n)
```

Encontra a solução ótima, mas explora mais estados do que A*.

### Greedy Best-First

Usa somente:

```text
h(n)
```

Ou seja, escolhe o estado que parece estar mais próximo do objetivo.

É eficiente neste exemplo, porém retorna uma solução subótima.

### A*

Combina:

```text
g(n) + h(n)
```

onde:

- `g(n)` representa o custo acumulado;
- `h(n)` representa a estimativa do custo restante.

Com uma heurística adequada, encontra a solução ótima explorando menos
estados do que UCS neste experimento.

## Relação com Tree of Thoughts

No Tree of Thoughts, o LLM também avalia estados intermediários para
decidir quais caminhos parecem mais promissores.

A diferença importante é que a função de avaliação de ToT é produzida
pelo próprio modelo de linguagem e não possui automaticamente as
propriedades matemáticas de uma heurística admissível usada por A*.

Consequentemente, um caminho correto pode ser avaliado como pouco
promissor e eliminado durante a busca.
