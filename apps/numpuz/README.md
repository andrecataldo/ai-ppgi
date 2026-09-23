# 🧩 NUMPUZ - Busca e Resolução de Problemas

Aplicação desenvolvida para a disciplina de **Inteligência Artificial** do PPGI/UNIRIO.

O projeto utiliza o clássico **8-puzzle** para demonstrar, de forma interativa, como diferentes algoritmos percorrem um espaço de estados para transformar um **estado inicial `I`** em um **estado objetivo `S`**.

---

## Objetivo

O problema é definido por dois estados:

```text
I = estado inicial
S = estado objetivo
```

A tarefa dos algoritmos de busca é encontrar uma sequência de movimentos válidos que transforme:

```text
I → S
```

O mesmo problema pode ser resolvido utilizando quatro estratégias:

- Breadth-First Search - BFS;
- Depth-First Search - DFS;
- A* + Misplaced Tiles;
- A* + Manhattan Distance.

Isso permite comparar como diferentes estratégias percorrem o mesmo espaço de estados.

---

## Formulação como problema de busca

| Elemento | Definição |
|---|---|
| Estado | configuração das 8 peças e do espaço vazio |
| Estado inicial `I` | configuração informada pelo usuário ou gerada aleatoriamente |
| Estado objetivo `S` | solução canônica ou configuração válida informada pelo usuário |
| Ações | mover o espaço vazio para cima, baixo, esquerda ou direita |
| Função sucessora | troca do vazio com uma peça adjacente |
| Teste objetivo | `estado_atual == S` |
| Custo | 1 por movimento |

O espaço vazio é representado internamente por `0` e visualmente por `_`.

---

## Estado objetivo canônico

A solução tradicional do 8-puzzle é:

```text
1 2 3
4 5 6
7 8 _
```

Representação interna:

```python
GOAL_STATE = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0,
)
```

Entretanto, essa configuração é apenas o **objetivo padrão**.

A aplicação permite que o usuário defina outro estado válido como `S`.

Exemplo:

```text
1 2 3
4 5 6
7 _ 8
```

Nesse caso, todos os algoritmos passam a procurar exatamente essa configuração.

---

## Entrada do problema

### Estado objetivo - S

O usuário pode escolher:

```text
Solução canônica
```

ou:

```text
Informar manualmente
```

Exemplo de entrada:

```text
123/456/7_8
```

### Estado inicial - I

O usuário pode:

```text
Informar manualmente
```

ou:

```text
Gerar aleatoriamente
```

Quando `I` é gerado aleatoriamente, a aplicação parte de `S` e executa uma sequência de movimentos válidos:

```text
S
↓
movimento válido
↓
movimento válido
↓
...
↓
I
```

Isso garante que o estado gerado pertence ao mesmo espaço alcançável de `S`.

---

## Validação dos estados

Tanto `I` quanto `S` devem conter exatamente:

```text
1 2 3 4 5 6 7 8 _
```

uma única vez cada.

Antes da execução da busca, a aplicação verifica também se:

```text
I pode alcançar S
```

através da paridade das configurações.

Caso os estados pertençam a componentes diferentes do espaço de estados, a busca não é executada.

---

## Teste de objetivo

O estado objetivo não é codificado diretamente dentro dos algoritmos.

Cada algoritmo recebe explicitamente:

```python
start
goal
```

Durante a busca, o estado selecionado é comparado com `S`:

```python
if current == goal:
    # solução encontrada
```

Assim, a condição de término é:

```text
estado_atual == S
```

e `S` pode ser tanto o objetivo canônico quanto um estado definido pelo usuário.

---

## Breadth-First Search - BFS

O BFS explora o espaço de estados por níveis.

```text
profundidade 0
↓
profundidade 1
↓
profundidade 2
↓
...
```

Como cada movimento possui custo 1:

```text
profundidade = custo
```

a primeira solução encontrada pelo BFS possui o menor número de movimentos.

O BFS é uma estratégia de **busca não informada**.

---

## Depth-First Search - DFS

O DFS explora um ramo em profundidade antes de retornar para explorar alternativas.

```text
I
│
├── estado
│   │
│   └── estado
│       │
│       └── ...
│
└── alternativas
```

O DFS também é uma estratégia de **busca não informada**.

Entretanto, diferentemente do BFS, ele:

```text
não garante a solução de menor número de movimentos
```

Por isso, dependendo da ordem de exploração dos sucessores, pode encontrar um caminho significativamente maior até `S`.

A ordem de movimentos utilizada pelo projeto é:

```text
UP
DOWN
LEFT
RIGHT
```

---

## A*

O A* combina o custo já percorrido com uma estimativa do custo restante:

```text
f(n) = g(n) + h(n)
```

onde:

```text
g(n) = custo desde I até n
h(n) = estimativa de n até S
f(n) = prioridade do estado
```

O projeto utiliza duas heurísticas.

---

## Misplaced Tiles

Conta quantas peças estão fora de sua posição no estado objetivo `S`.

O espaço vazio não é considerado.

Exemplo:

```text
Estado atual

1 2 3
4 5 6
_ 7 8
```

para:

```text
S

1 2 3
4 5 6
7 8 _
```

temos:

```text
h(n) = 2
```

---

## Manhattan Distance

Para cada peça é calculado:

```text
|linha atual - linha em S|
+
|coluna atual - coluna em S|
```

A heurística soma essas distâncias para todas as peças.

O espaço vazio é ignorado.

Diferentemente de uma implementação presa ao objetivo canônico, as posições utilizadas pela Manhattan Distance são calculadas a partir do próprio estado `S`.

---

## Quatro estratégias, um único problema

A comparação é realizada sempre sobre o mesmo par:

```text
(I, S)
```

Conceitualmente:

```text
                    I
                    │
        ┌───────────┼──────────────┐
        │           │              │
       BFS         DFS            A*
        │           │        ┌─────┴─────┐
        │           │    Misplaced    Manhattan
        │           │        │            │
        └───────────┴────────┴────────────┘
                    │
                    S
```

Para cada estratégia são coletadas as métricas:

- custo da solução encontrada;
- estados expandidos;
- estados gerados;
- tempo de execução.

---

## Optimalidade

Para movimentos com custo unitário:

| Estratégia | Garante menor número de movimentos? |
|---|---|
| BFS | Sim |
| DFS | Não |
| A* + Misplaced Tiles | Sim, com a heurística utilizada |
| A* + Manhattan Distance | Sim, com a heurística utilizada |

Portanto, o DFS pode apresentar um custo diferente das demais estratégias.

Essa diferença faz parte do comportamento esperado do algoritmo.

---

## Exemplo simples

Considere:

```text
I

1 2 3
4 5 6
7 _ 8
```

e:

```text
S

1 2 3
4 5 6
7 8 _
```

Existe uma solução de apenas um movimento:

```text
RIGHT
```

Agora considere o problema inverso:

```text
I

1 2 3
4 5 6
7 8 _
```

```text
S

1 2 3
4 5 6
7 _ 8
```

A solução passa a ser:

```text
LEFT
```

Esse exemplo demonstra que o objetivo da busca é efetivamente o estado `S` fornecido ao algoritmo.

---

## Jogo manual

Além da resolução automática, a interface permite mover manualmente as peças.

O usuário parte de `I` e tenta alcançar `S`.

A métrica:

```text
Manhattan até S
```

é atualizada em relação ao objetivo atualmente configurado.

---

## Arquitetura

```text
apps/numpuz/
├── app.py
├── compare_search.py
├── heuristics.py
├── puzzle.py
├── solver.py
├── README.md
└── tests/
    ├── test_heuristics.py
    ├── test_puzzle.py
    └── test_solver.py
```

### `puzzle.py`

Responsável por:

- representação dos estados;
- parsing da entrada do usuário;
- validação;
- movimentos válidos;
- função sucessora;
- teste de objetivo;
- solvabilidade entre `I` e `S`;
- geração de estados solucionáveis.

### `heuristics.py`

Implementa:

- Misplaced Tiles;
- Manhattan Distance.

Ambas recebem explicitamente o estado objetivo `S`.

### `solver.py`

Implementa:

- BFS;
- DFS;
- A*;
- reconstrução do caminho;
- métricas da busca.

Todos os algoritmos recebem:

```python
start
goal
```

### `compare_search.py`

Executa:

- BFS;
- DFS;
- A* + Misplaced Tiles;
- A* + Manhattan Distance;

sobre exatamente o mesmo problema.

### `app.py`

Interface Streamlit responsável pela:

- definição de `I`;
- definição de `S`;
- geração aleatória do estado inicial;
- exploração manual;
- escolha do algoritmo;
- visualização do caminho;
- comparação das quatro estratégias.

---

## Testes

Execute:

```bash
cd apps/numpuz
python -m unittest discover -s tests -v
```

A suíte cobre, entre outros aspectos:

- movimentos válidos;
- estados solucionáveis e não solucionáveis;
- objetivos personalizados;
- parsing da entrada;
- BFS;
- DFS;
- A*;
- heurísticas relativas a objetivos personalizados;
- consistência dos caminhos produzidos.

---

## Execução via terminal

```bash
python compare_search.py
```

---

## Execução da aplicação

```bash
streamlit run app.py
```

---

## Conceitos demonstrados

O projeto exemplifica:

- espaço de estados;
- estado inicial;
- estado objetivo;
- função sucessora;
- teste de objetivo;
- custo de caminho;
- busca em largura;
- busca em profundidade;
- busca informada;
- busca não informada;
- heurística;
- admissibilidade;
- optimalidade;
- controle de estados repetidos;
- solvabilidade;
- comparação experimental entre estratégias.

---

## Conclusão

O NUMPUZ demonstra que um problema de busca pode ser formulado independentemente de um estado objetivo específico.

O usuário define:

```text
I = estado inicial
S = estado objetivo
```

e diferentes algoritmos percorrem o espaço de estados acessível a partir de `I` até encontrar `S`.

BFS e DFS utilizam apenas a estrutura do espaço de busca, enquanto o A* utiliza informação heurística sobre a distância até o objetivo.

As duas heurísticas implementadas permitem ainda observar como a qualidade da informação utilizada pelo A* pode reduzir significativamente o número de estados explorados.

A comparação sobre o mesmo par `I → S` permite visualizar de forma prática como diferentes estratégias de busca podem produzir comportamentos e custos computacionais bastante distintos para resolver exatamente o mesmo problema.