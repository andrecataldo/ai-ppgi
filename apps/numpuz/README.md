# 🧩 NUMPUZ - Busca e Resolução de Problemas

Aplicação desenvolvida para a disciplina de **Inteligência Artificial** do PPGI/UNIRIO.

O projeto utiliza o clássico **8-puzzle** para demonstrar, de forma interativa, conceitos de busca não informada e busca informada.

A aplicação permite:

- jogar o NUMPUZ manualmente;
- resolver automaticamente com Breadth-First Search (BFS);
- resolver com A* utilizando diferentes heurísticas;
- acompanhar o caminho da solução passo a passo;
- comparar custo, estados expandidos, estados gerados e tempo de execução.

---

## Objetivo

Modelar o NUMPUZ como um problema clássico de espaço de estados e comparar:

- Breadth-First Search (BFS);
- A* + Misplaced Tiles;
- A* + Manhattan Distance.

O objetivo principal é observar como o uso de conhecimento heurístico pode reduzir o espaço de busca mantendo a optimalidade da solução.

---

## Formulação como problema de busca

| Elemento | Definição |
|---|---|
| Estado | configuração das 8 peças e do espaço vazio |
| Estado inicial | configuração embaralhada e solucionável |
| Ações | mover o espaço vazio para cima, baixo, esquerda ou direita |
| Função sucessora | troca do vazio com uma peça adjacente |
| Estado objetivo | `(1, 2, 3, 4, 5, 6, 7, 8, 0)` |
| Custo | 1 por movimento |

O espaço vazio é representado internamente por `0`.

---

## Estado objetivo

```text
1 2 3
4 5 6
7 8 _
```

Representação:

```python
GOAL_STATE = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0,
)
```

---

## Geração de puzzles solucionáveis

O puzzle não é criado por uma permutação completamente aleatória.

A aplicação começa no estado objetivo e executa uma sequência de movimentos válidos:

```text
GOAL
  ↓
movimento válido
  ↓
movimento válido
  ↓
...
  ↓
estado inicial
```

Dessa forma, toda configuração produzida pela aplicação possui solução.

O parâmetro `moves` representa a intensidade do embaralhamento e não necessariamente a distância ótima até o objetivo.

---

## Breadth-First Search — BFS

O BFS explora o espaço de estados por níveis de profundidade.

Como cada movimento do NUMPUZ possui custo:

```text
1
```

temos:

```text
profundidade = custo
```

Consequentemente, a primeira solução encontrada pelo BFS é ótima em número de movimentos.

O BFS não utiliza conhecimento sobre a posição do objetivo.

---

## A*

O A* utiliza:

```text
f(n) = g(n) + h(n)
```

onde `g(n)` é o custo acumulado desde o estado inicial e `h(n)` é uma estimativa do custo restante até o objetivo.

A aplicação implementa duas heurísticas.

---

## Misplaced Tiles

Conta quantas peças não estão em sua posição objetivo.

O espaço vazio não é considerado.

Exemplo:

```text
1 2 3
4 5 6
_ 7 8
```

Duas peças estão fora da posição:

```text
h(n) = 2
```

---

## Manhattan Distance

Para cada peça calcula:

```text
|linha atual - linha objetivo|
+
|coluna atual - coluna objetivo|
```

e soma as distâncias.

O espaço vazio é ignorado.

A Manhattan Distance fornece mais informação sobre o problema do que Misplaced Tiles, pois considera não apenas se uma peça está fora da posição, mas também quanto ela está distante do objetivo.

---

## Experimento

Para:

```text
shuffle moves = 20
seed = 42
```

foi gerado:

```text
8 7 2
5 3 6
_ 1 4
```

Os três algoritmos encontraram uma solução ótima com:

```text
20 movimentos
```

Resultados observados:

| Estratégia | Custo | Expandidos | Gerados |
|---|---:|---:|---:|
| BFS | 20 | 53.388 | 70.546 |
| A* + Misplaced Tiles | 20 | 2.822 | 4.476 |
| A* + Manhattan Distance | 20 | 129 | 216 |

Os tempos de execução variam de acordo com a máquina e não devem ser considerados valores absolutos.

---

## Interpretação

Os três métodos encontraram uma solução com o mesmo custo ótimo.

Entretanto, o esforço de busca foi muito diferente.

Comparando BFS com A* + Manhattan Distance:

```text
BFS
53.388 estados expandidos

A* + Manhattan
129 estados expandidos
```

Isso corresponde, neste experimento, a aproximadamente:

```text
99,76% menos estados expandidos
```

A diferença demonstra o valor da heurística.

O BFS explora o espaço praticamente sem informação sobre o objetivo, enquanto o A* utiliza conhecimento do domínio para priorizar estados mais promissores.

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

Responsável pela modelagem do problema:

- representação dos estados;
- movimentos válidos;
- função sucessora;
- teste de objetivo;
- solvabilidade;
- geração de puzzles.

### `heuristics.py`

Implementa:

- Misplaced Tiles;
- Manhattan Distance.

### `solver.py`

Implementa:

- Breadth-First Search;
- A*;
- reconstrução do caminho;
- métricas da busca.

### `compare_search.py`

Executa os três métodos sobre exatamente o mesmo estado para permitir comparação controlada.

### `app.py`

Interface Streamlit para:

- jogo manual;
- solução automática;
- visualização passo a passo;
- comparação dos algoritmos.

---

## Executar os testes

A partir de:

```bash
cd apps/numpuz
```

execute:

```bash
python -m unittest discover -s tests -v
```

---

## Executar comparação via terminal

```bash
python compare_search.py
```

---

## Executar aplicação

A partir de:

```bash
cd apps/numpuz
```

execute:

```bash
streamlit run app.py
```

A aplicação será disponibilizada normalmente em:

http://localhost:8501

---

## Conceitos demonstrados

O projeto exemplifica:

- espaço de estados;
- estado inicial;
- estado objetivo;
- função sucessora;
- custo de caminho;
- busca não informada;
- busca informada;
- heurística;
- admissibilidade;
- optimalidade;
- controle de estados repetidos;
- comparação de eficiência entre estratégias.

---

## Conclusão

O NUMPUZ mostra de forma prática a principal diferença entre busca não informada e busca informada.

O BFS encontra uma solução ótima, porém pode explorar uma grande parte do espaço de estados.

O A* utiliza uma heurística para direcionar a exploração.

No experimento apresentado, a heurística Manhattan Distance permitiu encontrar a mesma solução ótima do BFS expandindo apenas uma pequena fração dos estados.

Assim, o projeto demonstra que uma boa função heurística pode produzir um ganho expressivo de eficiência sem sacrificar a qualidade da solução.