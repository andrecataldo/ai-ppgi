"""NUMPUZ - interactive search laboratory.

Curso de Inteligência Artificial - PPGI/UNIRIO.

Execute:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from heuristics import (
    HEURISTICS,
    manhattan_distance,
    misplaced_tiles,
)
from puzzle import (
    BLANK,
    GOAL_STATE,
    SIZE,
    Move,
    State,
    apply_move,
    is_goal,
    is_solvable,
    parse_state,
    shuffle_state,
)
from solver import (
    SearchResult,
    astar,
    bfs,
    dfs,
)


# ================================================================
# CONFIGURAÇÃO
# ================================================================

st.set_page_config(
    page_title="NUMPUZ - Busca em IA",
    page_icon="🧩",
    layout="centered",
)


MOVE_LABELS: dict[Move, str] = {
    "UP": "↑ Cima",
    "DOWN": "↓ Baixo",
    "LEFT": "← Esquerda",
    "RIGHT": "→ Direita",
}


# ================================================================
# ESTADO DA INTERFACE
# ================================================================

def clear_solution() -> None:
    """Remove search results associated with the current problem."""
    st.session_state.search_result = None
    st.session_state.solution_index = 0
    st.session_state.search_algorithm = None
    st.session_state.search_heuristic = None
    st.session_state.comparison_results = None


def initialize_session() -> None:
    """Initialize Streamlit state once."""
    if "goal_state" not in st.session_state:
        st.session_state.goal_state = GOAL_STATE

    if "initial_state" not in st.session_state:
        initial = shuffle_state(
            moves=20,
            seed=42,
            start=st.session_state.goal_state,
        )

        st.session_state.initial_state = initial
        st.session_state.board = initial
        st.session_state.manual_moves = 0

    if "search_result" not in st.session_state:
        clear_solution()


initialize_session()


# ================================================================
# DEFINIÇÃO DO PROBLEMA
# ================================================================

def set_problem(
    initial: State,
    goal: State,
) -> None:
    """Set a new I -> S search problem."""
    if not is_solvable(initial, goal):
        raise ValueError(
            "O estado inicial I não pode alcançar "
            "o estado objetivo S informado."
        )

    st.session_state.initial_state = initial
    st.session_state.goal_state = goal
    st.session_state.board = initial
    st.session_state.manual_moves = 0

    clear_solution()


def reset_game() -> None:
    """Restore the configured initial state I."""
    st.session_state.board = st.session_state.initial_state
    st.session_state.manual_moves = 0

    clear_solution()


# ================================================================
# FUNÇÕES DE INTERFACE
# ================================================================

def tile_move(
    state: State,
    tile_index: int,
) -> Move | None:
    """Return the blank movement required to move a clicked tile."""
    blank_index = state.index(BLANK)

    blank_row, blank_col = divmod(
        blank_index,
        SIZE,
    )

    tile_row, tile_col = divmod(
        tile_index,
        SIZE,
    )

    distance = (
        abs(blank_row - tile_row)
        + abs(blank_col - tile_col)
    )

    if distance != 1:
        return None

    if tile_row < blank_row:
        return "UP"

    if tile_row > blank_row:
        return "DOWN"

    if tile_col < blank_col:
        return "LEFT"

    return "RIGHT"


def execute_manual_move(move: Move) -> None:
    """Apply one manual movement to the current board."""
    st.session_state.board = apply_move(
        st.session_state.board,
        move,
    )

    st.session_state.manual_moves += 1

    clear_solution()


def render_board(
    state: State,
    *,
    interactive: bool,
    key_prefix: str,
) -> None:
    """Render a 3x3 puzzle board."""
    for row in range(SIZE):
        columns = st.columns(SIZE)

        for col in range(SIZE):
            index = row * SIZE + col
            tile = state[index]

            move = (
                tile_move(state, index)
                if tile != BLANK
                else None
            )

            enabled = (
                interactive
                and tile != BLANK
                and move is not None
            )

            label = (
                "⬛"
                if tile == BLANK
                else str(tile)
            )

            with columns[col]:
                clicked = st.button(
                    label,
                    key=f"{key_prefix}_{index}",
                    use_container_width=True,
                    disabled=not enabled,
                    type=(
                        "primary"
                        if enabled
                        else "secondary"
                    ),
                )

                if clicked and move is not None:
                    execute_manual_move(move)
                    st.rerun()


# ================================================================
# BUSCAS
# ================================================================

def solve_problem(
    algorithm: str,
    heuristic_name: str,
) -> None:
    """Run a search algorithm from configured I to configured S."""
    initial = st.session_state.initial_state
    goal = st.session_state.goal_state

    if algorithm == "BFS":
        result = bfs(
            initial,
            goal,
        )

        st.session_state.search_algorithm = "BFS"
        st.session_state.search_heuristic = None

    elif algorithm == "DFS":
        result = dfs(
            initial,
            goal,
        )

        st.session_state.search_algorithm = "DFS"
        st.session_state.search_heuristic = None

    else:
        heuristic = HEURISTICS[
            heuristic_name
        ]

        result = astar(
            initial,
            goal,
            heuristic=heuristic,
        )

        st.session_state.search_algorithm = "A*"
        st.session_state.search_heuristic = heuristic_name

    st.session_state.search_result = result
    st.session_state.solution_index = 0


def compare_search_strategies() -> None:
    """Run all strategies over exactly the same I -> S problem."""
    initial = st.session_state.initial_state
    goal = st.session_state.goal_state

    bfs_result = bfs(
        initial,
        goal,
    )

    dfs_result = dfs(
        initial,
        goal,
    )

    misplaced_result = astar(
        initial,
        goal,
        heuristic=misplaced_tiles,
    )

    manhattan_result = astar(
        initial,
        goal,
        heuristic=manhattan_distance,
    )

    st.session_state.comparison_results = {
        "BFS": bfs_result,
        "DFS": dfs_result,
        "A* + Misplaced Tiles": misplaced_result,
        "A* + Manhattan Distance": manhattan_result,
    }


# ================================================================
# APRESENTAÇÃO DOS RESULTADOS
# ================================================================

def show_comparison() -> None:
    """Display the comparison between search strategies."""
    results = st.session_state.comparison_results

    if not results:
        return

    rows = []

    for strategy, result in results.items():
        rows.append(
            {
                "Estratégia": strategy,
                "Custo": result.cost,
                "Expandidos": result.expanded,
                "Gerados": result.generated,
                "Tempo (ms)": round(
                    result.elapsed_seconds * 1000,
                    3,
                ),
            }
        )

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
    )

    bfs_cost = results["BFS"].cost

    misplaced_cost = results[
        "A* + Misplaced Tiles"
    ].cost

    manhattan_cost = results[
        "A* + Manhattan Distance"
    ].cost

    if (
        bfs_cost
        == misplaced_cost
        == manhattan_cost
    ):
        st.success(
            "✅ BFS e as duas configurações de A* "
            "encontraram uma solução com o mesmo custo ótimo."
        )

    dfs_cost = results["DFS"].cost

    if dfs_cost == bfs_cost:
        st.info(
            "ℹ️ Neste caso, o DFS também encontrou "
            "uma solução com o mesmo custo."
        )
    else:
        st.info(
            "ℹ️ O DFS encontrou uma solução com custo "
            f"{dfs_cost}, enquanto o custo ótimo observado "
            f"foi {bfs_cost}. Isso é esperado: DFS não "
            "garante a solução de menor profundidade."
        )

    fewest_name, fewest_result = min(
        results.items(),
        key=lambda item: item[1].expanded,
    )

    st.info(
        "📌 **Menor número de estados expandidos neste "
        f"experimento:** {fewest_name} - "
        f"{fewest_result.expanded:,} estados."
    )

    bfs_result = results["BFS"]
    manhattan_result = results[
        "A* + Manhattan Distance"
    ]

    if bfs_result.expanded > 0:
        reduction = (
            1
            - (
                manhattan_result.expanded
                / bfs_result.expanded
            )
        ) * 100

        st.metric(
            "Redução de expansões - Manhattan vs. BFS",
            f"{reduction:.2f}%",
        )


def show_search_metrics(
    result: SearchResult,
) -> None:
    """Display search performance metrics."""
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Custo",
        result.cost,
        help="Quantidade de movimentos da solução encontrada.",
    )

    col2.metric(
        "Expandidos",
        result.expanded,
        help=(
            "Estados cujos sucessores foram "
            "efetivamente examinados."
        ),
    )

    col3.metric(
        "Gerados",
        result.generated,
        help="Estados distintos descobertos.",
    )

    col4.metric(
        "Tempo",
        f"{result.elapsed_seconds * 1000:.2f} ms",
    )


def show_search_state_analysis(
    state: State,
    step: int,
) -> None:
    """Explain how the selected strategy evaluates the state."""
    algorithm = st.session_state.search_algorithm
    goal = st.session_state.goal_state

    with st.expander(
        "🔎 Como a busca avalia este estado?"
    ):
        if algorithm == "BFS":
            st.markdown(
                f"""
**Algoritmo:** Breadth-First Search (BFS)

```text
g(n) = {step}
```

O BFS explora o espaço de estados em níveis crescentes
de profundidade.

Como cada movimento possui custo 1:

```text
profundidade = custo acumulado
```

O BFS não utiliza heurística.
"""
            )

            return

        if algorithm == "DFS":
            st.markdown(
                f"""
**Algoritmo:** Depth-First Search (DFS)

**Profundidade do estado no caminho encontrado:**

```text
g(n) = {step}
```

O DFS explora um ramo em profundidade antes de retornar
para explorar alternativas.

Ele não utiliza heurística e não garante encontrar a
solução com o menor número de movimentos.
"""
            )

            return

        heuristic_name = (
            st.session_state.search_heuristic
        )

        heuristic = HEURISTICS[
            heuristic_name
        ]

        g = step

        h = heuristic(
            state,
            goal,
        )

        f = g + h

        st.markdown(
            f"""
**Algoritmo:** A*

**Heurística:** {heuristic_name}

```text
g(n) = {g}
h(n) = {h}
f(n) = g(n) + h(n)
     = {g} + {h}
     = {f}
```

- `g(n)` = custo já percorrido desde `I`;
- `h(n)` = estimativa do custo restante até `S`;
- `f(n)` = prioridade usada pelo A*.
"""
        )


def show_solution(
    result: SearchResult,
) -> None:
    """Show the search solution one state at a time."""
    index = st.session_state.solution_index
    state = result.path[index]

    st.markdown(
        f"### Passo {index} de {result.cost}"
    )

    render_board(
        state,
        interactive=False,
        key_prefix=f"solution_{index}",
    )

    progress = (
        1.0
        if result.cost == 0
        else index / result.cost
    )

    st.progress(
        progress,
        text=f"{index}/{result.cost} movimentos",
    )

    previous_col, next_col = st.columns(2)

    with previous_col:
        if st.button(
            "⬅️ Passo anterior",
            use_container_width=True,
            disabled=index == 0,
        ):
            st.session_state.solution_index -= 1
            st.rerun()

    with next_col:
        if st.button(
            "Próximo passo ➡️",
            use_container_width=True,
            disabled=index == result.cost,
        ):
            st.session_state.solution_index += 1
            st.rerun()

    if index < len(result.moves):
        next_move = result.moves[index]

        st.caption(
            "Próximo movimento do espaço vazio: "
            f"**{MOVE_LABELS[next_move]}**"
        )
    else:
        st.success(
            "🎯 Estado objetivo S alcançado."
        )

    show_search_state_analysis(
        state,
        index,
    )


# ================================================================
# CABEÇALHO
# ================================================================

st.title("🧩 NUMPUZ - Busca e Resolução")

st.caption(
    "Laboratório de busca aplicado ao 8-puzzle. "
    "Introdução à Inteligência Artificial - PPGI/UNIRIO."
)

st.markdown(
    """
Defina um **estado inicial I** e um **estado objetivo S**.

O mesmo problema `I → S` pode ser resolvido utilizando:

- **BFS**;
- **DFS**;
- **A\\* + Misplaced Tiles**;
- **A\\* + Manhattan Distance**.
"""
)


# ================================================================
# DEFINIÇÃO DO PROBLEMA I -> S
# ================================================================

with st.expander(
    "⚙️ Definição do problema de busca — I → S",
    expanded=True,
):
    st.markdown("### 🎯 Estado objetivo — S")

    goal_mode = st.radio(
        "Como definir S?",
        options=[
            "Solução canônica",
            "Informar manualmente",
        ],
        horizontal=True,
        key="goal_mode",
    )

    custom_goal_text = None

    if goal_mode == "Informar manualmente":
        custom_goal_text = st.text_input(
            "Configuração de S",
            value="123/456/78_",
            help=(
                "Use os números de 1 a 8 e '_' para o vazio. "
                "Exemplo: 123/456/78_"
            ),
            key="custom_goal_text",
        )

    else:
        st.code(
            "1 2 3\n4 5 6\n7 8 _",
            language="text",
        )

    st.markdown("### 🚩 Estado inicial — I")

    initial_mode = st.radio(
        "Como definir I?",
        options=[
            "Gerar aleatoriamente",
            "Informar manualmente",
        ],
        horizontal=True,
        key="initial_mode",
    )

    shuffle_moves = 20
    seed: int | None = 42
    custom_initial_text = None

    if initial_mode == "Gerar aleatoriamente":
        shuffle_moves = st.slider(
            "Intensidade de embaralhamento",
            min_value=1,
            max_value=30,
            value=20,
            step=1,
            help=(
                "Número de movimentos aleatórios executados "
                "a partir de S para produzir I. Não representa "
                "necessariamente a distância ótima entre I e S."
            ),
            key="shuffle_moves",
        )

        reproducible = st.checkbox(
            "Usar seed reproduzível",
            value=True,
            key="reproducible",
        )

        if reproducible:
            seed = int(
                st.number_input(
                    "Seed",
                    min_value=0,
                    value=42,
                    step=1,
                    key="seed",
                )
            )

        else:
            seed = None

    else:
        custom_initial_text = st.text_input(
            "Configuração de I",
            value="123/456/7_8",
            help=(
                "Use os números de 1 a 8 e '_' para o vazio. "
                "Exemplo: 123/456/7_8"
            ),
            key="custom_initial_text",
        )

    if st.button(
        "✅ Aplicar problema I → S",
        type="primary",
        use_container_width=True,
        key="apply_problem",
    ):
        try:
            if goal_mode == "Solução canônica":
                goal = GOAL_STATE

            else:
                goal = parse_state(
                    custom_goal_text
                )

            if initial_mode == "Gerar aleatoriamente":
                initial = shuffle_state(
                    moves=shuffle_moves,
                    seed=seed,
                    start=goal,
                )

            else:
                initial = parse_state(
                    custom_initial_text
                )

            set_problem(
                initial,
                goal,
            )

            st.rerun()

        except ValueError as exc:
            st.error(str(exc))

# ================================================================
# PROBLEMA ATUAL
# ================================================================

st.divider()

st.subheader("🧭 Problema atual")

initial_col, goal_col = st.columns(2)

with initial_col:
    st.markdown("#### Estado inicial — I")

    render_board(
        st.session_state.initial_state,
        interactive=False,
        key_prefix="problem_initial",
    )

with goal_col:
    st.markdown("#### Estado objetivo — S")

    render_board(
        st.session_state.goal_state,
        interactive=False,
        key_prefix="problem_goal",
    )

if is_solvable(
    st.session_state.initial_state,
    st.session_state.goal_state,
):
    st.success(
        "✅ I e S pertencem ao mesmo espaço alcançável."
    )


# ================================================================
# JOGO MANUAL
# ================================================================

st.divider()

st.subheader("🎮 Exploração manual")

st.caption(
    "Clique em uma peça adjacente ao espaço vazio. "
    "O objetivo é transformar I em S."
)

render_board(
    st.session_state.board,
    interactive=not is_goal(
        st.session_state.board,
        st.session_state.goal_state,
    ),
    key_prefix="manual",
)

metric_col1, metric_col2 = st.columns(2)

metric_col1.metric(
    "Movimentos manuais",
    st.session_state.manual_moves,
)

metric_col2.metric(
    "Manhattan até S",
    manhattan_distance(
        st.session_state.board,
        st.session_state.goal_state,
    ),
)

if is_goal(
    st.session_state.board,
    st.session_state.goal_state,
):
    st.success(
        "🎉 Estado objetivo S alcançado!"
    )

if st.button(
    "↩️ Reiniciar exploração em I",
    use_container_width=True,
):
    reset_game()
    st.rerun()


# ================================================================
# RESOLUÇÃO AUTOMÁTICA
# ================================================================

st.divider()

st.subheader("🤖 Resolver I → S com busca")

algorithm = st.selectbox(
    "Algoritmo",
    options=[
        "BFS",
        "DFS",
        "A*",
    ],
)

heuristic_name = st.selectbox(
    "Heurística do A*",
    options=list(
        HEURISTICS.keys()
    ),
    disabled=algorithm != "A*",
    help=(
        "BFS e DFS são buscas não informadas e "
        "não utilizam função heurística."
    ),
)

if algorithm == "BFS":
    st.caption(
        "BFS explora o espaço por níveis. Como todos "
        "os movimentos custam 1, encontra uma solução "
        "ótima em número de movimentos."
    )

elif algorithm == "DFS":
    st.caption(
        "DFS explora um ramo em profundidade antes de "
        "retroceder. Não garante a solução de menor custo."
    )

elif heuristic_name == "Manhattan Distance":
    st.caption(
        "A* + Manhattan soma as distâncias horizontais "
        "e verticais de cada peça até sua posição em S."
    )

else:
    st.caption(
        "A* + Misplaced Tiles conta quantas peças "
        "estão fora de sua posição em S."
    )

st.info(
    "Todas as buscas automáticas partem do mesmo estado "
    "inicial I configurado acima e procuram o mesmo "
    "estado objetivo S."
)

if st.button(
    "🔍 Encontrar solução",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner(
            f"Executando {algorithm} de I até S..."
        ):
            solve_problem(
                algorithm,
                heuristic_name,
            )

    except ValueError as exc:
        st.error(str(exc))


# ================================================================
# RESULTADO
# ================================================================

result = st.session_state.search_result

if result is not None:
    st.divider()

    st.subheader("📊 Resultado da busca")

    st.markdown(
        f"**{result.algorithm}**"
    )

    show_search_metrics(result)

    st.markdown(
        "### 🧭 Caminho I → S"
    )

    show_solution(result)


# ================================================================
# COMPARAÇÃO DAS ESTRATÉGIAS
# ================================================================

st.divider()

st.subheader("⚖️ Comparar estratégias")

st.markdown(
    """
Execute as quatro estratégias sobre **exatamente o mesmo
estado inicial I e o mesmo estado objetivo S**:

- BFS;
- DFS;
- A\\* + Misplaced Tiles;
- A\\* + Manhattan Distance.

Assim é possível observar como diferentes estratégias
visitam o espaço de estados para resolver o mesmo problema.
"""
)

if st.button(
    "📊 Executar comparação",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Executando BFS, DFS e as duas "
            "configurações de A*..."
        ):
            compare_search_strategies()

    except ValueError as exc:
        st.error(str(exc))


show_comparison()


# ================================================================
# FUNDAMENTAÇÃO
# ================================================================

st.divider()

with st.expander(
    "📚 Formulação do problema de busca"
):
    st.markdown(
        """
| Elemento | NUMPUZ |
|---|---|
| **Estado** | configuração das 8 peças e do espaço vazio |
| **Estado inicial I** | configuração informada pelo usuário ou gerada aleatoriamente |
| **Estado objetivo S** | solução canônica ou configuração válida informada pelo usuário |
| **Ações** | mover o vazio para cima, baixo, esquerda ou direita |
| **Função sucessora** | troca o vazio com uma peça adjacente |
| **Teste objetivo** | `estado_atual == S` |
| **Custo** | `1` por movimento |
| **BFS** | busca não informada em largura |
| **DFS** | busca não informada em profundidade |
| **A\\*** | busca informada: `f(n) = g(n) + h(n)` |
| **Heurísticas** | Misplaced Tiles e Manhattan Distance calculadas em relação a S |
"""
    )

    st.markdown(
        """
Quando o estado inicial é gerado aleatoriamente, a aplicação
parte do próprio estado objetivo `S` e executa uma sequência
de movimentos válidos:

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

Dessa forma, o estado `I` gerado é garantidamente alcançável
a partir de `S` e vice-versa.

Durante cada algoritmo de busca, os estados são explorados até
que seja satisfeita a condição:

```python
estado_atual == S
```

O estado objetivo não precisa ser necessariamente a configuração
canônica do 8-puzzle.
"""
    )