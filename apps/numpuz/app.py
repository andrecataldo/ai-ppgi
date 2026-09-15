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
    shuffle_state,
)
from solver import SearchResult, astar, bfs

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
    """Remove search results associated with the current board."""
    st.session_state.search_result = None
    st.session_state.solution_index = 0
    st.session_state.search_algorithm = None
    st.session_state.search_heuristic = None
    st.session_state.comparison_results = None


def initialize_session() -> None:
    """Initialize Streamlit state once."""
    if "initial_state" not in st.session_state:
        initial = shuffle_state(
            moves=20,
            seed=42,
        )

        st.session_state.initial_state = initial
        st.session_state.board = initial
        st.session_state.manual_moves = 0

    if "search_result" not in st.session_state:
        clear_solution()


initialize_session()


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
                    type="primary" if enabled else "secondary",
                )

                if clicked and move is not None:
                    execute_manual_move(move)
                    st.rerun()


def new_game(
    shuffle_moves: int,
    seed: int | None,
) -> None:
    """Generate and start a new solvable puzzle."""
    initial = shuffle_state(
        moves=shuffle_moves,
        seed=seed,
    )

    st.session_state.initial_state = initial
    st.session_state.board = initial
    st.session_state.manual_moves = 0

    clear_solution()


def reset_game() -> None:
    """Restore the original puzzle state."""
    st.session_state.board = (
        st.session_state.initial_state
    )

    st.session_state.manual_moves = 0

    clear_solution()


def solve_current_state(
    algorithm: str,
    heuristic_name: str,
) -> None:
    """Run the selected search algorithm."""
    state = st.session_state.board

    if algorithm == "BFS":
        result = bfs(state)

        st.session_state.search_algorithm = "BFS"
        st.session_state.search_heuristic = None

    else:
        heuristic = HEURISTICS[
            heuristic_name
        ]

        result = astar(
            state,
            heuristic=heuristic,
        )

        st.session_state.search_algorithm = "A*"
        st.session_state.search_heuristic = heuristic_name

    st.session_state.search_result = result
    st.session_state.solution_index = 0

def compare_search_strategies() -> None:
    """Run all search strategies over exactly the same current state."""
    state = st.session_state.board

    bfs_result = bfs(state)

    misplaced_result = astar(
        state,
        heuristic=misplaced_tiles,
    )

    manhattan_result = astar(
        state,
        heuristic=manhattan_distance,
    )

    st.session_state.comparison_results = {
        "BFS": bfs_result,
        "A* + Misplaced Tiles": misplaced_result,
        "A* + Manhattan Distance": manhattan_result,
    }


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

    costs = {
        result.cost
        for result in results.values()
    }

    if len(costs) == 1:
        st.success(
            "✅ As três estratégias encontraram "
            "uma solução com o mesmo custo."
        )
    else:
        st.warning(
            "⚠️ As estratégias retornaram soluções "
            "com custos diferentes."
        )

    best_name, best_result = min(
        results.items(),
        key=lambda item: item[1].expanded,
    )

    st.info(
        f"🏆 **Menor número de estados expandidos:** "
        f"{best_name} - {best_result.expanded:,} estados."
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
        help="Quantidade de movimentos da solução.",
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
    """Explain the search evaluation for the displayed state."""
    algorithm = st.session_state.search_algorithm

    with st.expander(
        "🔎 Como a busca avalia este estado?"
    ):
        if algorithm == "BFS":
            st.markdown(
                f"""
**Algoritmo:** Breadth-First Search

**Profundidade atual:**

```text
g(n) = {step}
```

Como cada movimento possui custo 1, a profundidade
é também o custo acumulado até este estado.

O BFS não utiliza heurística: ele explora o espaço
de estados em níveis crescentes de profundidade.
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
            GOAL_STATE,
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

- `g(n)` = custo já percorrido;
- `h(n)` = estimativa do custo restante;
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
            "🎯 Estado objetivo alcançado."
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
Resolva o puzzle manualmente ou utilize algoritmos de
**busca não informada (BFS)** e **busca informada (A\\*)**
para encontrar automaticamente uma solução ótima.
"""
)


# ================================================================
# NOVO JOGO
# ================================================================

with st.expander(
    "⚙️ Configuração do puzzle",
    expanded=False,
):
    shuffle_moves = st.slider(
        "Intensidade de embaralhamento",
        min_value=2,
        max_value=30,
        value=20,
        step=1,
        help=(
            "Quantidade de movimentos aleatórios "
            "executados a partir do estado objetivo. "
            "Não representa necessariamente a "
            "distância ótima até a solução."
        ),
    )

    reproducible = st.checkbox(
        "Usar seed reproduzível",
        value=True,
    )

    seed = None

    if reproducible:
        seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                value=42,
                step=1,
            )
        )

    col_new, col_reset = st.columns(2)

    with col_new:
        if st.button(
            "🎲 Novo puzzle",
            type="primary",
            use_container_width=True,
        ):
            new_game(
                shuffle_moves,
                seed,
            )

            st.rerun()

    with col_reset:
        if st.button(
            "↩️ Reiniciar",
            use_container_width=True,
        ):
            reset_game()
            st.rerun()


# ================================================================
# JOGO MANUAL
# ================================================================

st.divider()

st.subheader("🎮 Jogo manual")

st.caption(
    "Clique em uma peça adjacente ao espaço vazio."
)

render_board(
    st.session_state.board,
    interactive=not is_goal(
        st.session_state.board
    ),
    key_prefix="manual",
)

metric_col1, metric_col2 = st.columns(2)

metric_col1.metric(
    "Movimentos manuais",
    st.session_state.manual_moves,
)

metric_col2.metric(
    "Manhattan atual",
    manhattan_distance(
        st.session_state.board
    ),
)

if is_goal(
    st.session_state.board
):
    st.success(
        "🎉 Puzzle resolvido!"
    )


# ================================================================
# RESOLUÇÃO AUTOMÁTICA
# ================================================================

st.divider()

st.subheader("🤖 Resolver com busca")

algorithm = st.selectbox(
    "Algoritmo",
    options=[
        "A*",
        "BFS",
    ],
)

heuristic_name = st.selectbox(
    "Heurística",
    options=list(
        HEURISTICS.keys()
    ),
    disabled=algorithm == "BFS",
    help="BFS não utiliza função heurística.",
)

if algorithm == "BFS":
    st.caption(
        "BFS explora o espaço por profundidade. "
        "Como todos os movimentos custam 1, "
        "a solução encontrada é ótima."
    )

elif heuristic_name == "Manhattan Distance":
    st.caption(
        "Manhattan soma as distâncias horizontais "
        "e verticais de cada peça até seu objetivo."
    )

else:
    st.caption(
        "Misplaced Tiles conta quantas peças "
        "estão fora de sua posição objetivo."
    )

st.info(
    "A busca será executada a partir do estado atual do tabuleiro. "
    "Movimentos feitos manualmente antes da busca serão considerados."
)

if st.button(
    "🔍 Encontrar solução",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner(
            f"Executando {algorithm}..."
        ):
            solve_current_state(
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
        "### 🧭 Caminho da solução"
    )

    show_solution(result)

# ================================================================
# COMPARAÇÃO DOS ALGORITMOS
# ================================================================

st.divider()

st.subheader("⚖️ Comparar estratégias")

st.markdown(
    """
Execute **BFS**, **A\\* + Misplaced Tiles** e
**A\\* + Manhattan Distance** sobre exatamente o mesmo
estado atual do puzzle.

Assim podemos comparar a eficiência das estratégias sem
alterar o problema de busca.
"""
)

if st.button(
    "📊 Executar comparação",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Executando BFS e as duas configurações de A*..."
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
| **Estado** | configuração das 8 peças + espaço vazio |
| **Estado inicial** | puzzle embaralhado e solucionável |
| **Ações** | mover o vazio para cima, baixo, esquerda ou direita |
| **Função sucessora** | troca o vazio com uma peça adjacente |
| **Teste objetivo** | `(1,2,3,4,5,6,7,8,0)` |
| **Custo** | `1` por movimento |
| **BFS** | busca não informada |
| **A\\*** | busca informada: `f(n) = g(n) + h(n)` |
| **Heurísticas** | Misplaced Tiles e Manhattan Distance |
"""
    )

    st.markdown(
        """
O puzzle é sempre gerado por uma sequência de movimentos
válidos a partir do estado objetivo. Portanto, toda configuração
gerada pela aplicação possui solução.
"""
    )