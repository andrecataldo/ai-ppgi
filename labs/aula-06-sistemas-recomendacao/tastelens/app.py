# ============================================================
# TASTELENS
# Interface Streamlit + painel didático
# ============================================================

"""
TasteLens

Demonstração didática de um sistema de recomendação baseado
em filtragem colaborativa entre usuários, inspirado no princípio
central do GroupLens.
"""

import io
import socket
import sqlite3
import time

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import qrcode
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from catalog import (
    ADMIN_PASSWORD,
    APP_NAME,
    APP_SUBTITLE,
    ITENS_ANCORA,
    ITENS_CANDIDATOS,
    LIMIAR_ARESTA_REDE,
    MIN_ANCORAS_RESPONDIDAS,
    MIN_EXTRAS_RESPONDIDOS,
    PORT,
)
from recommender import matriz_similaridade, recomendar_para
from storage import (
    carregar_dados,
    get_or_create_participante,
    limpar_tudo,
    recommendations_released,
    salvar_avaliacoes,
    set_recommendations_released,
)

# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🍽️",
    layout="wide",
)


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================


def obter_ip_local():
    """Descobre o endereço IP local utilizado pelo computador."""

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        sock.close()

    return ip


def criar_qr(url):
    """Gera um QR Code em memória."""

    imagem = qrcode.make(url)

    buffer = io.BytesIO()
    imagem.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def usuario_concluiu(
    matriz: pd.DataFrame,
    participante: dict,
) -> bool:
    """Verifica se o participante atingiu a quantidade mínima de avaliações."""

    pseudonimo = participante["pseudonym"]

    if matriz.empty:
        return False

    if pseudonimo not in matriz.index:
        return False

    linha = matriz.loc[pseudonimo]

    ancoras_respondidas = sum(
        1 for item in ITENS_ANCORA if item in matriz.columns and pd.notna(linha[item])
    )

    extras_respondidos = sum(
        1
        for item in participante["assigned_items"]
        if item in matriz.columns and pd.notna(linha[item])
    )

    return (
        ancoras_respondidas >= MIN_ANCORAS_RESPONDIDAS
        and extras_respondidos >= MIN_EXTRAS_RESPONDIDOS
    )


# ============================================================
# 3. VISUALIZAÇÕES DIDÁTICAS
# ============================================================


def cor_similaridade(valor):
    """Define uma cor de fundo para os valores da matriz de Pearson."""

    if pd.isna(valor):
        return ""

    if valor >= 0.70:
        return "background-color: #b7e4c7"

    if valor >= 0.30:
        return "background-color: #d8f3dc"

    if valor <= -0.70:
        return "background-color: #f5b7b1"

    if valor <= -0.30:
        return "background-color: #fadbd8"

    return "background-color: #f8f9fa"


def grafico_similaridade(sim: pd.DataFrame, usuario: str):
    """Mostra a correlação do usuário selecionado com os demais."""

    serie = sim.loc[usuario].drop(index=usuario).dropna().sort_values()

    cores = ["#c0392b" if valor < 0 else "#2e8b57" for valor in serie.values]

    fig = go.Figure(
        go.Bar(
            x=serie.values,
            y=serie.index,
            orientation="h",
            marker_color=cores,
            text=[f"{valor:+.2f}" for valor in serie.values],
            textposition="outside",
        )
    )

    fig.add_vline(x=0, line_width=1)

    fig.update_layout(
        title=f"Similaridade com {usuario}",
        xaxis_title="Correlação de Pearson",
        yaxis_title="",
        xaxis_range=[-1.05, 1.05],
        height=max(360, 45 * max(1, len(serie))),
    )

    return fig


def grafico_rede(sim: pd.DataFrame):
    """Cria uma rede de similaridade entre participantes."""

    rede = nx.Graph()

    for usuario in sim.index:
        rede.add_node(usuario)

    for i, usuario_1 in enumerate(sim.index):
        for usuario_2 in sim.index[i + 1 :]:
            valor = sim.loc[usuario_1, usuario_2]

            if pd.isna(valor):
                continue

            if abs(valor) < LIMIAR_ARESTA_REDE:
                continue

            rede.add_edge(
                usuario_1,
                usuario_2,
                similarity=float(valor),
                weight=abs(float(valor)),
            )

    posicoes = nx.spring_layout(
        rede,
        seed=8,
        k=1.2,
        iterations=100,
    )
    

    linhas = []

    for usuario_1, usuario_2, dados in rede.edges(data=True):
        x0, y0 = posicoes[usuario_1]
        x1, y1 = posicoes[usuario_2]
        valor = dados["similarity"]

        linhas.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line={
                    "width": 1 + 4 * abs(valor),
                    "color": "#2e8b57" if valor >= 0 else "#c0392b",
                    "dash": "solid" if valor >= 0 else "dash",
                },
                hoverinfo="text",
                text=[f"{usuario_1} ↔ {usuario_2}: {valor:+.2f}"] * 3,
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    nomes = []

    for usuario in rede.nodes():
        x, y = posicoes[usuario]
        node_x.append(x)
        node_y.append(y)
        nomes.append(usuario)

    nos = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=nomes,
        textposition="top center",
        marker={
            "size": 24,
            "color": "#3498db",
            "line": {"width": 1, "color": "#1f618d"},
        },
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(linhas + [nos])

    fig.update_layout(
        title="Rede de similaridade dos participantes",
        height=560,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )

    return fig


# ============================================================
# 4. ESTADO DA SESSÃO
# ============================================================

if "participant" not in st.session_state:
    st.session_state["participant"] = None


# ============================================================
# 5. CABEÇALHO
# ============================================================

st.title(f"🍽️ {APP_NAME}")
st.caption(APP_SUBTITLE)
st.info(
    "O TasteLens compara padrões de avaliação de diferentes "
    "participantes para estimar quais pratos cada pessoa "
    "provavelmente gostaria de experimentar."
)


# ============================================================
# 6. NAVEGAÇÃO
# ============================================================

modo = st.sidebar.radio(
    "Modo",
    [
        "👤 Participante",
        "📊 Painel da apresentação",
    ],
)


# ============================================================
# 7. MODO PARTICIPANTE
# ============================================================

if modo == "👤 Participante":
    if st.session_state["participant"] is None:
        st.subheader("Entre com um pseudônimo")
        st.write("Não é necessário informar seu nome verdadeiro.")

        pseudonimo = st.text_input(
            "Pseudônimo",
            max_chars=24,
            placeholder="Ex.: Gourmet42",
        ).strip()

        if st.button(
            "Entrar",
            type="primary",
            use_container_width=True,
        ):
            if len(pseudonimo) < 2:
                st.warning("Use um pseudônimo com pelo menos 2 caracteres.")
            else:
                try:
                    participante = get_or_create_participante(pseudonimo)
                    st.session_state["participant"] = participante
                    st.rerun()
                except sqlite3.Error as erro:
                    st.error("Não foi possível registrar o participante.")
                    st.code(str(erro))

        st.stop()

    participante = st.session_state["participant"]
    st.success(f"Você entrou como **{participante['pseudonym']}**")

    _, matriz = carregar_dados()
    concluiu = usuario_concluiu(matriz, participante)

    if not concluiu:
        st.subheader("Avalie os pratos")
        st.write("Use de **1 a 5 estrelas** conforme sua preferência.")
        st.caption("Se você não conhece um prato, pode deixá-lo sem avaliação.")

        st.markdown("### 🍽️ Itens comuns")
        st.caption("Todos os participantes recebem estes itens.")

        respostas_ancora = {}

        for item in ITENS_ANCORA:
            st.markdown(f"**{item}**")
            valor = st.feedback(
                "stars",
                key=f"anchor_{participante['id']}_{item}",
            )

            if valor is not None:
                respostas_ancora[item] = valor + 1

        st.markdown("---")
        st.markdown("### 🥘 Itens extras")
        st.caption("Estes itens variam entre os participantes.")

        respostas_extras = {}

        for item in participante["assigned_items"]:
            st.markdown(f"**{item}**")
            valor = st.feedback(
                "stars",
                key=f"extra_{participante['id']}_{item}",
            )

            if valor is not None:
                respostas_extras[item] = valor + 1

        quantidade_ancoras = len(respostas_ancora)
        quantidade_extras = len(respostas_extras)

        pronto = (
            quantidade_ancoras >= MIN_ANCORAS_RESPONDIDAS
            and quantidade_extras >= MIN_EXTRAS_RESPONDIDOS
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        c1.metric(
            "Itens comuns avaliados",
            f"{quantidade_ancoras}/{len(ITENS_ANCORA)}",
        )
        c2.metric(
            "Itens extras avaliados",
            f"{quantidade_extras}/{len(participante['assigned_items'])}",
        )

        if not pronto:
            st.info(
                "Avalie pelo menos "
                f"{MIN_ANCORAS_RESPONDIDAS} itens comuns e "
                f"{MIN_EXTRAS_RESPONDIDOS} itens extras."
            )

        if st.button(
            "Enviar avaliações",
            type="primary",
            use_container_width=True,
            disabled=not pronto,
        ):
            respostas = {
                **respostas_ancora,
                **respostas_extras,
            }

            salvar_avaliacoes(
                participante["id"],
                respostas,
            )

            st.success("✅ Avaliações registradas!")
            time.sleep(0.5)
            st.rerun()

    else:
        st.success("✅ Suas avaliações já foram registradas.")

        if not recommendations_released():
            st_autorefresh(
                interval=4000,
                key="espera_participante",
            )

            st.markdown("### ⏳ Aguardando as recomendações...")
            st.write(
                "Quando o apresentador liberar os resultados, "
                "sua recomendação aparecerá automaticamente."
            )

        else:
            _, matriz = carregar_dados()
            recomendacao = recomendar_para(
                matriz,
                participante["pseudonym"],
                ITENS_CANDIDATOS,
            )

            st.markdown("---")
            st.subheader("🎯 Sua recomendação")

            if recomendacao is None:
                st.warning(
                    "Ainda não existem dados colaborativos suficientes "
                    "para produzir uma recomendação."
                )
            else:
                item, previsao, influencias = recomendacao

                st.markdown(f"# 🍽️ {item}")
                st.metric("Nota prevista", f"{previsao:.2f} / 5")

                quantidade_influencias = len(influencias)
                st.caption(
                    "Esta previsão utilizou "
                    f"**{quantidade_influencias}** "
                    + (
                        "participante."
                        if quantidade_influencias == 1
                        else "participantes."
                    )
                )

                if influencias:
                    st.markdown("### Quem influenciou esta previsão?")

                    for nome, similaridade, nota in influencias[:5]:
                        relacao = (
                            "padrão semelhante" if similaridade > 0 else "padrão oposto"
                        )

                        st.write(
                            f"**{nome}** — {relacao}; "
                            f"similaridade `{similaridade:+.2f}`; "
                            f"avaliação `{int(nota)}/5`"
                        )


# ============================================================
# 8. PAINEL DA APRESENTAÇÃO
# ============================================================

else:
    senha = st.sidebar.text_input(
        "Senha do painel",
        type="password",
    )

    if senha != ADMIN_PASSWORD:
        st.info("Digite a senha do painel na barra lateral.")
        st.stop()

    st_autorefresh(
        interval=3000,
        key="painel_refresh",
    )

    # QR CODE
    ip = obter_ip_local()
    url = f"http://{ip}:{PORT}"

    st.subheader("📱 Entrada dos participantes")
    coluna_qr, coluna_info = st.columns([1, 2])

    with coluna_qr:
        st.image(criar_qr(url), width=250)

    with coluna_info:
        st.write("Os participantes devem estar conectados à **mesma rede Wi-Fi**.")
        st.write("Endereço da aplicação:")
        st.code(url)

    st.markdown("---")

    # EXPLICAÇÃO DIDÁTICA
    with st.expander("ℹ️ Como o TasteLens calcula as recomendações?"):
        st.markdown(
            """
            O TasteLens utiliza **filtragem colaborativa baseada em usuários**.

            **1. Avaliações**  
            Cada participante avalia alguns pratos de 1 a 5.

            **2. Matriz usuário × item**  
            As avaliações são organizadas em uma matriz. Células vazias
            representam preferências desconhecidas.

            **3. Similaridade de Pearson**  
            O sistema compara usuários que avaliaram itens em comum.

            - próximo de **+1** → padrões semelhantes;
            - próximo de **0** → pouca relação;
            - próximo de **−1** → padrões opostos.

            **4. Previsão**  
            Avaliações de outros participantes são ponderadas pela similaridade.

            **5. Recomendação**  
            Entre os itens ainda não avaliados, o sistema escolhe aquele com
            maior nota prevista.
            """
        )

    participantes, matriz = carregar_dados()
    liberado = recommendations_released()

    c1, c2, c3 = st.columns(3)
    c1.metric("Participantes", len(participantes))
    c2.metric("Usuários com avaliações", 0 if matriz.empty else len(matriz.index))
    c3.metric("Recomendações", "LIBERADAS" if liberado else "AGUARDANDO")

    # CONTROLES
    coluna_a, coluna_b = st.columns(2)

    with coluna_a:
        if st.button(
            "🚀 Calcular e liberar recomendações",
            type="primary",
            use_container_width=True,
        ):
            set_recommendations_released(True)
            st.success("Recomendações liberadas!")
            time.sleep(0.3)
            st.rerun()

    with coluna_b:
        if st.button(
            "⏸️ Ocultar recomendações",
            use_container_width=True,
        ):
            set_recommendations_released(False)
            st.rerun()

    if matriz.empty:
        st.warning("Aguardando as primeiras avaliações.")
        st.stop()

    # MATRIZ USUÁRIO × ITEM
    st.markdown("---")
    st.subheader("Matriz usuário × item")
    st.caption("Células vazias representam preferências ainda não observadas.")

    matriz_estilizada = matriz.style.format(
        "{:.0f}",
        na_rep="—",
    )

    st.dataframe(
        matriz_estilizada,
        use_container_width=True,
    )

    # MATRIZ DE SIMILARIDADE + VISUALIZAÇÕES
    if len(matriz.index) >= 2:
        sim = matriz_similaridade(matriz)

        st.subheader("Similaridade entre usuários")
        st.caption(
            "A correlação de Pearson varia de -1 a +1. "
            "Valores positivos indicam padrões semelhantes; "
            "valores negativos indicam padrões opostos."
        )

        sim_estilizada = sim.style.format(
            "{:.2f}",
            na_rep="—",
        ).map(cor_similaridade)

        st.dataframe(
            sim_estilizada,
            use_container_width=True,
        )

        st.markdown("### 🔎 Análise individual")
        usuario_foco = st.selectbox(
            "Escolha um participante",
            matriz.index.tolist(),
        )

        serie_usuario = sim.loc[usuario_foco].drop(index=usuario_foco).dropna()

        if not serie_usuario.empty:
            mais_semelhante = serie_usuario.idxmax()
            mais_oposto = serie_usuario.idxmin()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Mais semelhante",
                mais_semelhante,
                f"{serie_usuario[mais_semelhante]:+.2f}",
            )
            c2.metric(
                "Padrão mais oposto",
                mais_oposto,
                f"{serie_usuario[mais_oposto]:+.2f}",
            )
            c3.metric(
                "Usuários comparáveis",
                len(serie_usuario),
            )

            st.plotly_chart(
                grafico_similaridade(
                    sim,
                    usuario_foco,
                ),
                use_container_width=True,
            )

        st.markdown("### 🕸️ Rede de similaridade")
        st.caption(
            "Cada nó representa um participante. "
            "Linhas contínuas indicam correlação positiva; "
            "linhas tracejadas indicam correlação negativa. "
            "Quanto mais grossa a ligação, maior a intensidade da relação."
        )

        st.plotly_chart(
            grafico_rede(sim),
            use_container_width=True,
        )

    # MÉDIA DAS AVALIAÇÕES
    st.subheader("Média das avaliações")

    medias = matriz.mean().sort_values(ascending=False)

    fig_medias = go.Figure(
        go.Bar(
            x=medias.index,
            y=medias.values,
            text=[f"{valor:.2f}" for valor in medias.values],
            textposition="outside",
        )
    )

    fig_medias.update_layout(
        yaxis_title="Média",
        xaxis_title="",
        yaxis_range=[0, 5.5],
        height=420,
    )

    st.plotly_chart(
        fig_medias,
        use_container_width=True,
    )

    # PRÉVIA DAS RECOMENDAÇÕES
    st.subheader("Prévia das recomendações")
    st.caption(
        "Suporte indica quantos participantes contribuíram para a previsão. "
        "Uma nota alta com suporte pequeno deve ser interpretada com cautela."
    )

    resultados = []

    for usuario in matriz.index:
        recomendacao = recomendar_para(
            matriz,
            usuario,
            ITENS_CANDIDATOS,
        )

        if recomendacao is None:
            resultados.append(
                {
                    "Participante": usuario,
                    "Recomendação": "Sem dados suficientes",
                    "Nota prevista": None,
                    "Suporte": "—",
                }
            )
        else:
            item, previsao, influencias = recomendacao
            quantidade_influencias = len(influencias)

            resultados.append(
                {
                    "Participante": usuario,
                    "Recomendação": item,
                    "Nota prevista": round(previsao, 2),
                    "Suporte": (
                        f"{quantidade_influencias} "
                        + ("usuário" if quantidade_influencias == 1 else "usuários")
                    ),
                }
            )

    st.dataframe(
        pd.DataFrame(resultados),
        use_container_width=True,
        hide_index=True,
    )

    # RESET
    with st.expander("⚠️ Reiniciar demonstração"):
        confirmar = st.checkbox(
            "Confirmo que quero apagar todos os participantes e avaliações."
        )

        if st.button(
            "Apagar tudo",
            disabled=not confirmar,
        ):
            limpar_tudo()
            st.success("Demonstração reiniciada.")
            time.sleep(0.4)
            st.rerun()
