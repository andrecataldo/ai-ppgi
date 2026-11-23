"""Interface enxuta para projetar o simulador na sala."""

import pandas as pd
import streamlit as st

from motor import caracteristicas, demonstracoes, prever

st.set_page_config(page_title="ICL: simulador didático", page_icon="🔬", layout="wide")
st.title("ICL: predição condicionada por exemplos")
st.caption("Simulador DIDÁTICO de similaridade e pesos normalizados. Não representa um LLM treinado.")
st.info("O que muda? Apenas exemplos e parâmetros explícitos da regra. "
        "O que permanece? Código, funções e ausência de atualização de pesos.")

with st.sidebar:
    st.header("1. Condições do experimento")
    k = st.select_slider("Número de demonstrações (K)", options=[0, 1, 2, 4], value=4)
    invertidos = st.checkbox("Inverter os rótulos apresentados")
    ordem = st.checkbox("Inverter a ordem das demonstrações")
    consulta = st.text_input("Código para classificar", value="TOV-42")
    st.header("2. Parâmetros visíveis")
    peso_prefixo = st.slider("Peso do prefixo", 0.0, 4.0, 2.0, 0.25)
    peso_sufixo = st.slider("Peso do sufixo", 0.0, 4.0, 0.5, 0.25)
    vies_posicao = st.slider("Viés pela posição (opcional)", 0.0, 2.0, 0.0, 0.25)

if not consulta.strip():
    st.error("Digite uma palavra de teste para continuar o experimento.")
    st.stop()

exemplos = demonstracoes(k, invertidos=invertidos, ordem_inversa=ordem)
resultado = prever(consulta, exemplos, peso_prefixo, peso_sufixo, vies_posicao)
esquerda, direita = st.columns(2)
with esquerda:
    st.subheader("Passo 1 - Entrada e características")
    st.write(f"Consulta: `{consulta}`")
    st.json(caracteristicas(consulta))
    st.subheader("Passo 2 - Demonstrações do contexto")
    if exemplos:
        st.dataframe(pd.DataFrame([{"Entrada": x.palavra, "Rótulo": x.rotulo} for x in exemplos]),
                     hide_index=True, use_container_width=True)
    else:
        st.warning("Zero-shot: sem exemplos, usamos uma divisão neutra de pesos (50% / 50%) por convenção.")

with direita:
    st.subheader("Passo 3 - Regra explícita")
    st.latex(r"s_i = w_p \cdot [p(q)=p(x_i)] + w_s \cdot [s(q)=s(x_i)] + b\,\frac{i}{\max(K-1,1)}")
    st.latex(r"a_i=\frac{e^{s_i}}{\sum_j e^{s_j}}\qquad M(c\mid q)=\sum_{i:y_i=c} a_i")
    st.markdown("O colchete vale **1** quando a característica coincide e **0** quando difere.")
    st.caption("M(c|q) é massa de peso por classe, não probabilidade calibrada de acerto.")

st.subheader("Passo 4 - Cálculo auditável por demonstração")
if resultado["passos"]:
    df = pd.DataFrame(resultado["passos"])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"atencao": st.column_config.ProgressColumn("Peso de atenção", min_value=0, max_value=1, format="%.3f")})
    st.bar_chart(df.set_index("exemplo")["atencao"], horizontal=True)
else:
    st.info("Não há linhas de cálculo porque K = 0.")

st.subheader("Passo 5 - Resultado")
st.metric("Classe prevista", resultado["predicao"])
st.write("**Pesos normalizados por classe:**", {c: f"{p:.1%}" for c, p in resultado["pesos_por_classe"].items()})
if k == 1:
    st.caption(
        "Com K=1, a única demonstração recebe todo o peso. "
        "Isso NÃO significa certeza de 100%."
    )
else:
    st.caption(
        "Os pesos normalizados não representam "
        "probabilidades calibradas de acerto."
    )

st.caption("Experimento didático independente das métricas de Brown (2020), Bender (2021) e Hoffmann (2022).")
