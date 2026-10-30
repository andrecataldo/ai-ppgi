"""Dashboard Streamlit para explorar resultados previamente treinados com lab.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from common import RESULTS

st.set_page_config(page_title="GAN Mode Collapse Lab", page_icon="🧪", layout="wide")
st.title("🧪 GAN Mode Collapse Lab")
st.caption("IA — PPGI/UNIRIO · Aula 9 · GANs, cobertura de modos e WGAN-GP")
st.markdown("""**Pergunta-guia:** uma GAN que produz amostras plausíveis realmente aprendeu toda a distribuição?

Treine os modelos no terminal com `python lab.py --quick` ou `python lab.py --steps 1500 --seeds 0 1 2`.
A interface **não treina** modelos: analisa resultados salvos, mantendo a execução reproduzível.
""")

if not RESULTS.exists():
    st.warning("Nenhum resultado encontrado. Execute `python lab.py --quick` primeiro.")
    st.stop()

runs = []
for path in sorted(RESULTS.glob("*_seed*_steps*/summary.json")):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if (path.parent / "samples.npz").exists():
        runs.append((path.parent.name, path, obj))
if not runs:
    st.warning("Não há execuções completas. Execute `python lab.py --quick`.")
    st.stop()

st.sidebar.header("Seleção dos experimentos")
choices = [name for name, _, _ in runs]
selected = st.sidebar.multiselect("Execuções para comparar", choices, default=choices[-min(2,len(choices)):])
if not selected:
    st.info("Selecione ao menos uma execução.")
    st.stop()

st.subheader("1. Métricas do experimento")
st.markdown("""**Cobertura:** clusters com pelo menos 1% do total de pontos gerados, próximos ao centro real.
**Válidas:** fração de amostras até 0,45 do centro mais próximo.
**JS (bits):** divergência de Jensen–Shannon entre frequências dos **pontos válidos** e distribuição uniforme; menor é melhor.
**Atenção:** JS precisa ser interpretada **junto** da fração válida. Uma GAN com pouquíssimos pontos válidos pode enganar essa métrica.
""")
for name, path, obj in runs:
    if name not in selected:
        continue
    m, cfg = obj["metrics"], obj["config"]
    st.markdown(f"**{name}** — {cfg['n_critic']} atualização(ões) do discriminador/crítico por atualização do gerador")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cobertura", f"{m['coverage']}/8")
    c2.metric("Amostras válidas", f"{m['valid_fraction']:.1%}")
    c3.metric("JS (bits) ↓", f"{m['js_divergence_bits']:.3f}")
    c4.metric("Duração", f"{obj['duration_seconds']:.1f} s")
    if cfg["steps"] <= 250:
        st.caption("⚠️ Execução rápida de exploração: não é evidência de convergência.")

st.divider()
st.subheader("2. Geometria das distribuições")
cols = st.columns(min(len(selected), 2))
for i, chosen in enumerate(selected):
    if i and i % 2 == 0:
        cols = st.columns(2)
    folder = next(p.parent for name, p, _ in runs if name == chosen)
    with cols[i % 2]:
        full_extent = st.checkbox("Exibir todos os pontos (sem zoom)", key=f"extent-{chosen}")
        fig_name = "amostras_full.png" if full_extent and (folder / "amostras_full.png").exists() else "amostras.png"
        st.image(str(folder / fig_name), caption=f"Reais × geradas: {chosen}", use_container_width=True)
        with np.load(folder / "samples.npz", allow_pickle=False) as x:
            samples = x["fake"]
        if st.checkbox("Mostrar intervalo e resumo dos pontos gerados", key=f"rng-{chosen}"):
            st.write({"mínimo": samples.min(axis=0).round(2).tolist(),
                      "máximo": samples.max(axis=0).round(2).tolist(),
                      "média": samples.mean(axis=0).round(2).tolist()})

st.subheader("3. Ocupação de cada cluster")
fig, ax = plt.subplots(figsize=(10, 3.5))
width = 0.75 / len(selected)
for j, chosen in enumerate(selected):
    obj = next(ob for name, _, ob in runs if name == chosen)
    occup = obj["metrics"]["occupancy"]
    ax.bar(np.arange(8) + j * width - 0.375 + width / 2, occup, width=width, label=chosen)
ax.axhline(1/8, linestyle="--", color="grey", label="Uniforme (12,5%)")
ax.set(xticks=list(range(8)), xlabel="Cluster", ylabel="Fração dos pontos válidos", title="Ocupação condicional (ignora pontos inválidos)")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(bottom=0)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("4. Evolução durante o treinamento")
for chosen in selected:
    folder = next(p.parent for name, p, _ in runs if name == chosen)
    with st.expander(chosen, expanded=len(selected)==1):
        st.image(str(folder / "progresso.png"), use_container_width=True)
        st.caption("A escala da perda difere entre GAN clássica e WGAN-GP; não compare seus valores diretamente.")

st.divider()
st.subheader("5. Referências didáticas (não treinadas)")
ref = RESULTS / "referencias_didaticas"
if ref.exists():
    a, b = st.columns(2)
    with a:
        st.image(str(ref / "referencia_completa.png"), caption="Amostragem de 8 modos (referência artificial)", use_container_width=True)
    with b:
        st.image(str(ref / "referencia_colapsada.png"), caption="Amostragem de 2 modos (colapso artificial)", use_container_width=True)

st.info("""**Discussão:** uma melhoria de estabilidade não garante cobertura. Repita com sementes 0, 1 e 2 e
compare custo de treinamento, validade e equilíbrio de modos. Os modos são conhecidos neste laboratório, o
que não costuma ocorrer em aplicações reais. Este é um experimento pedagógico, não benchmark de SOTA.""")
