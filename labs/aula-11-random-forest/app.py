"""Painel de exploração dos resultados do LAB 11 (opcional).

Instale com: pip install -r requirements-ui.txt; streamlit run app.py.
Não treina nem modifica modelos automaticamente: leia results/ da execução CLI.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.data import load_data, make_features, CONDITIONS

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'data' / 'Exp2.csv'
RESULTS = ROOT / 'results'
st.set_page_config(page_title='LAB 11 | Human-AI Judgment Bias',layout='wide')
st.title('LAB 11 - Human-AI Judgment Bias')
st.caption('UNIRIO / PPGI · Decision Tree · Random Forest · XGBoost')
st.info('Os modelos preveem desvios de estimativas, **não** demonstram efeitos causais nem classificam pessoas como cognitivamente enviesadas.')

if not CSV.exists():
    st.error('Arquivo data/Exp2.csv não encontrado. Copie o CSV original e execute novamente.')
    st.stop()

@st.cache_data
def source():
    return make_features(load_data(CSV))

frame=source()
num_people=frame.subject.nunique()
c1,c2,c3=st.columns(3)
c1.metric('Registros',f'{len(frame):,}'.replace(',','.'))
c2.metric('Participantes',num_people)
c3.metric('Condições',frame.condition.nunique())

st.subheader('Distribuição de desvios humanos')
c=st.selectbox('Condição de análise',[0,1,2,3],format_func=lambda x:CONDITIONS[x])
x=frame.loc[frame.condition==c,['subject','trial_in_block','signed_error','absolute_error']].copy()
lo,hi=st.columns(2)
lo.metric('Desvio assinado médio',f'{x.signed_error.mean():+.2f} p.p.')
hi.metric('Erro absoluto médio',f'{x.absolute_error.mean():.2f} p.p.')
plot=x.groupby('trial_in_block',as_index=False)[['signed_error','absolute_error']].mean().set_index('trial_in_block')
st.line_chart(plot, y=['signed_error','absolute_error'])
st.caption('Cada posição agrega as tentativas dessa condição nos 120 participantes. Não são observações independentes.')

st.subheader('Modelos executados no conjunto de teste')
metrics=RESULTS/'METRICAS_MODELOS.csv'
if metrics.exists():
    m=pd.read_csv(metrics)
    st.dataframe(m[['tipo','modelo','cenario','MAE','RMSE','R2','cv_mae_mean','late_block_MAE']].round(3),hide_index=True,use_container_width=True)
    panel=m[m.tipo=='modelo'].copy()
    panel['modelo/cenario']=panel.modelo+' / '+panel.cenario
    st.bar_chart(panel.set_index('modelo/cenario')['MAE'])
else:
    st.warning('Resultados ainda não gerados. Execute no terminal: python run_lab.py')

imp=RESULTS/'INCERTEZA_HISTORICO.csv'
if imp.exists():
    st.subheader('Melhoria com histórico (bootstrap por participante)')
    st.dataframe(pd.read_csv(imp).round(3),hide_index=True,use_container_width=True)
    st.caption('ΔMAE = MAE do modelo sem histórico − MAE do modelo com histórico. Positivo favorece histórico.')

preds=RESULTS/'PREDICOES_TESTE.csv'
if preds.exists():
    st.subheader('Explorar previsões de um participante não visto')
    p=pd.read_csv(preds)
    pid=st.selectbox('Participante do conjunto de teste',sorted(p.subject.unique()),index=0)
    mdl=st.selectbox('Modelo', ['Random_Forest__historico','XGBoost__historico',
                               'Decision_Tree__historico','Random_Forest__contexto'])
    z=p[p.subject==pid][['trial_index','target',mdl]].set_index('trial_index')
    st.line_chart(z)
    st.caption('As estimativas de ML foram calculadas a partir do treino em outros participantes; o histórico de cada participante testado é estritamente anterior à tentativa prevista.')

st.divider()
st.markdown('**Metodologia:** comparação de regressão em dois cenários (contexto × histórico); divisão em pessoas, validação cruzada em pessoas, métricas no teste e incerteza agrupada. Leia `docs/PROTOCOLO_EXPERIMENTAL.md`.')
