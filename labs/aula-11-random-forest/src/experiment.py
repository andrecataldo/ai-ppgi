"""Execução auditável e sem vazamento temporal do Lab 11."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from .data import (load_data,audit_data,make_features,descriptive_by_subject,
                   file_sha256,TARGET,CONDITIONS)
from .models import (participant_split,features_for,make_pipeline,scores,
                     participant_ci_for_delta,group_cv_mae)

MODELS=['Decision Tree','Random Forest','XGBoost']


def make_figures(f: pd.DataFrame, rows: pd.DataFrame, pred: pd.DataFrame, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    agg=descriptive_by_subject(f)
    fig,ax=plt.subplots(figsize=(9,4.6))
    x=np.arange(4); mean=[agg.loc[agg.condition==c,'bias_mean'].mean() for c in range(4)]
    se=[agg.loc[agg.condition==c,'bias_mean'].std(ddof=1)/np.sqrt(120) for c in range(4)]
    ax.bar(x,mean,yerr=1.96*np.asarray(se),capsize=5)
    ax.axhline(0,linewidth=1,color='0.4');ax.set_xticks(x,[CONDITIONS[i] for i in range(4)])
    ax.set_ylabel('Desvio assinado médio (pontos percentuais)')
    ax.set_title('Exp2 — desvio médio por condição (IC normal aproximado, participantes)')
    fig.tight_layout();fig.savefig(out/'01_vies_por_condicao.png',dpi=160);plt.close(fig)

    # Resumo do desempenho, sem sugerir causalidade.
    work=rows[rows['tipo']=='modelo'].copy()
    fig,ax=plt.subplots(figsize=(9,5))
    labels=[f"{r.modelo} / {r.cenario}" for r in work.itertuples()]
    ax.barh(np.arange(len(work)),work.MAE)
    ax.set_yticks(np.arange(len(work)),labels)
    ax.invert_yaxis();ax.set_xlabel('MAE no teste (menor é melhor)')
    ax.set_title('Previsão do desvio humano — 30 participantes não vistos')
    fig.tight_layout();fig.savefig(out/'02_mae_modelos.png',dpi=160);plt.close(fig)

    # Mudança de viés por bloco de apresentação com os dados efetivos.
    obs=(f.groupby(['subject','condition','trial_in_block'],as_index=False)
         .agg(error=(TARGET,'mean')))
    by=obs.groupby(['condition','trial_in_block']).error.mean().reset_index()
    fig,ax=plt.subplots(figsize=(9,4.8))
    for c in (0,1,2,3):
        z=by[by.condition==c]
        ax.plot(z.trial_in_block,z.error,label=CONDITIONS[c])
    ax.axhline(0,color='0.4',lw=.8);ax.set_xlabel('Posição dentro do bloco (1–30)')
    ax.set_ylabel('Desvio assinado médio')
    ax.set_title('Evolução das estimativas dentro de cada condição')
    ax.legend();fig.tight_layout();fig.savefig(out/'03_tendencia_intra_bloco.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,5))
    best=pred['Random_Forest__historico'].to_numpy()
    y=pred['target'].to_numpy()
    ax.scatter(y,best,s=10,alpha=.26)
    low=min(y.min(),best.min());high=max(y.max(),best.max())
    ax.plot([low,high],[low,high],linestyle='--',color='black',lw=1)
    ax.set_xlabel('Desvio observado');ax.set_ylabel('Desvio previsto')
    ax.set_title('Random Forest com histórico — conjunto de teste')
    fig.tight_layout();fig.savefig(out/'04_real_vs_previsto.png',dpi=160);plt.close(fig)


def markdown_report(audit:dict, rows:pd.DataFrame, changes:pd.DataFrame,
                    desc:pd.DataFrame, out:Path, quick:bool, cv:bool,
                    n_train:int,n_test:int)->None:
    d=rows.copy()
    # Sem exigência da dependência tabulate.
    def md(df):
        return df.to_markdown(index=False,floatfmt='.3f')
    report=f'''# Relatório de execução — LAB 11

> Resultados computados com o `Exp2.csv` original. Este relatório registra resultados preditivos **exploratórios**; não constitui reprodução inferencial do artigo nem demonstra causalidade.

## Integridade e contexto

- SHA-256 do CSV: `{audit['sha256']}`.
- Participantes: **{audit['participants']}**; tentativas: **{audit['rows']}**.
- Treino: **{n_train} participantes**; teste: **{n_test} participantes**, sem interseção.
- Modo rápido: **{quick}**; validação cruzada no treino: **{cv}**.
- A ordem original das linhas dentro de participante foi preservada.
- Alvo: `response - evidence` (desvio assinado em pontos percentuais).
- Instante de previsão: após apresentar o estímulo e **antes da estimativa humana**.
- `responseAI` contemporânea não entra em nenhum preditor. Apenas conselhos já revelados em tentativas anteriores entram em features históricas.
- `subject` somente agrupamento; medidas retrospectivas `perceivedInfluence*` não entram no modelo.

## Descrição por condição

{md(desc)}

Os valores são médias descritivas, não testes de hipótese. Cada participante contribui com 30 tentativas por condição.

## Comparação de desempenho no conjunto de teste

{md(d[['tipo','modelo','cenario','MAE','RMSE','R2','cv_mae_mean','cv_mae_sd','late_block_MAE']])}

- Métrica principal: MAE, na escala do desvio assinado.
- `late_block_MAE`: somente tentativas 16–30 dos blocos com IA, em participantes que não estavam no treino.
- Média e desvio dos folds de CV agrupados referem-se **apenas ao treinamento**.
- Os hiperparâmetros foram fixados antes de avaliar o teste (sem seleção pelo teste).

## Histórico versus contexto: incerteza por participante

{md(changes)}

O intervalo é bootstrap sobre os **participantes do teste**. Valores positivos de `delta_mae` favorecem o histórico. Não é um teste sobre mecanismo causal, não corrige decisões pós-hoc e não incorpora incerteza de re-treinamento.

## Interpretação e limites

1. Random Forest/XGBoost operam aqui sobre dados tabulares de comportamento humano, não sobre neuroimagem.
2. A variável-alvo representa **erro assinado**, e não diagnóstico individual de viés cognitivo.
3. `condition` e `order` são metadados experimentais conhecidos ao pesquisador; não se afirma que participantes conheciam a condição técnica.
4. Histórico dentro do participante está disponível no teste **somente de forma retrospectiva e causalmente ordenada**: eventos de instantes anteriores a t podem ser observados antes de prever t. Este é um cenário *online* com histórico, não uma predição cold-start estrita.
5. Os resultados medem associação e generalização preditiva para **novos participantes**, não o efeito causal de exposição à IA.
6. A unidade independente primária é a pessoa. 14.400 linhas não equivalem a 14.400 participantes.
7. A importância das variáveis isoladamente não revela mecanismos nem causalidade (Strobl et al., 2007).
8. A condição "IA precisa" difere de `evidence` em ±1 ou 0, segundo a codificação original. Não alterar o dado bruto.
9. Dados reais podem refletir características de interface, tarefa perceptual e amostragem do estudo; resultados não generalizam automaticamente para LLMs.

## Referências

- Glickman, M.; Sharot, T. *How human–AI feedback loops alter human perceptual, emotional and social judgements*. Nature Human Behaviour 9, 345–359 (2025), publicado online em 18 dez. 2024. https://doi.org/10.1038/s41562-024-02077-2
- Dados originais: https://github.com/affective-brain-lab/BiasedHumanAI/tree/main/Exp2/Data
- Strobl, C. et al. (2007). *Bias in random forest variable importance measures*. BMC Bioinformatics 8, 25.
'''
    (out/'RELATORIO_RESULTADOS.md').write_text(report,encoding='utf-8')


def execute(csv_path:Path,output_dir:Path,quick=False,cv=True,seed=42):
    output_dir.mkdir(parents=True,exist_ok=True)
    frame=load_data(csv_path)
    aud=audit_data(frame,strict=True);aud['sha256']=file_sha256(csv_path)
    feats=make_features(frame)
    train_idx,test_idx=participant_split(feats,seed=seed)
    train=feats.iloc[train_idx].copy(); test=feats.iloc[test_idx].copy()
    assert set(train.subject).isdisjoint(test.subject)
    split_info = {'seed':seed,'train_subject_ids':sorted(int(x) for x in train.subject.unique()),
                  'test_subject_ids':sorted(int(x) for x in test.subject.unique())}
    (output_dir/'SPLIT_PARTICIPANTES.json').write_text(
        json.dumps(split_info,ensure_ascii=False,indent=2),encoding='utf-8')
    (output_dir/'AUDITORIA.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=descriptive_by_subject(feats).groupby('condition').agg(
        bias_medio=('bias_mean','mean'),erro_absoluto_medio=('mae','mean'),
        desvio_padrao_entre_participantes=('bias_mean','std'),participantes=('subject','nunique'))
    summary.index=summary.index.map(lambda x:f'{x}: {CONDITIONS[x]}')
    summary.reset_index(names='condicao').to_csv(output_dir/'ESTATISTICA_DESCRITIVA.csv',index=False)
    desc=pd.read_csv(output_dir/'ESTATISTICA_DESCRITIVA.csv')

    rows=[]; predictions=pd.DataFrame({'subject':test.subject.values,
        'trial_index':test.trial_index.values,'condition':test.condition.values,
        'target':test[TARGET].values})
    late=(test['block_index']>=1)&(test['trial_in_block']>=16)
    # Baselines: zero (sem desvio), média global treinada, persistência local.
    for name,arr in [
        ('Zero',np.zeros(len(test))),
        ('Média do treino',np.full(len(test),float(train[TARGET].mean()))),
        ('Persistência anterior',test['lag1_error'].fillna(0).to_numpy()),
    ]:
        z=scores(test[TARGET],arr); preds=np.asarray(arr)
        rows.append({'tipo':'baseline','modelo':name,'cenario':'referencia',**z,
                     'cv_mae_mean':np.nan,'cv_mae_sd':np.nan,
                     'late_block_MAE':float(np.mean(np.abs(test.loc[late,TARGET]-preds[late.to_numpy()])))})
        predictions['Baseline__'+name.replace(' ','_')]=preds

    for name in MODELS:
        for scenario in ('contexto','historico'):
            cols=features_for(scenario)
            # Proteção explícita contra colunas da tentativa corrente.
            assert not set(cols).intersection({'subject','response','responseAI','confidence','collaboration',
                 'perceivedInfluenceA','perceivedInfluenceB','perceivedInfluenceN',TARGET,'absolute_error'})
            est=make_pipeline(name,cols,seed,quick)
            est.fit(train[cols],train[TARGET]); yp=est.predict(test[cols])
            prediction_key=name.replace(' ','_')+'__'+scenario
            predictions[prediction_key]=yp
            z=scores(test[TARGET],yp)
            cv_stats=group_cv_mae(name,scenario,train,seed,quick,folds=3 if quick else 4) if cv else {'cv_mae_mean':np.nan,'cv_mae_sd':np.nan}
            rows.append({'tipo':'modelo','modelo':name,'cenario':scenario,**z,**cv_stats,
                         'late_block_MAE':float(np.mean(np.abs(test.loc[late,TARGET]-yp[late.to_numpy()])))})
    tab=pd.DataFrame(rows)
    tab.to_csv(output_dir/'METRICAS_MODELOS.csv',index=False)
    predictions.to_csv(output_dir/'PREDICOES_TESTE.csv',index=False)
    per_condition=[]
    for row in rows:
        name=row['modelo'];scenario=row['cenario']
        if row['tipo']=='modelo':key=name.replace(' ','_')+'__'+scenario
        else:key='Baseline__'+name.replace(' ','_')
        for cond in sorted(test.condition.unique()):
            mask=predictions.condition==cond
            measures=scores(predictions.loc[mask,'target'],predictions.loc[mask,key])
            per_condition.append({'modelo':name,'cenario':scenario,'condition':int(cond),
                                  'n_trials':int(mask.sum()),**measures})
    pd.DataFrame(per_condition).to_csv(output_dir/'METRICAS_POR_CONDICAO.csv',index=False)
    improvement=[]
    for name in MODELS:
        k=name.replace(' ','_')
        val=participant_ci_for_delta(test,predictions[k+'__contexto'].values,
                                     predictions[k+'__historico'].values,seed=seed)
        improvement.append({'modelo':name,'delta_mae':val['delta_mae_positive_favors_history'],
                            'ci95_low':val['ci95_low'],'ci95_high':val['ci95_high'],
                            'participantes':val['participants']})
    changes=pd.DataFrame(improvement)
    changes.to_csv(output_dir/'INCERTEZA_HISTORICO.csv',index=False)
    make_figures(feats,tab,predictions,output_dir)
    markdown_report(aud,tab,changes,desc,output_dir,quick,cv,train.subject.nunique(),test.subject.nunique())
    return {'audit':aud,'metrics':tab,'history_delta':changes,'description':desc,
            'train_subjects':sorted(train.subject.unique().tolist()),
            'test_subjects':sorted(test.subject.unique().tolist())}


def cli():
    parser=argparse.ArgumentParser(description='LAB 11: Human–AI Judgment Bias, RF e XGBoost.')
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--data',type=Path,default=root/'data'/'Exp2.csv')
    parser.add_argument('--out',type=Path,default=root/'results')
    parser.add_argument('--quick',action='store_true',help='Menos árvores e 3 folds: útil para validação rápida.')
    parser.add_argument('--no-cv',action='store_true',help='Desabilita CV; somente demonstração, não recomendado para relatório final.')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    summary=execute(args.data,args.out,quick=args.quick,cv=not args.no_cv,seed=args.seed)
    print('\nLAB 11 | auditoria:',summary['audit']['rows'],'linhas /',summary['audit']['participants'],'participantes')
    print('Treino:',len(summary['train_subjects']),'participantes; teste:',len(summary['test_subjects']))
    print(summary['metrics'][['tipo','modelo','cenario','MAE','RMSE','R2','cv_mae_mean','late_block_MAE']].round(3).to_string(index=False))
    print('\nGanho MAE por histórico (positivo favorece histórico):')
    print(summary['history_delta'].round(3).to_string(index=False))
    print('\nArtefatos em:',args.out)
