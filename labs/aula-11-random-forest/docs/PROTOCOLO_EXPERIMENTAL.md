# Protocolo experimental — LAB 11 (versão 1.0)

**Status:** pré-especificação operacional implementada e executada.

## Questão científica

Em condições de tarefas perceptuais com aconselhamento algorítmico, atributos estritamente retrospectivos de interação permitem prever **melhor** os erros assinados dos julgamentos humanos independentes do que os atributos de contexto experimental isolados?

A pergunta é **preditiva**, não causal. Não há um diagnóstico formal de viés individual.

## Base e unidade de análise

- Fonte: `Exp2.csv` (Glickman & Sharot).
- Observação: uma tentativa perceptual de uma pessoa.
- Independência: observações correlacionadas dentro da pessoa; validação e intervalo por sujeito.
- 120 indivíduos, 120 tentativas/indivíduo; 30 baseline, 30 por IA precisa/ enviesada/ ruidosa.
- As informações do CSV foram coletadas em condições controladas, não em uma aplicação LLM.
- Ordem de linhas dentro de cada participante representa as tentativas em sequência. Esta premissa é suportada pela ordenação dos blocos no código original dos autores.

## Linha temporal

```
TRIAL t: [estímulo, condition/order] → [response(t)] → [confidence(t)] →
         [collaboration(t)] → [revelação responseAI(t)]
TRIAL t+1: pode usar resposta, confiança e conselho do trial t
```

Predição: **antes de response(t)**. `evidence(t)` descreve o estímulo conhecido ao experimentador (parâmetro físico real do estímulo), não o conhecimento do participante.

## Alvo

`y_t = response_t - evidence_t` (pontos percentuais de desvio assinado).

- Grandezas positivas: sobrestimativa da proporção para a direita.
- Grandezas negativas: subestimativa.
- Como o valor do desvio pode flutuar de tentativa a tentativa, um desvio não caracteriza automaticamente *cognitive bias*; é o **padrão sistemático** que sustenta a interpretação.

## Cenários (ablação de histórico)

**Contexto**: `evidence_t, condition_t, order, trial_index_t, trial_in_block_t`.

**Contexto + histórico**: adicionar `lag1_error`, `rolling5_prior_error`, `prior_mean_error`, `rolling5_prior_absolute_error`, `lag1_confidence`, `rolling5_prior_confidence`, `lag1_shown_ai_bias`, `rolling5_prior_shown_ai_bias`, `prior_mean_shown_ai_bias`, `rolling5_prior_collaboration`.

Todas as features históricas são construídas **por sujeito** com `shift(1)`, seguido (quando aplicável) de `rolling` ou `expanding`. Colunas em `t` ou `t+1` não entram. Os valores `responseAI` no baseline são ignorados como aconselhamento (pois não houve revelação ao participante).

**Observação interpretativa:** o cenário histórico agrega comportamento humano **e** exposição anterior à IA. Ele verifica o valor do **histórico completo**, mas sozinho não separa essas duas contribuições. Uma extensão pré-registrada pode dividir histórico humano e histórico da IA em ablações adicionais.

## Modelos e referências

- Baseline zero, média do treino, persistência do erro anterior (sem ajuste).
- Decision Tree Regressor: max_depth=6, min_samples_leaf=25.
- RandomForestRegressor: n_estimators=180, max_depth=12, min_samples_leaf=6, max_features='sqrt'.
- XGBRegressor: n_estimators=180, max_depth=4, learning_rate=.055, subsample=.85, colsample_bytree=.9, reg_lambda=2.
- Modelo e pré-processamento em `Pipeline`; imputation aprendida somente no treino; one-hot de metadados de condição e ordem.
- Semente 42, hiperparâmetros fixos; não há ajuste pelo conjunto de teste.

## Divisão e avaliação

- 75% participantes para treino, 25% para teste com `GroupShuffleSplit` (90/30).
- `GroupKFold(4)` no treino, para comparar cenários e verificar estabilidade entre grupos.
- Preditores históricos do sujeito testado podem usar **somente seus eventos anteriores**, nunca a resposta atual ou futura. Esse é um cenário online com histórico, não cold start estrito.
- Avaliação secundária: apenas tentativas 16–30 dos blocos de interação, sempre nos participantes de teste.
- MAE principal; RMSE e R² complementares; erros assinados médios para interpretação.
- Comparação pareada `MAE(contexto) − MAE(histórico)` por pessoa, intervalo bootstrap de 95% reamostrando **30 pessoas do teste** (`n_boot=2000`). Interpretar como intervalo exploratório condicionado ao modelo treinado.

## Hipóteses exploratórias

- **H1:** histórico completo reduz MAE em novos participantes, quando seus eventos anteriores estão disponíveis.
- **H2:** ganho preditivo varia entre condições de interação.

Essas hipóteses não são equivalentes às hipóteses experimentais causais do artigo. Não realizar teste de causalidade usando apenas ablação de preditores.

## Ameaças à validade

1. **Dependência intrassujeitos**: mitigada pela divisão em pessoas e bootstrap agrupado.
2. **Estímulos compartilhados** entre participantes: possibilidade de transferência do mesmo item; avaliação por item/família de estímulo seria outra pergunta.
3. **Confusão temporal** entre condição, bloco e aprendizagem acumulada; medição descritiva de predição não resolve causalidade.
4. **Rótulo derivado da resposta humana**: certo para regressão de erro, mas cuidado para não usar a resposta atual como feature.
5. **Acesso ao histórico de pessoas de teste**: requisito explícito de um modelo de previsão online; performance cold-start com 0 histórico exige outro protocolo.
6. **Configuração de hiperparâmetros**: fixed defaults, não uma disputa de SOTA nem busca exaustiva.
7. **Importância de atributos**: Strobl et al. (2007) explicam que ranqueamentos podem ser enviesados; este lab prioriza avaliação por ablação e não causaliza rankings.
8. **Generalização externa**: tarefas perceptuais de RDK não são tarefas em linguagem natural com LLMs.
9. **Incerteza**: bootstrap de previsões de teste não engloba variabilidade por re-treino em outras amostras.
10. **Variante do dataset**: `Exp2-Accurate.csv` e `Exp2-Biased.csv` representam experimentos de acompanhamento; não integrar aos 120 participantes do arquivo consolidado.

## Condições para aceitar as conclusões

- Fonte corresponde ao hash esperado e à estrutura original.
- 12 testes automatizados passam.
- Três modelos rodam sob os mesmos splits e métricas.
- Desempenho com histórico é comparado ao contexto sem histórico para **cada** modelo.
- Relatório explicita se a evidência favorece ou não H1 e traz ressalvas.

## Reprodutibilidade

`python run_lab.py`, seguido de `python -m pytest -v`. Todos os artefatos da execução são emitidos em `results/`, inclusive o conjunto de IDs dos participantes de cada divisão. A execução fornecida foi validada em Python 3.13, pandas 2.2.3, numpy 2.3.5, scikit-learn 1.8.0, XGBoost 3.1.3, matplotlib 3.10.8 e pytest 9.0.2. A faixa de versões em requirements permite outras instalações compatíveis; isso pode gerar pequenas diferenças numéricas.
