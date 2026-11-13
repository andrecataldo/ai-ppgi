# Relatório de execução — LAB 11

> Resultados computados com o `Exp2.csv` original. Este relatório registra resultados preditivos **exploratórios**; não constitui reprodução inferencial do artigo nem demonstra causalidade.

## Integridade e contexto

- SHA-256 do CSV: `ec2207efc6fa936031107376857d48310990874d200de39c2880945ace574277`.
- Participantes: **120**; tentativas: **14400**.
- Treino: **90 participantes**; teste: **30 participantes**, sem interseção.
- Modo rápido: **False**; validação cruzada no treino: **True**.
- A ordem original das linhas dentro de participante foi preservada.
- Alvo: `response - evidence` (desvio assinado em pontos percentuais).
- Instante de previsão: após apresentar o estímulo e **antes da estimativa humana**.
- `responseAI` contemporânea não entra em nenhum preditor. Apenas conselhos já revelados em tentativas anteriores entram em features históricas.
- `subject` somente agrupamento; medidas retrospectivas `perceivedInfluence*` não entram no modelo.

## Descrição por condição

| condicao        |   bias_medio |   erro_absoluto_medio |   desvio_padrao_entre_participantes |   participantes |
|:----------------|-------------:|----------------------:|------------------------------------:|----------------:|
| 0: Sem IA       |        0.620 |                15.030 |                               6.277 |             120 |
| 1: IA precisa   |        1.268 |                13.483 |                               6.842 |             120 |
| 2: IA enviesada |        2.655 |                14.736 |                               7.972 |             120 |
| 3: IA ruidosa   |        1.155 |                14.358 |                               7.757 |             120 |

Os valores são médias descritivas, não testes de hipótese. Cada participante contribui com 30 tentativas por condição.

## Comparação de desempenho no conjunto de teste

| tipo     | modelo                | cenario    |    MAE |   RMSE |     R2 |   cv_mae_mean |   cv_mae_sd |   late_block_MAE |
|:---------|:----------------------|:-----------|-------:|-------:|-------:|--------------:|------------:|-----------------:|
| baseline | Zero                  | referencia | 14.176 | 19.224 | -0.002 |       nan     |     nan     |           13.171 |
| baseline | Média do treino       | referencia | 14.301 | 19.227 | -0.002 |       nan     |     nan     |           13.197 |
| baseline | Persistência anterior | referencia | 19.144 | 26.021 | -0.835 |       nan     |     nan     |           17.273 |
| modelo   | Decision Tree         | contexto   | 13.604 | 17.680 |  0.153 |        14.181 |       0.387 |           12.835 |
| modelo   | Decision Tree         | historico  | 13.026 | 17.268 |  0.192 |        13.845 |       0.431 |           12.240 |
| modelo   | Random Forest         | contexto   | 13.600 | 17.705 |  0.150 |        14.230 |       0.401 |           12.828 |
| modelo   | Random Forest         | historico  | 12.608 | 16.918 |  0.224 |        13.241 |       0.333 |           11.824 |
| modelo   | XGBoost               | contexto   | 13.535 | 17.618 |  0.159 |        14.116 |       0.401 |           12.819 |
| modelo   | XGBoost               | historico  | 12.674 | 16.870 |  0.229 |        13.342 |       0.165 |           11.842 |

- Métrica principal: MAE, na escala do desvio assinado.
- `late_block_MAE`: somente tentativas 16–30 dos blocos com IA, em participantes que não estavam no treino.
- Média e desvio dos folds de CV agrupados referem-se **apenas ao treinamento**.
- Os hiperparâmetros foram fixados antes de avaliar o teste (sem seleção pelo teste).

## Histórico versus contexto: incerteza por participante

| modelo        |   delta_mae |   ci95_low |   ci95_high |   participantes |
|:--------------|------------:|-----------:|------------:|----------------:|
| Decision Tree |       0.578 |      0.129 |       1.056 |              30 |
| Random Forest |       0.991 |      0.541 |       1.490 |              30 |
| XGBoost       |       0.861 |      0.323 |       1.433 |              30 |

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
