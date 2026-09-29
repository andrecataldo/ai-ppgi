# Filtro de Spam com Spambase (UCI)

**Disciplina:** Inteligência Artificial - Aula 5 - Aprendizado de Máquina Clássico  
**Aluno:** André Luiz Cataldo Falbo Santo

## Objetivo e método

O trabalho avaliou classificadores clássicos na base Spambase, com 4.601 e-mails, 57 atributos numéricos e 39,4% de spam. Foi realizado split estratificado de 80% para treino e 20% para teste. A comparação de modelos utilizou validação cruzada estratificada de 5 dobras apenas no conjunto de treino. O conjunto de teste foi mantido separado para avaliação final.

## Resultados

| Experimento | Resultado principal |
|---|---|
| Baseline | 60,60% de acurácia e 0% de recall para spam. Todos os 1.813 spams foram classificados como legítimos. |
| Modelos | Random Forest obteve o melhor resultado: acurácia 95,27% ± 0,61 p.p., F1 93,91% e ROC-AUC 98,61%. |
| k-NN | A padronização elevou a acurácia de 79,16% para 90,68% e o F1 de 73,37% para 87,90%. |
| Threshold | O limiar foi ajustado para 0,7267. No teste, o FPR ficou em 0,72%: 4 de 558 e-mails legítimos foram marcados como spam. O filtro bloqueou 82,92% dos spams e deixou passar 17,08%. |
| Remoção de features | Sem `george`, `650`, `hp` e `hpl`, a acurácia média caiu de 95,27% para 94,51% na validação cruzada e de 94,57% para 94,14% no teste. |

## Discussão e conclusão

A acurácia do baseline mostra que essa métrica isolada é insuficiente para avaliar o filtro. O Random Forest apresentou o melhor desempenho global entre os modelos testados. No k-NN, a diferença entre as escalas dos atributos prejudicava o cálculo de distância, e o `StandardScaler` produziu melhora expressiva.

O ajuste do threshold mostrou o custo assimétrico dos erros: reduzir falsos positivos protege e-mails legítimos, mas aumenta falsos negativos e permite a passagem de mais spam. A remoção das quatro features específicas provocou perda pequena e consistente. Como essas variáveis são muito mais frequentes em mensagens legítimas da origem da base, elas podem funcionar como atalhos estatísticos específicos do usuário. Assim, o bom desempenho na Spambase não garante desempenho equivalente em outra caixa postal.

**Limitações:** foram mantidas as linhas duplicadas para preservar os 4.601 registros do enunciado. A portabilidade para usuários diferentes não foi testada diretamente.
