# Análise acadêmica — LAB 11

**Tema:** impacto preditivo do histórico de interações humano–IA em julgamentos perceptuais: comparação de Decision Tree, Random Forest e XGBoost.

## Resumo

Esta atividade compara modelos de regressão baseados em árvores para prever o desvio assinado entre uma estimativa humana e a proporção real de pontos em movimento em uma tarefa perceptual. Empregou-se o dataset público `Exp2.csv` de Glickman e Sharot, contendo 14.400 tentativas de 120 participantes, submetidos a condições sem IA ou com aconselhamento algorítmico preciso, enviesado ou ruidoso. Compararam-se modelos com atributos de contexto e modelos acrescidos de histórico retrospectivo de julgamentos, confiança e conselhos de IA. Uma divisão agrupada por participante reservou 90 indivíduos para treino e 30 para teste, evitando compartilhar tentativas de uma mesma pessoa entre os dois conjuntos. A avaliação utilizou MAE, RMSE, R² e validação cruzada agrupada. Os três modelos melhoraram o MAE ao incluir histórico, com maior redução para Random Forest (13,600 para 12,605). O resultado é **preditivo**, não identificando isoladamente mecanismo causal de influência algorítmica.

## 1. Questão e motivação

O estudo de Glickman e Sharot investiga como a interação com algoritmos modifica julgamentos humanos e pode gerar ciclos de realimentação de vieses. Enquanto o artigo estima efeitos comportamentais com desenho experimental e modelos estatísticos, este laboratório faz uma pergunta complementar: a sequência de eventos anteriores permite melhorar a **previsão** de novos julgamentos?

A comparação de algoritmos segue os temas da disciplina de IA sobre árvores, impureza, sobreajuste, bagging e boosting. A abordagem conecta, ainda, a advertência de Strobl et al. de que medidas de importância de variáveis não necessariamente permitem interpretação confiável dos mecanismos geradores do fenômeno.

## 2. Material e método

A unidade observacional é uma tentativa (`trial`) de uma pessoa (`subject`). Cada participante contribuiu com 30 tentativas sem IA e 30 em cada uma das três condições de algoritmo. A sequência experimental foi preservada conforme a ordem do arquivo original. Não se juntaram arquivos de experimentos complementares.

A variável-alvo é `signed_error = response - evidence`, em pontos percentuais. Valores positivos representam sobrestimação, e negativos subestimação. Um julgamento isolado pode ser errado sem representar um viés sistemático.

Foram comparados dois conjuntos de atributos para os mesmos modelos. O primeiro, `contexto`, inclui características da tentativa e metadados de experimento. O segundo, `historico`, adiciona médias retrospectivas e lags das respostas, confiança e aconselhamento **revelado em tentativas anteriores**. A criação desses atributos utiliza `shift(1)` por pessoa. Não se utiliza a resposta contemporânea da IA para prever uma estimativa que a precede temporalmente.

A divisão foi agrupada por pessoa, com semente 42 e 25% dos participantes reservados para teste. A validação cruzada com quatro folds ocorreu somente nos participantes do treino. Foram utilizados modelos de referência (erro zero, média de treino e persistência do último erro), Decision Tree, Random Forest e XGBoost, com hiperparâmetros fixos antes de avaliar o teste.

## 3. Resultados da execução validada

| Modelo | MAE contexto | MAE histórico | Redução do MAE | R² histórico |
|---|---:|---:|---:|---:|
| Decision Tree | 13,604 | 13,026 | 4,2% | 0,192 |
| Random Forest | 13,600 | **12,605** | **7,3%** | 0,224 |
| XGBoost | 13,535 | 12,674 | 6,4% | **0,229** |

O baseline que prevê desvio zero apresentou MAE de 14,176 no teste, reforçando a necessidade de comparar métodos aprendidos contra referências simples. Embora a persistência do erro anterior explore informação histórica, teve desempenho inferior (MAE de 19,144), mostrando que histórico não garante melhoria se incorporado de forma ingênua.

As diferenças de MAE (cenário sem histórico menos cenário com histórico), com intervalos bootstrap agrupados por participante, foram: **0,578** (IC95% [0,129; 1,056]) para Decision Tree; **0,994** ([0,545; 1,493]) para Random Forest; **0,861** ([0,323; 1,433]) para XGBoost. Esses intervalos são condicionados aos modelos treinados e não estabelecem efeito causal.

### Resultados comportamentais descritivos

| Condição | Desvio médio (`response-evidence`) | MAE humano |
|---|---:|---:|
| Sem IA | +0,620 | 15,030 |
| IA precisa | +1,268 | 13,483 |
| IA enviesada | +2,655 | 14,736 |
| IA ruidosa | +1,155 | 14,358 |

A condição enviesada apresenta maior deslocamento positivo médio; a condição precisa apresenta menor erro absoluto médio. A direção dessas diferenças é compatível com as análises dos autores, mas os números aqui são descrições do CSV. Não foi executada uma reprodução completa dos testes inferenciais originais.

## 4. Discussão crítica

Os resultados indicam que o histórico contém informação preditiva adicional, em relação ao conjunto de atributos de contexto proposto. A maior melhoria ocorreu com Random Forest. O XGBoost apresentou R² ligeiramente superior ao de Random Forest no cenário histórico, ilustrando por que diferentes métricas podem produzir ordenações diferentes entre modelos.

A melhora observada **não separa** a influência de conselhos da IA da autocorrelação das respostas da própria pessoa. O histórico agrega ambas as informações. Uma etapa seguinte deve comparar: apenas histórico humano, apenas histórico de conselhos já revelados e histórico completo. A interpretação da variável `condition` também exige cuidado: ela é uma informação do pesquisador e sinaliza a condição experimental, mas não é uma explicação causal individual.

O melhor R² observado foi de cerca de 0,23, indicando que grande parte da variabilidade nos desvios permanece fora do alcance dos preditores e da especificação atual. A avaliação em participantes novos não garante generalização a outros experimentos ou a interações com LLMs. Todos os resultados dependem de escolhas de pré-processamento, divisão e hiperparâmetros.

Em termos epistemológicos, o laboratório diferencia **capacidade preditiva** de **inferência causal**: mesmo que atributos do histórico melhorem previsões, a ocorrência e a direção de um mecanismo de influência algorítmica devem ser estabelecidas pelo desenho experimental e por métodos estatísticos apropriados. A análise do laboratório é complementar, não substitutiva, ao estudo original.

## 5. Conclusão

Na execução com semente 42 e divisão agrupada por participante, os três algoritmos apresentaram menor MAE quando informações de julgamentos e interações anteriores foram adicionadas ao contexto experimental. Random Forest produziu o menor MAE no teste (12,605). Esses resultados apoiam a viabilidade de modelagem preditiva do comportamento observado, com restrições claras sobre disponibilidade temporal de atributos e generalização. Eles não permitem afirmar que Random Forest ou XGBoost detectaram a causa do viés cognitivo nem que seriam superiores em outros datasets.

## 6. Limitações e extensões

1. Comparar **histórico humano**, **histórico de conselho de IA** e **histórico completo** para investigar melhor qual conjunto gera o ganho preditivo.
2. Executar múltiplas divisões por participante, registrando distribuição dos resultados, sem selecionar apenas a divisão com melhor desempenho.
3. Experimentar regressão de `absolute_error` e modelos estatísticos hierárquicos, mantendo perguntas distintas.
4. Avaliar desempenho e calibração por condição, posição no bloco e pessoa.
5. Replicar análises comportamentais originais antes de comparar conclusões causais.
6. Reservar a primeira parte de cada nova interação como período de *warm-up* para avaliar cenários online mais realistas.

## Referências

- Glickman, M.; Sharot, T. (2025). *How human–AI feedback loops alter human perceptual, emotional and social judgements*. Nature Human Behaviour, 9, 345–359. https://doi.org/10.1038/s41562-024-02077-2
- Strobl, C.; Boulesteix, A.-L.; Zeileis, A.; Hothorn, T. (2007). *Bias in random forest variable importance measures: Illustrations, sources and a solution*. BMC Bioinformatics, 8, 25.
- Chen, T.; Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD.
