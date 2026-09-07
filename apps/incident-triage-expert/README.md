# Incident Triage Expert

Sistema especialista didático para diagnóstico inicial de incidentes em aplicações Web.

Projeto desenvolvido para a disciplina de Inteligência Artificial do PPGI/UNIRIO - 2026.2.

## Objetivo

Demonstrar conceitos clássicos de sistemas especialistas por meio de um domínio próximo de operações de software e engenharia de sistemas.

O sistema recebe sinais observáveis de uma aplicação e utiliza regras de produção para:

1. identificar diagnósticos possíveis;
2. gerar fatos intermediários;
3. recomendar ações de investigação ou mitigação;
4. explicar quais regras participaram da inferência.

O objetivo não é substituir ferramentas reais de observabilidade ou processos de Incident Management.

## Arquitetura conceitual

```text
Sinais observados
       ↓
Memória de trabalho
       ↓
Motor de inferência
       ↓
Regras R1-R9
       ↓
Diagnósticos
       ↓
Regras R10-R18
       ↓
Ações recomendadas
       ↓
Módulo de explicação
```

O sistema utiliza encadeamento progressivo (*forward chaining*).

## Tecnologias

- Python
- Experta
- Streamlit

## Executar

A partir da raiz do projeto:

```bash
source .venv/bin/activate

streamlit run \
  apps/incident-triage-expert/incident_triage_app.py
```

A aplicação ficará disponível normalmente em:

```text
http://localhost:8501
```

## Fatos utilizados

A memória de trabalho é composta por cinco grupos principais de fatos observáveis.

### Aplicação

```text
taxa_erros
http_5xx
latencia
reinicios
```

### Infraestrutura

```text
cpu
memoria
disco
```

### Banco de dados

```text
latencia
conexoes
```

### Dependência externa

```text
status
```

### Contexto

```text
deploy_recente
trafego
```

Além dos fatos de entrada, o próprio sistema produz fatos intermediários:

```text
Diagnostico
Acao
```

Esses fatos permitem demonstrar encadeamento entre regras.

## Base de conhecimento

O sistema possui 18 regras de produção.

### Regras de diagnóstico

| Regra | Evidências | Diagnóstico |
|---|---|---|
| R1 | Deploy recente + taxa de erros alta | Regressão após deploy |
| R2 | Dependência degradada ou fora | Falha/degradação de dependência externa |
| R3 | Latência da aplicação alta + latência do banco alta | Gargalo de banco |
| R4 | Conexões do banco esgotadas + HTTP 5xx alto | Pool de conexões esgotado |
| R5 | CPU alta + pico de tráfego | Saturação de capacidade |
| R6 | CPU alta + tráfego normal + deploy recente | Regressão de consumo de CPU |
| R7 | Memória alta + reinícios | Pressão de memória / possível OOM |
| R8 | Disco cheio | Esgotamento de disco |
| R9 | Erros e HTTP 5xx com demais subsistemas normais | Falha provavelmente localizada na aplicação |

### Regras de ação

| Regra | Diagnóstico intermediário | Ação sugerida |
|---|---|---|
| R10 | Regressão de deploy | Comparar versões e avaliar rollback |
| R11 | Dependência externa | Verificar fornecedor, fallback e failover |
| R12 | Gargalo de banco | Investigar queries, locks, índices e I/O |
| R13 | Pool de conexões esgotado | Investigar pool, limites e possíveis leaks |
| R14 | Saturação de capacidade | Avaliar scale-out e autoscaling |
| R15 | Regressão de CPU | Comparar profiling e versões |
| R16 | Pressão de memória / OOM | Investigar heap, GC, limites e memory leaks |
| R17 | Disco cheio | Liberar/expandir espaço e investigar crescimento |
| R18 | Falha da aplicação | Correlacionar logs, traces e exceções |

## Encadeamento progressivo

Um dos principais objetivos do experimento é mostrar que uma regra pode produzir um novo fato que ativa outra regra.

Exemplo:

```text
CPU = alta
Tráfego = pico
       ↓
R5
       ↓
Diagnostico(
    tipo="saturacao_capacidade"
)
       ↓
R14
       ↓
Acao(
    "Avaliar scale-out..."
)
```

A regra R14 não precisa conhecer diretamente os valores de CPU ou tráfego.

Ela reage ao fato intermediário produzido pela R5.

Esse comportamento caracteriza o encadeamento progressivo:

```text
fatos
  ↓
inferência
  ↓
novos fatos
  ↓
novas regras ativadas
```

## Salience

Algumas regras possuem prioridade maior de execução.

Exemplo:

```python
salience=30
```

utilizado na regra de disco cheio.

Outras condições críticas também possuem saliência elevada.

A saliência influencia a ordem de disparo das regras na agenda do motor de inferência.

Ela não significa que regras de menor prioridade deixarão necessariamente de executar.

## Agenda e conflito

Mais de uma regra pode ser aplicável aos mesmos fatos.

Por exemplo:

```text
Deploy recente
Taxa de erros alta
CPU alta
Tráfego normal
```

pode produzir simultaneamente:

```text
R1 → regressao_deploy
R6 → regressao_cpu
```

Isso é intencional.

Um incidente real pode possuir múltiplas hipóteses compatíveis com os sinais observados.

O motor de inferência mantém ativações em uma agenda e utiliza critérios como saliência para determinar a ordem de execução.

## Fatores de confiança

Os diagnósticos possuem valores como:

```text
0.80
0.90
0.95
```

Esses valores são heurísticos e foram definidos para fins didáticos.

Eles representam força relativa das regras dentro desta base de conhecimento.

Não foram estimados a partir de dados históricos e não devem ser interpretados como probabilidades estatisticamente calibradas.

## Explicabilidade

O sistema registra uma trilha das regras disparadas.

Exemplo:

```text
[R5]
CPU elevada durante pico de tráfego é compatível
com saturação de capacidade.

[R14]
ENCADEAMENTO: saturação de capacidade gerou
uma ação de escalabilidade e análise de carga.
```

Isso permite responder a uma pergunta típica de sistemas especialistas:

```text
Por que o sistema chegou a essa conclusão?
```

## Cenários de demonstração

### Cenário 1 - Regressão após deploy

```text
Taxa de erros       alta
HTTP 5xx            alto
Deploy recente      sim
```

Fluxo esperado:

```text
R1
 ↓
regressao_deploy
 ↓
R10
 ↓
investigar versão / rollback
```

### Cenário 2 - Gargalo de banco

```text
Latência aplicação  alta
Latência banco      alta
```

Fluxo esperado:

```text
R3 → R12
```

### Cenário 3 - Saturação por tráfego

```text
CPU       alta
Tráfego   pico
```

Fluxo esperado:

```text
R5 → R14
```

### Cenário 4 - Memória / OOM

```text
Memória      alta
Reinícios    sim
```

Fluxo esperado:

```text
R7 → R16
```

### Cenário 5 - Disco cheio

```text
Disco = cheio
```

Fluxo esperado:

```text
R8 → R17
```

### Cenário 6 - Ausência de cobertura

```text
Latência aplicação = alta
```

com os demais sinais normais.

Nenhuma regra deve necessariamente produzir um diagnóstico.

Esse cenário demonstra uma limitação fundamental:

> um sistema especialista não possui conhecimento sobre situações que não foram representadas em sua base de regras.

## Limitações

O modelo foi construído para fins educacionais.

Entre suas principais limitações estão:

- regras simplificadas;
- ausência de aprendizado automático;
- ausência de dados históricos;
- fatores de confiança heurísticos;
- cobertura limitada do domínio;
- ausência de contexto temporal;
- ausência de correlação real entre métricas, logs e traces;
- possibilidade de múltiplos diagnósticos simultâneos;
- conhecimento estático definido manualmente.

## Sistema especialista x Machine Learning

Este experimento utiliza conhecimento explícito:

```text
SE condição
ENTÃO conclusão
```

Não existe treinamento do modelo.

A inteligência do sistema está na base de conhecimento e no processo de inferência.

Em Machine Learning, por outro lado, padrões são normalmente aprendidos a partir de dados.

## Relação com o Sommelier Digital

Os dois experimentos utilizam o mesmo princípio:

```text
Fatos
 ↓
Motor de inferência
 ↓
Regras
 ↓
Conclusão
```

O Incident Triage Expert acrescenta uma camada explícita de conhecimento intermediário:

```text
Fatos
 ↓
Diagnóstico
 ↓
Ação
```

Isso torna mais visível o encadeamento entre regras.

## Origem

Atividade desenvolvida para a disciplina de Inteligência Artificial - PPGI/UNIRIO - 2026.2.

O sistema é exclusivamente didático e não deve ser utilizado como mecanismo automatizado de decisão em ambientes de produção.
