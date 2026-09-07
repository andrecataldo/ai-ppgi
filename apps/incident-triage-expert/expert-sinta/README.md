# Incident Triage Expert - Expert SINTA

Implementação em Expert SINTA de um sistema especialista para triagem inicial de incidentes em aplicações Web.

Projeto desenvolvido para a disciplina de Inteligência Artificial do PPGI/UNIRIO - 2026.2.

## Objetivo

O sistema recebe sinais observáveis relacionados à aplicação, infraestrutura, banco de dados, dependências externas e contexto operacional.

A partir desses fatos, a base de conhecimento busca inferir:

1. possíveis diagnósticos;
2. ações recomendadas.

O objetivo do projeto é demonstrar conceitos clássicos de sistemas especialistas, incluindo:

- representação explícita do conhecimento;
- regras de produção;
- inferência;
- fatores de confiança;
- variáveis-objetivo;
- encadeamento entre regras;
- explicabilidade;
- limitações de cobertura da base de conhecimento.

## Arquivo da base

A base de conhecimento está armazenada em:

```text
incident-triage.bcm
```

O arquivo pode ser aberto com o Expert SINTA 1.1b.

## Arquitetura conceitual

```text
Fatos observados
       ↓
Regras de diagnóstico
       ↓
Diagnóstico
       ↓
Regras de ação
       ↓
Ação recomendada
```

Um diagnóstico produzido por uma regra pode ser utilizado como premissa de outra regra.

Exemplo:

```text
cpu = alta
tráfego = pico
        ↓
R5
        ↓
diagnóstico = saturação de capacidade
        ↓
R14
        ↓
ação recomendada = avaliar scale-out
```

## Variáveis

A base possui 14 variáveis.

### Variáveis de entrada

| Variável | Valores |
|---|---|
| taxa de erros | normal, alta |
| http 5xx | normal, alto |
| latência da aplicação | normal, alta |
| reinícios da aplicação | não, sim |
| cpu | normal, alta |
| memória | normal, alta |
| disco | normal, cheio |
| latência do banco | normal, alta |
| conexões do banco | normal, esgotadas |
| dependência externa | ok, degradada, fora |
| deploy recente | não, sim |
| tráfego | normal, pico |

Todas as variáveis de entrada são univaloradas.

### Variáveis inferidas

```text
diagnóstico
ação recomendada
```

Essas duas variáveis são multivaloradas para permitir a coexistência de mais de uma hipótese ou recomendação.

## Variáveis-objetivo

A consulta possui dois objetivos:

```text
diagnóstico
ação recomendada
```

O sistema primeiro procura determinar possíveis diagnósticos e utiliza esses resultados para inferir as ações correspondentes.

## Interface de consulta

As 12 variáveis de entrada possuem perguntas associadas.

| Variável | Pergunta |
|---|---|
| taxa de erros | Como está a taxa de erros da aplicação? |
| http 5xx | Como está a ocorrência de respostas HTTP 5xx? |
| latência da aplicação | Como está a latência da aplicação? |
| reinícios da aplicação | A aplicação está apresentando reinícios? |
| cpu | Como está o consumo de CPU? |
| memória | Como está o consumo de memória? |
| disco | Como está a utilização do disco? |
| latência do banco | Como está a latência do banco de dados? |
| conexões do banco | Como está o pool de conexões do banco? |
| dependência externa | Qual é o estado da dependência externa? |
| deploy recente | Houve deploy recente antes do incidente? |
| tráfego | Como está o volume de tráfego? |

As respostas fornecidas pelo usuário são tratadas como fatos observados.

## Base de conhecimento

O modelo possui 18 regras conceituais.

No Expert SINTA existem 19 regras físicas, pois a regra conceitual R2 foi desdobrada em duas regras para preservar fatores de confiança distintos.

### Regras de diagnóstico

| Regra conceitual | Regra SINTA | Condições principais | Diagnóstico | CNF |
|---|---:|---|---|---:|
| R1 | 1 | deploy recente + erros altos | regressão de deploy | 90% |
| R2a | 2 | dependência degradada | dependência externa | 85% |
| R2b | 3 | dependência fora | dependência externa | 98% |
| R3 | 4 | latência app alta + banco alta | gargalo de banco | 88% |
| R4 | 5 | conexões esgotadas + HTTP 5xx alto | pool de conexões esgotado | 95% |
| R5 | 6 | CPU alta + pico de tráfego | saturação de capacidade | 90% |
| R6 | 7 | CPU alta + tráfego normal + deploy | regressão de CPU | 82% |
| R7 | 8 | memória alta + reinícios | pressão de memória ou OOM | 95% |
| R8 | 9 | disco cheio | disco cheio | 99% |
| R9 | 10 | erros altos com demais subsistemas normais | falha da aplicação | 80% |

### Regras de ação

As regras de ação utilizam CNF 100%. Assim, elas não introduzem nova incerteza e preservam o fator de confiança proveniente do diagnóstico.

| Regra conceitual | Regra SINTA | Diagnóstico | Ação |
|---|---:|---|---|
| R10 | 11 | regressão de deploy | avaliar rollback |
| R11 | 12 | dependência externa | verificar dependência externa |
| R12 | 13 | gargalo de banco | investigar banco |
| R13 | 14 | pool de conexões esgotado | investigar pool de conexões |
| R14 | 15 | saturação de capacidade | avaliar scale-out |
| R15 | 16 | regressão de CPU | investigar regressão de CPU |
| R16 | 17 | pressão de memória ou OOM | investigar memória e OOM |
| R17 | 18 | disco cheio | liberar ou expandir disco |
| R18 | 19 | falha da aplicação | investigar aplicação |

## Fatores de confiança

Os fatores de confiança utilizados na base são heurísticos e foram definidos para fins didáticos.

Eles não foram estimados a partir de dados históricos e não devem ser interpretados como probabilidades estatisticamente calibradas.

Nas regras de diagnóstico, cada conclusão recebe um CNF específico.

Exemplo:

```text
SE
    deploy recente = sim
E
    taxa de erros = alta
ENTÃO
    diagnóstico = regressão de deploy CNF 90%
```

As regras de ação utilizam:

```text
CNF 100%
```

Isso permite que a confiança do diagnóstico seja propagada para a ação sem introduzir uma redução adicional.

Exemplo validado:

```text
diagnóstico = regressão de deploy (90%)
        ↓
ação recomendada = avaliar rollback (90%)
```

## Cenários validados

### Regressão após deploy

Entrada principal:

```text
deploy recente = sim
taxa de erros = alta
```

Resultado:

```text
diagnóstico
└── regressão de deploy (90%)

ação recomendada
└── avaliar rollback (90%)
```

### Saturação por tráfego

Entrada principal:

```text
cpu = alta
tráfego = pico
```

Resultado:

```text
diagnóstico
└── saturação de capacidade (90%)

ação recomendada
└── avaliar scale-out (90%)
```

### Falha da aplicação

Cenário:

```text
taxa de erros = alta
http 5xx = alto
cpu = normal
memória = normal
disco = normal
latência do banco = normal
conexões do banco = normal
dependência externa = ok
deploy recente = não
```

Resultado:

```text
diagnóstico
└── falha da aplicação (80%)

ação recomendada
└── investigar aplicação (80%)
```

## Relação com a implementação Python

O mesmo domínio foi implementado também utilizando Python, Experta e Streamlit:

```text
../incident_triage_app.py
```

As duas implementações utilizam essencialmente a mesma base de conhecimento.

A versão Expert SINTA atende diretamente à atividade da disciplina.

A versão Python fornece uma implementação complementar com interface Web e maior visibilidade sobre:

- memória de trabalho;
- regras disparadas;
- diagnósticos;
- ações;
- trilha de explicação.

## Diferença de implementação

Na implementação Python, a regra de dependência externa pode calcular fatores de confiança diferentes dentro da mesma função.

No Expert SINTA, para preservar esses valores, a regra conceitual R2 foi representada como duas regras físicas:

```text
R2a
dependência degradada → CNF 85%

R2b
dependência fora → CNF 98%
```

Por esse motivo:

```text
18 regras conceituais
19 regras físicas no Expert SINTA
```

## Limitações

Esta base foi construída para fins educacionais.

Entre suas limitações estão:

- conhecimento definido manualmente;
- cobertura limitada do domínio;
- ausência de aprendizado automático;
- ausência de dados históricos;
- ausência de contexto temporal detalhado;
- fatores de confiança heurísticos;
- simplificação de relações causais;
- possibilidade de múltiplos diagnósticos compatíveis;
- ausência de integração com métricas, logs e traces reais.

## Disciplina

Inteligência Artificial  
PPGI/UNIRIO  
2026.2
