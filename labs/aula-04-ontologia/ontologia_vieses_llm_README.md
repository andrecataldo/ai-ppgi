# Ontologia - Vieses Cognitivos em Interações Humano-LLM

## 1. Contexto

Este laboratório foi desenvolvido para a disciplina de **Inteligência Artificial - PPGI/UNIRIO - 2026.2**.

O objetivo é construir e validar uma ontologia OWL para representar **vieses cognitivos em julgamentos e decisões mediados por interações entre humanos e Large Language Models (LLMs)**.

A modelagem foi construída manualmente no **Protégé 5.6.9**, com validação lógica utilizando o reasoner **HermiT 1.4.3.456**.

Arquivo principal da ontologia:

    ontologia_vieses_llm_v1.rdf

IRI da ontologia:

    http://www.semanticweb.org/andrecataldo/ontologies/2026/9/vieses-llm

---

## 2. Objetivo da modelagem

A ontologia busca representar, de forma explícita e semanticamente consultável:

- humanos e LLMs envolvidos em uma interação;
- tarefas de julgamento e decisão;
- prompts, respostas, evidências e âncoras presentes na interação;
- vieses cognitivos observados;
- fenômenos relacionados à confiança e sobredependência;
- julgamentos e decisões resultantes;
- diferentes formas pelas quais um viés pode ser associado ao humano, ao LLM ou emergir da própria interação.

O escopo desta versão é deliberadamente restrito. A ontologia **não pretende cobrir todos os vieses cognitivos nem todos os fenômenos de interação humano-IA**.

---

## 3. Estrutura conceitual

    owl:Thing
    ├── Agente
    │   ├── Humano
    │   └── SistemaIA
    │       └── LLM
    ├── InteracaoHumanoLLM
    ├── Tarefa
    │   ├── TarefaJulgamento
    │   └── TarefaDecisao
    ├── ArtefatoInteracao
    │   ├── Prompt
    │   ├── RespostaLLM
    │   │   └── RecomendacaoLLM
    │   ├── Evidencia
    │   └── Ancora
    ├── CrencaPrevia
    ├── JulgamentoDecisao
    ├── ViesCognitivo
    │   ├── ViesConfirmacao
    │   ├── ViesAncoragem
    │   └── ViesAutomacao
    └── FenomenoConfianca
        ├── Sobredependencia
        ├── ConfiancaCalibrada
        └── ConfiancaDescalibrada

Também foram criadas classes definidas por equivalência:

    InteracaoComVies
    InteracaoComViesConfirmacao
    InteracaoComViesAncoragem
    InteracaoComViesAutomacao
    InteracaoComSobredependencia
    ViesManifestadoPorHumano
    ViesManifestadoPorLLM
    ViesEmergenteDaInteracao

---

## 4. Decisões de modelagem

- `Sobredependencia` é modelada como `FenomenoConfianca`, e não automaticamente como `ViesCognitivo`.
- A presença de uma `Ancora` não implica, por si só, `ViesAncoragem`.
- Uma resposta alinhada a uma crença prévia não implica automaticamente `ViesConfirmacao`.
- `TarefaJulgamento` e `TarefaDecisao` não foram declaradas disjuntas.
- `Humano` foi declarado disjunto de `SistemaIA`.
- Domain e Range das Object Properties são utilizados com a semântica inferencial de OWL, e não como simples validação estrutural.

---

## 5. Object Properties

| Propriedade | Domain | Range | Inversa |
|---|---|---|---|
| `envolveHumano` | `InteracaoHumanoLLM` | `Humano` | `participaEm` |
| `participaEm` | `Humano` | `InteracaoHumanoLLM` | `envolveHumano` |
| `envolveLLM` | `InteracaoHumanoLLM` | `LLM` | `usadoEmInteracao` |
| `usadoEmInteracao` | `LLM` | `InteracaoHumanoLLM` | `envolveLLM` |
| `executaTarefa` | `InteracaoHumanoLLM` | `Tarefa` | - |
| `possuiPrompt` | `InteracaoHumanoLLM` | `Prompt` | - |
| `produzResposta` | `InteracaoHumanoLLM` | `RespostaLLM` | - |
| `geradaPorLLM` | `RespostaLLM` | `LLM` | - |
| `resultaEm` | `InteracaoHumanoLLM` | `JulgamentoDecisao` | - |
| `possuiCrencaPrevia` | `Humano` | `CrencaPrevia` | - |
| `expressaCrenca` | `Prompt` | `CrencaPrevia` | - |
| `respostaAlinhadaCom` | `RespostaLLM` | `CrencaPrevia` | - |
| `contemAncora` | `ArtefatoInteracao` | `Ancora` | - |
| `influenciaJulgamento` | `Ancora` | `JulgamentoDecisao` | - |
| `confrontaResposta` | `Evidencia` | `RespostaLLM` | - |
| `baseiaSeEm` | `JulgamentoDecisao` | `ArtefatoInteracao` | - |
| `temVies` | `InteracaoHumanoLLM` | `ViesCognitivo` | `ocorreEmInteracao` |
| `ocorreEmInteracao` | `ViesCognitivo` | `InteracaoHumanoLLM` | `temVies` |
| `manifestadoPor` | `ViesCognitivo` | `Agente` | `manifestaVies` |
| `manifestaVies` | `Agente` | `ViesCognitivo` | `manifestadoPor` |
| `emergeDaInteracao` | `ViesCognitivo` | `InteracaoHumanoLLM` | - |
| `temFenomenoConfianca` | `InteracaoHumanoLLM` | `FenomenoConfianca` | - |

---

## 6. Data Properties

| Propriedade | Domain | Range |
|---|---|---|
| `codigoParticipante` | `Humano` | `xsd:string` |
| `nomeModelo` | `LLM` | `xsd:string` |
| `versaoModelo` | `LLM` | `xsd:string` |
| `descricaoTarefa` | `Tarefa` | `xsd:string` |
| `dominioTarefa` | `Tarefa` | `xsd:string` |
| `textoPrompt` | `Prompt` | `xsd:string` |
| `textoResposta` | `RespostaLLM` | `xsd:string` |
| `descricaoCrenca` | `CrencaPrevia` | `xsd:string` |
| `valorAncora` | `Ancora` | `xsd:string` |
| `descricaoEvidencia` | `Evidencia` | `xsd:string` |
| `resultadoJulgamento` | `JulgamentoDecisao` | `xsd:string` |
| `dataHoraInteracao` | `InteracaoHumanoLLM` | `xsd:dateTime` |
| `numeroTurnos` | `InteracaoHumanoLLM` | `xsd:nonNegativeInteger` |
| `nivelConfianca` | `FenomenoConfianca` | `xsd:decimal` |

---

## 7. Axiomas principais

    InteracaoHumanoLLM SubClassOf envolveHumano some Humano
    InteracaoHumanoLLM SubClassOf envolveLLM some LLM
    InteracaoHumanoLLM SubClassOf executaTarefa some (TarefaJulgamento or TarefaDecisao)
    
    RespostaLLM SubClassOf geradaPorLLM some LLM
    
    Humano DisjointWith SistemaIA

Classes definidas:

    InteracaoComVies
    EquivalentTo InteracaoHumanoLLM and temVies some ViesCognitivo
    
    InteracaoComViesConfirmacao
    EquivalentTo InteracaoHumanoLLM and temVies some ViesConfirmacao
    
    InteracaoComViesAncoragem
    EquivalentTo InteracaoHumanoLLM and temVies some ViesAncoragem
    
    InteracaoComViesAutomacao
    EquivalentTo InteracaoHumanoLLM and temVies some ViesAutomacao
    
    InteracaoComSobredependencia
    EquivalentTo InteracaoHumanoLLM
                 and temFenomenoConfianca some Sobredependencia
    
    ViesManifestadoPorHumano
    EquivalentTo ViesCognitivo and manifestadoPor some Humano
    
    ViesManifestadoPorLLM
    EquivalentTo ViesCognitivo and manifestadoPor some LLM
    
    ViesEmergenteDaInteracao
    EquivalentTo ViesCognitivo
                 and emergeDaInteracao some InteracaoHumanoLLM

---

## 8. Regras conceituais do domínio

1. Um LLM é um tipo de sistema de IA.
2. Toda interação humano-LLM envolve pelo menos um humano e um LLM.
3. Toda interação ocorre no contexto de pelo menos uma tarefa de julgamento ou decisão.
4. Uma interação pode possuir um ou mais prompts e respostas.
5. Uma interação pode resultar em julgamento ou decisão humana.
6. Um humano pode possuir uma crença prévia.
7. Um prompt pode expressar uma crença prévia.
8. Um artefato de interação pode conter uma âncora capaz de influenciar um julgamento posterior.
9. Uma resposta de LLM pode estar alinhada a uma crença prévia e pode ser confrontada por evidências.
10. Confirmação, ancoragem e automação são modeladas como vieses cognitivos.
11. Um viés pode ser manifestado por um humano, por um LLM ou emergir da interação.
12. Uma interação pode apresentar zero, um ou mais vieses cognitivos.
13. O viés de confirmação pode estar associado ao favorecimento de informações consistentes com crenças prévias.
14. O viés de ancoragem pode estar associado à influência de uma informação inicial sobre um julgamento posterior.
15. Sobredependência, confiança calibrada e confiança descalibrada são tratadas como fenômenos de confiança, e não automaticamente como vieses cognitivos.

Nem todas essas regras foram transformadas em inferências automáticas nesta versão.

---

## 9. Cenário 1 - Viés de Confirmação

Indivíduos principais:

    Participante_P01
    LLM_Exemplo
    Tarefa_Avaliacao_Texto
    Vies_Confirmacao_01
    Interacao_Confirmacao_01

Relações:

    Interacao_Confirmacao_01 envolveHumano Participante_P01
    Interacao_Confirmacao_01 envolveLLM LLM_Exemplo
    Interacao_Confirmacao_01 executaTarefa Tarefa_Avaliacao_Texto
    Interacao_Confirmacao_01 temVies Vies_Confirmacao_01
    Vies_Confirmacao_01 manifestadoPor Participante_P01

Inferências verificadas:

    Interacao_Confirmacao_01 rdf:type InteracaoComViesConfirmacao
    Vies_Confirmacao_01 rdf:type ViesManifestadoPorHumano
    Participante_P01 participaEm Interacao_Confirmacao_01
    Vies_Confirmacao_01 ocorreEmInteracao Interacao_Confirmacao_01

---

## 10. Cenário 2 - Viés de Ancoragem

Indivíduos principais:

    Prompt_Ancoragem_01
    Ancora_01
    Julgamento_Ancoragem_01
    Vies_Ancoragem_01
    Interacao_Ancoragem_01

Relações:

    Prompt_Ancoragem_01 contemAncora Ancora_01
    Ancora_01 influenciaJulgamento Julgamento_Ancoragem_01
    
    Interacao_Ancoragem_01 envolveHumano Participante_P01
    Interacao_Ancoragem_01 envolveLLM LLM_Exemplo
    Interacao_Ancoragem_01 executaTarefa Tarefa_Avaliacao_Texto
    Interacao_Ancoragem_01 possuiPrompt Prompt_Ancoragem_01
    Interacao_Ancoragem_01 resultaEm Julgamento_Ancoragem_01
    Interacao_Ancoragem_01 temVies Vies_Ancoragem_01
    
    Vies_Ancoragem_01 emergeDaInteracao Interacao_Ancoragem_01

Inferências verificadas:

    Interacao_Ancoragem_01 rdf:type InteracaoComViesAncoragem
    Vies_Ancoragem_01 rdf:type ViesEmergenteDaInteracao
    Vies_Ancoragem_01 ocorreEmInteracao Interacao_Ancoragem_01

A presença de `Ancora_01` não é utilizada para inferir automaticamente `ViesAncoragem`.

---

## 11. Consultas DL validadas

    InteracaoHumanoLLM and temVies some ViesConfirmacao

Resultado:

    Interacao_Confirmacao_01

    ViesCognitivo and manifestadoPor some Humano

Resultado:

    Vies_Confirmacao_01

    Humano and participaEm some
        (InteracaoHumanoLLM and temVies some ViesConfirmacao)

Resultado:

    Participante_P01

    LLM and usadoEmInteracao some
        (InteracaoHumanoLLM and temVies some ViesConfirmacao)

Resultado:

    LLM_Exemplo

    InteracaoHumanoLLM and envolveHumano value Participante_P01

Resultados após a inclusão dos dois cenários:

    Interacao_Confirmacao_01
    Interacao_Ancoragem_01

    ViesCognitivo and emergeDaInteracao some InteracaoHumanoLLM

Resultado:

    Vies_Ancoragem_01

    InteracaoHumanoLLM and temVies some ViesAncoragem

Resultado:

    Interacao_Ancoragem_01

    Ancora and influenciaJulgamento some JulgamentoDecisao

Resultado:

    Ancora_01

    Prompt and contemAncora some
        (Ancora and influenciaJulgamento some JulgamentoDecisao)

Resultado:

    Prompt_Ancoragem_01

---

## 12. Validação

A ontologia foi validada no Protégé utilizando o HermiT.

Foram verificados:

- classificação automática de indivíduos;
- propriedades inversas;
- classes definidas com `EquivalentTo`;
- consultas DL;
- justificativas de inferência;
- efeitos semânticos de Domain e Range.

Durante os testes, relações inseridas na direção incorreta produziram classificações indesejadas, que foram identificadas pelas justificativas do reasoner e corrigidas.

---

## 13. Limitações e próximos passos

Esta versão é uma prova de conceito. Extensões possíveis:

- ampliar cenários de `ViesAutomacao` e `Sobredependencia`;
- representar evidências contrárias e respostas/recomendações do LLM;
- adicionar `rdfs:label` e `rdfs:comment`;
- alinhar conceitos com ontologias existentes;
- utilizar SHACL para validação de dados;
- expandir a ABox com dados empíricos.

---

## 14. Ferramentas

- Protégé Desktop 5.6.9
- OWL / RDF/XML
- HermiT 1.4.3.456
- DL Query

---

## 15. Arquivos da entrega

    labs/aula-04-ontologia/
    ├── README.md
    ├── relatorio-academico.md
    └── ontologia_vieses_llm_v1.rdf
