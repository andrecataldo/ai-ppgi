# Ontologia de Vieses Cognitivos em Interações Humano-LLM

**Disciplina:** Inteligência Artificial - PPGI/UNIRIO - 2026.2  
**Professora:** Ana Cristina Bicharra Garcia  
**Aluno:** André Luiz Cataldo Falbo Santo  
**Ferramenta:** Protégé 5.6.9  
**Representação:** OWL / RDF/XML  
**Reasoner:** HermiT 1.4.3.456

## Resumo

Este trabalho apresenta uma ontologia para representar vieses cognitivos em julgamentos e decisões mediados por interações entre humanos e Large Language Models (LLMs). A modelagem procura explicitar agentes, tarefas, artefatos de interação, crenças prévias, julgamentos, vieses cognitivos e fenômenos de confiança. A ontologia foi implementada em OWL no Protégé e validada com o reasoner HermiT. Dois cenários sintéticos foram instanciados: um de viés de confirmação, modelado como manifestado por um humano, e outro de viés de ancoragem, modelado como emergente da interação humano-LLM. As inferências e consultas DL demonstram que a ontologia é capaz de classificar automaticamente indivíduos a partir de axiomas, relações e classes definidas, indo além da simples representação de um grafo de dados.

## 1. Objetivo

O objetivo do trabalho é desenvolver uma ontologia inicial para o domínio de **vieses cognitivos em julgamentos mediados por LLMs**, com ênfase na interação humano-LLM.

A proposta não busca representar todos os vieses cognitivos nem todos os aspectos de sistemas de IA generativa. O recorte concentra-se em conceitos úteis para estudar situações nas quais um ser humano interage com um LLM durante tarefas de julgamento ou decisão, podendo ocorrer vieses como confirmação, ancoragem e automação.

A ontologia também diferencia vieses cognitivos de fenômenos relacionados à confiança. Dessa forma, `Sobredependencia`, `ConfiancaCalibrada` e `ConfiancaDescalibrada` são modeladas como subclasses de `FenomenoConfianca`, sem assumir que sejam automaticamente vieses cognitivos.

## 2. Modelagem

A conceitualização central considera `InteracaoHumanoLLM` como unidade de análise. Uma interação deve envolver pelo menos um `Humano`, um `LLM` e uma tarefa de julgamento ou decisão.

Os principais grupos conceituais são:

- **Agentes:** `Humano`, `SistemaIA` e `LLM`;
- **Interação:** `InteracaoHumanoLLM`;
- **Tarefas:** `TarefaJulgamento` e `TarefaDecisao`;
- **Artefatos:** `Prompt`, `RespostaLLM`, `RecomendacaoLLM`, `Evidencia` e `Ancora`;
- **Aspectos cognitivos:** `CrencaPrevia` e `JulgamentoDecisao`;
- **Vieses:** `ViesConfirmacao`, `ViesAncoragem` e `ViesAutomacao`;
- **Confiança:** `Sobredependencia`, `ConfiancaCalibrada` e `ConfiancaDescalibrada`.

A distinção entre vieses manifestados por agentes e vieses emergentes da interação é representada pelas classes definidas `ViesManifestadoPorHumano`, `ViesManifestadoPorLLM` e `ViesEmergenteDaInteracao`.

## 3. Regras e axiomas

A ontologia foi orientada por regras conceituais do domínio, entre elas: um LLM é um sistema de IA; toda interação humano-LLM envolve pelo menos um humano e um LLM; toda interação ocorre no contexto de uma tarefa de julgamento ou decisão; um humano pode possuir uma crença prévia; um prompt pode expressar essa crença; artefatos de interação podem conter âncoras; e vieses podem estar associados a humanos, LLMs ou à própria interação.

Alguns desses compromissos foram formalizados como axiomas OWL:

    InteracaoHumanoLLM SubClassOf envolveHumano some Humano
    InteracaoHumanoLLM SubClassOf envolveLLM some LLM
    InteracaoHumanoLLM SubClassOf executaTarefa some
        (TarefaJulgamento or TarefaDecisao)
    
    RespostaLLM SubClassOf geradaPorLLM some LLM
    
    Humano DisjointWith SistemaIA

Classes equivalentes permitem inferência automática:

    InteracaoComViesConfirmacao
    EquivalentTo InteracaoHumanoLLM
                 and temVies some ViesConfirmacao
    
    InteracaoComViesAncoragem
    EquivalentTo InteracaoHumanoLLM
                 and temVies some ViesAncoragem
    
    ViesManifestadoPorHumano
    EquivalentTo ViesCognitivo
                 and manifestadoPor some Humano
    
    ViesEmergenteDaInteracao
    EquivalentTo ViesCognitivo
                 and emergeDaInteracao some InteracaoHumanoLLM

Uma decisão importante foi não automatizar conclusões conceitualmente fortes sem evidência suficiente. Assim, a presença de uma âncora em um prompt não é suficiente para classificar automaticamente um caso como `ViesAncoragem`; de forma semelhante, o alinhamento de uma resposta do LLM a uma crença prévia não é tratado como prova automática de `ViesConfirmacao`.

## 4. Cenários instanciados

### 4.1 Viés de confirmação

O primeiro cenário representa uma interação entre `Participante_P01` e `LLM_Exemplo` durante `Tarefa_Avaliacao_Texto`. A interação possui o indivíduo `Vies_Confirmacao_01` por meio da propriedade `temVies`, e esse viés é associado ao participante pela propriedade `manifestadoPor`.

O HermiT inferiu:

    Interacao_Confirmacao_01
        rdf:type InteracaoComViesConfirmacao
    
    Vies_Confirmacao_01
        rdf:type ViesManifestadoPorHumano

As propriedades inversas também permitiram inferir que o participante `participaEm` a interação e que o viés `ocorreEmInteracao`.

### 4.2 Viés de ancoragem

O segundo cenário representa um prompt contendo uma informação inicial utilizada como âncora. Foram criados `Prompt_Ancoragem_01`, `Ancora_01`, `Julgamento_Ancoragem_01`, `Vies_Ancoragem_01` e `Interacao_Ancoragem_01`.

Relações principais:

    Prompt_Ancoragem_01
        contemAncora Ancora_01
    
    Ancora_01
        influenciaJulgamento Julgamento_Ancoragem_01
    
    Interacao_Ancoragem_01
        possuiPrompt Prompt_Ancoragem_01
    
    Interacao_Ancoragem_01
        resultaEm Julgamento_Ancoragem_01
    
    Interacao_Ancoragem_01
        temVies Vies_Ancoragem_01
    
    Vies_Ancoragem_01
        emergeDaInteracao Interacao_Ancoragem_01

O reasoner inferiu:

    Interacao_Ancoragem_01
        rdf:type InteracaoComViesAncoragem
    
    Vies_Ancoragem_01
        rdf:type ViesEmergenteDaInteracao

Esse cenário demonstra a distinção entre representar uma **âncora** e afirmar a existência de **viés de ancoragem**.

## 5. Inferências e consultas DL

Exemplos validados:

    InteracaoHumanoLLM and temVies some ViesConfirmacao

Resultado:

    Interacao_Confirmacao_01

    ViesCognitivo and manifestadoPor some Humano

Resultado:

    Vies_Confirmacao_01

    ViesCognitivo and emergeDaInteracao some InteracaoHumanoLLM

Resultado:

    Vies_Ancoragem_01

    InteracaoHumanoLLM and temVies some ViesAncoragem

Resultado:

    Interacao_Ancoragem_01

    Ancora and influenciaJulgamento some JulgamentoDecisao

Resultado:

    Ancora_01

Consulta composta:

    Prompt and contemAncora some
        (Ancora and influenciaJulgamento some JulgamentoDecisao)

Resultado:

    Prompt_Ancoragem_01

## 6. Validação e discussão

O reasoner HermiT foi utilizado durante a construção da ontologia para classificação e verificação das inferências.

A validação permitiu observar diretamente uma característica importante da semântica OWL: `Domain` e `Range` não funcionam apenas como regras de validação de dados. Eles também produzem inferências sobre os tipos dos indivíduos.

Durante os testes, relações inseridas na direção incorreta levaram o reasoner a classificar entidades de forma indesejada. A inspeção das justificativas fornecidas pelo Protégé permitiu identificar e corrigir esses casos.

Os dois cenários demonstram usos diferentes da ontologia: no primeiro, o viés é associado a um agente humano; no segundo, é modelado como emergente da própria interação.

## 7. Limitações

Esta é uma versão inicial e exploratória da ontologia. Os indivíduos utilizados são cenários sintéticos e não constituem evidência empírica sobre ocorrência de vieses.

A ontologia ainda pode ser expandida para:

- representar respostas e recomendações produzidas por LLMs;
- incorporar evidências contrárias;
- modelar de forma mais detalhada viés de automação e sobredependência;
- utilizar SHACL para validação estrutural de dados;
- incorporar ou alinhar conceitos com ontologias existentes;
- receber dados provenientes de estudos empíricos sobre interação humano-LLM.

## 8. Conclusão

A ontologia desenvolvida fornece uma primeira estrutura formal para representar vieses cognitivos em interações humano-LLM. A modelagem distingue agentes, artefatos, tarefas, julgamentos, vieses e fenômenos de confiança, permitindo representar diferentes formas de manifestação de um viés.

Os testes com HermiT e DL Query mostraram que a representação é capaz de produzir novas classificações a partir dos axiomas definidos. Os cenários de confirmação e ancoragem demonstraram tanto inferências baseadas em propriedades inversas quanto classificação por classes equivalentes.

Como resultado, o trabalho apresenta uma base inicial extensível para futuras investigações sobre vieses cognitivos, confiança e julgamentos mediados por LLMs.

## Referências

GRUBER, T. R. A translation approach to portable ontology specifications. *Knowledge Acquisition*, v. 5, n. 2, p. 199-220, 1993.

HOGAN, A. et al. Knowledge graphs. *ACM Computing Surveys*, v. 54, n. 4, 2021.

MUSEN, M. A. The Protégé project: a look back and a look forward. *AI Matters*, v. 1, n. 4, p. 4-12, 2015.
