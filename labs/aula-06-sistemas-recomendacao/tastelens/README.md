# TasteLens

Sistema interativo de recomendação desenvolvido para a disciplina de **Inteligência Artificial - PPGI/UNIRIO**.

O TasteLens demonstra, de forma prática, uma abordagem de **filtragem colaborativa baseada em usuários**, inspirada no princípio central do **GroupLens**: usuários que apresentaram padrões semelhantes de avaliação no passado podem fornecer informação útil para prever preferências futuras.

Neste projeto, o domínio escolhido é **comida/pratos**, evitando o uso de filmes e permitindo uma dinâmica simples de avaliação em sala.

---

## 1. Objetivo

O objetivo do TasteLens é tornar visível o funcionamento de um sistema de recomendação baseado em filtragem colaborativa.

Durante a demonstração:

1. participantes entram com um pseudônimo;
2. avaliam pratos utilizando notas de 1 a 5;
3. as avaliações são armazenadas em SQLite;
4. o sistema monta uma matriz **usuário × item**;
5. calcula a similaridade entre usuários com **correlação de Pearson**;
6. utiliza essas similaridades para prever avaliações ainda não observadas;
7. recomenda, para cada participante, o item com maior nota prevista.

Além da recomendação final, o painel da apresentação permite visualizar a matriz de avaliações, a matriz de similaridade, relações positivas e negativas entre usuários, uma rede de similaridade, médias dos itens e o suporte de cada recomendação.

---

## 2. Relação com o GroupLens

O trabalho de Resnick et al. (1994), *GroupLens: An Open Architecture for Collaborative Filtering of Netnews*, apresenta uma arquitetura de filtragem colaborativa baseada na ideia de que pessoas que concordaram em avaliações anteriores podem continuar apresentando padrões de preferência relacionados no futuro.

O TasteLens **não reproduz a arquitetura distribuída completa do GroupLens**. Trata-se de uma implementação didática do princípio de filtragem colaborativa baseada em usuários.

A ideia central representada no projeto é:

```text
avaliações
    ↓
matriz usuário × item
    ↓
similaridade entre usuários
    ↓
previsão de avaliações ausentes
    ↓
recomendação personalizada
```

A aplicação de referência apresentada por Carol Guzman na disciplina também utiliza uma dinâmica interativa para demonstrar filtragem colaborativa baseada em usuários. O TasteLens preserva o princípio acadêmico da atividade, mas utiliza outro domínio, modulariza o código e explicita o uso de correlações positivas e negativas na previsão.

---

## 3. Domínio da aplicação

O TasteLens trabalha com preferências alimentares.

### 3.1 Itens-âncora

Todos os participantes recebem os mesmos quatro itens:

- Pizza
- Sushi
- Feijoada
- Hambúrguer

Os itens-âncora criam uma base comum de avaliações para permitir a comparação dos padrões de preferência.

### 3.2 Itens candidatos

Cada participante recebe três itens adicionais selecionados deterministicamente a partir do pseudônimo.

O catálogo atual contém:

- Moqueca
- Lasanha
- Churrasco
- Yakisoba
- Risoto
- Acarajé
- Tacos
- Poke
- Curry indiano
- Escondidinho

Os itens extras criam células vazias diferentes na matriz usuário × item, permitindo que o algoritmo tente prever preferências ainda não observadas.

### 3.3 Quantidade mínima de avaliações

Para concluir a etapa de avaliação, o participante precisa avaliar pelo menos:

- **3 dos 4 itens-âncora**;
- **2 dos 3 itens extras**.

Um item desconhecido pode ser deixado sem avaliação.

---

## 4. Arquitetura

O projeto está organizado em módulos com responsabilidades separadas:

```text
tastelens/
├── app.py
├── catalog.py
├── recommender.py
├── storage.py
├── requirements.txt
├── README.md
├── INICIAR_APP.bat
├── iniciar_app.sh
├── data/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_recommender.py
    └── test_storage.py
```

### Responsabilidades

| Arquivo | Responsabilidade |
|---|---|
| `app.py` | Interface Streamlit, fluxo dos participantes e painel didático |
| `catalog.py` | Itens do domínio e parâmetros da demonstração |
| `recommender.py` | Pearson, matriz de similaridade, previsão e recomendação |
| `storage.py` | Persistência SQLite |
| `tests/test_recommender.py` | Testes do algoritmo |
| `tests/test_storage.py` | Testes de persistência |
| `data/` | Banco SQLite gerado durante a execução |

Fluxo arquitetural:

```text
catalog.py
    │
    ├──────────────┐
    ↓              ↓
storage.py    recommender.py
    │              │
    └──────┬───────┘
           ↓
         app.py
           ↓
       Streamlit
```

---

## 5. Matriz usuário × item

As avaliações são organizadas em uma matriz.

Exemplo:

| Usuário | Pizza | Sushi | Feijoada | Hambúrguer | Moqueca |
|---|---:|---:|---:|---:|---:|
| Ana | 5 | 4 | 1 | 5 | 5 |
| André | 5 | 4 | 1 | 5 | - |
| Bruno | 1 | 2 | 5 | 1 | 1 |

Cada linha representa um usuário e cada coluna representa um item.

O símbolo `-` indica uma preferência ainda não observada.

O problema de recomendação consiste em estimar valores plausíveis para essas células ausentes e selecionar itens com maior previsão.

---

## 6. Similaridade de Pearson

A similaridade entre dois usuários é calculada somente sobre itens avaliados por ambos.

A correlação de Pearson varia entre `-1` e `+1`:

- próximo de `+1`: padrões de avaliação semelhantes;
- próximo de `0`: pouca relação linear;
- próximo de `-1`: padrões de avaliação opostos.

A fórmula é:

```text
              Σ (xᵢ - x̄)(yᵢ - ȳ)
r(x,y) = -------------------------------
         √Σ(xᵢ - x̄)² · √Σ(yᵢ - ȳ)²
```

Por padrão, o TasteLens exige pelo menos **3 avaliações em comum** para calcular Pearson.

Em uma demonstração pequena, correlações calculadas com poucos itens devem ser interpretadas com cautela: valores extremos como `+1` ou `-1` podem surgir com pouca evidência.

---

## 7. Previsão de avaliações

Para prever a nota que um usuário `u` daria ao item `i`, o TasteLens utiliza uma média centrada ponderada pela similaridade:

```text
                    Σ sim(u,v) · (r(v,i) - média(v))
r^(u,i) = média(u) + --------------------------------
                           Σ |sim(u,v)|
```

Onde:

- `média(u)` é a média das avaliações do usuário-alvo;
- `sim(u,v)` é a correlação de Pearson entre o usuário-alvo e o vizinho;
- `r(v,i)` é a avaliação do vizinho para o item;
- `média(v)` é a média das avaliações do vizinho.

A previsão é limitada à escala de **1 a 5**.

---

## 8. Correlações negativas

O TasteLens não descarta automaticamente usuários com correlação negativa.

Uma correlação negativa pode fornecer informação útil.

Exemplo:

```text
Usuário A gosta de itens que Usuário B costuma rejeitar.
```

Se essa relação for consistente, uma nota baixa de B pode representar evidência positiva para A.

Por isso, o sistema utiliza:

```text
similaridade com sinal
```

no numerador da previsão e:

```text
valor absoluto da similaridade
```

no denominador.

Na rede de similaridade:

- linha verde contínua = correlação positiva;
- linha vermelha tracejada = correlação negativa;
- espessura da linha = intensidade da relação.

---

## 9. Suporte da recomendação

Além da nota prevista, o TasteLens informa o **suporte** da recomendação.

Exemplo:

```text
Recomendação: Lasanha
Nota prevista: 5.00 / 5
Suporte: 1 usuário
```

O suporte representa quantos participantes contribuíram efetivamente para aquela previsão.

Isso permite distinguir:

```text
nota prevista alta
```

de:

```text
evidência colaborativa forte
```

Uma recomendação `5.00` baseada em apenas um vizinho deve ser interpretada com mais cautela do que uma previsão semelhante apoiada por vários usuários.

---

## 10. Painel didático

O modo **Painel da apresentação** exibe:

- QR Code para entrada dos participantes;
- quantidade de participantes;
- estado da liberação das recomendações;
- explicação resumida do algoritmo;
- matriz usuário × item;
- matriz de similaridade colorida;
- análise individual de um usuário;
- gráfico horizontal de Pearson;
- rede de similaridade;
- média das avaliações dos itens;
- prévia das recomendações;
- nota prevista;
- suporte de cada previsão;
- opção para reiniciar a demonstração.

A matriz de similaridade utiliza cores para facilitar a leitura:

```text
vermelho  ← -1 -------- 0 -------- +1 → verde
            oposto                 semelhante
```

---

# 11. Instalação

## 11.1 Pré-requisitos

Recomenda-se:

- Python 3.11;
- Git;
- navegador web;
- terminal;
- rede Wi-Fi compartilhada caso os participantes utilizem celulares.

---

## 11.2 Clonar o repositório

```bash
git clone https://github.com/andrecataldo/ai-ppgi.git
cd ai-ppgi
```

---

## 11.3 Criar ambiente virtual

Na raiz do repositório:

```bash
python3 -m venv .venv
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

No Windows CMD:

```bat
.venv\Scripts\activate.bat
```

Quando o ambiente estiver ativo, o terminal normalmente mostrará:

```text
(.venv)
```

---

## 11.4 Instalar as dependências

A partir da raiz do repositório:

```bash
python -m pip install -r labs/aula-06-sistemas-recomendacao/tastelens/requirements.txt
```

Ou, entrando primeiro no projeto:

```bash
cd labs/aula-06-sistemas-recomendacao/tastelens
python -m pip install -r requirements.txt
```

---

# 12. Executar a aplicação

Entre na pasta:

```bash
cd labs/aula-06-sistemas-recomendacao/tastelens
```

Para uso somente no computador:

```bash
streamlit run app.py
```

O Streamlit normalmente disponibilizará:

```text
http://localhost:8501
```

---

## 12.1 Executar para acesso pelo celular

Para permitir acesso por outros dispositivos da mesma rede:

```bash
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

O terminal exibirá algo semelhante a:

```text
Local URL: http://localhost:8501
Network URL: http://192.168.0.45:8501
```

O endereço `Network URL` é utilizado pelos celulares.

---

# 13. Passo a passo de uso

## 13.1 Preparação do apresentador

1. Conecte o computador à rede Wi-Fi que será utilizada na demonstração.
2. Ative o ambiente virtual.
3. Entre na pasta `tastelens`.
4. Execute:

```bash
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

5. Abra o navegador.
6. Selecione:

```text
📊 Painel da apresentação
```

7. Informe a senha padrão do painel:

```text
tastelens2026
```

> A senha é apenas uma proteção simples para a demonstração local e pode ser alterada em `catalog.py`.

8. Projete o QR Code exibido pelo painel.

---

## 13.2 Entrada dos participantes

Cada participante deve:

1. conectar o celular à mesma rede Wi-Fi do computador;
2. escanear o QR Code;
3. abrir o endereço no navegador;
4. manter selecionado o modo:

```text
👤 Participante
```

5. informar um pseudônimo;
6. clicar em **Entrar**.

Não é necessário utilizar o nome verdadeiro.

---

## 13.3 Avaliar os itens

O participante recebe:

```text
4 itens comuns
+
3 itens extras
```

Cada item pode receber de:

```text
★☆☆☆☆  1
```

até:

```text
★★★★★  5
```

Se o participante não conhecer um item, pode deixá-lo sem avaliação.

Para concluir, é necessário avaliar pelo menos:

```text
3 itens comuns
+
2 itens extras
```

Depois, clique em:

```text
Enviar avaliações
```

A aplicação mostrará:

```text
✅ Suas avaliações já foram registradas.
```

e entrará em estado de espera.

---

## 13.4 Acompanhar a turma

No painel, o apresentador poderá acompanhar a formação da matriz usuário × item.

À medida que mais participantes respondem, tornam-se disponíveis:

- correlações de Pearson;
- relações positivas e negativas;
- análise individual;
- rede de similaridade;
- previsões de avaliações;
- recomendações.

É recomendável aguardar vários participantes antes de liberar os resultados.

---

## 13.5 Liberar recomendações

Quando houver dados suficientes, o apresentador deve clicar em:

```text
🚀 Calcular e liberar recomendações
```

Os navegadores dos participantes são atualizados automaticamente.

Cada usuário poderá receber algo semelhante a:

```text
🎯 Sua recomendação

🍽️ Churrasco

Nota prevista
4.29 / 5

Esta previsão utilizou 2 participantes.
```

Também são mostrados os usuários que influenciaram a previsão, sua similaridade e a avaliação dada ao item.

---

## 13.6 Interpretar a recomendação

A recomendação não significa:

```text
"o sistema sabe que você gostará deste prato"
```

Ela significa:

```text
"com base nos padrões de avaliação disponíveis,
este é o item ainda não avaliado com maior nota estimada"
```

O valor deve ser interpretado juntamente com o suporte da previsão.

---

## 13.7 Ocultar recomendações

O apresentador pode utilizar:

```text
⏸️ Ocultar recomendações
```

para voltar ao estado de espera.

---

## 13.8 Reiniciar a demonstração

No final do painel existe a seção:

```text
⚠️ Reiniciar demonstração
```

Marque a confirmação e clique em:

```text
Apagar tudo
```

Isso remove participantes e avaliações e volta o sistema ao estado inicial.

---

# 14. Acesso por QR Code e rede local

O QR Code aponta para o IP local da máquina.

Exemplo:

```text
http://192.168.0.45:8501
```

Por isso:

- computador e celulares precisam estar na mesma rede;
- o IP pode mudar quando o computador muda de rede;
- o QR Code deve ser gerado novamente em cada ambiente;
- firewalls podem bloquear o acesso ao Streamlit.

Se o aplicativo abrir no computador, mas não no celular, verifique:

1. se ambos estão no mesmo Wi-Fi;
2. se a aplicação foi iniciada com `--server.address 0.0.0.0`;
3. se o celular utiliza o `Network URL`, e não `localhost`;
4. se o firewall permite conexões de entrada para Python/Streamlit na rede privada.

---

# 15. Banco de dados

O sistema utiliza SQLite.

O banco é criado automaticamente em:

```text
data/tastelens.db
```

O banco armazena:

- identificador do participante;
- pseudônimo;
- itens extras atribuídos;
- avaliações;
- estado de liberação das recomendações.

O arquivo `.db` é dado de execução e não precisa ser versionado no Git.

---

# 16. Testes

O projeto possui testes automatizados do algoritmo e da persistência.

Executar:

```bash
pytest -v
```

Os testes verificam, entre outros pontos:

- usuários com padrões semelhantes;
- usuários com padrões opostos;
- correlação de Pearson;
- participação de correlação negativa na previsão;
- construção da matriz de similaridade;
- escolha da recomendação;
- persistência de participantes e avaliações;
- liberação e bloqueio das recomendações.

---

# 17. Qualidade de código

Durante o desenvolvimento, o projeto utiliza Ruff para análise estática e formatação.

Instalação:

```bash
python -m pip install ruff
```

Verificação:

```bash
ruff check .
```

Correção automática de problemas seguros:

```bash
ruff check . --fix
```

Formatação:

```bash
ruff format .
```

Validação de sintaxe:

```bash
python -m py_compile app.py catalog.py recommender.py storage.py
```

Um checkpoint esperado antes da entrega é:

```text
Ruff       → All checks passed!
Pytest     → todos os testes passaram
PyCompile  → sem erros
```

---

# 18. Limitações

O TasteLens possui finalidade acadêmica e didática.

Principais limitações:

- quantidade pequena de participantes;
- catálogo reduzido;
- uso apenas de avaliações explícitas;
- Pearson calculado sobre poucos itens em alguns casos;
- correlações extremas podem surgir com pouca evidência;
- uma recomendação pode possuir suporte de apenas um usuário;
- não existem fatores temporais ou contextuais;
- não existem perfis nutricionais;
- restrições alimentares, alergias e condições de saúde não são consideradas;
- o sistema não possui autenticação real;
- o SQLite é utilizado por simplicidade;
- a execução multiusuário ocorre em rede local;
- não reproduz a infraestrutura distribuída histórica do GroupLens.

Portanto, o TasteLens não deve ser interpretado como um sistema de recomendação de alimentos para decisões nutricionais ou de saúde. O domínio de pratos é utilizado apenas para demonstrar preferências subjetivas e filtragem colaborativa.

---

# 19. Decisões didáticas do projeto

Algumas escolhas foram feitas especificamente para melhorar a demonstração:

### Quatro itens-âncora

Aumentam a quantidade de avaliações compartilhadas entre participantes.

### Três itens extras

Criam esparsidade e itens ainda não avaliados.

### Mínimo de três avaliações comuns

Reduz o risco de calcular Pearson com evidência excessivamente pequena.

### Correlações negativas

São preservadas porque padrões opostos também podem fornecer informação preditiva.

### Suporte

Mostra que uma previsão alta não implica necessariamente evidência colaborativa forte.

### Painel visual

Permite observar o algoritmo em diferentes níveis:

```text
dados
  ↓
matriz
  ↓
similaridade
  ↓
rede
  ↓
previsão
  ↓
recomendação
```

---

# 20. Roteiro sugerido para demonstração em sala

Uma sequência possível:

1. apresentar o problema de recomendação;
2. relacionar o TasteLens ao princípio do GroupLens;
3. pedir aos participantes que entrem pelo QR Code;
4. coletar as avaliações;
5. mostrar a matriz usuário × item;
6. explicar as células vazias;
7. mostrar a matriz de Pearson;
8. selecionar um participante na análise individual;
9. explicar relações positivas e negativas;
10. mostrar a rede de similaridade;
11. observar a prévia das recomendações;
12. destacar a diferença entre **nota prevista** e **suporte**;
13. liberar as recomendações;
14. pedir aos participantes que comparem os resultados;
15. discutir limitações, esparsidade e partida a frio.

---

# 21. Referência principal

RESNICK, P.; IACOVOU, N.; SUCHAK, M.; BERGSTROM, P.; RIEDL, J.  
**GroupLens: An Open Architecture for Collaborative Filtering of Netnews.**  
Proceedings of the ACM Conference on Computer Supported Cooperative Work - CSCW, 1994, p. 175–186.

---

## 22. Contexto acadêmico

Projeto desenvolvido para a disciplina de **Inteligência Artificial - PPGI/UNIRIO**.

Tema:

```text
Sistemas de Recomendação
```

Foco:

```text
Filtragem colaborativa baseada em usuários
```

Domínio:

```text
Preferências por pratos
```

Aplicação:

```text
TasteLens
```
