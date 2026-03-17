# Projeto Dataset AI - Versão 02

# 1. Importação das bibliotecas

```python
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
```

Essas linhas importam as bibliotecas utilizadas no projeto.

* **pandas**: manipulação e leitura de dados em formato de tabela.
* **streamlit**: criação da interface web do aplicativo.
* **plotly.express**: geração de gráficos interativos.
* **RandomForestRegressor**: algoritmo de machine learning utilizado para prever preços.
* **train_test_split**: divide os dados em treino e teste.
* **r2_score**: métrica usada para avaliar a qualidade do modelo.

---

# 2. Função para carregar os dados

```python
@st.cache_data
def get_data():
    data = pd.read_csv("data.csv")
    return data
```

Essa função carrega o arquivo **data.csv**.

* `pd.read_csv()` lê o arquivo de dados.
* `@st.cache_data` faz o Streamlit guardar o resultado em memória para não precisar recarregar o arquivo sempre.

---

# 3. Função para treinar o modelo

```python
@st.cache_resource
def train_model():
```

Define uma função responsável por treinar o modelo de machine learning.

O decorator `@st.cache_resource` evita que o modelo seja treinado novamente toda vez que o aplicativo atualizar.

---

# 4. Seleção das variáveis

```python
features = ["CRIM","INDUS","CHAS","NOX","RM","PTRATIO"]
```

Define quais colunas do dataset serão usadas como variáveis de entrada para o modelo.

Essas variáveis representam características da região onde o imóvel está localizado.

---

# 5. Separação das variáveis de entrada e saída

```python
X = data[features]
y = data["MEDV"]
```

* **X** contém as variáveis usadas para prever o preço.
* **y** contém o valor real das casas (MEDV).

MEDV representa o valor mediano das casas no dataset.

---

# 6. Divisão entre dados de treino e teste

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

Divide os dados em dois conjuntos:

* **80%** para treinar o modelo
* **20%** para testar o modelo

`random_state=42` garante que a divisão seja sempre a mesma.

---

# 7. Criação do modelo

```python
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)
```

Cria o modelo de machine learning.

Parâmetros utilizados:

* **n_estimators**: número de árvores usadas pelo algoritmo.
* **max_depth**: profundidade máxima das árvores.
* **random_state**: garante resultados reproduzíveis.

---

# 8. Treinamento do modelo

```python
model.fit(X_train, y_train)
```

Essa linha treina o modelo usando os dados de treino.

O algoritmo aprende a relação entre as variáveis e o preço dos imóveis.

---

# 9. Avaliação do modelo

```python
predictions = model.predict(X_test)

score = r2_score(y_test, predictions)
```

O modelo faz previsões usando os dados de teste.

Depois é calculado o **R²**, que mede a qualidade das previsões.

Valores de R² próximos de **1** indicam melhor desempenho.

---

# 10. Interface da aplicação

```python
st.title("Prevendo valores de imóveis")
```

Define o título da aplicação exibido na página.

---

# 11. Exibição dos dados

```python
st.dataframe(data.head())
```

Mostra as primeiras linhas do dataset na interface.

Isso permite visualizar os dados utilizados no modelo.

---

# 12. Criação do gráfico

```python
grafico = px.histogram(data, x="MEDV", nbins=50)

st.plotly_chart(grafico)
```

Cria um histograma mostrando a distribuição dos preços das casas.

---

# 13. Entrada de dados do usuário

```python
crim = st.sidebar.number_input(...)
```

Essas linhas criam campos de entrada na barra lateral da aplicação.

O usuário pode informar características do imóvel para gerar uma previsão.

---

# 14. Conversão da variável CHAS

```python
chas = 1 if chas == "Sim" else 0
```

Converte a resposta do usuário para um valor numérico:

* Sim → 1
* Não → 0

---

# 15. Previsão do preço

```python
entrada = [[crim, indus, chas, nox, rm, ptratio]]

resultado = model.predict(entrada)
```

Cria um vetor com os dados informados pelo usuário e envia para o modelo realizar a previsão.

---

# 16. Exibição do resultado

```python
st.write("US$", round(resultado[0]*1000,2))
```

Mostra o valor estimado do imóvel na tela.

O valor é multiplicado para ajustar a escala do dataset.

---

# Resumo do funcionamento

O fluxo completo da aplicação é:

1. Carregar os dados.
2. Treinar o modelo de machine learning.
3. Mostrar informações do dataset.
4. Receber dados do usuário.
5. Usar o modelo para prever o preço do imóvel.
6. Mostrar o resultado na interface.


## Estrutura do Projeto

A estrutura da pasta deve ficar assim:

```
projetoAssistido
│
├── app.py
├── data.csv
└── README.md
```

---

# Passo a Passo para Executar o Projeto

## 8. Abrir a aplicação

O terminal mostrará:

```
Local URL: http://localhost:8501
```

Abra esse endereço no navegador. arrume isso e me mande em readme 