import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# -----------------------------
# carregar dados
# -----------------------------
@st.cache_data
def get_data():
    data = pd.read_csv("data.csv")
    return data


# -----------------------------
# treinar modelo
# -----------------------------
@st.cache_resource
def train_model():

    data = get_data()

    features = ["CRIM","INDUS","CHAS","NOX","RM","PTRATIO"]

    X = data[features]
    y = data["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    return model, score


# carregar dados
data = get_data()

# treinar modelo
model, score = train_model()


# -----------------------------
# interface
# -----------------------------
st.title("Prevendo valores de imóveis")

st.markdown("Aplicação de Machine Learning usando dataset Boston Housing.")


# mostrar qualidade do modelo
st.subheader("Precisão do modelo")

st.write("R²:", round(score,3))


# mostrar dados
st.subheader("Visualização do dataset")

st.dataframe(data.head())


# gráfico
st.subheader("Distribuição de preços")

grafico = px.histogram(data, x="MEDV", nbins=50)

st.plotly_chart(grafico)


# -----------------------------
# entrada do usuário
# -----------------------------
st.sidebar.header("Dados do imóvel")

crim = st.sidebar.number_input(
    "Taxa de criminalidade [CRIM]",
    value=float(data.CRIM.mean())
)

indus = st.sidebar.number_input(
    "Proporção de indústrias [INDUS]",
    value=float(data.INDUS.mean())
)

chas = st.sidebar.selectbox(
    "Faz limite com rio? [CHAS]",
    ("Sim","Não")
)

chas = 1 if chas == "Sim" else 0

nox = st.sidebar.number_input(
    "Óxido nítrico [NOX]",
    value=float(data.NOX.mean())
)

rm = st.sidebar.number_input(
    "Número de quartos [RM]",
    value=float(data.RM.mean())
)

ptratio = st.sidebar.number_input(
    "Índice alunos/professores",
    value=float(data.PTRATIO.mean())
)


# -----------------------------
# previsão
# -----------------------------
if st.sidebar.button("Prever preço"):

    entrada = [[crim, indus, chas, nox, rm, ptratio]]

    resultado = model.predict(entrada)

    st.subheader("Valor previsto do imóvel")

    st.write("US$", round(resultado[0]*1000,2))



# -----------------------------
# Citações
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido por Victor Gabryel da Silva.")