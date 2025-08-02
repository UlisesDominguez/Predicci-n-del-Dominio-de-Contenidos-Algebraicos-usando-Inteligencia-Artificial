import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Predicción en Álgebra", layout="wide")

# Cargar datos con cache
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_algebra.csv")

# Cargar dataset
ds = cargar_datos()

st.title("Predicción de Riesgo de Dominio en Temas de Álgebra")
st.write("Vista previa del dataset:")
st.dataframe(ds.head())

# Codificación
ds_encode = ds.copy()
label_cols = ['Participacion', 'Domina_Algebra']
le = LabelEncoder()
for col in label_cols:
    ds_encode[col] = le.fit_transform(ds_encode[col])

# Definir variables
X = ds_encode.drop("Domina_Algebra", axis=1)
y = ds_encode["Domina_Algebra"]

# Convertir todas las columnas a numérico
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(X_train, y_train)

# Evaluación
score = modelo.score(X_test, y_test)
st.subheader(f"Precisión del modelo: {score:.2f}")

# Matriz de confusión
y_pred = modelo.predict(X_test)
mc = confusion_matrix(y_test, y_pred)
st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Importancia de características
importancias = modelo.feature_importances_
st.subheader("Importancia de las características")
st.bar_chart(pd.DataFrame({
    "Característica": X.columns,
    "Importancia": importancias
}).set_index("Característica"))

# Formulario
st.subheader("Formulario de Predicción")
with st.form("formulario"):
    ecuaciones = st.number_input("Ecuaciones correctas", 0, 100, 50)
    inecuaciones = st.number_input("Inecuaciones correctas", 0, 100, 50)
    matrices = st.number_input("Matrices correctas", 0, 100, 50)
    determinantes = st.number_input("Determinantes correctos", 0, 100, 50)
    intentos_prom = st.number_input("Promedio de intentos", 1.0, 10.0, 3.0)
    tiempo_prom = st.number_input("Tiempo promedio (min)", 0.0, 20.0, 5.0)
    preguntas_docente = st.slider("Preguntas al docente", 0, 20, 5)
    calificacion = st.slider("Calificación final", 0.0, 10.0, 7.0)
    participacion = st.selectbox("Nivel de Participación", ["Baja", "Media", "Alta"])
    submit = st.form_submit_button("Predecir")

if submit:
    participacion_map = {"Baja": 0, "Media": 1, "Alta": 2}
    entrada = pd.DataFrame([{
        "Ecuaciones": ecuaciones,
        "Inecuaciones": inecuaciones,
        "Matrices": matrices,
        "Determinantes": determinantes,
        "Intentos_Promedio": intentos_prom,
        "Tiempo_Promedio": tiempo_prom,
        "Participacion": participacion_map[participacion],
        "Preguntas_Al_Docente": preguntas_docente,
        "Calificacion_Final": calificacion
    }])

    # Reordenar columnas según entrenamiento
    entrada = entrada[X.columns]

    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][pred]
    resultado = "Domina" if pred == 1 else "No domina"
    st.success(f"El estudiante {resultado} con una probabilidad de {prob:.2f}")