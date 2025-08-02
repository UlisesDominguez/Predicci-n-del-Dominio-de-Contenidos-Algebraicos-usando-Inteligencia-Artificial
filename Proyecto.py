import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Predicción en Álgebra", layout="wide")

# Cargar los datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_algebra.csv")

# Dataset original
ds = cargar_datos()
st.title("Predicción de Dominio en Álgebra con IA")
st.write("Vista previa del dataset:")
st.dataframe(ds.head())

# Codificación de columnas categóricas
ds_encode = ds.copy()
le = LabelEncoder()
ds_encode['Participacion'] = le.fit_transform(ds_encode['Participacion'])
ds_encode['Domina_Algebra'] = le.fit_transform(ds_encode['Domina_Algebra'])

# Variables predictoras y objetivo
X = ds_encode.drop(["ID_Estudiante", "Domina_Algebra"], axis=1)
y = ds_encode["Domina_Algebra"]

# Limpieza de datos
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(X_train, y_train)

# Precisión
score = modelo.score(X_test, y_test)
st.subheader(f"🎯 Precisión del modelo: {score:.2f}")

# Matriz de Confusión
y_pred = modelo.predict(X_test)
mc = confusion_matrix(y_test, y_pred)
st.subheader("📊 Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Importancia de características
importancias = modelo.feature_importances_
st.subheader("🔍 Importancia de las Características")
importancia_ds = pd.DataFrame({"Característica": X.columns, "Importancia": importancias})
st.bar_chart(importancia_ds.set_index("Característica"))

# Formulario de predicción
st.subheader("📝 Formulario de Predicción")
with st.form("formulario"):
    ecuaciones = st.number_input("Ecuaciones correctas", 0, 100, 50)
    inecuaciones = st.number_input("Inecuaciones correctas", 0, 100, 50)
    matrices = st.number_input("Matrices correctas", 0, 100, 50)
    determinantes = st.number_input("Determinantes correctos", 0, 100, 50)
    intentos_prom = st.number_input("Promedio de intentos", 1.0, 10.0, 3.0)
    tiempo_prom = st.number_input("Tiempo promedio por pregunta (min)", 0.0, 20.0, 5.0)
    preguntas_docente = st.slider("Preguntas al docente", 0, 20, 5)
    calificacion = st.slider("Calificación final", 0.0, 10.0, 7.0)
    participacion = st.selectbox("Nivel de Participación", ["Baja", "Media", "Alta"])
    submit = st.form_submit_button("Predecir")

if submit:
    # Mapeo de participación
    participacion_map = {"Baja": 0, "Media": 1, "Alta": 2}

    # Crear entrada de predicción
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

    # Asegurar columnas coincidan
    for col in X.columns:
        if col not in entrada.columns:
            entrada[col] = 0
    entrada = entrada[X.columns]

    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][pred]
    resultado = "Domina" if pred == 1 else "No domina"
    st.success(f"✅ El estudiante **{resultado}** con una probabilidad de {prob:.2f}")