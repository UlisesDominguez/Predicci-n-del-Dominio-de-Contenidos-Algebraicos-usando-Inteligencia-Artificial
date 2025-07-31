import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Mostrar o cargar los datos
@st.cache_data
def cargar_datos():
    # Cambia la ruta por donde tengas guardado tu CSV
    ds = pd.read_csv("dataset_algebra.csv")
    return ds

# Cargar los datos
ds = cargar_datos()

st.title("Predicción de Riesgo de Dominio en Temas de Álgebra")
st.write("Vista previa de los datos:")
st.dataframe(ds.head())

# Procesamiento de datos
ds_encode = ds.copy()

# Asegúrate que las columnas 'Historial_Credito' y 'Nivel_Educacion' existan en tu dataset
# y sea necesario codificarlas
label_cols = ['Participacion', 'Domina_Algebra']
le = LabelEncoder()
for col in label_cols:
    ds_encode[col] = le.fit_transform(ds_encode[col])

# Define tus variables predictoras y la variable objetivo
# Cambia 'Riesgo_Financiero' por el nombre correcto de la columna en tu dataset
x = ds_encode.drop("Domina_Algebra", axis=1)
y = ds_encode["Domina_Algebra"]
# Si 'Riesgo_Financiero' aún no está codificada, también la codificamos
y = LabelEncoder().fit_transform(y)

# Dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(x_train, y_train)
score = modelo.score(x_test, y_test)

st.subheader(f"Precisión del modelo: {score:.2f}")

# Matriz de confusión
y_pred = modelo.predict(x_test)
mc = confusion_matrix(y_test, y_pred)
st.subheader('Matriz de Confusión')
fig, ax = plt.subplots()
sns.heatmap(mc, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Importancia de las características
importancias = modelo.feature_importances_
st.subheader("Importancia de las características")
importancia_ds = pd.DataFrame({"Característica": x.columns, "Importancia": importancias})
st.bar_chart(importancia_ds.set_index("Característica"))

# Formulario para predicción
st.subheader("Formulario de Predicción")
with st.form("formulario"):
    ecuaciones = st.number_input("Cantidad de Ecuaciones Correctas", min_value=0, max_value=100, value=50)
    inecuaciones = st.number_input("Cantidad de Inecuaciones Correctas", min_value=0, max_value=100, value=50)
    matrices = st.number_input("Cantidad de Matrices Correctas", min_value=0, max_value=100, value=50)
    determinantes = st.number_input("Cantidad de Determinantes Correctos", min_value=0, max_value=100, value=50)
    intentos_promedio = st.number_input("Promedio de Intentos por Pregunta", min_value=1.0, max_value=10.0, value=3.0)
    tiempo_promedio = st.number_input("Tiempo Promedio por Pregunta (minutos)", min_value=0.0, max_value=20.0, value=5.0)
    preguntas_docente = st.slider("Preguntas al Docente (número)", min_value=0, max_value=20, value=5)
    calificacion_final = st.slider("Calificación Final", min_value=0.0, max_value=10.0, value=7.0)
    # Añadimos el selectbox para nivel de participación
    participacion = st.selectbox(
        "Nivel de Participación",
        ("Baja", "Media", "Alta")
    )
    submit = st.form_submit_button("Predecir")

if submit:
    # Mapear la participación a un valor numérico
    participacion_mapping = {"Baja": 0, "Media": 1, "Alta": 2}
    participacion_cod = participacion_mapping[participacion]
    
    # Crear el DataFrame de entrada con los datos ingresados
    entrada = pd.DataFrame([{
        'Ecuaciones': ecuaciones,
        'Inecuaciones': inecuaciones,
        'Matrices': matrices,
        'Determinantes': determinantes,
        'Intentos_Promedio': intentos_promedio,
        'Tiempo_Promedio': tiempo_promedio,
        'Participacion': participacion_cod,
        'Preguntas_Al_Docente': preguntas_docente,
        'Calificacion_Final': calificacion_final
        # Añade más columnas si tu dataset las requiere
    }])
    
    # Asegúrate que los nombres de columnas en 'entrada' coincidan exactamente con los utilizados en tu entrenamiento
    pred = modelo.predict(entrada)[0]
    riesgo = {0: "No domina", 1: "Domina"}.get(pred, "Desconocido")
    prob = modelo.predict_proba(entrada)[0][pred]
    st.success(f"El estudiante {riesgo} con una probabilidad de {prob:.2f}")