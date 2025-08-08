import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Predicción en Álgebra", layout="wide")


# Cargar los datos

@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_algebra.csv")


# Preprocesar los datos

def preprocesar_datos(df):
    le = LabelEncoder()
    df = df.copy()
    df["Participacion"] = le.fit_transform(df["Participacion"])
    df["Domina_Algebra"] = le.fit_transform(df["Domina_Algebra"])
    X = df.drop(["ID_Estudiante", "Domina_Algebra"], axis=1)
    y = df["Domina_Algebra"]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())
    return X, y

# Entrenar el modelo

def entrenar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=0)
    modelo.fit(X_train, y_train)
    return modelo


# Evaluar el modelo

def evaluar_modelo(modelo, X_test, y_test):
    score = modelo.score(X_test, y_test)
    y_pred = modelo.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred)
    return score, matriz


# Mostrar importancia de variables

def mostrar_importancia(modelo, columnas):
    importancias = modelo.feature_importances_
    df_imp = pd.DataFrame({
        "Característica": columnas,
        "Importancia": importancias
    })
    st.subheader("🔍 Importancia de las Características")
    st.bar_chart(df_imp.set_index("Característica"))


# Guardar resultado en base de datos y CSV

def guardar_en_base_de_datos(data: dict):
    conn = sqlite3.connect("resultados.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            ecuaciones INTEGER,
            inecuaciones INTEGER,
            matrices INTEGER,
            determinantes INTEGER,
            intentos_prom REAL,
            tiempo_prom REAL,
            participacion TEXT,
            preguntas_docente INTEGER,
            calificacion REAL,
            resultado TEXT,
            probabilidad REAL
        )
    """)

    cursor.execute("""
        INSERT INTO predicciones (
            fecha, ecuaciones, inecuaciones, matrices, determinantes,
            intentos_prom, tiempo_prom, participacion, preguntas_docente,
            calificacion, resultado, probabilidad
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["Ecuaciones"],
        data["Inecuaciones"],
        data["Matrices"],
        data["Determinantes"],
        data["Intentos_Promedio"],
        data["Tiempo_Promedio"],
        data["Participacion_Texto"],
        data["Preguntas_Al_Docente"],
        data["Calificacion_Final"],
        data["Resultado"],
        data["Probabilidad"]
    ))

    conn.commit()
    conn.close()

    # Guardar en CSV externo
    archivo_csv = "resultados.csv"
    fila = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ecuaciones": data["Ecuaciones"],
        "inecuaciones": data["Inecuaciones"],
        "matrices": data["Matrices"],
        "determinantes": data["Determinantes"],
        "intentos_prom": data["Intentos_Promedio"],
        "tiempo_prom": data["Tiempo_Promedio"],
        "participacion": data["Participacion_Texto"],
        "preguntas_docente": data["Preguntas_Al_Docente"],
        "calificacion": data["Calificacion_Final"],
        "resultado": data["Resultado"],
        "probabilidad": data["Probabilidad"]
    }
    escribir_encabezado = not os.path.exists(archivo_csv)
    df_nueva = pd.DataFrame([fila])
    df_nueva.to_csv(archivo_csv, mode='a', header=escribir_encabezado, index=False)

# Formulario de predicción

def formulario_prediccion(X, modelo):
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

        for col in X.columns:
            if col not in entrada.columns:
                entrada[col] = 0
        entrada = entrada[X.columns]

        pred = modelo.predict(entrada)[0]
        prob = modelo.predict_proba(entrada)[0][pred]
        resultado = "Domina" if pred == 1 else "No domina"
        st.success(f"✅ El estudiante **{resultado}** con una probabilidad de {prob:.2f}")

        datos_guardar = {
            "Ecuaciones": ecuaciones,
            "Inecuaciones": inecuaciones,
            "Matrices": matrices,
            "Determinantes": determinantes,
            "Intentos_Promedio": intentos_prom,
            "Tiempo_Promedio": tiempo_prom,
            "Participacion_Texto": participacion,
            "Preguntas_Al_Docente": preguntas_docente,
            "Calificacion_Final": calificacion,
            "Resultado": resultado,
            "Probabilidad": prob
        }
        guardar_en_base_de_datos(datos_guardar)

# Mostrar y exportar historial de predicciones

def mostrar_historial_exportar():
    conn = sqlite3.connect("resultados.db")
    cursor = conn.cursor()

    # Verificamos si la tabla existe
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='predicciones'
    """)
    existe = cursor.fetchone()

    if existe:
        df = pd.read_sql_query("SELECT * FROM predicciones ORDER BY fecha DESC", conn)
        st.subheader("📋 Historial de Predicciones")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar como CSV",
            data=csv,
            file_name="historial_predicciones.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no hay predicciones guardadas.")

    conn.close()


# App principal

def main():
    st.title("Predicción de Dominio en Álgebra con IA")

    ds = cargar_datos()
    st.write("Vista previa del dataset:")
    st.dataframe(ds.head())

    X, y = preprocesar_datos(ds)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    modelo = entrenar_modelo(X_train, y_train)
    score, matriz = evaluar_modelo(modelo, X_test, y_test)
    st.subheader(f"🎯 Precisión del modelo: {score:.2f}")

    st.subheader("📊 Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    mostrar_importancia(modelo, X.columns)
    formulario_prediccion(X, modelo)
    mostrar_historial_exportar()

if __name__ == "__main__":
    main()