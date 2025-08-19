import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sqlite3
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE  # 👈 NEW

st.set_page_config(page_title="Predicción en Álgebra", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_algebra.csv")

def preprocesar_datos(df: pd.DataFrame):
    df = df.copy()
    # y = 0/1 (ojo con el acento en "Sí")
    y = df["Domina_Algebra"].map({"No": 0, "Sí": 1})
    # X sin ID, sin target y SIN Participacion (en tu CSV no es % real)
    X = df.drop(columns=["ID_Estudiante", "Domina_Algebra", "Participacion"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean(numeric_only=True))
    return X, y

def entrenar_modelo(X_train, y_train, metodo_balanceo: str):
    if metodo_balanceo == "SMOTE":
        smote = SMOTE(random_state=0)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        modelo = RandomForestClassifier(
            n_estimators=200,
            random_state=0
        )
        modelo.fit(X_train, y_train)
        info = "Entrenado con SMOTE (oversampling de la clase 'Sí')."

    elif metodo_balanceo == "class_weight":
        modelo = RandomForestClassifier(
            n_estimators=200,
            random_state=0,
            class_weight="balanced"
        )
        modelo.fit(X_train, y_train)
        info = "Entrenado con class_weight='balanced'."

    else:
        modelo = RandomForestClassifier(
            n_estimators=200,
            random_state=0
        )
        modelo.fit(X_train, y_train)
        info = "Entrenado SIN balanceo."

    return modelo, info

def evaluar_modelo_con_umbral(modelo, X_test, y_test, umbral: float):
    # Probabilidades de clase 1 (Domina)
    probas = modelo.predict_proba(X_test)[:, list(modelo.classes_).index(1)]
    y_pred_thr = (probas >= umbral).astype(int)

    score = (y_pred_thr == y_test).mean()
    matriz = confusion_matrix(y_test, y_pred_thr)
    reporte = classification_report(
        y_test, y_pred_thr, target_names=["No domina", "Domina"], output_dict=False
    )
    return score, matriz, reporte

def mostrar_importancia(modelo, columnas):
    importancias = modelo.feature_importances_
    imp = pd.DataFrame({"Característica": columnas, "Importancia": importancias}).sort_values("Importancia", ascending=False)
    st.subheader("🔍 Importancia de las Características")
    st.bar_chart(imp.set_index("Característica"))

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
            preguntas_docente INTEGER,
            calificacion REAL,
            resultado TEXT,
            probabilidad REAL
        )
    """)
    cursor.execute("""
        INSERT INTO predicciones (
            fecha, ecuaciones, inecuaciones, matrices, determinantes,
            intentos_prom, tiempo_prom, preguntas_docente,
            calificacion, resultado, probabilidad
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["Ecuaciones"],
        data["Inecuaciones"],
        data["Matrices"],
        data["Determinantes"],
        data["Intentos_Promedio"],
        data["Tiempo_Promedio"],
        data["Preguntas_Al_Docente"],
        data["Calificacion_Final"],
        data["Resultado"],
        data["Probabilidad"]
    ))
    conn.commit()
    conn.close()

def formulario_prediccion(X, modelo, umbral: float):
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
        submit = st.form_submit_button("Predecir")

    if submit:
        entrada = pd.DataFrame([{
            "Ecuaciones": ecuaciones,
            "Inecuaciones": inecuaciones,
            "Matrices": matrices,
            "Determinantes": determinantes,
            "Intentos_Promedio": intentos_prom,
            "Tiempo_Promedio": tiempo_prom,
            "Preguntas_Al_Docente": preguntas_docente,
            "Calificacion_Final": calificacion
        }]).astype(float)

        # asegurar orden idéntico al entrenamiento
        entrada = entrada[X.columns]

        # Probabilidad clase 1 (Domina)
        idx_cls1 = list(modelo.classes_).index(1)
        prob_domina = float(modelo.predict_proba(entrada)[0][idx_cls1])

        # Decisión con umbral configurable
        if prob_domina >= umbral:
            resultado = "Domina"
            prob_mostrada = prob_domina
        else:
            resultado = "No domina"
            prob_mostrada = 1 - prob_domina

        st.success(f"✅ El estudiante **{resultado}** con una probabilidad de {prob_mostrada:.2f}")

        datos_guardar = {
            "Ecuaciones": ecuaciones,
            "Inecuaciones": inecuaciones,
            "Matrices": matrices,
            "Determinantes": determinantes,
            "Intentos_Promedio": intentos_prom,
            "Tiempo_Promedio": tiempo_prom,
            "Preguntas_Al_Docente": preguntas_docente,
            "Calificacion_Final": calificacion,
            "Resultado": resultado,
            "Probabilidad": prob_mostrada
        }
        guardar_en_base_de_datos(datos_guardar)

def mostrar_historial_exportar():
    conn = sqlite3.connect("resultados.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predicciones'")
    existe = cursor.fetchone()
    if existe:
        dfh = pd.read_sql_query("SELECT * FROM predicciones ORDER BY fecha DESC", conn)
        st.subheader("📋 Historial de Predicciones")
        st.dataframe(dfh)
        st.download_button("📥 Descargar como CSV", data=dfh.to_csv(index=False).encode("utf-8"),
                           file_name="historial_predicciones.csv", mime="text/csv")
    else:
        st.info("Aún no hay predicciones guardadas.")
    conn.close()

def main():
    st.title("Predicción de Dominio en Álgebra con IA")

    # --------- Sidebar: controles ----------
    st.sidebar.header("⚙️ Configuración")
    metodo_balanceo = st.sidebar.selectbox(
        "Método de balanceo",
        options=["SMOTE", "class_weight", "none"],
        format_func=lambda x: {"SMOTE":"SMOTE (oversampling)", "class_weight":"Class weight", "none":"Sin balanceo"}[x]
    )
    umbral = st.sidebar.slider("Umbral para 'Domina'", 0.05, 0.95, 0.35, 0.01)
    st.sidebar.caption("Con datasets desbalanceados, valores entre 0.25 y 0.40 suelen mejorar el recall de 'Domina'.")

    ds = cargar_datos()
    st.write("Vista previa del dataset:")
    st.dataframe(ds.head())

    X, y = preprocesar_datos(ds)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    modelo, info = entrenar_modelo(X_train, y_train, metodo_balanceo)
    st.info(info)

    # Evaluación con el umbral elegido
    score, matriz, reporte = evaluar_modelo_con_umbral(modelo, X_test, y_test, umbral)
    st.subheader(f"🎯 Accuracy (con umbral {umbral:.2f}): {score:.2f}")

    st.subheader("📊 Matriz de Confusión")
    fig, ax = plt.subplots()
    im = ax.imshow(matriz, interpolation="nearest")
    ax.set_title("Matriz de Confusión (umbral aplicado)")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No domina", "Domina"], rotation=45)
    ax.set_yticklabels(["No domina", "Domina"])
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, format(matriz[i, j], "d"), ha="center", va="center")
    ax.set_ylabel("Real"); ax.set_xlabel("Predicción")
    st.pyplot(fig)

    mostrar_importancia(modelo, X.columns)
    with st.expander("📄 Classification report (con umbral)"):
        st.text(reporte)

    # Diagnóstico rápido
    with st.expander("🔧 Diagnóstico rápido"):
        st.write("Distribución y (0=No, 1=Sí):", y.value_counts().to_dict())
        st.write("Orden de clases del modelo:", list(modelo.classes_))
        cols = list(X.columns)
        fila_extrema = pd.DataFrame([{
            "Ecuaciones": 100, "Inecuaciones": 100, "Matrices": 100, "Determinantes": 100,
            "Intentos_Promedio": 1.0, "Tiempo_Promedio": 1.0, "Preguntas_Al_Docente": 0, "Calificacion_Final": 10.0
        }])[cols]
        idx1 = list(modelo.classes_).index(1)
        p_ext = float(modelo.predict_proba(fila_extrema)[0][idx1])
        pred_ext = int(p_ext >= umbral)
        st.write(f"Fila extrema → prob_domina: {p_ext:.3f}  |  pred (umbral {umbral:.2f}): {pred_ext}")

        ej_si = ds[ds["Domina_Algebra"] == "Sí"].iloc[0]
        fila_real = ej_si.drop(labels=["ID_Estudiante", "Domina_Algebra"]).to_frame().T
        if "Participacion" in fila_real.columns and "Participacion" not in X.columns:
            fila_real = fila_real.drop(columns=["Participacion"])
        fila_real = fila_real[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        p_real = float(modelo.predict_proba(fila_real)[0][idx1])
        pred_real = int(p_real >= umbral)
        st.write(f"Fila real 'Sí' → prob_domina: {p_real:.3f}  |  pred (umbral {umbral:.2f}): {pred_real}")

    formulario_prediccion(X, modelo, umbral)
    mostrar_historial_exportar()

if __name__ == "__main__":
    main()