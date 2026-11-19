import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

os.makedirs("../graficos", exist_ok=True)


def main():
    print("=" * 80)
    print("FASE 4A: MODELO FINAL DEL PROYECTO CON SCIKIT-LEARN - ALMACÉN CARTAGO")
    print("=" * 80)

    # 1. Cargar los datos limpios del Avance 2
    df = pd.read_csv("../data/proyecto_limpio.csv")

    # 2. Seleccionar entradas (X) y salida (y) sin fuga de datos (Data Leakage)
    columnas_base = ["ganancia_unitaria", "dias_sin_venta", "calificacion_cliente", "stock_actual", "categoria"]
    X = pd.get_dummies(df[columnas_base], columns=["categoria"], prefix="cat", dtype=int)
    y = df["decision_abastecer"]

    # Guardamos el orden exacto de las columnas para usarlas en usar_modelo.py
    columnas_entrenamiento = X.columns.tolist()

    # 3. Separar datos: 80% entrenamiento y 20% prueba (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Productos para entrenar: {len(X_train)} | Productos para probar: {len(X_test)}")

    # 4. Definir los 3 modelos dentro de un Pipeline (Escalado + Clasificador)
    modelos = {
        "Regresión Logística": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", LogisticRegression(random_state=42))
        ]),
        "KNN (3 vecinos)": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", KNeighborsClassifier(n_neighbors=3))
        ]),
        "Árbol de Decisión (prof=4)": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42))
        ])
    }

    # 5. Entrenar cada modelo, medir en prueba y ejecutar validación cruzada (cv=5)
    resultados = []
    for nombre, pipeline in modelos.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        cv_score = cross_val_score(pipeline, X, y, cv=5).mean()

        resultados.append({
            "Modelo": nombre,
            "Exactitud": accuracy_score(y_test, y_pred),
            "Precisión": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "Val. cruzada (5-fold)": cv_score
        })

    tabla = pd.DataFrame(resultados)
    print("\nCOMPARACIÓN DE MODELOS (DATOS DE PRUEBA Y VALIDACIÓN CRUZADA):")
    print("-" * 80)
    print(tabla.round(3).to_string(index=False))
    print("-" * 80)

    # 6. Elegir el mejor modelo según F1 y Val. cruzada
    tabla["Puntaje_Global"] = (tabla["F1"] + tabla["Val. cruzada (5-fold)"]) / 2
    idx_mejor = tabla["Puntaje_Global"].idxmax()
    nombre_mejor = tabla.loc[idx_mejor, "Modelo"]
    mejor_modelo = modelos[nombre_mejor]
    print(f"\n[GANADOR] Mejor modelo seleccionado: {nombre_mejor}")

    # 7. Comparación de pesos lineales (Regresión Logística vs Perceptrón manual)
    reg = modelos["Regresión Logística"].named_steps["modelo"]
    print("\nCOMPARACIÓN DE PESOS LINEALES (REGRESIÓN LOGÍSTICA VS PERCEPTRÓN):")
    for col, peso in zip(columnas_entrenamiento[:4], reg.coef_[0][:4]):
        direccion = "favorece abastecer (+)" if peso > 0 else "penaliza abastecer (-)"
        print(f"  -> Peso {col:22s}: {peso:+.4f} ({direccion})")
    print(f"  -> Sesgo (intercepto b):        {reg.intercept_[0]:+.4f}")

    # 8. Matriz de confusión del mejor modelo
    matriz = confusion_matrix(y_test, mejor_modelo.predict(X_test))
    print("\nMatriz de confusión (filas = valor real, columnas = predicción):")
    print(matriz)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predijo 0 (No Abastecer)", "Predijo 1 (Abastecer)"],
        yticklabels=["Real 0 (No Abastecer)", "Real 1 (Abastecer)"]
    )
    plt.title(f"Matriz de Confusión - {nombre_mejor}")
    plt.tight_layout()
    plt.savefig("../graficos/7_matriz_confusion.png", dpi=150)
    plt.close()
    print("\n[OK] Gráfico guardado: ../graficos/7_matriz_confusion.png")

    # 9. Guardar el Pipeline completo y las columnas esperadas con joblib
    paquete_modelo = {
        "pipeline": mejor_modelo,
        "columnas": columnas_entrenamiento,
        "nombre_modelo": nombre_mejor
    }
    ruta_salida = "modelo_proyecto.joblib"
    joblib.dump(paquete_modelo, ruta_salida)
    print(f"[EXITO] Modelo guardado en: {os.path.abspath(ruta_salida)}")


if __name__ == "__main__":
    main()