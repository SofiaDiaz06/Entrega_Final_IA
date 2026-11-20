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

    df = pd.read_csv("../data/proyecto_limpio.csv")

    # ============================================================
    # CORRECCIÓN CLAVE: Eliminación de fuga de información (leakage)
    # El target se calcula con (unidades_vendidas, margen_ganancia, dias_sin_venta).
    # Por lo tanto, esas variables NO pueden ser entradas del modelo.
    # Usamos solo variables conocidas ANTES de decidir:
    # precio_venta, costo_compra, stock_actual, calificacion_cliente, categoria
    # ============================================================
    columnas_base = ["precio_venta", "costo_compra", "stock_actual", "calificacion_cliente", "categoria"]
    X = pd.get_dummies(df[columnas_base], columns=["categoria"], prefix="cat", dtype=int)
    y = df["decision_abastecer"]

    columnas_entrenamiento = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Productos para entrenar: {len(X_train)} | Productos para probar: {len(X_test)}")

    modelos = {
        "Regresión Logística": Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", LogisticRegression(random_state=42, max_iter=1000))
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

    # Criterio oficial de la Clase 12: mayor F1
    idx_mejor = tabla["F1"].idxmax()
    nombre_mejor = tabla.loc[idx_mejor, "Modelo"]
    mejor_modelo = modelos[nombre_mejor]
    print(f"\n[GANADOR] Mejor modelo seleccionado por F1: {nombre_mejor}")

    reg = modelos["Regresión Logística"].named_steps["modelo"]
    print("\nCOMPARACIÓN DE PESOS LINEALES (REGRESIÓN LOGÍSTICA VS PERCEPTRÓN):")
    for col, peso in zip(columnas_entrenamiento[:4], reg.coef_[0][:4]):
        direccion = "favorece abastecer (+)" if peso > 0 else "penaliza abastecer (-)"
        print(f"  -> Peso {col:22s}: {peso:+.4f} ({direccion})")
    print(f"  -> Sesgo (intercepto b):        {reg.intercept_[0]:+.4f}")

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