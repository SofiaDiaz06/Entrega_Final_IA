import joblib
import pandas as pd

def main():
    print("=" * 85)
    print("DEMOSTRACIÓN EN VIVO: SISTEMA DE RECOMENDACIÓN DE ABASTECIMIENTO (CARTAGO)")
    print("=" * 85)

    # 1. Cargar el paquete del modelo guardado (incluye StandardScaler + Modelo ganador)
    paquete = joblib.load("modelo_proyecto.joblib")
    modelo = paquete["pipeline"]
    columnas_entrenamiento = paquete["columnas"]
    print(f"[INFO] Modelo cargado: {paquete['nombre_modelo']}\n")

    # 2. Definir 4 productos nuevos realistas que llegan a cotización al almacén
    casos_nuevos = pd.DataFrame({
        "producto": [
            "Set_Boligrafos_Gel (Alta rotación)",
            "Adorno_Navideño (Estancado)",
            "Agenda_Ejecutiva (Rotación media)",
            "Esmalte_Acrilico_Neon (Alta demanda)"
        ],
        "ganancia_unitaria": [3200, 8500, 6500, 4000],
        "dias_sin_venta": [1, 34, 9, 2],
        "calificacion_cliente": [4.8, 3.8, 4.5, 4.7],
        "stock_actual": [15, 48, 22, 18],
        "categoria": ["Papeleria", "Fiesta", "Papeleria", "Belleza"]
    })

    # 3. Aplicar la misma codificación One-Hot y alinear columnas con el entrenamiento
    X_nuevos = pd.get_dummies(
        casos_nuevos[["ganancia_unitaria", "dias_sin_venta", "calificacion_cliente", "stock_actual", "categoria"]],
        columns=["categoria"],
        prefix="cat",
        dtype=int
    )
    # Rellenar con 0 las categorías que no aparezcan en estos 4 ejemplos nuevos
    X_nuevos = X_nuevos.reindex(columns=columnas_entrenamiento, fill_value=0)

    # 4. Predecir probabilidad y decisión final
    probabilidades = modelo.predict_proba(X_nuevos)[:, 1]
    predicciones = modelo.predict(X_nuevos)

    casos_nuevos["Prob_Abastecer_%"] = (probabilidades * 100).round(1)
    casos_nuevos["Recomendación"] = [
        "ABASTECER (1)" if p == 1 else "NO ABASTECER (0)" for p in predicciones
    ]

    columnas_vista = [
        "producto", "categoria", "ganancia_unitaria", "dias_sin_venta", "Prob_Abastecer_%", "Recomendación"
    ]
    print(casos_nuevos[columnas_vista].to_string(index=False))
    print("\n" + "=" * 85)

if __name__ == "__main__":
    main()