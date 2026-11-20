import joblib
import pandas as pd


def main():
    print("=" * 85)
    print("DEMOSTRACIÓN EN VIVO: SISTEMA DE RECOMENDACIÓN DE ABASTECIMIENTO (CARTAGO)")
    print("=" * 85)

    paquete = joblib.load("modelo_proyecto.joblib")
    modelo = paquete["pipeline"]
    columnas_entrenamiento = paquete["columnas"]
    print(f"[INFO] Modelo cargado: {paquete['nombre_modelo']}\n")

    # Casos nuevos: solo variables conocidas ANTES de decidir
    casos_nuevos = pd.DataFrame({
        "producto": [
            "Set_Boligrafos_Gel (Alta rotación esperada)",
            "Adorno_Navideño (Riesgo estancamiento)",
            "Agenda_Ejecutiva (Rotación media)",
            "Esmalte_Acrilico_Neon (Alta demanda)"
        ],
        "precio_venta": [6000, 12000, 15000, 8000],
        "costo_compra": [2800, 7000, 8500, 3500],
        "stock_actual": [40, 55, 22, 18],
        "calificacion_cliente": [4.7, 3.8, 4.5, 4.7],
        "categoria": ["Papeleria", "Fiesta", "Papeleria", "Belleza"]
    })

    X_nuevos = pd.get_dummies(
        casos_nuevos[["precio_venta", "costo_compra", "stock_actual", "calificacion_cliente", "categoria"]],
        columns=["categoria"],
        prefix="cat",
        dtype=int
    )
    X_nuevos = X_nuevos.reindex(columns=columnas_entrenamiento, fill_value=0)

    probabilidades = modelo.predict_proba(X_nuevos)[:, 1]
    predicciones = modelo.predict(X_nuevos)

    casos_nuevos["Prob_Abastecer_%"] = (probabilidades * 100).round(1)
    casos_nuevos["Recomendación"] = [
        "ABASTECER (1)" if p == 1 else "NO ABASTECER (0)" for p in predicciones
    ]

    columnas_vista = [
        "producto", "categoria", "precio_venta", "stock_actual", "Prob_Abastecer_%", "Recomendación"
    ]
    print(casos_nuevos[columnas_vista].to_string(index=False))
    print("\n" + "=" * 85)


if __name__ == "__main__":
    main()