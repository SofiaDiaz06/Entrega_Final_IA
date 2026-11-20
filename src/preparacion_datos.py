import os
import numpy as np
import pandas as pd

def cargar_datos(ruta_csv):
    """Carga el dataset original con manejo de excepciones."""
    try:
        df = pd.read_csv(ruta_csv)
        print(f"[OK] Archivo cargado: {df.shape[0]} filas y {df.shape[1]} columnas.")
        return df
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo en: {ruta_csv}")
        return None
    except Exception as e:
        print(f"[ERROR] Ocurrió un problema al leer el CSV: {e}")
        return None


def limpiar_y_transformar(df):
    """Limpia erratas, verifica nulos y crea variables financieras del negocio."""
    df_limpio = df.copy()

    # 1. Revisión de valores nulos
    nulos = df_limpio.isnull().sum().sum()
    print(f"[INFO] Valores nulos detectados: {nulos}")
    if nulos > 0:
        df_limpio = df_limpio.fillna(df_limpio.median(numeric_only=True))

    # 2. Corrección de errores tipográficos en categorías
    df_limpio["categoria"] = df_limpio["categoria"].str.strip().replace({
        "Juquetes": "Juguetes"
    })
    print("[OK] Categorías estandarizadas:", df_limpio["categoria"].unique().tolist())

    # 3. Creación de columnas derivadas (Ingeniería de Características)
    df_limpio["ganancia_unitaria"] = df_limpio["precio_venta"] - df_limpio["costo_compra"]
    df_limpio["margen_ganancia"] = ((df_limpio["ganancia_unitaria"] / df_limpio["precio_venta"]) * 100).round(2)
    df_limpio["rotacion_diaria"] = (df_limpio["unidades_vendidas"] / 30).round(2)

    # 4. Variable objetivo (Target): 1 = Abastecer, 0 = No abastecer
    condicion_principal = (
        (df_limpio["unidades_vendidas"] >= 85)
        & (df_limpio["margen_ganancia"] >= 35)
        & (df_limpio["dias_sin_venta"] <= 10)
    )
    condicion_alta_rotacion = (
        (df_limpio["unidades_vendidas"] >= 120)
        & (df_limpio["margen_ganancia"] >= 30)
        & (df_limpio["dias_sin_venta"] <= 10)
    )
    df_limpio["decision_abastecer"] = np.where(
        condicion_principal | condicion_alta_rotacion,
        1,
        0
    )

    print("[OK] Variables derivadas y etiqueta 'decision_abastecer' creadas.")
    print(f"     Distribución de clases: {df_limpio['decision_abastecer'].value_counts().to_dict()}")
    return df_limpio


def preparar_para_ml(df_limpio):
    """Aplica One-Hot Encoding y elimina identificadores para dejar listo para ML."""
    df_ml = df_limpio.copy()
    df_ml = pd.get_dummies(df_ml, columns=["categoria"], prefix="cat", dtype=int)
    df_ml = df_ml.drop(columns=["id_producto", "nombre"])
    print(f"[OK] Dataset preparado para ML con {df_ml.shape[1]} columnas numéricas.")
    return df_ml


def main():
    print("=" * 65)
    print("FASE 2A: LIMPIEZA Y PREPARACIÓN DE DATOS - ALMACÉN CARTAGO")
    print("=" * 65)

    ruta_origen = "../data/datos_productos.csv"
    df_raw = cargar_datos(ruta_origen)
    if df_raw is None:
        return

    df_limpio = limpiar_y_transformar(df_raw)
    df_ml = preparar_para_ml(df_limpio)

    df_limpio.to_csv("../data/proyecto_limpio.csv", index=False)
    df_ml.to_csv("../data/proyecto_ml.csv", index=False)

    print("\n[EXITO] Archivos guardados en ../data/:")
    print("  1. proyecto_limpio.csv (con categorías limpias y métricas de negocio)")
    print("  2. proyecto_ml.csv (con One-Hot Encoding listo para algoritmos)")


if __name__ == "__main__":
    main()