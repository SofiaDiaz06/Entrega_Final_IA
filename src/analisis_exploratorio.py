import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo visual profesional
sns.set_theme(style="whitegrid")
os.makedirs("../graficos", exist_ok=True)
os.makedirs("../docs", exist_ok=True)

def ejecutar_eda():
    print("=" * 65)
    print("FASE 2B: ANÁLISIS EXPLORATORIO DE DATOS (EDA) Y VISUALIZACIÓN")
    print("=" * 65)

    # 1. Cargar dataset limpio
    df = pd.read_csv("../data/proyecto_limpio.csv")

    print("\n1. PRIMERAS 5 FILAS DEL DATASET LIMPIO:")
    print(df.head().to_string(index=False))

    print("\n2. INFORMACIÓN DE TIPOS DE DATOS:")
    print(df.info())

    print("\n3. ESTADÍSTICAS DESCRIPTIVAS GENERALES:")
    print(df.describe().round(2).to_string())

    # 4. Cálculos vectorizados con NumPy (Requisito Clase 4)
    unidades = df["unidades_vendidas"].to_numpy()
    ganancias = df["ganancia_unitaria"].to_numpy()
    dias_sin_venta = df["dias_sin_venta"].to_numpy()

    stats_numpy = {
        "Unidades Vendidas": {
            "media": np.mean(unidades),
            "mediana": np.median(unidades),
            "std": np.std(unidades),
            "min": np.min(unidades),
            "max": np.max(unidades),
        },
        "Ganancia Unitaria (COP)": {
            "media": np.mean(ganancias),
            "mediana": np.median(ganancias),
            "std": np.std(ganancias),
            "min": np.min(ganancias),
            "max": np.max(ganancias),
        },
        "Días Sin Venta": {
            "media": np.mean(dias_sin_venta),
            "mediana": np.median(dias_sin_venta),
            "std": np.std(dias_sin_venta),
            "min": np.min(dias_sin_venta),
            "max": np.max(dias_sin_venta),
        }
    }

    # Agrupación por categoría (Requisito Clase 7)
    resumen_cat = df.groupby("categoria")[["unidades_vendidas", "margen_ganancia", "dias_sin_venta"]].mean().round(2)
    print("\n4. PROMEDIOS POR CATEGORÍA:")
    print(resumen_cat)

    # 5. Generar informe en Markdown (Requisito Clase 3)
    with open("../docs/reporte_eda.md", "w", encoding="utf-8") as f:
        f.write("# Reporte de Análisis Exploratorio de Datos (EDA)\n\n")
        f.write("**Contexto:** Almacén de variedades en el centro de Cartago, Valle del Cauca.\n\n")
        f.write("## Estadísticas Clave (NumPy)\n\n")
        for var, vals in stats_numpy.items():
            f.write(f"### {var}\n")
            f.write(f"- **Media:** {vals['media']:.2f}\n")
            f.write(f"- **Mediana:** {vals['mediana']:.2f}\n")
            f.write(f"- **Desviación Estándar:** {vals['std']:.2f}\n")
            f.write(f"- **Rango (Mín - Máx):** {vals['min']:.2f} - {vals['max']:.2f}\n\n")
        f.write("## Promedios por Categoría\n\n")
        f.write(resumen_cat.to_markdown() + "\n")
    print("\n[OK] Informe generado en: ../docs/reporte_eda.md")

    # =========================================================================
    # 6. GENERACIÓN DE GRÁFICOS CON SEABORN (Requisito Clase 7)
    # =========================================================================

    # Gráfico 1: Histograma de Unidades Vendidas
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="unidades_vendidas", bins=8, kde=True, color="#2b5c8f")
    plt.axvline(df["unidades_vendidas"].mean(), color="red", linestyle="--", label=f"Media ({df['unidades_vendidas'].mean():.1f})")
    plt.title("Distribución de Unidades Vendidas Mensuales")
    plt.xlabel("Unidades Vendidas")
    plt.ylabel("Cantidad de Productos")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../graficos/1_distribucion_ventas.png", dpi=150)
    plt.close()

    # Gráfico 2: Gráfico de Barras - Unidades Vendidas Promedio por Categoría
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="categoria", y="unidades_vendidas", estimator=np.mean, errorbar=None, hue="categoria", palette="viridis", legend=False)
    plt.title("Promedio de Unidades Vendidas por Categoría en Cartago")
    plt.xlabel("Categoría de Producto")
    plt.ylabel("Unidades Vendidas (Promedio)")
    plt.tight_layout()
    plt.savefig("../graficos/2_ventas_por_categoria.png", dpi=150)
    plt.close()

    # Gráfico 3: Dispersión - Días sin venta vs Unidades Vendidas (coloreado por decisión)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="dias_sin_venta",
        y="unidades_vendidas",
        hue="decision_abastecer",
        style="decision_abastecer",
        palette={0: "#d9534f", 1: "#5cb85c"},
        s=120
    )
    plt.title("Relación: Días sin Venta vs. Unidades Vendidas")
    plt.xlabel("Días sin Venta (Estancamiento)")
    plt.ylabel("Unidades Vendidas al Mes")
    plt.legend(title="Decisión (1=Abastecer)")
    plt.tight_layout()
    plt.savefig("../graficos/3_dias_vs_ventas.png", dpi=150)
    plt.close()

    # Gráfico 4: Mapa de Calor de Correlaciones
    plt.figure(figsize=(9, 7))
    cols_num = ["precio_venta", "costo_compra", "unidades_vendidas", "stock_actual", "dias_sin_venta", "calificacion_cliente", "ganancia_unitaria", "margen_ganancia", "decision_abastecer"]
    corr = df[cols_num].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Mapa de Calor de Correlaciones entre Variables")
    plt.tight_layout()
    plt.savefig("../graficos/4_mapa_correlaciones.png", dpi=150)
    plt.close()

    print("[EXITO] 4 gráficos guardados en la carpeta ../graficos/")

if __name__ == "__main__":
    ejecutar_eda()