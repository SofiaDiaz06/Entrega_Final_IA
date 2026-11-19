import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("../graficos", exist_ok=True)


# 1. Función de activación Sigmoide (suave y derivable)
def sigmoide(z):
    # Recortamos z para evitar desbordamiento numérico en exp
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))


# 2. Función de pérdida (Binary Cross-Entropy)
def calcular_perdida(y_real, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))


def entrenar_perceptron(X, y, tasa_aprendizaje=0.1, epocas=1000):
    """Entrena un perceptrón con descenso del gradiente desde cero."""
    n_muestras, n_caracteristicas = X.shape

    # Inicializar pesos en cero y sesgo en cero
    pesos = np.zeros(n_caracteristicas)
    sesgo = 0.0
    historial_error = []

    for epoca in range(epocas):
        # Propagación hacia adelante (Forward pass)
        z = np.dot(X, pesos) + sesgo
        y_pred = sigmoide(z)

        # Cálculo del error (pérdida logarítmica)
        perdida = calcular_perdida(y, y_pred)
        historial_error.append(perdida)

        # Cálculo de gradientes (derivadas)
        error = y_pred - y
        dw = (1 / n_muestras) * np.dot(X.T, error)
        db = (1 / n_muestras) * np.sum(error)

        # Actualización de parámetros (Descenso del gradiente)
        pesos -= tasa_aprendizaje * dw
        sesgo -= tasa_aprendizaje * db

    return pesos, sesgo, historial_error


def main():
    print("=" * 65)
    print("FASE 3: PERCEPTRÓN DESDE CERO (NUMPY PURO) - ALMACÉN CARTAGO")
    print("=" * 65)

    # 1. Cargar dataset limpio
    df = pd.read_csv("../data/proyecto_limpio.csv")

    # Usamos 2 variables independientes (sin fuga de datos) para graficar en 2D
    caracteristicas = ["ganancia_unitaria", "dias_sin_venta"]
    X_raw = df[caracteristicas].to_numpy(dtype=float)
    y = df["decision_abastecer"].to_numpy(dtype=float)

    # 2. Estandarización Z-score manual (media = 0, std = 1) con protección ante std=0
    media = np.mean(X_raw, axis=0)
    desviacion = np.std(X_raw, axis=0)
    desviacion_segura = np.where(desviacion == 0, 1.0, desviacion)
    X_norm = (X_raw - media) / desviacion_segura

    # 3. Entrenar la neurona artificial
    tasa = 0.1
    epocas = 1000
    pesos, sesgo, historial_error = entrenar_perceptron(X_norm, y, tasa, epocas)

    # 4. Evaluar predicciones sobre el dataset de entrenamiento
    probabilidades = sigmoide(np.dot(X_norm, pesos) + sesgo)
    predicciones = (probabilidades >= 0.5).astype(int)
    exactitud = np.mean(predicciones == y) * 100

    print(f"\n[RESULTADOS DEL ENTRENAMIENTO ({epocas} épocas, alpha={tasa})]")
    print(f"  -> Error inicial (Época 1):    {historial_error[0]:.4f}")
    print(f"  -> Error final (Época {epocas}): {historial_error[-1]:.4f}")
    print(f"  -> Peso w1 (ganancia_unitaria): {pesos[0]:+.4f}")
    print(f"  -> Peso w2 (dias_sin_venta):    {pesos[1]:+.4f}")
    print(f"  -> Sesgo (bias b):              {sesgo:+.4f}")
    print(f"  -> Exactitud en entrenamiento:  {exactitud:.2f}% ({int(np.sum(predicciones == y))}/{len(y)} productos)")

    # Interpretación automática de los signos aprendidos
    print("\n[INTERPRETACIÓN DE LOS PESOS]")
    for nombre_var, w in zip(caracteristicas, pesos):
        efecto = "aumenta la probabilidad de ABASTECER" if w > 0 else "reduce la probabilidad de ABASTECER"
        print(f"  -> {nombre_var}: signo {'+' if w > 0 else '-'} ({w:+.4f}) -> a mayor valor, {efecto}.")

    # 5. Gráfico 1: Curva de aprendizaje (Disminución del error)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epocas + 1), historial_error, color="#d9534f", linewidth=2)
    plt.title("Curva de Aprendizaje del Perceptrón (Descenso del Gradiente)")
    plt.xlabel("Época")
    plt.ylabel("Pérdida Logarítmica (Error)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../graficos/5_perceptron_curva_error.png", dpi=150)
    plt.close()

    # 6. Gráfico 2: Frontera de Decisión en 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X_norm[y == 0, 0], X_norm[y == 0, 1], color="#d9534f", label="No Abastecer (0)", s=100, edgecolors="k")
    plt.scatter(X_norm[y == 1, 0], X_norm[y == 1, 1], color="#5cb85c", label="Abastecer (1)", s=100, edgecolors="k")

    # Línea de frontera donde z = w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
    if abs(pesos[1]) > 1e-8:
        x1_vals = np.linspace(X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5, 100)
        x2_vals = -(pesos[0] * x1_vals + sesgo) / pesos[1]
        plt.plot(x1_vals, x2_vals, color="#2b5c8f", linestyle="--", linewidth=2, label="Frontera de Decisión (z=0)")

    plt.ylim(X_norm[:, 1].min() - 0.5, X_norm[:, 1].max() + 0.5)
    plt.title("Frontera de Decisión del Perceptrón (Datos Estandarizados)")
    plt.xlabel("Ganancia Unitaria (Estandarizada)")
    plt.ylabel("Días sin Venta (Estandarizados)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("../graficos/6_perceptron_frontera_decision.png", dpi=150)
    plt.close()

    print("\n[EXITO] Gráficos del perceptrón guardados en ../graficos/:")
    print("  - 5_perceptron_curva_error.png")
    print("  - 6_perceptron_frontera_decision.png")


if __name__ == "__main__":
    main()