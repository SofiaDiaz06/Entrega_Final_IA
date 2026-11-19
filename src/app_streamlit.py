import os
import joblib
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Abastecimiento Cartago",
    page_icon="🛒",
    layout="centered"
)

# Encabezado limpio y visual
st.markdown(
    """
    <div style="padding: 16px 20px; border-radius: 12px; background-color: #f4f6f9; border-left: 6px solid #1f77b4; margin-bottom: 16px;">
        <h1 style="margin: 0; font-size: 1.8rem;">🛒 Sistema Inteligente de Abastecimiento</h1>
        <p style="margin: 4px 0 0 0; font-weight: 600; color: #333;">Almacén de Variedades — Centro de Cartago, Valle del Cauca</p>
        <p style="margin: 6px 0 0 0; color: #555; font-size: 0.95rem;">
            Apoyo objetivo para decidir si conviene abastecer un producto o evitar capital inmovilizado por estancamiento.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def cargar_modelo():
    ruta = "modelo_proyecto.joblib"
    if not os.path.exists(ruta):
        ruta = "src/modelo_proyecto.joblib"

    cargado = joblib.load(ruta)

    if isinstance(cargado, dict):
        modelo = cargado.get("pipeline", cargado.get("modelo"))
        columnas = cargado.get("columnas", None)
    else:
        modelo = cargado
        columnas = [
            "ganancia_unitaria", "dias_sin_venta", "calificacion_cliente",
            "stock_actual", "cat_Accesorios", "cat_Belleza", "cat_Fiesta",
            "cat_Hogar", "cat_Juguetes", "cat_Papeleria"
        ]
    return modelo, columnas


modelo, columnas_entrenamiento = cargar_modelo()

# Barra lateral organizada por bloques
st.sidebar.header("📋 Parámetros del Producto")

st.sidebar.markdown("**1. Datos financieros**")
precio_venta = st.sidebar.number_input("Precio de Venta (COP)", min_value=0.0, value=15000.0, step=500.0)
costo_compra = st.sidebar.number_input("Costo de Compra (COP)", min_value=0.0, value=7000.0, step=500.0)

ganancia_calculada = precio_venta - costo_compra
margen_pct = (ganancia_calculada / precio_venta * 100) if precio_venta > 0 else 0.0

if ganancia_calculada >= 0:
    st.sidebar.success(f"💡 Ganancia unitaria: **${ganancia_calculada:,.0f} COP** ({margen_pct:.1f}%)")
else:
    st.sidebar.warning(f"⚠️ Pérdida unitaria: **${ganancia_calculada:,.0f} COP**")

st.sidebar.markdown("---")
st.sidebar.markdown("**2. Rotación y categoría**")
unidades_vendidas = st.sidebar.number_input("Unidades Vendidas (Mensuales)", min_value=0.0, value=150.0, step=5.0)
dias = st.sidebar.number_input("Días sin venta (Estancamiento)", min_value=0, value=1, step=1)
calificacion = st.sidebar.slider("Calificación del cliente", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
stock = st.sidebar.number_input("Stock actual en bodega", min_value=0, value=15, step=1)
categoria = st.sidebar.selectbox(
    "Categoría del producto",
    ["Accesorios", "Belleza", "Fiesta", "Hogar", "Juguetes", "Papeleria"]
)

# Resumen rápido del producto en el panel principal
st.markdown("#### Resumen del producto a evaluar")
res1, res2, res3, res4 = st.columns(4)
res1.metric("Ganancia / Ud.", f"${ganancia_calculada:,.0f}")
res2.metric("Días sin venta", f"{dias} d")
res3.metric("Stock bodega", f"{stock} uds")
res4.metric("Categoría", categoria)

st.write("")
analizar = st.button("🔍 Analizar y Recomendar", type="primary", use_container_width=True)

if analizar:
    caso_nuevo = pd.DataFrame({
        "ganancia_unitaria": [ganancia_calculada],
        "dias_sin_venta": [dias],
        "calificacion_cliente": [calificacion],
        "stock_actual": [stock],
        "categoria": [categoria]
    })

    # One-hot encoding de la categoría
    X_nuevo = pd.get_dummies(caso_nuevo, columns=["categoria"], prefix="cat", dtype=int)

    # Asegurar que tenga exactamente las mismas columnas con las que se entrenó
    if columnas_entrenamiento is not None:
        X_nuevo = X_nuevo.reindex(columns=columnas_entrenamiento, fill_value=0)

    # Predicción y cálculo de probabilidades
    probabilidad = float(modelo.predict_proba(X_nuevo)[0, 1] * 100)
    prediccion = int(modelo.predict(X_nuevo)[0])

    st.markdown("---")
    st.subheader("Resultado del Análisis")

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.metric("Probabilidad de Éxito", f"{probabilidad:.1f}%")
        st.progress(min(max(probabilidad / 100.0, 0.0), 1.0))

    with col2:
        if prediccion == 1:
            st.success("### ✅ RECOMENDACIÓN: ABASTECER")
            st.info(
                f"El producto ({unidades_vendidas:,.0f} uds/mes de referencia) presenta condiciones favorables "
                "de margen y rotación para el comercio en Cartago."
            )
        else:
            st.error("### ❌ RECOMENDACIÓN: NO ABASTECER")
            st.warning(
                "El modelo detecta riesgo de estancamiento o margen insuficiente. Conviene evitar "
                "inmovilizar capital en este momento."
            )