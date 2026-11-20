import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Recomendador de Abastacimiento",
    
    layout="centered"
)

st.markdown(
    """
    <div style="padding: 16px 20px; border-radius: 12px; background-color: #f4f6f9; border-left: 6px solid #1f77b4; margin-bottom: 16px;">
        <h1 style="margin: 0; font-size: 1.8rem;">𝐑𝐞𝐜𝐨𝐦𝐞𝐧𝐝𝐚𝐝𝐨𝐫 𝐝𝐞 𝐀𝐛𝐚𝐬𝐭𝐞𝐜𝐢𝐦𝐢𝐞𝐧𝐭𝐨</h1>
        <p style="margin: 4px 0 0 0; font-weight: 600; color: #333;"> ✶ Almacén de Variedades de Cartago ✶ </p>
        <p style="margin: 6px 0 0 0; color: #555; font-size: 0.95rem;">
            Es rentable abastecer un producto o no.
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
            "precio_venta", "costo_compra", "stock_actual", "calificacion_cliente",
            "cat_Accesorios", "cat_Belleza", "cat_Fiesta",
            "cat_Hogar", "cat_Juguetes", "cat_Papeleria"
        ]
    return modelo, columnas


modelo, columnas_entrenamiento = cargar_modelo()

st.sidebar.header("Parámetros del Producto")

st.sidebar.markdown("**1. Datos financieros**")
precio_venta = st.sidebar.number_input("Precio de Venta (COP)", min_value=0.0, value=15000.0, step=500.0)
costo_compra = st.sidebar.number_input("Costo de Compra (COP)", min_value=0.0, value=7000.0, step=500.0)

ganancia_calculada = precio_venta - costo_compra
margen_pct = (ganancia_calculada / precio_venta * 100) if precio_venta > 0 else 0.0

if ganancia_calculada >= 0:
    st.sidebar.success(f"Ganancia unitaria: **${ganancia_calculada:,.0f} COP** ({margen_pct:.1f}%)")
else:
    st.sidebar.warning(f"Pérdida unitaria: **${ganancia_calculada:,.0f} COP**")

st.sidebar.markdown("---")
st.sidebar.markdown("**2. Inventario y categoría**")
calificacion = st.sidebar.slider("Calificación del cliente", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
stock = st.sidebar.number_input("Stock actual en bodega", min_value=0, value=15, step=1)
categoria = st.sidebar.selectbox(
    "Categoría del producto",
    ["Accesorios", "Belleza", "Fiesta", "Hogar", "Juguetes", "Papeleria"]
)

st.markdown("#### Resumen del producto a evaluar")
res1, res2, res3, res4 = st.columns(4)
res1.metric("Precio / Ud.", f"${precio_venta:,.0f}")
res2.metric("Costo / Ud.", f"${costo_compra:,.0f}")
res3.metric("Stock bodega", f"{stock} uds")
res4.metric("Categoría", categoria)

st.write("")
analizar = st.button("Analizar y Recomendar", type="primary", use_container_width=True)

if analizar:
    caso_nuevo = pd.DataFrame({
        "precio_venta": [precio_venta],
        "costo_compra": [costo_compra],
        "stock_actual": [stock],
        "calificacion_cliente": [calificacion],
        "categoria": [categoria]
    })

    X_nuevo = pd.get_dummies(caso_nuevo, columns=["categoria"], prefix="cat", dtype=int)

    if columnas_entrenamiento is not None:
        X_nuevo = X_nuevo.reindex(columns=columnas_entrenamiento, fill_value=0)

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
                "El producto presenta condiciones favorables de precio, costo, stock y calidad percibida "
                "para el comercio en Cartago."
            )
        else:
            st.error("### ❌ RECOMENDACIÓN: NO ABASTECER")
            st.warning(
                "El modelo detecta riesgo de estancamiento o margen insuficiente. Conviene evitar "
                "inmovilizar capital en este momento."
            )