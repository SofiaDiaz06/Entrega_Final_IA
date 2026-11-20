# Recomendador de Abastecimiento para almacén de variedades en Cartago
Herramienta de apoyo para ayudar a los comerciantes a decidir si es conveniente abastecer un producto o no.
## Descripción 
* **Preparación de datos (`src/preparacion_datos.py`):** Limpieza de datos, cálculo de variables y generación de la `decision_abastecer`.
* **Análisis exploratorio (`src/analisis_exploratorio.py`):** Estadísticas con NumPy y Pandas, reporte en `docs/reporte_eda.md` y gráficos.
* **Perceptrón desde cero (`src/perceptron_proyecto.py`):** Implementación con NumPy puro (función sigmoide y descenso del gradiente), curva de error y frontera de decisión.
* **Modelos de Scikit-Learn (`src/modelo_proyecto.py`):** Comparación de modelos dentro de `Pipeline` y guardado del modelo seleccionado (`src/modelo_proyecto.joblib`).
* **Interfaz (`src/usar_modelo.py` y `src/app_streamlit.py`):** Predicción de casos nuevos usando una interfaz sencilla en Streamlit.
## Estructura del Repositorio
```text
Entrega_Final_IA/
├── data/
│   ├── datos_productos.csv
│   ├── proyecto_limpio.csv
│   └── proyecto_ml.csv
├── docs/
│   └── reporte_eda.md
├── graficos/
├── src/
│   ├── analisis_exploratorio.py
│   ├── preparacion_datos.py
│   ├── perceptron_proyecto.py
│   ├── modelo_proyecto.py
│   ├── usar_modelo.py
│   ├── app_streamlit.py
│   └── modelo_proyecto.joblib
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
``` 

## Requisitos Previos

- **Git** instalado para clonar el repositorio.
- **Docker** y **Docker Compose** (`docker compose`) instalados y activos en el equipo.
- **Puerto `8501` disponible** para la interfaz de Streamlit.
- Terminal **Bash / Linux / WSL** para los comandos de prueba (en Windows sin WSL se pueden ejecutar directamente los comandos `git` y `docker compose` en PowerShell o Git Bash).

## Cómo Hacer la Prueba Paso a Paso con Docker

1. Crea una carpeta temporal de prueba y clona el repositorio:

   Bash
   ```
   mkdir -p /tmp/prueba_entrega
   cd /tmp/prueba_entrega
   git clone https://github.com/SofiaDiaz06/Entrega_Final_IA.git
   cd Entrega_Final_IA

   ```

2. Levanta el contenedor desde cero:

   Bash
   ```
   docker compose up -d --build

   ```

3. Verifica que el contenedor

   Bash
   ```
   docker compose ps

   ```

4. Ejecuta la demostración con casos nuevos sin reentrenar:

   Bash
   ```
   docker compose exec python bash -c "cd /app/src && python usar_modelo.py"

   ```

5. *(Opcional)* Si deseas probar la interfaz de Streamlit (con el puerto `8501` mapeado en `docker-compose.yml`):

   Bash
   ```
   docker compose exec python bash -c "cd /app/src && streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501"

   ```

   *(Nota: si el puerto `8501` ya está definido en `docker-compose.yml`, omite `-p 8501:8501` y usa `docker compose exec <NOMBRE_SERVICIO> bash -c "cd /app/src && streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501"`).*

6. Al finalizar la prueba, detén y limpia el entorno:

   Bash
   ```
   docker compose down

   ```

## Resultados Principales

- Se limpiaron 22 registros de productos, generando variables nuevas y la `decision_abastecer`.
- Se entrenó y evaluó el perceptrón desde cero con NumPy y se compararon modelos de clasificación en `Pipeline` con `StandardScaler`, validación cruzada y métricas de evaluación, guardando el modelo ganador en `src/modelo_proyecto.joblib`.
- `src/usar_modelo.py` carga directamente el modelo guardado y genera predicciones de probabilidad y recomendación para productos nuevos.

