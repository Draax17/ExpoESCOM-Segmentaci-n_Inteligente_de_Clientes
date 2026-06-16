# Sistema de Segmentacion Inteligente de Clientes

Proyecto para segmentar clientes mediante clustering y apoyar estrategias de marketing.

## Alcance

- Deteccion y tratamiento de registros duplicados
- Deteccion y tratamiento de outliers
- Escalamiento de variables
- Seleccion de variables
- Clustering con K-means
- Clustering jerarquico
- Evaluacion con metodo del codo y Silhouette Score
- Interfaz grafica
- Simulador de pipeline de clustering

## Estructura

```text
src/
  app/             # Interfaz grafica en Streamlit
  pipeline/        # Logica de preparacion, clustering y evaluacion
  utils/           # Utilidades comunes
  visualization/   # Graficos y visualizaciones

data/
  raw/             # Dataset original
  processed/       # Datos transformados
  external/        # Fuentes externas o auxiliares

models/            # Modelos entrenados y artefactos
reports/figures/   # Graficos exportados
notebooks/         # Exploracion y analisis
```

## Instalacion

1. Crear un entorno virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
streamlit run src/app/streamlit_app.py
```

## Siguientes pasos sugeridos

- Colocar el dataset en `data/raw/`
- Completar las funciones del pipeline con tu logica de negocio
- Ajustar la interfaz para mostrar resultados reales de cada etapa
- Guardar modelos y metricas en `models/` y `reports/`
