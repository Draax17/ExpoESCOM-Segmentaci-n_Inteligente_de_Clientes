from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline.data_loader import generate_demo_customer_dataset, load_dataset
from src.pipeline.duplicates import duplicate_summary
from src.pipeline.simulator import PipelineSimulator
from src.visualization.charts import elbow_chart, silhouette_chart


st.set_page_config(
    page_title="Segmentacion Inteligente de Clientes",
    page_icon="",
    layout="wide",
)


def _load_input_data(uploaded_file: object | None) -> pd.DataFrame:
    if uploaded_file is None:
        return generate_demo_customer_dataset()

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Solo se admiten archivos CSV o Excel.")

    temporary_path = Path(uploaded_file.name)
    temporary_path.write_bytes(uploaded_file.getbuffer())
    try:
        return load_dataset(temporary_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    st.title("Sistema de Segmentacion Inteligente de Clientes")
    st.write("Explora un flujo ordenado para limpieza, clustering y evaluacion de segmentos de clientes.")

    with st.sidebar:
        st.header("Configuracion del simulador")
        scaling_method = st.selectbox("Escalamiento", ["standard", "minmax"], index=0)
        k_min = st.number_input("K minimo", min_value=2, max_value=20, value=2, step=1)
        k_max = st.number_input("K maximo", min_value=3, max_value=20, value=8, step=1)
        linkage = st.selectbox("Linkage jerarquico", ["ward", "complete", "average", "single"], index=0)
        uploaded_file = st.file_uploader("Carga tu dataset", type=["csv", "xlsx", "xls"])

    if k_max < k_min:
        st.error("K maximo debe ser mayor o igual que K minimo.")
        return

    data = _load_input_data(uploaded_file)

    left_column, right_column = st.columns([1.2, 1])
    with left_column:
        st.subheader("Vista previa del dataset")
        st.dataframe(data.head(20), use_container_width=True)

    with right_column:
        summary = duplicate_summary(data)
        st.subheader("Indicadores rapidos")
        st.metric("Filas", summary["total_rows"])
        st.metric("Duplicados", summary["duplicate_rows"])
        st.metric("Columnas", data.shape[1])

    st.divider()
    st.subheader("Simulador de Pipeline de Clustering")
    st.caption("Ejecuta un pipeline base para revisar cada etapa antes de personalizarla.")

    if st.button("Ejecutar simulador", type="primary"):
        simulator = PipelineSimulator(
            scaling_method=scaling_method,
            k_min=int(k_min),
            k_max=int(k_max),
            linkage=linkage,
        )
        result = simulator.run(data)

        tab_data, tab_metrics, tab_plots, tab_summary = st.tabs(
            ["Datos", "Metricas", "Graficas", "Resumen"]
        )

        with tab_data:
            st.write("Datos limpios")
            st.dataframe(result.cleaned_data.head(20), use_container_width=True)
            st.write("Variables seleccionadas")
            st.write(result.selected_features)

        with tab_metrics:
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.markdown("**K-means**")
                st.json(result.kmeans_metrics)
            with metric_col2:
                st.markdown("**Jerarquico**")
                st.json(result.hierarchical_metrics)

        with tab_plots:
            st.plotly_chart(elbow_chart(result.elbow_table), use_container_width=True)
            st.plotly_chart(silhouette_chart(result.silhouette_table), use_container_width=True)

        with tab_summary:
            st.subheader("Metadata del pipeline")
            st.json(result.metadata)
            st.subheader("Resumen por cluster")
            st.dataframe(result.cluster_summary, use_container_width=True)
    else:
        st.info("Carga un dataset o usa el dataset demo y ejecuta el simulador para ver los resultados.")


if __name__ == "__main__":
    main()
