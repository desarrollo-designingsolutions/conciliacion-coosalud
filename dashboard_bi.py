"""
Dashboard BI - Conciliacion Cuentas Medicas Coosalud.
Lee datos de Snowflake y muestra graficos interactivos, filtros y exportaciones.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from snowflake_conn import get_sf_conn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COP_FORMAT = "${:,.0f}"


def _fmt_cop(val):
    """Formatea un numero como moneda colombiana."""
    if pd.isna(val):
        return "$0"
    return f"${val:,.0f}"


def _pct(num, den):
    if den == 0:
        return 0.0
    return round(num / den * 100, 1)


MODALIDAD_CASE = """
CASE
    WHEN UPPER(aud.MODALIDAD) LIKE '%PPE%CAPITA%' THEN 'PPE Capita'
    WHEN UPPER(aud.MODALIDAD) LIKE '%PPE%EVENTO%' THEN 'PPE Evento'
    WHEN UPPER(aud.MODALIDAD) LIKE '%PGE%' THEN 'PGE'
    WHEN UPPER(aud.MODALIDAD) LIKE '%PGP%' OR UPPER(aud.MODALIDAD) LIKE '%PAGO GLOBAL PROSPECTIVO%' THEN 'PGP'
    WHEN UPPER(aud.MODALIDAD) LIKE '%PRESUPUESTO GLOBAL%' THEN 'Presupuesto Global'
    WHEN UPPER(aud.MODALIDAD) LIKE '%MIA%' OR UPPER(aud.MODALIDAD) LIKE '%MAIS%' THEN 'MIA'
    WHEN UPPER(aud.MODALIDAD) LIKE '%PAQUETE%' THEN 'Paquete Integral'
    WHEN UPPER(aud.MODALIDAD) LIKE '%CAPITA%' THEN 'Capita'
    WHEN UPPER(aud.MODALIDAD) LIKE '%EVENT%' OR UPPER(aud.MODALIDAD) LIKE '%COVID%' OR UPPER(aud.MODALIDAD) LIKE '%NO POS%' THEN 'Evento'
    WHEN UPPER(aud.MODALIDAD) LIKE '%PAGO POR SERVICIO%' THEN 'Pago por Servicio'
    WHEN UPPER(aud.MODALIDAD) LIKE '%EPISODIO%' THEN 'Pago por episodio'
    ELSE 'Sin identificar'
END
"""

COLOR_PALETTE = [
    "#0068c9", "#83c9ff", "#ff2b2b", "#ffabab",
    "#29b09d", "#7defa1", "#ff8700", "#ffd16a",
    "#6d3fc0", "#d5dae5",
]

# ---------------------------------------------------------------------------
# Data queries (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner="Consultando Snowflake...")
def load_kpi_general():
    """KPIs generales: totales por estado."""
    sql = """
    SELECT
        eafr.ESTADO,
        COUNT(DISTINCT aud.FACTURA_ID) AS cantidad_facturas,
        COUNT(aud.ID) AS cantidad_servicios,
        SUM(aud.VALOR_TOTAL_SERVICIO) AS valor_total,
        SUM(aud.VALOR_GLOSA) AS valor_glosa,
        SUM(aud.VALOR_APROBADO) AS valor_aprobado
    FROM SYNC_AUDITORY_FINAL_REPORTS aud
    INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
    GROUP BY eafr.ESTADO
    ORDER BY eafr.ESTADO
    """
    conn = get_sf_conn()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


@st.cache_data(ttl=600, show_spinner="Consultando prestadores...")
def load_prestadores_resumen():
    """Resumen de conciliacion por prestador (solo contabilizadas con glosa)."""
    sql = f"""
    WITH servicios AS (
        SELECT
            aud.NIT,
            aud.RAZON_SOCIAL,
            aud.FACTURA_ID,
            aud.ID AS SERVICIO_ID,
            aud.VALOR_GLOSA,
            aud.VALOR_TOTAL_SERVICIO,
            aud.VALOR_APROBADO,
            {MODALIDAD_CASE} AS MODALIDAD_HOMOLOGADA,
            aud.REGIMEN,
            DATE_TRUNC('month', aud.CREATED_AT) AS MES_CREACION,
            MAX(CASE WHEN ci.ID IS NOT NULL THEN 1 ELSE 0 END) AS ES_CONCILIADO,
            SUM(COALESCE(ci.ACCEPTED_VALUE_EPS, 0)) AS ACCEPTED_VALUE_EPS,
            SUM(COALESCE(ci.ACCEPTED_VALUE_IPS, 0)) AS ACCEPTED_VALUE_IPS,
            SUM(COALESCE(ci.EPS_RATIFIED_VALUE, 0)) AS EPS_RATIFIED_VALUE
        FROM SYNC_AUDITORY_FINAL_REPORTS aud
        INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
        INNER JOIN SYNC_INVOICE_AUDITS ia ON ia.ID = aud.FACTURA_ID
        INNER JOIN SYNC_THIRDS t ON t.ID = ia.THIRD_ID
        LEFT JOIN SYNC_CONCILIATION_RESULTS ci ON ci.AUDITORY_FINAL_REPORT_ID = aud.ID
        WHERE eafr.ESTADO = 'contabilizada' AND aud.VALOR_GLOSA > 0
        GROUP BY aud.NIT, aud.RAZON_SOCIAL, aud.FACTURA_ID,
                 aud.ID, aud.VALOR_GLOSA, aud.VALOR_TOTAL_SERVICIO,
                 aud.VALOR_APROBADO, MODALIDAD_HOMOLOGADA,
                 aud.REGIMEN, MES_CREACION
    )
    SELECT
        NIT,
        RAZON_SOCIAL,
        MODALIDAD_HOMOLOGADA,
        REGIMEN,
        MES_CREACION,
        COUNT(DISTINCT FACTURA_ID) AS cantidad_facturas,
        COUNT(*) AS cantidad_servicios,
        SUM(VALOR_TOTAL_SERVICIO) AS valor_total,
        SUM(VALOR_GLOSA) AS valor_glosa,
        SUM(VALOR_APROBADO) AS valor_aprobado,
        SUM(ES_CONCILIADO) AS servicios_conciliados,
        SUM(CASE WHEN ES_CONCILIADO = 0 THEN 1 ELSE 0 END) AS servicios_no_conciliados,
        SUM(CASE WHEN ES_CONCILIADO = 1 THEN VALOR_GLOSA ELSE 0 END) AS glosa_conciliada,
        SUM(CASE WHEN ES_CONCILIADO = 0 THEN VALOR_GLOSA ELSE 0 END) AS glosa_no_conciliada,
        SUM(ACCEPTED_VALUE_EPS) AS valor_aceptado_eps,
        SUM(ACCEPTED_VALUE_IPS) AS valor_aceptado_ips,
        SUM(EPS_RATIFIED_VALUE) AS valor_ratificado
    FROM servicios
    GROUP BY NIT, RAZON_SOCIAL, MODALIDAD_HOMOLOGADA, REGIMEN, MES_CREACION
    ORDER BY NIT
    """
    conn = get_sf_conn()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


@st.cache_data(ttl=600, show_spinner="Consultando clasificacion de conciliacion...")
def load_clasificacion_conciliacion():
    """Clasificacion detallada de servicios contabilizados con glosa."""
    sql = f"""
    SELECT
        CASE
            WHEN ci.ID IS NULL THEN 'Sin conciliar'
            WHEN ci.EPS_RATIFIED_VALUE > 0 AND ci.ACCEPTED_VALUE_IPS = 0 AND ci.ACCEPTED_VALUE_EPS = 0 THEN 'Ratificada'
            WHEN ci.EPS_RATIFIED_VALUE > 0 AND (ci.ACCEPTED_VALUE_IPS > 0 OR ci.ACCEPTED_VALUE_EPS > 0) THEN 'Conciliada y ratificada'
            WHEN ci.EPS_RATIFIED_VALUE = 0 AND (ci.ACCEPTED_VALUE_IPS > 0 OR ci.ACCEPTED_VALUE_EPS > 0) THEN 'Conciliada'
            ELSE 'En proceso'
        END AS clasificacion,
        COUNT(DISTINCT aud.FACTURA_ID) AS cantidad_facturas,
        COUNT(aud.ID) AS cantidad_servicios,
        SUM(aud.VALOR_GLOSA) AS valor_glosa,
        SUM(COALESCE(ci.ACCEPTED_VALUE_EPS, 0)) AS valor_aceptado_eps,
        SUM(COALESCE(ci.ACCEPTED_VALUE_IPS, 0)) AS valor_aceptado_ips,
        SUM(COALESCE(ci.EPS_RATIFIED_VALUE, 0)) AS valor_ratificado
    FROM SYNC_AUDITORY_FINAL_REPORTS aud
    INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
    LEFT JOIN SYNC_CONCILIATION_RESULTS ci ON ci.AUDITORY_FINAL_REPORT_ID = aud.ID
    WHERE eafr.ESTADO = 'contabilizada' AND aud.VALOR_GLOSA > 0
    GROUP BY clasificacion
    ORDER BY valor_glosa DESC
    """
    conn = get_sf_conn()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def apply_filters(df, nits, modalidades, regimenes, mes_desde, mes_hasta):
    """Aplica filtros al dataframe de prestadores."""
    filtered = df.copy()
    if nits:
        filtered = filtered[filtered["nit"].isin(nits)]
    if modalidades:
        filtered = filtered[filtered["modalidad_homologada"].isin(modalidades)]
    if regimenes:
        filtered = filtered[filtered["regimen"].isin(regimenes)]
    if mes_desde is not None:
        filtered = filtered[filtered["mes_creacion"] >= pd.Timestamp(mes_desde)]
    if mes_hasta is not None:
        filtered = filtered[filtered["mes_creacion"] <= pd.Timestamp(mes_hasta)]
    return filtered


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_top_prestadores_glosa(df_agg, top_n=15):
    top = df_agg.nlargest(top_n, "valor_glosa")
    fig = px.bar(
        top,
        x="valor_glosa",
        y="razon_social",
        orientation="h",
        color="glosa_conciliada",
        color_continuous_scale=["#ff2b2b", "#29b09d"],
        labels={
            "valor_glosa": "Valor Glosa",
            "razon_social": "Prestador",
            "glosa_conciliada": "Glosa conciliada",
        },
        title=f"Top {top_n} prestadores por valor de glosa",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(title="Conciliada"),
    )
    return fig


def chart_estado_pie(df_kpi):
    fig = px.pie(
        df_kpi,
        values="cantidad_servicios",
        names="estado",
        title="Distribucion de servicios por estado",
        color_discrete_sequence=COLOR_PALETTE,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def chart_modalidad_bar(df_agg):
    mod = (
        df_agg.groupby("modalidad_homologada", as_index=False)
        .agg(valor_glosa=("valor_glosa", "sum"), cantidad_facturas=("cantidad_facturas", "sum"))
        .sort_values("valor_glosa", ascending=False)
    )
    fig = px.bar(
        mod,
        x="modalidad_homologada",
        y="valor_glosa",
        color="modalidad_homologada",
        color_discrete_sequence=COLOR_PALETTE,
        labels={"modalidad_homologada": "Modalidad", "valor_glosa": "Valor Glosa"},
        title="Valor de glosa por modalidad",
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickangle=-30,
    )
    return fig


def chart_conciliacion_donut(df_agg):
    total_conc = df_agg["glosa_conciliada"].sum()
    total_no_conc = df_agg["glosa_no_conciliada"].sum()
    fig = go.Figure(data=[go.Pie(
        labels=["Conciliada", "Sin conciliar"],
        values=[total_conc, total_no_conc],
        hole=0.5,
        marker_colors=["#29b09d", "#ff2b2b"],
        textinfo="percent+label",
    )])
    fig.update_layout(
        title="Glosa conciliada vs sin conciliar",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def chart_clasificacion_bar(df_clas):
    fig = px.bar(
        df_clas,
        x="clasificacion",
        y="valor_glosa",
        color="clasificacion",
        color_discrete_sequence=COLOR_PALETTE,
        labels={"clasificacion": "Clasificacion", "valor_glosa": "Valor Glosa"},
        title="Valor de glosa por clasificacion de conciliacion",
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def chart_tendencia_mensual(df_agg):
    mensual = (
        df_agg.groupby("mes_creacion", as_index=False)
        .agg(
            valor_glosa=("valor_glosa", "sum"),
            glosa_conciliada=("glosa_conciliada", "sum"),
            glosa_no_conciliada=("glosa_no_conciliada", "sum"),
        )
        .sort_values("mes_creacion")
    )
    if mensual.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mensual["mes_creacion"], y=mensual["glosa_conciliada"],
        mode="lines+markers", name="Conciliada",
        line=dict(color="#29b09d", width=2),
        fill="tozeroy", fillcolor="rgba(41,176,157,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=mensual["mes_creacion"], y=mensual["glosa_no_conciliada"],
        mode="lines+markers", name="Sin conciliar",
        line=dict(color="#ff2b2b", width=2),
        fill="tozeroy", fillcolor="rgba(255,43,43,0.10)",
    ))
    fig.update_layout(
        title="Tendencia mensual de glosa",
        xaxis_title="Mes",
        yaxis_title="Valor ($)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_regimen_pie(df_agg):
    reg = (
        df_agg.groupby("regimen", as_index=False)
        .agg(valor_glosa=("valor_glosa", "sum"))
    )
    reg = reg[reg["regimen"].notna() & (reg["regimen"] != "")]
    fig = px.pie(
        reg,
        values="valor_glosa",
        names="regimen",
        title="Distribucion de glosa por regimen",
        color_discrete_sequence=COLOR_PALETTE,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def chart_top_prestadores_conciliacion(df_agg, top_n=15):
    top = df_agg.nlargest(top_n, "valor_glosa")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["razon_social"],
        x=top["glosa_conciliada"],
        name="Conciliada",
        orientation="h",
        marker_color="#29b09d",
    ))
    fig.add_trace(go.Bar(
        y=top["razon_social"],
        x=top["glosa_no_conciliada"],
        name="Sin conciliar",
        orientation="h",
        marker_color="#ff2b2b",
    ))
    fig.update_layout(
        barmode="stack",
        title=f"Top {top_n} prestadores: conciliado vs sin conciliar",
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_dashboard_page():
    """Renderiza la pagina del Dashboard BI."""
    st.title("Dashboard BI - Conciliacion")

    # --- Load data ---
    with st.spinner("Cargando datos desde Snowflake..."):
        df_kpi = load_kpi_general()
        df_prest = load_prestadores_resumen()
        df_clas = load_clasificacion_conciliacion()

    if df_prest.empty:
        st.warning("No se encontraron datos en Snowflake.")
        return

    # --- Sidebar filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros Dashboard")

    nit_options = sorted(df_prest["nit"].unique().tolist())
    razon_map = df_prest.drop_duplicates("nit").set_index("nit")["razon_social"].to_dict()
    nit_display = [f"{n} - {razon_map.get(n, '')}" for n in nit_options]
    selected_display = st.sidebar.multiselect("Prestador (NIT)", nit_display)
    selected_nits = [nit_options[nit_display.index(d)] for d in selected_display] if selected_display else []

    modalidad_options = sorted(df_prest["modalidad_homologada"].unique().tolist())
    selected_modalidades = st.sidebar.multiselect("Modalidad", modalidad_options)

    regimen_options = sorted([r for r in df_prest["regimen"].unique().tolist() if r and str(r) != "nan"])
    selected_regimenes = st.sidebar.multiselect("Regimen", regimen_options)

    meses_disponibles = df_prest["mes_creacion"].dropna().sort_values().unique()
    mes_desde = None
    mes_hasta = None
    if len(meses_disponibles) > 0:
        col_f1, col_f2 = st.sidebar.columns(2)
        with col_f1:
            mes_desde = st.date_input(
                "Desde",
                value=pd.Timestamp(meses_disponibles[0]),
                min_value=pd.Timestamp(meses_disponibles[0]),
                max_value=pd.Timestamp(meses_disponibles[-1]),
            )
        with col_f2:
            mes_hasta = st.date_input(
                "Hasta",
                value=pd.Timestamp(meses_disponibles[-1]),
                min_value=pd.Timestamp(meses_disponibles[0]),
                max_value=pd.Timestamp(meses_disponibles[-1]),
            )

    if st.sidebar.button("Limpiar cache", help="Recargar datos desde Snowflake"):
        load_kpi_general.clear()
        load_prestadores_resumen.clear()
        load_clasificacion_conciliacion.clear()
        st.rerun()

    # --- Apply filters ---
    df_filtered = apply_filters(df_prest, selected_nits, selected_modalidades, selected_regimenes, mes_desde, mes_hasta)

    df_agg = (
        df_filtered.groupby(["nit", "razon_social", "modalidad_homologada", "regimen", "mes_creacion"], as_index=False, dropna=False)
        .agg(
            cantidad_facturas=("cantidad_facturas", "sum"),
            cantidad_servicios=("cantidad_servicios", "sum"),
            valor_total=("valor_total", "sum"),
            valor_glosa=("valor_glosa", "sum"),
            valor_aprobado=("valor_aprobado", "sum"),
            servicios_conciliados=("servicios_conciliados", "sum"),
            servicios_no_conciliados=("servicios_no_conciliados", "sum"),
            glosa_conciliada=("glosa_conciliada", "sum"),
            glosa_no_conciliada=("glosa_no_conciliada", "sum"),
            valor_aceptado_eps=("valor_aceptado_eps", "sum"),
            valor_aceptado_ips=("valor_aceptado_ips", "sum"),
            valor_ratificado=("valor_ratificado", "sum"),
        )
    )

    df_agg_prestador = (
        df_filtered.groupby(["nit", "razon_social"], as_index=False)
        .agg(
            cantidad_facturas=("cantidad_facturas", "sum"),
            cantidad_servicios=("cantidad_servicios", "sum"),
            valor_total=("valor_total", "sum"),
            valor_glosa=("valor_glosa", "sum"),
            valor_aprobado=("valor_aprobado", "sum"),
            servicios_conciliados=("servicios_conciliados", "sum"),
            servicios_no_conciliados=("servicios_no_conciliados", "sum"),
            glosa_conciliada=("glosa_conciliada", "sum"),
            glosa_no_conciliada=("glosa_no_conciliada", "sum"),
            valor_aceptado_eps=("valor_aceptado_eps", "sum"),
            valor_aceptado_ips=("valor_aceptado_ips", "sum"),
            valor_ratificado=("valor_ratificado", "sum"),
        )
    )

    # ===================================================================
    # TABS
    # ===================================================================
    tab_overview, tab_conciliacion, tab_detalle = st.tabs([
        "Vision General", "Conciliacion", "Detalle y Exportacion"
    ])

    # -------------------------------------------------------------------
    # TAB 1: Vision General
    # -------------------------------------------------------------------
    with tab_overview:
        # KPI cards - row 1
        st.subheader("Indicadores generales")

        total_facturas = int(df_agg_prestador["cantidad_facturas"].sum())
        total_servicios = int(df_agg_prestador["cantidad_servicios"].sum())
        total_valor = df_agg["valor_total"].sum()
        total_glosa = df_agg["valor_glosa"].sum()
        total_aprobado = df_agg["valor_aprobado"].sum()
        pct_glosa = _pct(total_glosa, total_valor) if total_valor else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Prestadores", f"{df_agg_prestador['nit'].nunique():,}")
        c2.metric("Facturas", f"{total_facturas:,}")
        c3.metric("Servicios", f"{total_servicios:,}")
        c4.metric("Valor Total", _fmt_cop(total_valor))
        c5.metric("Valor Glosa", _fmt_cop(total_glosa))
        c6.metric("% Glosa", f"{pct_glosa}%")

        st.markdown("---")

        # Charts row 1
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(chart_estado_pie(df_kpi), use_container_width=True)
        with col_right:
            st.plotly_chart(chart_modalidad_bar(df_agg), use_container_width=True)

        # Charts row 2
        col_left2, col_right2 = st.columns(2)
        with col_left2:
            st.plotly_chart(
                chart_top_prestadores_glosa(df_agg_prestador),
                use_container_width=True,
            )
        with col_right2:
            st.plotly_chart(chart_regimen_pie(df_agg), use_container_width=True)

        # Tendencia mensual
        fig_tend = chart_tendencia_mensual(df_agg)
        if fig_tend:
            st.plotly_chart(fig_tend, use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 2: Conciliacion
    # -------------------------------------------------------------------
    with tab_conciliacion:
        st.subheader("Estado de conciliacion")

        total_conc = df_agg["glosa_conciliada"].sum()
        total_no_conc = df_agg["glosa_no_conciliada"].sum()
        serv_conc = int(df_agg["servicios_conciliados"].sum())
        serv_no_conc = int(df_agg["servicios_no_conciliados"].sum())
        pct_conc_valor = _pct(total_conc, total_conc + total_no_conc)
        pct_conc_serv = _pct(serv_conc, serv_conc + serv_no_conc)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Servicios conciliados", f"{serv_conc:,}")
        c2.metric("Sin conciliar", f"{serv_no_conc:,}")
        c3.metric("% Conciliado (serv.)", f"{pct_conc_serv}%")
        c4.metric("Glosa conciliada", _fmt_cop(total_conc))
        c5.metric("Glosa sin conciliar", _fmt_cop(total_no_conc))
        c6.metric("% Conciliado ($)", f"{pct_conc_valor}%")

        st.markdown("---")

        # Valores de conciliacion
        st.subheader("Valores de conciliacion")
        v1, v2, v3 = st.columns(3)
        v1.metric("Aceptado EPS", _fmt_cop(df_agg["valor_aceptado_eps"].sum()))
        v2.metric("Aceptado IPS", _fmt_cop(df_agg["valor_aceptado_ips"].sum()))
        v3.metric("Ratificado EPS", _fmt_cop(df_agg["valor_ratificado"].sum()))

        st.markdown("---")

        # Charts
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(chart_conciliacion_donut(df_agg), use_container_width=True)
        with col_right:
            st.plotly_chart(chart_clasificacion_bar(df_clas), use_container_width=True)

        st.plotly_chart(
            chart_top_prestadores_conciliacion(df_agg_prestador),
            use_container_width=True,
        )

        # Tabla clasificacion
        st.subheader("Detalle por clasificacion")
        df_clas_display = df_clas.copy()
        for col in ["valor_glosa", "valor_aceptado_eps", "valor_aceptado_ips", "valor_ratificado"]:
            if col in df_clas_display.columns:
                df_clas_display[col] = df_clas_display[col].apply(_fmt_cop)
        st.dataframe(
            df_clas_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "clasificacion": st.column_config.Column("Clasificacion"),
                "cantidad_facturas": st.column_config.Column("Facturas"),
                "cantidad_servicios": st.column_config.Column("Servicios"),
                "valor_glosa": st.column_config.Column("Valor Glosa"),
                "valor_aceptado_eps": st.column_config.Column("Aceptado EPS"),
                "valor_aceptado_ips": st.column_config.Column("Aceptado IPS"),
                "valor_ratificado": st.column_config.Column("Ratificado"),
            },
        )

    # -------------------------------------------------------------------
    # TAB 3: Detalle y Exportacion
    # -------------------------------------------------------------------
    with tab_detalle:
        st.subheader("Detalle por prestador")

        df_table = df_agg_prestador.copy()
        df_table = df_table.sort_values("valor_glosa", ascending=False)

        # Formatear para display
        df_display = df_table.copy()
        money_cols = [
            "valor_total", "valor_glosa", "valor_aprobado",
            "glosa_conciliada", "glosa_no_conciliada",
            "valor_aceptado_eps", "valor_aceptado_ips", "valor_ratificado",
        ]
        for col in money_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(_fmt_cop)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "nit": st.column_config.Column("NIT", width="small"),
                "razon_social": st.column_config.Column("Razon Social", width="medium"),
                "cantidad_facturas": st.column_config.Column("Facturas"),
                "cantidad_servicios": st.column_config.Column("Servicios"),
                "valor_total": st.column_config.Column("Valor Total"),
                "valor_glosa": st.column_config.Column("Valor Glosa"),
                "valor_aprobado": st.column_config.Column("Valor Aprobado"),
                "servicios_conciliados": st.column_config.Column("Serv. Conciliados"),
                "servicios_no_conciliados": st.column_config.Column("Serv. Sin Conciliar"),
                "glosa_conciliada": st.column_config.Column("Glosa Conciliada"),
                "glosa_no_conciliada": st.column_config.Column("Glosa Sin Conciliar"),
                "valor_aceptado_eps": st.column_config.Column("Aceptado EPS"),
                "valor_aceptado_ips": st.column_config.Column("Aceptado IPS"),
                "valor_ratificado": st.column_config.Column("Ratificado"),
            },
        )

        st.markdown("---")
        st.subheader("Exportar datos")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_prestador = df_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar resumen por prestador (CSV)",
                data=csv_prestador,
                file_name="dashboard_resumen_prestador.csv",
                mime="text/csv",
            )
        with col_exp2:
            csv_detalle = df_agg.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar detalle completo (CSV)",
                data=csv_detalle,
                file_name="dashboard_detalle_completo.csv",
                mime="text/csv",
            )

        # Detalle por modalidad exportable
        st.markdown("---")
        st.subheader("Detalle por modalidad")
        df_mod = (
            df_agg.groupby("modalidad_homologada", as_index=False)
            .agg(
                cantidad_facturas=("cantidad_facturas", "sum"),
                cantidad_servicios=("cantidad_servicios", "sum"),
                valor_total=("valor_total", "sum"),
                valor_glosa=("valor_glosa", "sum"),
                valor_aprobado=("valor_aprobado", "sum"),
                glosa_conciliada=("glosa_conciliada", "sum"),
                glosa_no_conciliada=("glosa_no_conciliada", "sum"),
            )
            .sort_values("valor_glosa", ascending=False)
        )
        df_mod_display = df_mod.copy()
        for col in ["valor_total", "valor_glosa", "valor_aprobado", "glosa_conciliada", "glosa_no_conciliada"]:
            df_mod_display[col] = df_mod_display[col].apply(_fmt_cop)
        st.dataframe(df_mod_display, use_container_width=True, hide_index=True)

        csv_mod = df_mod.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar detalle por modalidad (CSV)",
            data=csv_mod,
            file_name="dashboard_detalle_modalidad.csv",
            mime="text/csv",
        )
