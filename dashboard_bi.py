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


def _swap_sep(s):
    """Intercambia separadores: , -> . y . -> , (formato colombiano)."""
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_cop(val):
    """Formatea un numero como moneda colombiana."""
    if pd.isna(val):
        return "$0"
    return "$" + _swap_sep(f"{val:,.0f}")


def _fmt_cop_compact(val):
    """Formatea moneda de forma compacta para tarjetas de indicadores.

    Ejemplos: $4,1B (billones), $12.320M (millones), $850K (miles).
    """
    if pd.isna(val):
        return "$0"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1_000_000_000_000:
        return f"{sign}$" + _swap_sep(f"{abs_val / 1_000_000_000_000:,.1f}") + "B"
    if abs_val >= 1_000_000_000:
        return f"{sign}$" + _swap_sep(f"{abs_val / 1_000_000_000:,.1f}") + "KM"
    if abs_val >= 1_000_000:
        return f"{sign}$" + _swap_sep(f"{abs_val / 1_000_000:,.0f}") + "M"
    if abs_val >= 1_000:
        return f"{sign}$" + _swap_sep(f"{abs_val / 1_000:,.0f}") + "K"
    return f"{sign}$" + _swap_sep(f"{abs_val:,.0f}")


def _fmt_num_compact(val):
    """Formatea numeros enteros grandes de forma compacta.

    Ejemplos: 6,9M, 1,4M, 18K.
    """
    if pd.isna(val):
        return "0"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1_000_000:
        return sign + _swap_sep(f"{abs_val / 1_000_000:,.1f}") + "M"
    if abs_val >= 10_000:
        return sign + _swap_sep(f"{abs_val / 1_000:,.1f}") + "K"
    return sign + _swap_sep(f"{val:,}")


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


@st.cache_data(ttl=600, show_spinner="Consultando facturas del prestador...")
def load_facturas_prestador(nit: str):
    """Facturas individuales de un prestador con detalle de conciliacion."""
    modalidad_escaped = MODALIDAD_CASE.replace("%", "%%")
    sql = f"""
    SELECT
        ia.INVOICE_NUMBER AS numero_factura,
        ia.CREATED_AT AS fecha_factura,
        {modalidad_escaped} AS MODALIDAD_HOMOLOGADA,
        aud.REGIMEN,
        COUNT(aud.ID) AS cantidad_servicios,
        SUM(aud.VALOR_TOTAL_SERVICIO) AS valor_total,
        SUM(aud.VALOR_GLOSA) AS valor_glosa,
        SUM(aud.VALOR_APROBADO) AS valor_aprobado,
        SUM(CASE WHEN ci.ID IS NOT NULL THEN 1 ELSE 0 END) AS servicios_conciliados,
        SUM(CASE WHEN ci.ID IS NULL THEN 1 ELSE 0 END) AS servicios_no_conciliados,
        SUM(CASE WHEN ci.ID IS NOT NULL THEN aud.VALOR_GLOSA ELSE 0 END) AS glosa_conciliada,
        SUM(CASE WHEN ci.ID IS NULL THEN aud.VALOR_GLOSA ELSE 0 END) AS glosa_no_conciliada,
        SUM(COALESCE(ci.ACCEPTED_VALUE_EPS, 0)) AS valor_aceptado_eps,
        SUM(COALESCE(ci.ACCEPTED_VALUE_IPS, 0)) AS valor_aceptado_ips,
        SUM(COALESCE(ci.EPS_RATIFIED_VALUE, 0)) AS valor_ratificado,
        eafr.ESTADO AS estado
    FROM SYNC_AUDITORY_FINAL_REPORTS aud
    INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
    INNER JOIN SYNC_INVOICE_AUDITS ia ON ia.ID = aud.FACTURA_ID
    LEFT JOIN SYNC_CONCILIATION_RESULTS ci ON ci.AUDITORY_FINAL_REPORT_ID = aud.ID
    WHERE aud.NIT = %s
    GROUP BY ia.INVOICE_NUMBER, ia.CREATED_AT, MODALIDAD_HOMOLOGADA, aud.REGIMEN, eafr.ESTADO
    ORDER BY ia.CREATED_AT DESC
    """
    conn = get_sf_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, (nit,))
        rows = cur.fetchall()
        cols = [desc[0].lower() for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()
    numeric_cols = [
        "cantidad_servicios", "valor_total", "valor_glosa", "valor_aprobado",
        "servicios_conciliados", "servicios_no_conciliados",
        "glosa_conciliada", "glosa_no_conciliada",
        "valor_aceptado_eps", "valor_aceptado_ips", "valor_ratificado",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=600, show_spinner="Consultando devoluciones...")
def load_devoluciones_resumen():
    """Resumen de devoluciones por prestador (estado = devolucion)."""
    sql = f"""
    SELECT
        aud.NIT,
        aud.RAZON_SOCIAL,
        {MODALIDAD_CASE} AS MODALIDAD_HOMOLOGADA,
        COUNT(DISTINCT aud.FACTURA_ID) AS cantidad_facturas,
        COUNT(aud.ID) AS cantidad_servicios,
        SUM(aud.VALOR_TOTAL_SERVICIO) AS valor_total,
        SUM(aud.VALOR_GLOSA) AS valor_glosa,
        SUM(aud.VALOR_APROBADO) AS valor_aprobado
    FROM SYNC_AUDITORY_FINAL_REPORTS aud
    INNER JOIN SYNC_ESTADOS_AUDITORY_FINAL_REPORTS eafr ON eafr.ID = aud.ID
    WHERE eafr.ESTADO = 'devolucion'
    GROUP BY aud.NIT, aud.RAZON_SOCIAL, MODALIDAD_HOMOLOGADA
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


def chart_composicion_valor(total_glosa, total_aprobado):
    fig = go.Figure(data=[go.Pie(
        labels=["Valor Glosa", "Valor Aprobado"],
        values=[total_glosa, total_aprobado],
        hole=0.5,
        marker_colors=["#ff2b2b", "#29b09d"],
        textinfo="percent+label",
    )])
    fig.update_layout(
        title="Composicion del valor (contabilizada)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
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


def chart_top_prestadores_devolucion(df_dev, top_n=15):
    top = df_dev.nlargest(top_n, "valor_glosa")
    fig = px.bar(
        top,
        x="valor_glosa",
        y="razon_social",
        orientation="h",
        color_discrete_sequence=["#ff8700"],
        labels={"valor_glosa": "Valor Glosa", "razon_social": "Prestador"},
        title=f"Top {top_n} prestadores por valor de glosa en devolucion",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
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

_METRIC_CSS = """
<style>
[data-testid="stMetric"] {
    padding: 10px 8px;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 0.8rem;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.15rem;
}
</style>
"""


def render_dashboard_page():
    """Renderiza la pagina del Dashboard BI."""
    st.markdown(_METRIC_CSS, unsafe_allow_html=True)
    st.title("Dashboard BI - Conciliacion")

    # --- Load data ---
    with st.spinner("Cargando datos desde Snowflake..."):
        df_prest = load_prestadores_resumen()
        df_clas = load_clasificacion_conciliacion()
        df_dev = load_devoluciones_resumen()

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

    if st.sidebar.button("Limpiar cache", help="Recargar datos desde Snowflake"):
        load_prestadores_resumen.clear()
        load_clasificacion_conciliacion.clear()
        load_facturas_prestador.clear()
        load_devoluciones_resumen.clear()
        st.rerun()

    # --- Apply filters ---
    df_filtered = apply_filters(df_prest, selected_nits, [], [], None, None)

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
    tab_overview, tab_conciliacion, tab_facturas, tab_devoluciones, tab_detalle = st.tabs([
        "Vision General", "Conciliacion", "Facturas por Prestador", "Devoluciones", "Detalle y Exportacion",
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
        c2.metric("Facturas", _fmt_num_compact(total_facturas))
        c3.metric("Servicios", _fmt_num_compact(total_servicios))
        c4.metric("Valor Total", _fmt_cop_compact(total_valor))
        c5.metric("Valor Glosa", _fmt_cop_compact(total_glosa))
        c6.metric("% Glosa", f"{pct_glosa}%")

        st.markdown("---")

        # Charts row 1
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(
                chart_composicion_valor(total_glosa, total_aprobado),
                use_container_width=True,
            )
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

        # Conciliacion donut + top prestadores conciliacion
        col_conc1, col_conc2 = st.columns(2)
        with col_conc1:
            st.plotly_chart(chart_conciliacion_donut(df_agg), use_container_width=True, key="overview_conc_donut")
        with col_conc2:
            st.plotly_chart(
                chart_top_prestadores_conciliacion(df_agg_prestador),
                use_container_width=True,
                key="overview_top_conc",
            )

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
        c1.metric("Servicios conciliados", _fmt_num_compact(serv_conc))
        c2.metric("Sin conciliar", _fmt_num_compact(serv_no_conc))
        c3.metric("% Conciliado (serv.)", f"{pct_conc_serv}%")
        c4.metric("Glosa conciliada", _fmt_cop_compact(total_conc))
        c5.metric("Glosa sin conciliar", _fmt_cop_compact(total_no_conc))
        c6.metric("% Conciliado ($)", f"{pct_conc_valor}%")

        st.markdown("---")

        # Valores de conciliacion
        st.subheader("Valores de conciliacion")
        v1, v2, v3 = st.columns(3)
        v1.metric("Aceptado EPS", _fmt_cop_compact(df_agg["valor_aceptado_eps"].sum()))
        v2.metric("Aceptado IPS", _fmt_cop_compact(df_agg["valor_aceptado_ips"].sum()))
        v3.metric("Ratificado EPS", _fmt_cop_compact(df_agg["valor_ratificado"].sum()))

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
    # TAB 3: Facturas por Prestador
    # -------------------------------------------------------------------
    with tab_facturas:
        st.subheader("Facturas por Prestador")
        st.caption("Seleccione un prestador para ver el detalle de sus facturas.")

        # Selector de prestador
        nit_opts_fact = sorted(df_agg_prestador["nit"].unique().tolist())
        razon_map_fact = df_agg_prestador.set_index("nit")["razon_social"].to_dict()
        nit_display_fact = [f"{n} - {razon_map_fact.get(n, '')}" for n in nit_opts_fact]
        selected_fact = st.selectbox("Prestador", nit_display_fact, index=None, placeholder="Seleccione un prestador...")

        if selected_fact:
            nit_sel = nit_opts_fact[nit_display_fact.index(selected_fact)]

            with st.spinner("Consultando facturas..."):
                df_fact = load_facturas_prestador(nit_sel)

            if df_fact.empty:
                st.info("No se encontraron facturas para este prestador.")
            else:
                # KPIs del prestador
                fact_total = df_fact["numero_factura"].nunique()
                fact_servicios = int(df_fact["cantidad_servicios"].sum())
                fact_valor = df_fact["valor_total"].sum()
                fact_glosa = df_fact["valor_glosa"].sum()
                fact_conc = df_fact["glosa_conciliada"].sum()
                fact_pct = _pct(fact_conc, fact_glosa)

                fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
                fc1.metric("Facturas", f"{fact_total:,}")
                fc2.metric("Servicios", _fmt_num_compact(fact_servicios))
                fc3.metric("Valor Total", _fmt_cop_compact(fact_valor))
                fc4.metric("Valor Glosa", _fmt_cop_compact(fact_glosa))
                fc5.metric("Glosa Conciliada", _fmt_cop_compact(fact_conc))
                fc6.metric("% Conciliado", f"{fact_pct}%")

                st.markdown("---")

                # Grafico resumen de valores del prestador
                labels_vals = [
                    "Valor Total", "Valor Glosa", "Valor Aprobado",
                    "Aceptado EPS", "Aceptado IPS", "Ratificado",
                ]
                values_vals = [
                    fact_valor,
                    fact_glosa,
                    df_fact["valor_aprobado"].sum(),
                    df_fact["valor_aceptado_eps"].sum(),
                    df_fact["valor_aceptado_ips"].sum(),
                    df_fact["valor_ratificado"].sum(),
                ]
                colors_vals = ["#0068c9", "#ff2b2b", "#29b09d", "#83c9ff", "#7defa1", "#ff8700"]
                fig_vals = go.Figure(data=[go.Bar(
                    x=labels_vals,
                    y=values_vals,
                    marker_color=colors_vals,
                    text=[_fmt_cop_compact(v) for v in values_vals],
                    textposition="outside",
                )])
                fig_vals.update_layout(
                    title="Resumen de valores del prestador",
                    yaxis_title="Valor ($)",
                    height=450,
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_vals, use_container_width=True)

                # Tabla de facturas
                st.subheader("Detalle de facturas")

                # Buscador de factura
                buscar_factura = st.text_input("Buscar factura", placeholder="Escriba el numero de factura...", key="buscar_factura")
                df_fact_show = df_fact.copy()
                if buscar_factura:
                    df_fact_show = df_fact_show[df_fact_show["numero_factura"].astype(str).str.contains(buscar_factura, case=False, na=False)]

                df_fact_display = df_fact_show.sort_values("valor_glosa", ascending=False).copy()
                money_cols_fact = [
                    "valor_total", "valor_glosa", "valor_aprobado",
                    "glosa_conciliada", "glosa_no_conciliada",
                    "valor_aceptado_eps", "valor_aceptado_ips", "valor_ratificado",
                ]
                for col in money_cols_fact:
                    if col in df_fact_display.columns:
                        df_fact_display[col] = df_fact_display[col].apply(_fmt_cop)

                st.dataframe(
                    df_fact_display,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                    column_config={
                        "numero_factura": st.column_config.Column("No. Factura", width="small"),
                        "fecha_factura": st.column_config.Column("Fecha", width="small"),
                        "modalidad_homologada": st.column_config.Column("Modalidad"),
                        "regimen": st.column_config.Column("Regimen"),
                        "estado": st.column_config.Column("Estado"),
                        "cantidad_servicios": st.column_config.Column("Servicios"),
                        "valor_total": st.column_config.Column("Valor Total"),
                        "valor_glosa": st.column_config.Column("Valor Glosa"),
                        "valor_aprobado": st.column_config.Column("Valor Aprobado"),
                        "servicios_conciliados": st.column_config.Column("Serv. Conciliados"),
                        "servicios_no_conciliados": st.column_config.Column("Sin Conciliar"),
                        "glosa_conciliada": st.column_config.Column("Glosa Conciliada"),
                        "glosa_no_conciliada": st.column_config.Column("Glosa Sin Conciliar"),
                        "valor_aceptado_eps": st.column_config.Column("Aceptado EPS"),
                        "valor_aceptado_ips": st.column_config.Column("Aceptado IPS"),
                        "valor_ratificado": st.column_config.Column("Ratificado"),
                    },
                )

                # Exportar
                csv_fact = df_fact_show.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Descargar facturas (CSV)",
                    data=csv_fact,
                    file_name=f"facturas_{nit_sel}.csv",
                    mime="text/csv",
                )

    # -------------------------------------------------------------------
    # TAB 4: Devoluciones
    # -------------------------------------------------------------------
    with tab_devoluciones:
        st.subheader("Devoluciones")
        st.caption("Facturas con estado **devolucion** — no incluidas en los indicadores generales.")

        if df_dev.empty:
            st.info("No se encontraron devoluciones.")
        else:
            dev_prestadores = df_dev["nit"].nunique()
            dev_facturas = int(df_dev["cantidad_facturas"].sum())
            dev_servicios = int(df_dev["cantidad_servicios"].sum())
            dev_valor_total = df_dev["valor_total"].sum()
            dev_valor_glosa = df_dev["valor_glosa"].sum()
            dev_pct_glosa = _pct(dev_valor_glosa, dev_valor_total) if dev_valor_total else 0

            d1, d2, d3, d4, d5, d6 = st.columns(6)
            d1.metric("Prestadores", f"{dev_prestadores:,}")
            d2.metric("Facturas", _fmt_num_compact(dev_facturas))
            d3.metric("Servicios", _fmt_num_compact(dev_servicios))
            d4.metric("Valor Total", _fmt_cop_compact(dev_valor_total))
            d5.metric("Valor Glosa", _fmt_cop_compact(dev_valor_glosa))
            d6.metric("% Glosa", f"{dev_pct_glosa}%")

            st.markdown("---")

            # Agregar por prestador para chart y tabla
            df_dev_prest = (
                df_dev.groupby(["nit", "razon_social"], as_index=False)
                .agg(
                    cantidad_facturas=("cantidad_facturas", "sum"),
                    cantidad_servicios=("cantidad_servicios", "sum"),
                    valor_total=("valor_total", "sum"),
                    valor_glosa=("valor_glosa", "sum"),
                    valor_aprobado=("valor_aprobado", "sum"),
                )
            )

            col_dev1, col_dev2 = st.columns(2)
            with col_dev1:
                st.plotly_chart(
                    chart_top_prestadores_devolucion(df_dev_prest),
                    use_container_width=True,
                )
            with col_dev2:
                # Devoluciones por modalidad
                dev_mod = (
                    df_dev.groupby("modalidad_homologada", as_index=False)
                    .agg(valor_glosa=("valor_glosa", "sum"))
                    .sort_values("valor_glosa", ascending=False)
                )
                fig_dev_mod = px.bar(
                    dev_mod,
                    x="modalidad_homologada",
                    y="valor_glosa",
                    color="modalidad_homologada",
                    color_discrete_sequence=COLOR_PALETTE,
                    labels={"modalidad_homologada": "Modalidad", "valor_glosa": "Valor Glosa"},
                    title="Devoluciones por modalidad",
                )
                fig_dev_mod.update_layout(
                    showlegend=False,
                    height=500,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_dev_mod, use_container_width=True)

            # Tabla detalle
            st.subheader("Detalle por prestador")
            df_dev_display = df_dev_prest.sort_values("valor_glosa", ascending=False).copy()
            for col in ["valor_total", "valor_glosa", "valor_aprobado"]:
                df_dev_display[col] = df_dev_display[col].apply(_fmt_cop)
            st.dataframe(
                df_dev_display,
                use_container_width=True,
                hide_index=True,
                height=500,
                column_config={
                    "nit": st.column_config.Column("NIT", width="small"),
                    "razon_social": st.column_config.Column("Razon Social", width="medium"),
                    "cantidad_facturas": st.column_config.Column("Facturas"),
                    "cantidad_servicios": st.column_config.Column("Servicios"),
                    "valor_total": st.column_config.Column("Valor Total"),
                    "valor_glosa": st.column_config.Column("Valor Glosa"),
                    "valor_aprobado": st.column_config.Column("Valor Aprobado"),
                },
            )

            # Exportar
            csv_dev = df_dev_prest.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar devoluciones (CSV)",
                data=csv_dev,
                file_name="dashboard_devoluciones.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------------
    # TAB 5: Detalle y Exportacion
    # -------------------------------------------------------------------
    with tab_detalle:
        st.subheader("Detalle por prestador")

        buscar_prestador = st.text_input("Buscar por NIT o nombre", placeholder="Escriba NIT o razon social...", key="buscar_prestador_detalle")

        df_table = df_agg_prestador.copy()
        if buscar_prestador:
            mask = (
                df_table["nit"].astype(str).str.contains(buscar_prestador, case=False, na=False)
                | df_table["razon_social"].astype(str).str.contains(buscar_prestador, case=False, na=False)
            )
            df_table = df_table[mask]
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
