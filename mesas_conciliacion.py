"""
Mesas de Conciliacion - Modulo para gestionar mesas de conciliacion.
Importacion Excel, tabla editable con persistencia MySQL e historial de cambios.
"""

import io
import os
import datetime
from decimal import Decimal

import pandas as pd
import plotly.express as px
import pymysql
import streamlit as st


def get_db_config():
    """Retorna (cfg_dict, missing_keys) leyendo variables de entorno MySQL."""
    _ENV = {
        "host": "MYSQL_HOST",
        "port": "MYSQL_PORT",
        "database": "MYSQL_DB",
        "user": "MYSQL_USER",
        "password": "MYSQL_PASSWORD",
    }
    cfg = {}
    missing = []
    for key, env in _ENV.items():
        value = os.getenv(env, "")
        cfg[key] = value
        if not value and key != "port":
            missing.append(env)
    port = cfg.get("port")
    try:
        cfg["port"] = int(port) if port else 3306
    except ValueError as exc:
        raise ValueError(f"MYSQL_PORT inválido: {port}") from exc
    return cfg, missing

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DB_COLUMNS = [
    "regional",
    "fecha",
    "hora_atencion",
    "nit",
    "ips",
    "red",
    "acta_utam",
    "estado_presentacion",
    "acta_reunion_coosalud",
    "observacion",
    "valor_glosa_conciliar",
    "valor_aceptado_eps",
    "valor_aceptado_ips",
    "valor_adicional",
    "valor_aceptado_eps_antiguas",
    "valor_aceptado_ips_antiguas",
    "valor_conciliar_antiguas_ratificadas",
    "eps_acepta",
    "ips_acepta",
    "glosa_conciliar",
    "eps_acepta_nuevas",
    "ips_acepta_nuevas",
    "glosa_pendiente_conciliar",
]

# Columnas monetarias (DECIMAL) — para formateo y parsing
MONEY_COLUMNS = [
    "valor_glosa_conciliar",
    "valor_aceptado_eps",
    "valor_aceptado_ips",
    "valor_adicional",
    "valor_aceptado_eps_antiguas",
    "valor_aceptado_ips_antiguas",
    "valor_conciliar_antiguas_ratificadas",
    "eps_acepta",
    "ips_acepta",
    "glosa_conciliar",
    "eps_acepta_nuevas",
    "ips_acepta_nuevas",
    "glosa_pendiente_conciliar",
]

REGIONAL_OPTIONS = ["CARIBE NORTE", "VALLE DEL CAUCA", "ANTIOQUIA", "NORTE DE SANTANDER"]
ESTADO_OPTIONS = ["CONCILIADA", "INASISTENCIA", "EN CONCILIACIÓN", "SIN INFORMACIÓN"]
RED_OPTIONS = ["Privada", "Mixta", "Pública"]

# Mapeo posicional Excel -> nombre de columna DB
EXCEL_POS_MAP = {
    0: "regional",
    1: "fecha",
    2: "hora_atencion",
    3: "nit",
    4: "ips",
    5: "red",
    6: "acta_utam",
    7: "estado_presentacion",
    8: "acta_reunion_coosalud",
    9: "observacion",
    10: "valor_glosa_conciliar",
    11: "valor_aceptado_eps",
    12: "valor_aceptado_ips",
    13: "valor_adicional",
    14: "valor_aceptado_eps_antiguas",
    15: "valor_aceptado_ips_antiguas",
    16: "valor_conciliar_antiguas_ratificadas",
    17: "eps_acepta",
    18: "ips_acepta",
    19: "glosa_conciliar",
    20: "eps_acepta_nuevas",
    21: "ips_acepta_nuevas",
    22: "glosa_pendiente_conciliar",
}

# ---------------------------------------------------------------------------
# DDL — auto-creacion de tablas
# ---------------------------------------------------------------------------

DDL_MAIN = """
CREATE TABLE IF NOT EXISTS mesas_conciliacion (
    id INT AUTO_INCREMENT PRIMARY KEY,
    regional VARCHAR(100),
    fecha DATE,
    hora_atencion TIME,
    nit VARCHAR(20) NOT NULL,
    ips VARCHAR(255) NOT NULL,
    red VARCHAR(50),
    acta_utam VARCHAR(100),
    estado_presentacion VARCHAR(100),
    acta_reunion_coosalud VARCHAR(255),
    observacion TEXT,
    valor_glosa_conciliar DECIMAL(18,2),
    valor_aceptado_eps DECIMAL(18,2),
    valor_aceptado_ips DECIMAL(18,2),
    valor_adicional DECIMAL(18,2),
    valor_aceptado_eps_antiguas DECIMAL(18,2),
    valor_aceptado_ips_antiguas DECIMAL(18,2),
    valor_conciliar_antiguas_ratificadas DECIMAL(18,2),
    eps_acepta DECIMAL(18,2),
    ips_acepta DECIMAL(18,2),
    glosa_conciliar DECIMAL(18,2),
    eps_acepta_nuevas DECIMAL(18,2),
    ips_acepta_nuevas DECIMAL(18,2),
    glosa_pendiente_conciliar DECIMAL(18,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

DDL_HISTORY = """
CREATE TABLE IF NOT EXISTS mesas_conciliacion_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    mesa_id INT NOT NULL,
    field_name VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by VARCHAR(255),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_mesa_changed (mesa_id, changed_at),
    CONSTRAINT fk_mesa_history FOREIGN KEY (mesa_id)
        REFERENCES mesas_conciliacion(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_connection():
    """Crea conexion MySQL usando la configuracion del proyecto."""
    cfg, missing = get_db_config()
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")
    return pymysql.connect(**cfg, cursorclass=pymysql.cursors.DictCursor)


def _ensure_tables():
    """Crea las tablas si no existen."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL_MAIN)
            cur.execute(DDL_HISTORY)
        conn.commit()
    finally:
        conn.close()


def _swap_sep(s):
    """Intercambia separadores: , -> . y . -> , (formato colombiano)."""
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_cop(val):
    """Formatea un numero como moneda colombiana."""
    if pd.isna(val) or val is None:
        return "$0"
    try:
        return "$" + _swap_sep(f"{float(val):,.0f}")
    except (ValueError, TypeError):
        return "$0"


def _parse_money(val):
    """Convierte un valor mixto (int, float, str con separadores) a float o None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float, Decimal)):
        return float(val)
    # Limpiar string: quitar $, espacios, puntos de miles
    s = str(val).strip().replace("$", "").replace(" ", "")
    if not s or s.lower() == "nan":
        return None
    # Si tiene coma como decimal (formato colombiano)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _load_data():
    """Carga todos los registros de mesas_conciliacion y retorna un DataFrame."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, " + ", ".join(DB_COLUMNS) + ", created_at, updated_at, created_by "
                "FROM mesas_conciliacion ORDER BY id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=["id"] + DB_COLUMNS + ["created_at", "updated_at", "created_by"])

    df = pd.DataFrame(rows)
    # Asegurar tipos correctos
    for col in MONEY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    return df


def _load_history(limit=200, user_filter=None, field_filter=None, date_start=None, date_end=None):
    """Carga registros del historial de cambios con filtros opcionales."""
    conn = _get_connection()
    try:
        where_clauses = ["1=1"]
        params = []

        if user_filter:
            where_clauses.append("h.changed_by = %s")
            params.append(user_filter)
        if field_filter:
            where_clauses.append("h.field_name = %s")
            params.append(field_filter)
        if date_start:
            where_clauses.append("DATE(h.changed_at) >= %s")
            params.append(date_start)
        if date_end:
            where_clauses.append("DATE(h.changed_at) <= %s")
            params.append(date_end)

        where_sql = " AND ".join(where_clauses)
        sql = (
            "SELECT h.id, h.mesa_id, h.field_name, h.old_value, h.new_value, "
            "h.changed_by, h.changed_at, m.nit, m.ips "
            "FROM mesas_conciliacion_history h "
            "LEFT JOIN mesas_conciliacion m ON m.id = h.mesa_id "
            f"WHERE {where_sql} "
            "ORDER BY h.changed_at DESC "
            f"LIMIT {int(limit)}"
        )
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            "id", "mesa_id", "field_name", "old_value", "new_value",
            "changed_by", "changed_at", "nit", "ips",
        ])
    return pd.DataFrame(rows)


def _get_history_filter_options():
    """Obtiene valores unicos de usuario y campo para filtros del historial."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT changed_by FROM mesas_conciliacion_history ORDER BY changed_by")
            users = [r["changed_by"] for r in cur.fetchall() if r["changed_by"]]
            cur.execute("SELECT DISTINCT field_name FROM mesas_conciliacion_history ORDER BY field_name")
            fields = [r["field_name"] for r in cur.fetchall() if r["field_name"]]
    finally:
        conn.close()
    return users, fields


def _export_to_excel(df):
    """Genera un archivo Excel en memoria con formato COP para columnas monetarias."""
    output = io.BytesIO()
    # Preparar DataFrame para exportar con etiquetas legibles
    df_export = df.copy()

    # Renombrar columnas con etiquetas legibles
    rename_map = {col: DISPLAY_LABELS.get(col, col) for col in df_export.columns}
    df_export = df_export.rename(columns=rename_map)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Mesas Conciliación")

        # Aplicar formato COP a columnas monetarias
        ws = writer.sheets["Mesas Conciliación"]
        money_labels = {DISPLAY_LABELS.get(c, c) for c in MONEY_COLUMNS}

        for col_idx, col_name in enumerate(df_export.columns, 1):
            if col_name in money_labels:
                for row_idx in range(2, len(df_export) + 2):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if cell.value is not None:
                        cell.number_format = '$#,##0'

        # Ajustar ancho de columnas
        for col_idx, col_name in enumerate(df_export.columns, 1):
            max_len = max(len(str(col_name)), 12)
            ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 30)

    output.seek(0)
    return output


def _import_excel_df(uploaded_file):
    """Lee un archivo Excel usando mapeo posicional y retorna DataFrame limpio."""
    raw = pd.read_excel(uploaded_file, header=0)

    if raw.shape[1] < 23:
        raise ValueError(
            f"El archivo tiene {raw.shape[1]} columnas, se esperan al menos 23."
        )

    # Mapeo posicional: usar iloc para cada columna
    data = {}
    for pos, db_col in EXCEL_POS_MAP.items():
        if pos < raw.shape[1]:
            data[db_col] = raw.iloc[:, pos].values
        else:
            data[db_col] = None

    df = pd.DataFrame(data)

    # Limpiar tipos
    # Fecha
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

    # Hora — puede venir como datetime.time o string
    def _parse_time(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, datetime.time):
            return v
        try:
            return pd.to_datetime(str(v), format="%H:%M:%S").time()
        except Exception:
            try:
                return pd.to_datetime(str(v), format="%H:%M").time()
            except Exception:
                return None

    df["hora_atencion"] = df["hora_atencion"].apply(_parse_time)

    # NIT como string
    df["nit"] = df["nit"].apply(lambda v: str(int(v)) if pd.notna(v) and isinstance(v, (int, float)) else str(v) if pd.notna(v) else "")

    # Columnas monetarias
    for col in MONEY_COLUMNS:
        df[col] = df[col].apply(_parse_money)

    return df


def _bulk_insert(df, user_email):
    """Inserta un DataFrame completo en mesas_conciliacion."""
    if df.empty:
        return 0

    cols = DB_COLUMNS + ["created_by"]
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO mesas_conciliacion ({', '.join(cols)}) VALUES ({placeholders})"

    conn = _get_connection()
    try:
        inserted = 0
        with conn.cursor() as cur:
            batch = []
            for _, row in df.iterrows():
                vals = []
                for c in DB_COLUMNS:
                    v = row.get(c)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        vals.append(None)
                    elif isinstance(v, datetime.time):
                        vals.append(v.strftime("%H:%M:%S"))
                    elif isinstance(v, datetime.date):
                        vals.append(v.isoformat())
                    else:
                        vals.append(v)
                vals.append(user_email)
                batch.append(tuple(vals))

                if len(batch) >= 500:
                    cur.executemany(sql, batch)
                    inserted += len(batch)
                    batch = []

            if batch:
                cur.executemany(sql, batch)
                inserted += len(batch)

        conn.commit()
        return inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _save_changes(original_df, edited_df, user_email):
    """Detecta cambios entre el DataFrame original y el editado.
    Aplica UPDATE, INSERT y DELETE en transaccion. Registra historial."""

    conn = _get_connection()
    try:
        updated_count = 0
        inserted_count = 0
        deleted_count = 0
        history_records = []

        orig_ids = set(original_df["id"].dropna().astype(int)) if not original_df.empty else set()
        edit_ids = set(edited_df["id"].dropna().astype(int)) if "id" in edited_df.columns and not edited_df.empty else set()

        with conn.cursor() as cur:
            # --- DELETE: filas que estaban en original pero no en editado ---
            ids_to_delete = orig_ids - edit_ids
            if ids_to_delete:
                # Registrar historial antes de borrar
                for del_id in ids_to_delete:
                    history_records.append((int(del_id), "_deleted", "true", "row deleted", user_email))
                placeholders = ", ".join(["%s"] * len(ids_to_delete))
                cur.execute(f"DELETE FROM mesas_conciliacion WHERE id IN ({placeholders})", tuple(ids_to_delete))
                deleted_count = len(ids_to_delete)

            # --- UPDATE: filas que existen en ambos ---
            ids_to_check = orig_ids & edit_ids
            for row_id in ids_to_check:
                orig_row = original_df[original_df["id"] == row_id].iloc[0]
                edit_row = edited_df[edited_df["id"] == row_id].iloc[0]

                changes = {}
                for col in DB_COLUMNS:
                    old_val = orig_row.get(col)
                    new_val = edit_row.get(col)

                    # Normalizar NaN/None
                    old_is_null = old_val is None or (isinstance(old_val, float) and pd.isna(old_val))
                    new_is_null = new_val is None or (isinstance(new_val, float) and pd.isna(new_val))

                    if old_is_null and new_is_null:
                        continue
                    if old_is_null != new_is_null or str(old_val) != str(new_val):
                        changes[col] = (old_val, new_val)

                if changes:
                    set_clause = ", ".join([f"{c} = %s" for c in changes])
                    vals = []
                    for c in changes:
                        v = changes[c][1]  # new value
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            vals.append(None)
                        elif isinstance(v, datetime.time):
                            vals.append(v.strftime("%H:%M:%S"))
                        elif isinstance(v, datetime.date):
                            vals.append(v.isoformat())
                        else:
                            vals.append(v)
                    vals.append(int(row_id))
                    cur.execute(
                        f"UPDATE mesas_conciliacion SET {set_clause} WHERE id = %s",
                        tuple(vals),
                    )
                    updated_count += 1

                    for col, (old_v, new_v) in changes.items():
                        old_str = None if (old_v is None or (isinstance(old_v, float) and pd.isna(old_v))) else str(old_v)
                        new_str = None if (new_v is None or (isinstance(new_v, float) and pd.isna(new_v))) else str(new_v)
                        history_records.append((int(row_id), col, old_str, new_str, user_email))

            # --- INSERT: filas nuevas (sin id o id que no estaba en original) ---
            new_rows = edited_df[
                edited_df["id"].isna() | ~edited_df["id"].isin(orig_ids)
            ] if "id" in edited_df.columns else edited_df

            if not new_rows.empty:
                cols_insert = DB_COLUMNS + ["created_by"]
                placeholders = ", ".join(["%s"] * len(cols_insert))
                insert_sql = f"INSERT INTO mesas_conciliacion ({', '.join(cols_insert)}) VALUES ({placeholders})"

                for _, row in new_rows.iterrows():
                    vals = []
                    for c in DB_COLUMNS:
                        v = row.get(c)
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            vals.append(None)
                        elif isinstance(v, datetime.time):
                            vals.append(v.strftime("%H:%M:%S"))
                        elif isinstance(v, datetime.date):
                            vals.append(v.isoformat())
                        else:
                            vals.append(v)
                    vals.append(user_email)
                    cur.execute(insert_sql, tuple(vals))
                    inserted_count += 1

            # --- Historial ---
            if history_records:
                cur.executemany(
                    "INSERT INTO mesas_conciliacion_history "
                    "(mesa_id, field_name, old_value, new_value, changed_by) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    history_records,
                )

        conn.commit()
        return updated_count, inserted_count, deleted_count

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# UI — Pagina principal
# ---------------------------------------------------------------------------

# Etiquetas legibles para la tabla
DISPLAY_LABELS = {
    "id": "ID",
    "regional": "Regional",
    "fecha": "Fecha",
    "hora_atencion": "Hora Atención",
    "nit": "NIT",
    "ips": "IPS",
    "red": "RED",
    "acta_utam": "Acta UTAM",
    "estado_presentacion": "Estado Presentación",
    "acta_reunion_coosalud": "Acta Reunión Coosalud",
    "observacion": "Observación",
    "valor_glosa_conciliar": "Valor Glosa a Conciliar",
    "valor_aceptado_eps": "Valor Aceptado EPS",
    "valor_aceptado_ips": "Valor Aceptado IPS",
    "valor_adicional": "Valor Adicional",
    "valor_aceptado_eps_antiguas": "Valor Aceptado EPS (Antiguas)",
    "valor_aceptado_ips_antiguas": "Valor Aceptado IPS (Antiguas)",
    "valor_conciliar_antiguas_ratificadas": "Valor Conciliar Antiguas Ratificadas",
    "eps_acepta": "EPS Acepta",
    "ips_acepta": "IPS Acepta",
    "glosa_conciliar": "Glosa a Conciliar",
    "eps_acepta_nuevas": "EPS Acepta (Nuevas)",
    "ips_acepta_nuevas": "IPS Acepta (Nuevas)",
    "glosa_pendiente_conciliar": "Glosa Pendiente de Conciliar",
    "created_at": "Creado",
    "updated_at": "Actualizado",
    "created_by": "Creado por",
}


def _build_column_config():
    """Construye column_config para st.data_editor."""
    config = {}

    config["id"] = st.column_config.NumberColumn("ID", disabled=True)
    config["created_at"] = st.column_config.DatetimeColumn("Creado", disabled=True)
    config["updated_at"] = st.column_config.DatetimeColumn("Actualizado", disabled=True)
    config["created_by"] = st.column_config.TextColumn("Creado por", disabled=True)

    config["regional"] = st.column_config.SelectboxColumn(
        "Regional", options=REGIONAL_OPTIONS, required=False
    )
    config["estado_presentacion"] = st.column_config.SelectboxColumn(
        "Estado Presentación", options=ESTADO_OPTIONS, required=False
    )
    config["red"] = st.column_config.SelectboxColumn(
        "RED", options=RED_OPTIONS, required=False
    )
    config["fecha"] = st.column_config.DateColumn("Fecha", format="YYYY-MM-DD")
    config["nit"] = st.column_config.TextColumn("NIT", required=True)
    config["ips"] = st.column_config.TextColumn("IPS", required=True)
    config["acta_utam"] = st.column_config.TextColumn("Acta UTAM")
    config["acta_reunion_coosalud"] = st.column_config.TextColumn("Acta Reunión Coosalud")
    config["observacion"] = st.column_config.TextColumn("Observación", width="large")

    for col in MONEY_COLUMNS:
        label = DISPLAY_LABELS.get(col, col)
        config[col] = st.column_config.NumberColumn(
            label, format="$ %,.0f"
        )

    return config


def _render_filters(df):
    """Muestra filtros y retorna el DataFrame filtrado."""
    st.subheader("Filtros")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        regionales = ["Todas"] + sorted(df["regional"].dropna().unique().tolist())
        sel_regional = st.selectbox("Regional", regionales, key="mc_filter_regional")

    with c2:
        estados = ["Todos"] + sorted(df["estado_presentacion"].dropna().unique().tolist())
        sel_estado = st.selectbox("Estado", estados, key="mc_filter_estado")

    with c3:
        redes = ["Todas"] + sorted(df["red"].dropna().unique().tolist())
        sel_red = st.selectbox("RED", redes, key="mc_filter_red")

    with c4:
        fechas_validas = df["fecha"].dropna()
        if not fechas_validas.empty:
            min_f = min(fechas_validas)
            max_f = max(fechas_validas)
            sel_fechas = st.date_input(
                "Rango de fechas",
                value=(min_f, max_f),
                min_value=min_f,
                max_value=max_f,
                key="mc_filter_fechas",
            )
        else:
            sel_fechas = None

    filtered = df.copy()
    if sel_regional != "Todas":
        filtered = filtered[filtered["regional"] == sel_regional]
    if sel_estado != "Todos":
        filtered = filtered[filtered["estado_presentacion"] == sel_estado]
    if sel_red != "Todas":
        filtered = filtered[filtered["red"] == sel_red]
    if sel_fechas and isinstance(sel_fechas, tuple) and len(sel_fechas) == 2:
        start, end = sel_fechas
        mask = filtered["fecha"].apply(
            lambda v: start <= v <= end if pd.notna(v) else False
        )
        filtered = filtered[mask]

    return filtered


def _render_tab_tabla(user_email):
    """Tab 1: Tabla editable con filtros y guardado."""
    df_full = _load_data()

    if df_full.empty:
        st.info("No hay datos cargados. Use la pestaña **Importar Excel** para cargar datos.")
        return

    st.caption(f"Total registros: {len(df_full)}")

    df_filtered = _render_filters(df_full)
    st.caption(f"Registros filtrados: {len(df_filtered)}")

    # Guardar copia original para detectar cambios
    if "mesas_original_df" not in st.session_state or st.session_state.get("mesas_reload", False):
        st.session_state["mesas_original_df"] = df_filtered.copy()
        st.session_state["mesas_reload"] = False

    column_config = _build_column_config()

    # Columnas a mostrar en el editor (ocultar metadata)
    display_cols = ["id"] + DB_COLUMNS
    df_edit = df_filtered[display_cols].copy()

    edited_df = st.data_editor(
        df_edit,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="mesas_data_editor",
    )

    col_save, col_reload = st.columns([1, 1])

    with col_save:
        if st.button("Guardar cambios", type="primary", key="mc_save_btn"):
            try:
                original = st.session_state.get("mesas_original_df", df_filtered)
                updated, inserted, deleted = _save_changes(original, edited_df, user_email)
                total = updated + inserted + deleted
                if total == 0:
                    st.info("No se detectaron cambios.")
                else:
                    parts = []
                    if updated:
                        parts.append(f"{updated} actualizados")
                    if inserted:
                        parts.append(f"{inserted} insertados")
                    if deleted:
                        parts.append(f"{deleted} eliminados")
                    st.success(f"Guardado exitoso: {', '.join(parts)}.")
                    st.session_state["mesas_reload"] = True
                    st.session_state.pop("mesas_original_df", None)
            except Exception as e:
                st.error(f"Error al guardar: {e}")

    with col_reload:
        if st.button("Recargar datos", key="mc_reload_btn"):
            st.session_state["mesas_reload"] = True
            st.session_state.pop("mesas_original_df", None)
            st.rerun()

    # --- Exportar a Excel ---
    st.divider()
    excel_data = _export_to_excel(df_filtered)
    st.download_button(
        label="Descargar Excel",
        data=excel_data,
        file_name=f"mesas_conciliacion_{datetime.date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="mc_export_excel",
    )

    # --- Historial de cambios ---
    with st.expander("Historial de cambios recientes", expanded=False):
        try:
            users, fields = _get_history_filter_options()
        except Exception:
            users, fields = [], []

        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            sel_user = st.selectbox(
                "Usuario", ["Todos"] + users, key="mc_hist_user"
            )
        with hc2:
            sel_field = st.selectbox(
                "Campo", ["Todos"] + fields, key="mc_hist_field"
            )
        with hc3:
            hist_date_start = st.date_input(
                "Desde", value=None, key="mc_hist_date_start"
            )
        with hc4:
            hist_date_end = st.date_input(
                "Hasta", value=None, key="mc_hist_date_end"
            )

        try:
            df_hist = _load_history(
                limit=200,
                user_filter=sel_user if sel_user != "Todos" else None,
                field_filter=sel_field if sel_field != "Todos" else None,
                date_start=hist_date_start,
                date_end=hist_date_end,
            )
        except Exception as e:
            st.error(f"Error al cargar historial: {e}")
            df_hist = pd.DataFrame()

        if df_hist.empty:
            st.info("No hay cambios registrados con los filtros seleccionados.")
        else:
            # Renombrar para presentacion
            df_hist_display = df_hist.rename(columns={
                "mesa_id": "ID Mesa",
                "nit": "NIT",
                "ips": "IPS",
                "field_name": "Campo",
                "old_value": "Valor anterior",
                "new_value": "Valor nuevo",
                "changed_by": "Usuario",
                "changed_at": "Fecha/Hora",
            })
            # Traducir nombres de campo a etiquetas legibles
            df_hist_display["Campo"] = df_hist_display["Campo"].map(
                lambda x: DISPLAY_LABELS.get(x, x)
            )
            st.dataframe(
                df_hist_display[["ID Mesa", "NIT", "IPS", "Campo", "Valor anterior", "Valor nuevo", "Usuario", "Fecha/Hora"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Mostrando {len(df_hist_display)} cambios (máximo 200).")


def _render_tab_import(user_email):
    """Tab 2: Importar datos desde archivo Excel."""
    st.subheader("Importar archivo Excel")
    st.caption(
        "Suba un archivo Excel (.xlsx) con la estructura de mesas de conciliación. "
        "Se usa mapeo posicional (por columna, no por nombre) para las 23 columnas."
    )

    uploaded = st.file_uploader(
        "Seleccione archivo Excel",
        type=["xlsx", "xls"],
        key="mc_excel_upload",
    )

    if uploaded is not None:
        try:
            df_preview = _import_excel_df(uploaded)
            st.success(f"Archivo leído: {len(df_preview)} filas, {df_preview.shape[1]} columnas.")

            st.subheader("Vista previa")
            # Mostrar primeras 20 filas
            st.dataframe(
                df_preview.head(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    col: st.column_config.NumberColumn(
                        DISPLAY_LABELS.get(col, col), format="$ %,.0f"
                    )
                    for col in MONEY_COLUMNS
                },
            )
            if len(df_preview) > 20:
                st.caption(f"Mostrando 20 de {len(df_preview)} filas.")

            # Opciones de importacion
            modo = st.radio(
                "Modo de importación",
                ["Agregar a datos existentes", "Reemplazar todos los datos"],
                key="mc_import_mode",
            )

            if st.button("Confirmar importación", type="primary", key="mc_confirm_import"):
                with st.spinner("Importando datos..."):
                    conn = _get_connection()
                    try:
                        if modo == "Reemplazar todos los datos":
                            with conn.cursor() as cur:
                                cur.execute("DELETE FROM mesas_conciliacion_history")
                                cur.execute("DELETE FROM mesas_conciliacion")
                            conn.commit()

                        inserted = _bulk_insert(df_preview, user_email)
                        st.success(f"Importación exitosa: {inserted} registros insertados.")
                        st.session_state["mesas_reload"] = True
                        st.session_state.pop("mesas_original_df", None)
                    except Exception as e:
                        st.error(f"Error durante la importación: {e}")
                    finally:
                        conn.close()

        except ValueError as e:
            st.error(f"Error de formato: {e}")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")


def _fmt_cop_compact(val):
    """Formatea moneda de forma compacta para tarjetas KPI."""
    if pd.isna(val) or val is None:
        return "$0"
    abs_val = abs(float(val))
    sign = "-" if float(val) < 0 else ""
    if abs_val >= 1_000_000_000:
        return f"{sign}$" + _swap_sep(f"{abs_val / 1_000_000_000:,.1f}") + "KM"
    if abs_val >= 1_000_000:
        return f"{sign}$" + _swap_sep(f"{abs_val / 1_000_000:,.0f}") + "M"
    if abs_val >= 1_000:
        return f"{sign}$" + _swap_sep(f"{abs_val / 1_000:,.0f}") + "K"
    return f"{sign}$" + _swap_sep(f"{abs_val:,.0f}")


COLOR_PALETTE = [
    "#0068c9", "#83c9ff", "#ff2b2b", "#ffabab",
    "#29b09d", "#7defa1", "#ff8700", "#ffd16a",
    "#6d3fc0", "#d5dae5",
]


def _render_tab_dashboard():
    """Tab 3: Dashboard con 5 indicadores principales."""
    df = _load_data()

    if df.empty:
        st.info("No hay datos para mostrar. Importe un archivo Excel primero.")
        return

    # =====================================================================
    # 4.1 — KPI Cards: Glosa a Conciliar vs Aceptado EPS vs Aceptado IPS
    # =====================================================================
    st.subheader("Indicadores principales")

    total_glosa = df["valor_glosa_conciliar"].sum()
    total_eps = df["valor_aceptado_eps"].sum()
    total_ips = df["valor_aceptado_ips"].sum()
    total_pendiente = df["glosa_pendiente_conciliar"].sum()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Glosa a Conciliar", _fmt_cop_compact(total_glosa))
        st.caption(_fmt_cop(total_glosa))
    with k2:
        st.metric("Valor Aceptado EPS", _fmt_cop_compact(total_eps))
        st.caption(_fmt_cop(total_eps))
    with k3:
        st.metric("Valor Aceptado IPS", _fmt_cop_compact(total_ips))
        st.caption(_fmt_cop(total_ips))
    with k4:
        st.metric("Glosa Pendiente", _fmt_cop_compact(total_pendiente))
        st.caption(_fmt_cop(total_pendiente))

    st.divider()

    # =====================================================================
    # 4.2 — % Conciliado por Regional
    # =====================================================================
    st.subheader("% Conciliado por Regional")

    regional_agg = (
        df.groupby("regional", dropna=False)
        .agg(
            glosa_total=("valor_glosa_conciliar", "sum"),
            eps_total=("valor_aceptado_eps", "sum"),
        )
        .reset_index()
    )
    regional_agg["regional"] = regional_agg["regional"].fillna("Sin Regional")
    regional_agg["pct_conciliado"] = regional_agg.apply(
        lambda r: round(r["eps_total"] / r["glosa_total"] * 100, 1) if r["glosa_total"] and r["glosa_total"] > 0 else 0.0,
        axis=1,
    )

    if not regional_agg.empty:
        fig_pct = px.bar(
            regional_agg.sort_values("pct_conciliado", ascending=True),
            x="pct_conciliado",
            y="regional",
            orientation="h",
            text="pct_conciliado",
            labels={"pct_conciliado": "% Conciliado", "regional": "Regional"},
            title="% Conciliado por Regional (Valor Aceptado EPS / Glosa a Conciliar)",
            color_discrete_sequence=[COLOR_PALETTE[0]],
        )
        fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_pct.update_layout(
            xaxis=dict(range=[0, max(regional_agg["pct_conciliado"].max() * 1.15, 10)], ticksuffix="%"),
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    st.divider()

    # =====================================================================
    # 4.3 — Estado de mesas por Regional
    # =====================================================================
    st.subheader("Estado de mesas por Regional")

    estado_regional = (
        df.groupby(["regional", "estado_presentacion"], dropna=False)
        .size()
        .reset_index(name="cantidad")
    )
    estado_regional["regional"] = estado_regional["regional"].fillna("Sin Regional")
    estado_regional["estado_presentacion"] = estado_regional["estado_presentacion"].fillna("Sin Estado")

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        if not estado_regional.empty:
            fig_estado = px.bar(
                estado_regional,
                x="regional",
                y="cantidad",
                color="estado_presentacion",
                barmode="stack",
                text_auto=True,
                labels={
                    "regional": "Regional",
                    "cantidad": "Cantidad",
                    "estado_presentacion": "Estado",
                },
                title="Estado de presentación por Regional",
                color_discrete_sequence=COLOR_PALETTE,
            )
            fig_estado.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(fig_estado, use_container_width=True)

    with col_table:
        # Tabla pivote: regional x estado
        pivot = estado_regional.pivot_table(
            index="regional",
            columns="estado_presentacion",
            values="cantidad",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)

    st.divider()

    # =====================================================================
    # 4.4 — Glosa Pendiente de Conciliar por RED
    # =====================================================================
    st.subheader("Glosa Pendiente de Conciliar por RED")

    red_agg = (
        df.groupby("red", dropna=False)
        .agg(
            glosa_pendiente=("glosa_pendiente_conciliar", "sum"),
            glosa_total=("valor_glosa_conciliar", "sum"),
            registros=("id", "count"),
        )
        .reset_index()
    )
    red_agg["red"] = red_agg["red"].fillna("Sin RED")

    if not red_agg.empty:
        # Agregar texto formateado en COP
        red_agg["texto"] = red_agg["glosa_pendiente"].apply(_fmt_cop)

        fig_red = px.bar(
            red_agg.sort_values("glosa_pendiente", ascending=False),
            x="red",
            y="glosa_pendiente",
            text="texto",
            labels={"red": "RED", "glosa_pendiente": "Glosa Pendiente"},
            title="Glosa Pendiente de Conciliar por RED (Privada / Mixta / Pública)",
            color="red",
            color_discrete_sequence=[COLOR_PALETTE[0], COLOR_PALETTE[4], COLOR_PALETTE[6]],
        )
        fig_red.update_traces(textposition="outside")
        fig_red.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(tickformat="$,.0f"),
            showlegend=False,
        )
        st.plotly_chart(fig_red, use_container_width=True)

    st.divider()

    # =====================================================================
    # 4.5 — Evolución por fecha (línea de tiempo)
    # =====================================================================
    st.subheader("Evolución por fecha")

    # Conteo de fechas nulas
    null_dates = df["fecha"].isna().sum()
    valid_dates = df[df["fecha"].notna()].copy()

    if null_dates > 0:
        st.info(f"Se excluyen {null_dates} registros sin fecha del gráfico de evolución.")

    if valid_dates.empty:
        st.warning("No hay registros con fecha para mostrar la evolución temporal.")
    else:
        # Convertir fecha a datetime para agrupacion
        valid_dates["fecha_dt"] = pd.to_datetime(valid_dates["fecha"])

        timeline = (
            valid_dates.groupby("fecha_dt")
            .agg(
                glosa_conciliar=("valor_glosa_conciliar", "sum"),
                aceptado_eps=("valor_aceptado_eps", "sum"),
                aceptado_ips=("valor_aceptado_ips", "sum"),
                pendiente=("glosa_pendiente_conciliar", "sum"),
                registros=("id", "count"),
            )
            .reset_index()
            .sort_values("fecha_dt")
        )

        # Melt para grafico de lineas multiples
        timeline_melted = timeline.melt(
            id_vars=["fecha_dt"],
            value_vars=["glosa_conciliar", "aceptado_eps", "aceptado_ips", "pendiente"],
            var_name="indicador",
            value_name="valor",
        )

        # Etiquetas legibles
        label_map = {
            "glosa_conciliar": "Glosa a Conciliar",
            "aceptado_eps": "Aceptado EPS",
            "aceptado_ips": "Aceptado IPS",
            "pendiente": "Glosa Pendiente",
        }
        timeline_melted["indicador"] = timeline_melted["indicador"].map(label_map)

        fig_timeline = px.line(
            timeline_melted,
            x="fecha_dt",
            y="valor",
            color="indicador",
            markers=True,
            labels={"fecha_dt": "Fecha", "valor": "Valor (COP)", "indicador": "Indicador"},
            title="Evolución de valores por fecha de mesa",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig_timeline.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            hovermode="x unified",
        )
        fig_timeline.update_traces(
            hovertemplate="%{y:$,.0f}"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_mesas_page():
    """Punto de entrada del modulo — llamado desde app_streamlit.py."""
    st.title("Mesas de Conciliación")
    st.caption("Gestión de mesas de conciliación: importar, editar y analizar datos.")

    # Asegurar que las tablas existen
    try:
        _ensure_tables()
    except Exception as e:
        st.error(f"Error al verificar tablas en la base de datos: {e}")
        return

    user_email = st.session_state.get("user_email", "desconocido")

    tab_tabla, tab_import, tab_dashboard = st.tabs(
        ["Tabla Editable", "Importar Excel", "Dashboard"]
    )

    with tab_tabla:
        _render_tab_tabla(user_email)

    with tab_import:
        _render_tab_import(user_email)

    with tab_dashboard:
        _render_tab_dashboard()
