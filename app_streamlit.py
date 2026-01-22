# app_streamlit.py
import os
import time
import base64
import subprocess
from pathlib import Path

from decimal import Decimal, ROUND_HALF_UP
from io import StringIO
import pandas as pd
import pymysql
import streamlit as st
from csv_conciliation_loader import EXPECTED_HEADERS

USERS_ENV_VAR = "APP_USERS"
DB_ENV_VARS = {
    "host": "MYSQL_HOST",
    "port": "MYSQL_PORT",
    "database": "MYSQL_DB",
    "user": "MYSQL_USER",
    "password": "MYSQL_PASSWORD",
}

STATE_OPTIONS = [
    "Glosa notificada",
    "Acta en Firma",
    "Pendiente notificar falta correo",
    "Actas en elaboración",
    "Acta Firmada",
    "Reprogramación",
    "En validación de soportes",
    "Inasistencia",
    "Acta Firmada - acta en elaboración",
    "Acta Firmada - acta en Firma",
    "Acta Firmada - Elaboración de acta",
]

SUMMARY_SQL = """
SELECT
  ts.nit,
  ts.razon_social,
  ts.cantidad_facturas,
  ts.cantidad_servicios,
  ts.valor_glosa,
  ts.valor_aceptado_eps,
  ts.valor_aceptado_ips,
  ts.valor_ratificado,
  ns.estado       AS estado_reciente,
  ns.comentario   AS comentario_reciente,
  ns.user         AS estado_usuario,
  ns.created_at   AS estado_fecha
FROM thirds_summary ts
LEFT JOIN (
  SELECT n1.*
  FROM nit_states n1
  JOIN (
    SELECT nit, MAX(created_at) AS mx
    FROM nit_states
    GROUP BY nit
  ) x
    ON x.nit = n1.nit AND x.mx = n1.created_at
) ns
  ON ns.nit = ts.nit
"""

APP_TITLE = "actas de conciliacion Coosalud"
IN_DIR = Path("/data/in")
OUT_DIR = Path("/data/out")
LOG_FILE = OUT_DIR / "runtime.log"

# ------------------------- Utils -------------------------

def ensure_dirs():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_users_from_env():
    raw = os.getenv(USERS_ENV_VAR, "")
    users = {}
    invalid_entries = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            invalid_entries.append(chunk)
            continue
        email, password = chunk.split(":", 1)
        email = email.strip().lower()
        password = password.strip()
        if not email or not password:
            invalid_entries.append(chunk)
            continue
        users[email] = password
    return users, invalid_entries


def rerun_app():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    elif hasattr(st, "rerun"):
        st.rerun()
    else:
        raise RuntimeError("La versión de Streamlit instalada no soporta recargar la app.")


def require_login():
    users, invalid_entries = load_users_from_env()

    if invalid_entries:
        st.warning("Entradas inválidas en APP_USERS: " + ", ".join(invalid_entries))

    if not users:
        st.error(
            "No hay usuarios configurados. Define APP_USERS (correo:contraseña, separados por comas)."
        )
        st.stop()

    if st.session_state.get("auth_ok"):
        return st.session_state.get("user_email")

    last_email = st.session_state.get("last_email", "")
    login_error = st.session_state.get("login_error")

    st.title(APP_TITLE)
    st.caption("Inicia sesión para continuar.")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Correo electrónico", value=last_email)
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")

    if submitted:
        trimmed_email = email.strip().lower()
        expected_password = users.get(trimmed_email)
        if expected_password and password == expected_password:
            st.session_state.auth_ok = True
            st.session_state.user_email = trimmed_email
            st.session_state.login_error = ""
            st.session_state.last_email = trimmed_email
            rerun_app()
        else:
            st.session_state.login_error = "Correo o contraseña incorrectos."
            st.session_state.last_email = email.strip()
            rerun_app()

    if login_error:
        st.error(login_error)

    st.stop()


def get_db_config():
    cfg = {}
    missing = []
    for key, env in DB_ENV_VARS.items():
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


@st.cache_data(ttl=300, show_spinner=False)
def load_summary_dataframe(cfg: dict):
    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        cursorclass=pymysql.cursors.Cursor,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(SUMMARY_SQL)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=columns)
    numeric_cols = [
        "cantidad_facturas",
        "cantidad_servicios",
        "valor_glosa",
        "valor_aceptado_eps",
        "valor_aceptado_ips",
        "valor_ratificado",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def clear_state_form(prefix: str) -> None:
    st.session_state.pop(f"{prefix}_estado", None)
    st.session_state.pop(f"{prefix}_comentario", None)


def clear_state_selection() -> None:
    st.session_state["selected_nit"] = None
    st.session_state["selected_razon"] = None
    st.session_state["show_state_modal"] = False


def insert_nit_state(cfg: dict, nit: str, estado: str, comentario: str, user_email: str):
    comentario = comentario.strip() or None
    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        autocommit=False,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO nit_states (nit, estado, comentario, user)
                VALUES (%s, %s, %s, %s)
                """,
                (nit, estado, comentario, user_email),
            )
            last_id = cur.lastrowid
        conn.commit()
        return last_id
    finally:
        conn.close()


def insert_acta_pdf_record(cfg: dict, id_str: str, nit: str, razon_social: str, file_name: str, file_path: str, usuario: str, nit_state_id: int | None):
    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        autocommit=False,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conciliation_acta_files_pdf
                    (id, nit, razon_social, file_name, file_path, usuario, nit_state_id, created_at, updated_at)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """,
                (id_str, nit, razon_social, file_name, file_path, usuario, nit_state_id),
            )
        conn.commit()
    finally:
        conn.close()




def parse_decimal_value(value):
    if value in (None, ""):
        return Decimal("0")
    text = str(value).strip().replace("$", "").replace(" ", "")
    if text.count(",") and text.count("."):
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    if text == "":
        return Decimal("0")
    return Decimal(text)


def format_decimal_value(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}".replace(".", ",")


def generate_percentage_sheet(file_obj, pct_ips: float, pct_eps: float, pct_rat: float):
    """Devuelve (df_out, resumen_str, csv_bytes).
       - Lee Excel (openpyxl), valida columnas mínimas.
       - Distribuye VALOR_GLOSA: IPS→EPS→RAT (tope por fila).
       - Ajusta redondeos para casar objetivos exactos.
       - Sobrescribe OBSERVACIONES según reglas.
       - Devuelve CSV (sep=';', coma decimal) en bytes."""
    import pandas as pd
    from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
    import io

    # ---- lectura ----
    try:
        df = pd.read_excel(file_obj, dtype=str, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"No fue posible leer el Excel: {exc}")

    # ---- validación mínima ----
    REQUIRED_COLS = {"ID","NIT","RAZON_SOCIAL","NUMERO_FACTURA",
                     "VALOR_GLOSA","VALOR_ACEPTADO_POR_IPS",
                     "VALOR_ACEPTADO_POR_EPS","VALOR_RATIFICADO_EPS","OBSERVACIONES"}
    headers = [(h or "").strip() for h in list(df.columns)]
    miss = [c for c in REQUIRED_COLS if c not in headers]
    if miss:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(sorted(miss)))

    # ---- utilidades ----
    def _to_dec(s):
        t = str(s).strip().replace("$","").replace(" ","")
        if t == "": return Decimal("0")
        if "," in t: t = t.replace(".","").replace(",",".")
        return Decimal(t)
    q = Decimal("0.01")

    # ---- totales y objetivos ----
    vals = []
    for v in df["VALOR_GLOSA"].tolist():
        try:
            d = _to_dec(v)
            if d < 0: d = Decimal("0")
        except InvalidOperation:
            d = Decimal("0")
        vals.append(d)
    total_glosa = sum(vals)
    objetivo_ips = (total_glosa * Decimal(pct_ips)/Decimal(100)).quantize(q, rounding=ROUND_HALF_UP)
    objetivo_eps = (total_glosa * Decimal(pct_eps)/Decimal(100)).quantize(q, rounding=ROUND_HALF_UP)
    objetivo_rat = (total_glosa * Decimal(pct_rat)/Decimal(100)).quantize(q, rounding=ROUND_HALF_UP)

    # ---- distribución greedy ----
    df2 = df.copy()
    for col in ["VALOR_ACEPTADO_POR_IPS","VALOR_ACEPTADO_POR_EPS","VALOR_RATIFICADO_EPS"]:
        df2[col] = "0"
    rem_ips, rem_eps, rem_rat = objetivo_ips, objetivo_eps, objetivo_rat
    for i in range(len(df2)):
        glosa = _to_dec(df2.at[i,"VALOR_GLOSA"])
        cap = glosa
        if rem_ips>0 and cap>0:
            take=min(cap,rem_ips).quantize(q,ROUND_HALF_UP)
            df2.at[i,"VALOR_ACEPTADO_POR_IPS"]=str(take); rem_ips-=take; cap-=take
        if rem_eps>0 and cap>0:
            take=min(cap,rem_eps).quantize(q,ROUND_HALF_UP)
            df2.at[i,"VALOR_ACEPTADO_POR_EPS"]=str(take); rem_eps-=take; cap-=take
        if rem_rat>0 and cap>0:
            take=min(cap,rem_rat).quantize(q,ROUND_HALF_UP)
            df2.at[i,"VALOR_RATIFICADO_EPS"]=str(take); rem_rat-=take; cap-=take

    # ---- ajuste por redondeo para casar objetivos ----
    def _ajusta(df2,col,obj):
        actual=sum(_to_dec(x) for x in df2[col]); diff=obj-actual
        if diff==0: return
        for j in reversed(range(len(df2))):
            glosa=_to_dec(df2.at[j,"VALOR_GLOSA"])
            ips=_to_dec(df2.at[j,"VALOR_ACEPTADO_POR_IPS"])
            eps=_to_dec(df2.at[j,"VALOR_ACEPTADO_POR_EPS"])
            rat=_to_dec(df2.at[j,"VALOR_RATIFICADO_EPS"])
            usado=ips+eps+rat
            if diff>0:
                cap=glosa-usado
                if cap<=0: continue
                delta=min(cap,diff).quantize(q,ROUND_HALF_UP)
                df2.at[j, col]=str((_to_dec(df2.at[j,col])+delta).quantize(q,ROUND_HALF_UP)); break
            else:
                cap=_to_dec(df2.at[j,col])
                if cap<=0: continue
                delta=min(cap,-diff).quantize(q,ROUND_HALF_UP)
                df2.at[j, col]=str((_to_dec(df2.at[j,col])-delta).quantize(q,ROUND_HALF_UP)); break
    _ajusta(df2,"VALOR_ACEPTADO_POR_IPS",objetivo_ips)
    _ajusta(df2,"VALOR_ACEPTADO_POR_EPS",objetivo_eps)
    _ajusta(df2,"VALOR_RATIFICADO_EPS",objetivo_rat)

    # ---- OBSERVACIONES ----
    obs=[]
    for i in range(len(df2)):
        glosa=_to_dec(df2.at[i,"VALOR_GLOSA"])
        ips=_to_dec(df2.at[i,"VALOR_ACEPTADO_POR_IPS"])
        eps=_to_dec(df2.at[i,"VALOR_ACEPTADO_POR_EPS"])
        rat=_to_dec(df2.at[i,"VALOR_RATIFICADO_EPS"])
        if ips==glosa and eps==0 and rat==0:
            txt="EL PRESTADOR ACEPTA EL VALOR"
        elif eps==glosa and ips==0 and rat==0:
            txt="EL PRESTADOR ADJUNTA LOS SOPORTES NECESARIOS PARA LA GLOSA APLICADA Y ADICIONAN EL CONTRATO Y ANEXOS DEL MISMO"
        elif ips==glosa:
            txt="SE LEVANTA GLOSA, EL PRESTADOR PRESENTA LOS SOPORTES NECESARIOS"
        elif ips>0 and eps>0:
            txt="SE LEVANTA GLOSA PARCIAL, EL PRESTADOR ADJUNTA LOS SOPORTES NECESARIOS PARA LA GLOSA APLICADA Y ADICIONAN EL CONTRATO Y ANEXOS DEL MISMO"
        elif rat>0 and ips==0 and eps==0:
            txt="SE RATIFICA LA GLOSA"
        elif ips>0 and rat>0:
            txt="SE LEVANTA GLOSA PARCIAL, EL PRESTADOR PRESENTA LOS SOPORTES NECESARIOS, SE RATIFICA GLOSA PARCIAL"
        elif eps>0 and rat>0:
            txt="EPS ACEPTA GLOSA PARCIAL, SE RATIFICA GLOSA PARCIAL"
        else:
            raw_obs = df2.at[i, "OBSERVACIONES"]
            txt = str(raw_obs or "").strip()
            if txt.lower() == "nan":
                txt = ""
        obs.append(txt)
    df2["OBSERVACIONES"]=obs

    # ---- resumen y CSV (sep=';' + coma decimal) ----
    sum_ips=sum(_to_dec(x) for x in df2["VALOR_ACEPTADO_POR_IPS"])
    sum_eps=sum(_to_dec(x) for x in df2["VALOR_ACEPTADO_POR_EPS"])
    sum_rat=sum(_to_dec(x) for x in df2["VALOR_RATIFICADO_EPS"])
    resumen=f"total_glosa={total_glosa} | objetivo ips/eps/rat=({objetivo_ips},{objetivo_eps},{objetivo_rat}) | asignado=({sum_ips},{sum_eps},{sum_rat})"

    num_cols=["VALOR_ACEPTADO_POR_IPS","VALOR_ACEPTADO_POR_EPS","VALOR_RATIFICADO_EPS","VALOR_GLOSA","VALOR_TOTAL_SERVICIO"]
    df_out=df2.copy()
    for c in num_cols:
        if c in df_out.columns:
            df_out[c]=df_out[c].astype(str).str.replace(".",",",regex=False)

    buf=io.StringIO()
    df_out.to_csv(buf,index=False,sep=";",lineterminator="\n")
    return df_out, resumen, buf.getvalue().encode("utf-8")

def render_state_form(cfg: dict, nit: str, razon: str | None, prefix: str) -> None:
    estado_key = f"{prefix}_estado"
    comentario_key = f"{prefix}_comentario"
    save_key = f"{prefix}_save_button"
    cancel_key = f"{prefix}_cancel_button"

    st.markdown(f"**NIT:** {nit}")
    st.markdown(f"**Razón social:** {razon or 'Sin información'}")

    st.selectbox(
        "Estado",
        STATE_OPTIONS,
        key=estado_key,
    )
    estado_actual = st.session_state.get(estado_key)
    st.text_area(
        "Comentario (opcional)",
        key=comentario_key,
        height=150,
    )

    # Si el estado seleccionado es "Acta Firmada", solicitar el PDF
    pdf_key = f"{prefix}_acta_pdf"
    if estado_actual == "Acta Firmada":
        st.file_uploader("Cargar PDF del acta (obligatorio)", type=["pdf"], key=pdf_key)

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Guardar", key=save_key):
            estado = st.session_state.get(estado_key)
            if not estado:
                st.warning("Debes seleccionar un estado.")
            else:
                comentario = st.session_state.get(comentario_key, "")
                user_email = st.session_state.get("user_email", "")
                # Validar PDF si el estado es "Acta Firmada"
                need_pdf = (estado == "Acta Firmada")
                pdf_file = st.session_state.get(pdf_key)
                if need_pdf and not pdf_file:
                    st.error("Debes cargar el PDF del acta para 'Acta Firmada'.")
                else:
                    try:
                        nit_state_id = insert_nit_state(cfg, nit, estado, comentario, user_email)
                        # Si aplica, guardar PDF a disco y registrar en BD
                        if need_pdf and pdf_file is not None:
                            from pathlib import Path
                            import uuid
                            pdf_dir = OUT_DIR / "actas_pdf"
                            try:
                                pdf_dir.mkdir(parents=True, exist_ok=True)
                            except Exception:
                                pass
                            original_name = Path(getattr(pdf_file, "name", "acta.pdf")).name
                            safe_nit = "".join(ch for ch in str(nit) if ch.isalnum()) or "sin_nit"
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            file_name = f"acta_pdf_{safe_nit}_{ts}.pdf"
                            file_path = str(pdf_dir / file_name)
                            with open(file_path, "wb") as f:
                                f.write(pdf_file.getbuffer())
                            acta_pdf_id = str(uuid.uuid4())
                            insert_acta_pdf_record(
                                cfg,
                                acta_pdf_id,
                                nit,
                                razon or "",
                                original_name,
                                file_path,
                                user_email or "",
                                nit_state_id,
                            )
                        load_summary_dataframe.clear()
                        clear_state_form(prefix)
                        clear_state_selection()
                        st.success("Estado guardado correctamente.")
                        st.session_state["menu_option"] = "Resumen"
                        rerun_app()
                    except Exception as exc:
                        st.error(f"No se pudo guardar el estado: {exc}")
    with col_cancel:
        if st.button("Cancelar", key=cancel_key):
            clear_state_form(prefix)
            clear_state_selection()
            st.session_state["menu_option"] = "Resumen"
            rerun_app()


def render_summary_page():
    st.title("Resumen por NIT")
    st.caption("Consulta agregada de facturas y conciliaciones.")

    try:
        cfg, missing = get_db_config()
    except ValueError as exc:
        st.error(str(exc))
        return

    if missing:
        st.error(
            "Faltan variables de entorno para la conexión MySQL: "
            + ", ".join(missing)
        )
        return

    with st.spinner("Cargando datos…"):
        try:
            df = load_summary_dataframe(cfg)
        except Exception as exc:
            st.error(
                "No se pudo obtener la información de la base de datos. "
                f"Detalle: {exc}"
            )
            return

    if df.empty:
        st.info("No se encontraron resultados para los criterios actuales.")
        return

    df["nit"] = df["nit"].astype(str)

    st.session_state.setdefault("selected_nit", None)
    st.session_state.setdefault("selected_razon", None)
    st.session_state.setdefault("show_state_modal", False)

    unique_nits = sorted(df["nit"].unique())
    selected_nits = st.multiselect(
        "Filtrar por NIT",
        options=unique_nits,
        default=[],
    )

    filtered = df[df["nit"].isin(selected_nits)] if selected_nits else df

    df_view = filtered.copy()
    df_view["Acciones"] = False

    # Agregar columnas de descargas por NIT (Excel/PDF)
    link_cols: list[str] = []
    try:
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            cursorclass=pymysql.cursors.Cursor,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT nit, file_name, file_path, created_at
                    FROM conciliation_acta_files
                    ORDER BY created_at DESC
                    """
                )
                excel_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT nit, file_name, file_path, created_at
                    FROM conciliation_acta_files_pdf
                    ORDER BY created_at DESC
                    """
                )
                pdf_rows = cur.fetchall()
        finally:
            conn.close()

        excel_map: dict[str, list[str]] = {}
        for nit_val, _fname, fpath, _created in excel_rows:
            try:
                p = Path(str(fpath))
                if p.exists():
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                        # Incluir sugerencia de nombre para descarga
                        safe_name = str(_fname or "acta.xlsx").replace(",", "_")
                        url = (
                            "data:application/"
                            f"vnd.openxmlformats-officedocument.spreadsheetml.sheet;name={safe_name};base64,"
                            + b64
                        )
                else:
                    url = ""
            except Exception:
                url = ""
            excel_map.setdefault(str(nit_val), []).append(url)

        pdf_map: dict[str, list[str]] = {}
        for nit_val, _fname, fpath, _created in pdf_rows:
            try:
                p = Path(str(fpath))
                if p.exists():
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                        safe_name = str(_fname or "acta.pdf").replace(",", "_")
                        # octet-stream para forzar descarga y sugerir nombre
                        url = f"data:application/octet-stream;name={safe_name};base64,{b64}"
                else:
                    url = ""
            except Exception:
                url = ""
            pdf_map.setdefault(str(nit_val), []).append(url)

        max_excel = max((len(v) for v in excel_map.values()), default=0)
        max_pdf = max((len(v) for v in pdf_map.values()), default=0)

        for i in range(max_excel):
            col_name = ("Actas excel" if i == 0 else f"Actas excel {i+1}")
            link_cols.append(col_name)
            df_view[col_name] = df_view["nit"].map(lambda n: (excel_map.get(str(n), []) + [""] * max_excel)[i])
        for i in range(max_pdf):
            col_name = ("Actas pdf" if i == 0 else f"Actas pdf {i+1}")
            link_cols.append(col_name)
            df_view[col_name] = df_view["nit"].map(lambda n: (pdf_map.get(str(n), []) + [""] * max_pdf)[i])
    except Exception:
        link_cols = []

    # Configurar etiquetas amigables para columnas base (reemplazar '_' por ' ')
    pretty_labels = {
        col: st.column_config.Column(col.replace("_", " "))
        for col in df_view.columns
        if col != "Acciones" and col not in link_cols
    }

    editor_result = st.data_editor(
        df_view,
        width='stretch',
        hide_index=True,
        disabled=[col for col in df_view.columns if col != "Acciones"],
        column_config={
            "Acciones": st.column_config.CheckboxColumn(
                "➕",
                help="Agregar estado al NIT",
                default=False,
            ),
            **{
                col: st.column_config.LinkColumn(
                    col,
                    display_text=(
                        "Descargar Acta" if col.lower().startswith("actas excel") else "Acta firmada"
                    ),
                )
                for col in link_cols
            },
            **pretty_labels,
        },
        key="summary_table_editor",
    )

    st.download_button(
        "Descargar CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="resumen_conciliacion.csv",
        mime="text/csv",
    )

    clicked_rows = editor_result[editor_result["Acciones"] == True]
    if not clicked_rows.empty:
        idx = clicked_rows.index[0]
        row = editor_result.loc[idx]
        st.session_state["selected_nit"] = str(row.get("nit", ""))
        st.session_state["selected_razon"] = str(row.get("razon_social", ""))
        st.session_state["menu_option"] = "Detalle NIT"
        st.session_state["show_state_modal"] = False
        rerun_app()

    selected_nit = st.session_state.get("selected_nit")
    selected_razon = st.session_state.get("selected_razon")

    if st.session_state.get("show_state_modal") and selected_nit:
        # Enviar siempre a la vista "Detalle NIT" para evitar problemas con modales.
        st.session_state["menu_option"] = "Detalle NIT"
        st.session_state.show_state_modal = False
        rerun_app()


def render_nit_detail_page():
    st.title("Detalle NIT")

    selected_nit = st.session_state.get("selected_nit")
    selected_razon = st.session_state.get("selected_razon")

    if not selected_nit:
        st.warning("Selecciona un NIT desde el resumen para agregar un estado.")
        if st.button("Volver al resumen", key="detail_back_to_summary"):
            st.session_state["menu_option"] = "Resumen"
            rerun_app()
        return

    try:
        cfg, missing = get_db_config()
    except ValueError as exc:
        st.error(str(exc))
        return

    if missing:
        st.error(
            "Faltan variables de entorno para la conexión MySQL: "
            + ", ".join(missing)
        )
        return

    render_state_form(cfg, selected_nit, selected_razon, prefix="detail")


def tail_file(path: Path, from_bytes=0, max_lines=4000):
    if not path.exists():
        return from_bytes, ""
    with open(path, "rb") as f:
        f.seek(from_bytes)
        chunk = f.read()
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()[-max_lines:]
    return from_bytes + len(chunk), '\n'.join(lines)



def run_job(csv_path: Path, verbose: bool = True, dry_run: bool = False):
    cmd = [
        "python",
        "csv_conciliation_loader.py",
        "--csv", str(csv_path),
        "--out-dir", str(OUT_DIR),
    ]
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    try:
        if LOG_FILE.exists():
            LOG_FILE.unlink()
    except Exception:
        pass
    env = os.environ.copy()
    user_email = st.session_state.get("user_email") or env.get("ACTA_GENERATION_USER")
    if user_email:
        env["ACTA_GENERATION_USER"] = user_email
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


def render_confirmation_ui(df_sum: "pd.DataFrame") -> None:
    """Muestra una ventana modal con el resumen y Confirmar/Cancelar.
    Prioriza st.modal (context manager). Si no existe, usa st.dialog como decorador.
    Si nada de eso existe, cae a un bloque en la página.
    """
    has_modal = hasattr(st, "modal")
    has_dialog = hasattr(st, "dialog")

    def _content():
        st.subheader("Resumen de conciliación a registrar")
        pretty_cols = {c: st.column_config.Column(c.replace("_", " ")) for c in df_sum.columns}
        st.data_editor(df_sum, hide_index=True, disabled=True, column_config=pretty_cols, width='stretch')
        st.markdown("")
        st.info("¿Está seguro de cargar los datos y generar el acta de conciliación?")

        confirm_col, cancel_col = st.columns([1, 1])
        with confirm_col:
            if st.button("Confirmar y registrar en SQL", type="primary"):
                csv_path = st.session_state.get("last_csv_path")
                if not csv_path:
                    st.error("No se encontró la ruta del CSV cargado.")
                else:
                    st.session_state.proc = run_job(Path(csv_path), verbose=st.session_state.get("verbose", True), dry_run=False)
                    st.session_state.running = True
                    st.session_state.log_offset = 0
                    st.session_state.last_result = "neutral"
                    st.session_state.already_conciliated_detected = False
                    st.session_state.phase = "inserting"
                    st.session_state.show_confirm_modal = False
                    st.success("Iniciando inserción en SQL…")
                    rerun_app()
        with cancel_col:
            if st.button("Cancelar"):
                st.session_state.phase = "idle"
                st.session_state.pop("dry_run_df", None)
                st.session_state.dry_run_summary_loaded = False
                st.session_state.show_confirm_modal = False
                st.info("Operación cancelada. No se insertaron datos.")
                rerun_app()

    if has_modal:
        with st.modal("Confirmar conciliación"):
            _content()
    elif has_dialog:
        # st.dialog es un decorador: lo aplicamos a _content y lo invocamos.
        try:
            show_dialog = st.dialog("Confirmar conciliación")(_content)
            show_dialog()
        except Exception:
            # Fallback a bloque en página si la invocación falla
            st.warning("No fue posible usar st.dialog; mostrando confirmación en la página.")
            _content()
    else:
        st.warning("Tu versión de Streamlit no soporta modales; mostrando confirmación en la página.")
        _content()

def file_to_download_button(label: str, path: Path, color: str = "neutral"):
    if not path.exists():
        return
    mime = "text/csv" if path.suffix.lower() == ".csv" else "text/plain"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    colors = {
        "red": "#dc2626",
        "green": "#16a34a",
        "neutral": "#374151",
    }
    bg = colors.get(color, colors["neutral"])

    btn_html = f"""
    <a download="{path.name}"
       href="data:{mime};base64,{b64}"
       style="
         text-decoration:none; padding:0.6rem 1rem; border-radius:10px;
         background:{bg}; color:white; font-weight:600; display:inline-block;
         box-shadow:0 2px 6px rgba(0,0,0,.12);
       ">
       {label}
    </a>
    """
    st.markdown(btn_html, unsafe_allow_html=True)


def render_reports_page():
    st.title("Reportes de Auditoría")
    st.caption("Busca por NIT y/o factura, aplica filtros y descarga el reporte completo.")

    try:
        cfg, missing = get_db_config()
    except ValueError as exc:
        st.error(str(exc))
        return

    if missing:
        st.error(
            "Faltan variables de entorno para la conexión MySQL: "
            + ", ".join(missing)
        )
        return

    st.header("📦 Generación masiva de reportes por NIT")
    st.caption("Selecciona múltiples NITs para generar archivos CSV individuales")
    
    with st.spinner("Cargando lista de NITs..."):
        try:
            conn = pymysql.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                password=cfg["password"],
                database=cfg["database"],
                cursorclass=pymysql.cursors.DictCursor,
            )
            
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT DISTINCT t.id, t.name 
                        FROM thirds t
                        INNER JOIN invoice_audits ia ON ia.third_id = t.id
                        INNER JOIN auditory_final_reports afr ON afr.factura_id = ia.id
                        ORDER BY t.id
                    """)
                    nit_rows = cursor.fetchall()
            finally:
                conn.close()
            
            if nit_rows:
                nit_options = {f"{row['id']} - {row['name']}": row['id'] for row in nit_rows}
                
                col_select1, col_select2 = st.columns([3, 1])
                with col_select1:
                    selected_nit_labels = st.multiselect(
                        "Selecciona los NITs para generar reportes",
                        options=list(nit_options.keys()),
                        default=st.session_state.get("selected_all_nits", []),
                        help=f"Total de NITs disponibles: {len(nit_options)}"
                    )
                with col_select2:
                    st.write("")
                    st.write("")
                    if st.button("✅ Seleccionar todos", use_container_width=True):
                        st.session_state["selected_all_nits"] = list(nit_options.keys())
                        rerun_app()
                
                selected_nits = [nit_options[label] for label in selected_nit_labels]
                
                if st.session_state.get("selected_all_nits") and not selected_nit_labels:
                    st.session_state.pop("selected_all_nits", None)
                
                report_type = st.radio(
                    "Tipo de reporte a generar",
                    options=["Agrupado por factura", "Detallado (Ver todo)"],
                    horizontal=True,
                    help="Agrupado: Totales por factura | Detallado: Registro línea por línea con todos los servicios"
                )
                
                estado_filter = st.multiselect(
                    "Filtrar por estados",
                    options=["contabilizada", "devolución", "eliminada"],
                    default=["contabilizada", "devolución", "eliminada"],
                    help="Selecciona los estados que deseas incluir en el reporte. Si no seleccionas ninguno, se incluirán todos los registros."
                )
                
                if selected_nits:
                    st.info(f"📊 NITs seleccionados: **{len(selected_nits)}** | Tipo: **{report_type}**")
                    
                    col_gen1, col_gen2 = st.columns([1, 3])
                    with col_gen1:
                        if st.button("🚀 Generar reportes", type="primary", use_container_width=True):
                            st.session_state["batch_generation_nits"] = selected_nits
                            st.session_state["batch_generation_report_type"] = report_type
                            st.session_state["batch_generation_estado_filter"] = estado_filter
                            st.session_state["batch_generation_status"] = {}
                            st.session_state["batch_generation_running"] = True
                            rerun_app()
                    
                    with col_gen2:
                        if st.button("🗑️ Limpiar selección"):
                            st.session_state.pop("batch_generation_nits", None)
                            st.session_state.pop("batch_generation_report_type", None)
                            st.session_state.pop("batch_generation_estado_filter", None)
                            st.session_state.pop("batch_generation_status", None)
                            st.session_state.pop("batch_generation_running", None)
                            st.session_state.pop("selected_all_nits", None)
                            rerun_app()
                    
                    if st.session_state.get("batch_generation_running"):
                        batch_nits = st.session_state.get("batch_generation_nits", [])
                        batch_report_type = st.session_state.get("batch_generation_report_type", "Agrupado por factura")
                        batch_estado_filter = st.session_state.get("batch_generation_estado_filter", [])
                        batch_status = st.session_state.get("batch_generation_status", {})
                        
                        reports_dir = OUT_DIR / "reportes_masivos"
                        reports_dir.mkdir(parents=True, exist_ok=True)
                        
                        progress_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_table_placeholder = st.empty()
                        
                        total_nits = len(batch_nits)
                        is_agrupado = batch_report_type == "Agrupado por factura"
                        
                        for idx, nit in enumerate(batch_nits):
                            if nit in batch_status and batch_status[nit]["status"] == "completado":
                                continue
                            
                            current_count = idx + 1
                            progress_percentage = int((current_count / total_nits) * 100)
                            progress_placeholder.info(f"🔄 Procesando {current_count}/{total_nits} NITs | **{nit}** | {progress_percentage}% completado")
                            progress_bar.progress(current_count / total_nits)
                            
                            batch_status[nit] = {
                                "status": "procesando",
                                "archivo": None,
                                "error": None
                            }
                            
                            status_table_placeholder.dataframe(
                                pd.DataFrame([{
                                    "NIT": k,
                                    "Estado": v["status"],
                                    "Archivo": Path(v.get("archivo", "")).name if v.get("archivo") else ""
                                } for k, v in batch_status.items()]),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            try:
                                conn = pymysql.connect(
                                    host=cfg["host"],
                                    port=cfg["port"],
                                    user=cfg["user"],
                                    password=cfg["password"],
                                    database=cfg["database"],
                                    cursorclass=pymysql.cursors.DictCursor,
                                )
                                
                                try:
                                    with conn.cursor() as cursor:
                                        if is_agrupado:
                                            query_agrupado = """
                                        SELECT
                                            aud.factura_id,
                                            aud.origin,
                                            aud.nit,
                                            aud.razon_social,
                                            aud.numero_factura,
                                            aud.fecha_inicio,
                                            aud.fecha_fin,
                                            aud.modalidad,
                                            aud.regimen,
                                            aud.cobertura,
                                            aud.contrato,
                                            SUM(aud.valor_total_servicio) AS valor_total_servicios,
                                            GROUP_CONCAT(DISTINCT aud.codigos_glosa ORDER BY aud.codigos_glosa SEPARATOR '-') AS codigos_glosa,
                                            SUM(aud.valor_glosa) AS valor_glosa,
                                            SUM(aud.valor_aprobado) AS valor_aprobado,
                                            GROUP_CONCAT(DISTINCT eafr.ESTADO ORDER BY eafr.ESTADO SEPARATOR '-') AS estado,
                                            SUM(ci.accepted_value_eps) AS valor_aceptado_eps,
                                            SUM(ci.accepted_value_ips) AS valor_aceptado_ips,
                                            SUM(ci.eps_ratified_value) AS valor_ratificado,
                                            MAX(ci.observation) AS observacion_conciliacion,
                                            MAX(ci.created_at) AS fecha_creacion,
                                            CASE
                                                WHEN MAX(eafr.ESTADO = 'contabilizada') = 1 AND SUM(aud.valor_glosa) > 0 THEN
                                                    CASE
                                                        WHEN COUNT(ci.id) > 0 THEN 'CONCILIADO'
                                                        ELSE 'NO CONCILIADA'
                                                    END
                                                ELSE 'NO REQUIERE CONCILIACION'
                                            END AS estado_conciliacion
                                        FROM auditory_final_reports aud
                                        INNER JOIN estados_auditory_final_reports eafr
                                            ON eafr.ID = aud.id
                                        INNER JOIN invoice_audits ia
                                            ON ia.id = aud.factura_id
                                        INNER JOIN thirds t
                                            ON t.id = ia.third_id
                                        LEFT JOIN conciliation_results ci
                                            ON ci.auditory_final_report_id = aud.id
                                        WHERE t.id = %s
                                        """
                                            
                                            if batch_estado_filter:
                                                placeholders = ','.join(['%s'] * len(batch_estado_filter))
                                                query_agrupado += f" AND eafr.ESTADO IN ({placeholders})"
                                            
                                            query_agrupado += """
                                        GROUP BY
                                            aud.factura_id,
                                            aud.origin,
                                            aud.nit,
                                            aud.razon_social,
                                            aud.numero_factura,
                                            aud.fecha_inicio,
                                            aud.fecha_fin,
                                            aud.modalidad,
                                            aud.regimen,
                                            aud.cobertura,
                                            aud.contrato
                                        ORDER BY aud.id DESC
                                        """
                                            
                                            query_params = [nit] + batch_estado_filter if batch_estado_filter else [nit]
                                            cursor.execute(query_agrupado, query_params)
                                            rows = cursor.fetchall()
                                        else:
                                            query_detallado = """
                                        SELECT
                                            aud.id,
                                            aud.factura_id,
                                            aud.servicio_id,
                                            aud.origin,
                                            aud.nit,
                                            aud.razon_social,
                                            aud.numero_factura,
                                            aud.fecha_inicio,
                                            aud.fecha_fin,
                                            aud.modalidad,
                                            aud.regimen,
                                            aud.cobertura,
                                            aud.contrato,
                                            aud.tipo_documento,
                                            aud.numero_documento,
                                            aud.primer_nombre,
                                            aud.segundo_nombre,
                                            aud.primer_apellido,
                                            aud.segundo_apellido,
                                            aud.genero,
                                            aud.codigo_servicio,
                                            aud.descripcion_servicio,
                                            aud.cantidad_servicio,
                                            aud.valor_unitario_servicio,
                                            aud.valor_total_servicio,
                                            aud.codigos_glosa,
                                            aud.observaciones_glosas,
                                            aud.valor_glosa,
                                            aud.valor_aprobado,
                                            eafr.ESTADO AS estado,
                                            ci.accepted_value_eps AS valor_aceptado_eps,
                                            ci.accepted_value_ips AS valor_aceptado_ips,
                                            ci.eps_ratified_value AS valor_ratificado,
                                            ci.observation AS observacion_conciliacion,
                                            ci.created_at AS fecha_creacion,
                                            CASE
                                                WHEN eafr.ESTADO = 'contabilizada' AND aud.valor_glosa > 0 THEN
                                                    CASE
                                                        WHEN ci.id IS NOT NULL THEN 'CONCILIADO'
                                                        ELSE 'NO CONCILIADA'
                                                    END
                                                ELSE 'NO REQUIERE CONCILIACION'
                                            END AS estado_conciliacion
                                        FROM auditory_final_reports aud
                                        INNER JOIN estados_auditory_final_reports eafr
                                            ON eafr.ID = aud.id
                                        INNER JOIN invoice_audits ia
                                            ON ia.id = aud.factura_id
                                        INNER JOIN thirds t
                                            ON t.id = ia.third_id
                                        LEFT JOIN conciliation_results ci
                                            ON ci.auditory_final_report_id = aud.id
                                        WHERE t.id = %s
                                        """
                                            
                                            if batch_estado_filter:
                                                placeholders = ','.join(['%s'] * len(batch_estado_filter))
                                                query_detallado += f" AND eafr.ESTADO IN ({placeholders})"
                                            
                                            query_detallado += " ORDER BY aud.id DESC"
                                            
                                            query_params = [nit] + batch_estado_filter if batch_estado_filter else [nit]
                                            cursor.execute(query_detallado, query_params)
                                            rows = cursor.fetchall()
                                finally:
                                    conn.close()
                                
                                safe_nit = "".join(ch for ch in str(nit) if ch.isalnum())
                                filename_suffix = "agrupado" if is_agrupado else "detalle"
                                filename = f"{safe_nit}_{filename_suffix}.csv"
                                filepath = reports_dir / filename
                                
                                if rows:
                                    df = pd.DataFrame(rows)
                                    df.to_csv(filepath, index=False, sep=";", encoding="utf-8")
                                    batch_status[nit]["archivo"] = str(filepath)
                                    batch_status[nit]["status"] = "completado"
                                else:
                                    batch_status[nit]["status"] = "sin datos"
                                    batch_status[nit]["error"] = "No se encontraron registros para este NIT"
                                
                            except Exception as e:
                                batch_status[nit]["status"] = "error"
                                batch_status[nit]["error"] = str(e)
                            
                            st.session_state["batch_generation_status"] = batch_status
                            
                            status_table_placeholder.dataframe(
                                pd.DataFrame([{
                                    "NIT": k,
                                    "Estado": v["status"],
                                    "Archivo": Path(v.get("archivo", "")).name if v.get("archivo") else ""
                                } for k, v in batch_status.items()]),
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        st.session_state["batch_generation_running"] = False
                        progress_placeholder.success(f"✅ Proceso completado: {total_nits}/{total_nits} NITs procesados")
                        rerun_app()
                    
                    if st.session_state.get("batch_generation_status"):
                        st.subheader("📊 Estado de generación de reportes")
                        
                        batch_status = st.session_state.get("batch_generation_status", {})
                        batch_report_type = st.session_state.get("batch_generation_report_type", "Agrupado por factura")
                        
                        status_data = []
                        for nit, info in batch_status.items():
                            status_data.append({
                                "NIT": nit,
                                "Estado": info["status"],
                                "Archivo": info.get("archivo", ""),
                                "Error": info.get("error", "")
                            })
                        
                        df_status = pd.DataFrame(status_data)
                        
                        def make_download_link(filepath):
                            if not filepath or not Path(filepath).exists():
                                return ""
                            try:
                                with open(filepath, "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode()
                                filename = Path(filepath).name
                                return f"data:text/csv;name={filename};base64,{b64}"
                            except Exception:
                                return ""
                        
                        df_status["Descargar"] = df_status["Archivo"].apply(make_download_link)
                        
                        display_cols = ["NIT", "Estado", "Descargar", "Error"]
                        df_display = df_status[display_cols]
                        
                        st.dataframe(
                            df_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "NIT": st.column_config.TextColumn("NIT"),
                                "Estado": st.column_config.TextColumn("Estado"),
                                "Descargar": st.column_config.LinkColumn(
                                    f"📥 {batch_report_type}",
                                    display_text="Descargar"
                                ),
                                "Error": st.column_config.TextColumn("Error")
                            }
                        )
                        
                        completed = sum(1 for info in batch_status.values() if info["status"] == "completado")
                        errors = sum(1 for info in batch_status.values() if info["status"] == "error")
                        total = len(batch_status)
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Total procesados", f"{total}")
                        with col_m2:
                            st.metric("Completados", f"{completed}", delta=f"{int(completed/total*100)}%" if total > 0 else "0%")
                        with col_m3:
                            st.metric("Errores", f"{errors}")
            else:
                st.warning("No se encontraron NITs en la base de datos.")
                
        except Exception as exc:
            st.error(f"Error al cargar NITs: {exc}")
            import traceback
            st.code(traceback.format_exc())
    
    st.divider()
    st.header("🔍 Búsqueda individual de registros")
    st.subheader("Filtros de búsqueda")
    col1, col2 = st.columns(2)
    
    with col1:
        nit_filter = st.text_input(
            "Buscar por NIT",
            placeholder="Ingresa el NIT completo o parcial",
            help="Puedes buscar por NIT completo o parcial"
        ).strip()
    
    with col2:
        factura_filter = st.text_input(
            "Buscar por Número de Factura",
            placeholder="Ingresa el número de factura",
            help="Busca por número de factura completo o parcial"
        ).strip()
    
    if not nit_filter and not factura_filter:
        st.info("👆 Ingresa al menos un criterio de búsqueda (NIT y/o número de factura) para ver los registros.")
        return

    st.subheader("Opciones de visualización")
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        btn_ver_todo = st.button(
            "📋 Ver todo",
            help="Muestra todos los registros con el detalle completo de servicios, glosas y conciliación",
            use_container_width=True
        )
    
    with col_btn2:
        btn_ver_glosados = st.button(
            "⚠️ Ver glosados",
            help="Muestra solo los registros que tienen valor de glosa mayor a cero",
            use_container_width=True
        )
    
    with col_btn3:
        btn_ver_por_estado = st.button(
            "🔍 Ver por estado",
            help="Filtra registros glosados por estado específico (contabilizada, devolución, eliminada)",
            use_container_width=True
        )
    
    with col_btn4:
        agrupado = st.checkbox(
            "📊 Agrupar por factura",
            help="Agrupa los resultados por factura en lugar de mostrar el detalle línea por línea",
            value=False
        )
    
    estado_filter = None
    if btn_ver_por_estado:
        estado_filter = st.selectbox(
            "Selecciona el estado",
            options=["contabilizada", "devolución", "eliminada"],
            help="Filtra los registros glosados por el estado seleccionado"
        )
    
    page_size = 100
    if "reports_page" not in st.session_state:
        st.session_state.reports_page = 0
    
    filter_type = None
    if btn_ver_todo:
        filter_type = "ver_todo"
        st.session_state.reports_page = 0
    elif btn_ver_glosados:
        filter_type = "ver_glosados"
        st.session_state.reports_page = 0
    elif btn_ver_por_estado and estado_filter:
        filter_type = "ver_por_estado"
        st.session_state.reports_page = 0
        st.session_state.reports_estado_filter = estado_filter
    
    if filter_type or "reports_data_full" in st.session_state:
        if filter_type:
            with st.spinner("Consultando registros..."):
                try:
                    conn = pymysql.connect(
                        host=cfg["host"],
                        port=cfg["port"],
                        user=cfg["user"],
                        password=cfg["password"],
                        database=cfg["database"],
                        cursorclass=pymysql.cursors.DictCursor,
                    )
                    
                    try:
                        with conn.cursor() as cursor:
                            if agrupado:
                                query = """
                                SELECT
                                    aud.factura_id,
                                    aud.origin,
                                    aud.nit,
                                    aud.razon_social,
                                    aud.numero_factura,
                                    aud.fecha_inicio,
                                    aud.fecha_fin,
                                    aud.modalidad,
                                    aud.regimen,
                                    aud.cobertura,
                                    aud.contrato,
                                    SUM(aud.valor_total_servicio) AS valor_total_servicios,
                                    GROUP_CONCAT(DISTINCT aud.codigos_glosa ORDER BY aud.codigos_glosa SEPARATOR '-') AS codigos_glosa,
                                    SUM(aud.valor_glosa) AS valor_glosa,
                                    SUM(aud.valor_aprobado) AS valor_aprobado,
                                    GROUP_CONCAT(DISTINCT eafr.ESTADO ORDER BY eafr.ESTADO SEPARATOR '-') AS estado,
                                    SUM(ci.accepted_value_eps) AS valor_aceptado_eps,
                                    SUM(ci.accepted_value_ips) AS valor_aceptado_ips,
                                    SUM(ci.eps_ratified_value) AS valor_ratificado,
                                    MAX(ci.observation) AS observacion_conciliacion,
                                    MAX(ci.created_at) AS fecha_creacion,
                                    CASE
                                        WHEN MAX(eafr.ESTADO = 'contabilizada') = 1 AND SUM(aud.valor_glosa) > 0 THEN
                                            CASE
                                                WHEN COUNT(ci.id) > 0 THEN 'CONCILIADO'
                                                ELSE 'NO CONCILIADA'
                                            END
                                        ELSE 'NO REQUIERE CONCILIACION'
                                    END AS estado_conciliacion
                                FROM auditory_final_reports aud
                                INNER JOIN estados_auditory_final_reports eafr
                                    ON eafr.ID = aud.id
                                INNER JOIN invoice_audits ia
                                    ON ia.id = aud.factura_id
                                INNER JOIN thirds t
                                    ON t.id = ia.third_id
                                LEFT JOIN conciliation_results ci
                                    ON ci.auditory_final_report_id = aud.id
                                WHERE 1=1
                                """
                            else:
                                query = """
                                SELECT
                                    aud.id,
                                    aud.factura_id,
                                    aud.servicio_id,
                                    aud.origin,
                                    aud.nit,
                                    aud.razon_social,
                                    aud.numero_factura,
                                    aud.fecha_inicio,
                                    aud.fecha_fin,
                                    aud.modalidad,
                                    aud.regimen,
                                    aud.cobertura,
                                    aud.contrato,
                                    aud.tipo_documento,
                                    aud.numero_documento,
                                    aud.primer_nombre,
                                    aud.segundo_nombre,
                                    aud.primer_apellido,
                                    aud.segundo_apellido,
                                    aud.genero,
                                    aud.codigo_servicio,
                                    aud.descripcion_servicio,
                                    aud.cantidad_servicio,
                                    aud.valor_unitario_servicio,
                                    aud.valor_total_servicio,
                                    aud.codigos_glosa,
                                    aud.observaciones_glosas,
                                    aud.valor_glosa,
                                    aud.valor_aprobado,
                                    eafr.ESTADO AS estado,
                                    ci.accepted_value_eps AS valor_aceptado_eps,
                                    ci.accepted_value_ips AS valor_aceptado_ips,
                                    ci.eps_ratified_value AS valor_ratificado,
                                    ci.observation AS observacion_conciliacion,
                                    ci.created_at AS fecha_creacion,
                                    CASE
                                        WHEN eafr.ESTADO = 'contabilizada' AND aud.valor_glosa > 0 THEN
                                            CASE
                                                WHEN ci.id IS NOT NULL THEN 'CONCILIADO'
                                                ELSE 'NO CONCILIADA'
                                            END
                                        ELSE 'NO REQUIERE CONCILIACION'
                                    END AS estado_conciliacion
                                FROM auditory_final_reports aud
                                INNER JOIN estados_auditory_final_reports eafr
                                    ON eafr.ID = aud.id
                                INNER JOIN invoice_audits ia
                                    ON ia.id = aud.factura_id
                                INNER JOIN thirds t
                                    ON t.id = ia.third_id
                                LEFT JOIN conciliation_results ci
                                    ON ci.auditory_final_report_id = aud.id
                                WHERE 1=1
                                """
                            
                            params = []
                            if nit_filter:
                                query += " AND t.id LIKE %s"
                                params.append(f"%{nit_filter}%")
                            
                            if factura_filter:
                                query += " AND aud.numero_factura LIKE %s"
                                params.append(f"%{factura_filter}%")
                            
                            if filter_type == "ver_glosados":
                                query += " AND aud.valor_glosa > 0"
                            elif filter_type == "ver_por_estado":
                                query += " AND aud.valor_glosa > 0"
                                query += " AND eafr.ESTADO = %s"
                                params.append(estado_filter)
                            
                            if agrupado:
                                query += """
                                GROUP BY
                                    aud.factura_id,
                                    aud.origin,
                                    aud.nit,
                                    aud.razon_social,
                                    aud.numero_factura,
                                    aud.fecha_inicio,
                                    aud.fecha_fin,
                                    aud.modalidad,
                                    aud.regimen,
                                    aud.cobertura,
                                    aud.contrato
                                """
                            
                            query += " ORDER BY aud.id DESC"
                            
                            cursor.execute(query, params)
                            rows = cursor.fetchall()
                    finally:
                        conn.close()
                    
                    if not rows:
                        st.warning("No se encontraron registros con los criterios de búsqueda especificados.")
                        st.session_state.pop("reports_data_full", None)
                        return
                    
                    df_full = pd.DataFrame(rows)
                    st.session_state["reports_data_full"] = df_full
                    st.session_state["reports_filter_type"] = filter_type
                    st.session_state["reports_agrupado"] = agrupado
                    
                except Exception as exc:
                    st.error(f"Error al consultar los datos: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        if "reports_data_full" in st.session_state:
            df_full = st.session_state["reports_data_full"]
            total_records = len(df_full)
            
            st.success(f"✅ Total de registros encontrados: **{total_records:,}**")
            
            total_pages = (total_records + page_size - 1) // page_size
            current_page = st.session_state.reports_page
            
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, total_records)
            df_page = df_full.iloc[start_idx:end_idx]
            
            st.subheader(f"Resultados (página {current_page + 1} de {total_pages})")
            st.caption(f"Mostrando registros {start_idx + 1} a {end_idx} de {total_records:,} totales")
            
            pretty_cols = {c: st.column_config.Column(c.replace("_", " ")) for c in df_page.columns}
            st.dataframe(
                df_page,
                use_container_width=True,
                hide_index=True,
                column_config=pretty_cols,
            )
            
            if total_pages > 1:
                col_prev, col_info, col_next = st.columns([1, 2, 1])
                with col_prev:
                    if st.button("⬅️ Anterior", disabled=(current_page == 0)):
                        st.session_state.reports_page = max(0, current_page - 1)
                        rerun_app()
                with col_info:
                    st.write(f"Página {current_page + 1} de {total_pages}")
                with col_next:
                    if st.button("Siguiente ➡️", disabled=(current_page >= total_pages - 1)):
                        st.session_state.reports_page = min(total_pages - 1, current_page + 1)
                        rerun_app()
            
            st.subheader("Descargar reporte completo")
            
            csv_data = df_full.to_csv(index=False, sep=";").encode("utf-8")
            
            filename_parts = []
            if nit_filter:
                filename_parts.append(f"nit_{nit_filter}")
            if factura_filter:
                filename_parts.append(f"fact_{factura_filter}")
            filter_type_name = st.session_state.get("reports_filter_type", "todo")
            filename_parts.append(filter_type_name)
            if st.session_state.get("reports_agrupado"):
                filename_parts.append("agrupado")
            
            filename = f"reporte_auditoria_{'_'.join(filename_parts)}.csv"
            
            st.download_button(
                label=f"📥 Descargar CSV completo ({total_records:,} registros)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help="Descarga todos los registros que coinciden con tus criterios de búsqueda (no paginado)"
            )


def render_loader_page():
    st.title(APP_TITLE)
    st.caption("Sube tu CSV, mira el progreso en vivo y descarga los resultados.")

    ensure_dirs()

    uploaded = st.file_uploader("Sube el archivo CSV", type=["csv"])
    col1, col2 = st.columns([1, 1])
    with col1:
        verbose = st.checkbox("Verbose", value=True)
    with col2:
        start_btn = st.button("Iniciar proceso")

    st.session_state.setdefault("running", False)
    st.session_state.setdefault("proc", None)
    st.session_state.setdefault("log_offset", 0)
    st.session_state.setdefault("last_result", "neutral")
    st.session_state.setdefault("already_conciliated_detected", False)
    st.session_state.setdefault("phase", "idle")  # idle | dry_run | confirm | inserting
    st.session_state.setdefault("last_csv_path", "")
    st.session_state.setdefault("dry_run_summary_loaded", False)
    st.session_state.setdefault("show_confirm_modal", False)

    if start_btn:
        if uploaded is None:
            st.error("Primero sube un CSV.")
        else:
            dest = IN_DIR / (uploaded.name or "archivo.csv")
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Archivo guardado: {dest}")
            st.session_state.last_csv_path = str(dest)
            st.session_state.verbose = bool(verbose)
            # Fase 1: DRY-RUN
            st.session_state.proc = run_job(dest, verbose=verbose, dry_run=True)
            st.session_state.running = True
            st.session_state.log_offset = 0
            st.session_state.last_result = "neutral"
            st.session_state.already_conciliated_detected = False
            st.session_state.phase = "dry_run"
            st.session_state.dry_run_summary_loaded = False

    log_box = st.empty()
    status_box = st.empty()

    if st.session_state.running and st.session_state.proc:
        status_box.info("Proceso en ejecución… leyendo logs en tiempo real.")
        for _ in range(2000):
            if st.session_state.proc.poll() is None:
                st.session_state.log_offset, new_text = tail_file(LOG_FILE, st.session_state.log_offset)
                if new_text:
                    log_box.code(new_text, language="log")
                    text_l = new_text.lower()
                    if (
                        "ids ya conciliados detectados" in text_l
                        or "ya estaban conciliados" in text_l
                        or "ya se encuentran conciliadas" in text_l
                    ):
                        st.session_state.already_conciliated_detected = True
                else:
                    if st.session_state.proc.stdout:
                        line = st.session_state.proc.stdout.readline()
                        if line:
                            log_box.code(line, language="log")
                            text_l = str(line).lower()
                            if (
                                "ids ya conciliados detectados" in text_l
                                or "ya estaban conciliados" in text_l
                                or "ya se encuentran conciliadas" in text_l
                            ):
                                st.session_state.already_conciliated_detected = True
                time.sleep(0.3)
            else:
                st.session_state.log_offset, new_text = tail_file(LOG_FILE, st.session_state.log_offset)
                if new_text:
                    log_box.code(new_text, language="log")
                    text_l = new_text.lower()
                    if (
                        "ids ya conciliados detectados" in text_l
                        or "ya estaban conciliados" in text_l
                        or "ya se encuentran conciliadas" in text_l
                    ):
                        st.session_state.already_conciliated_detected = True
                exit_code = st.session_state.proc.returncode
                phase = st.session_state.get("phase", "idle")
                if phase == "dry_run":
                    if exit_code == 0:
                        status_box.success("Validación y resumen listos (DRY-RUN).")
                        st.session_state.last_result = "neutral"
                        # Pasar a fase de confirmación explícita y abrir modal
                        st.session_state.phase = "confirm"
                        st.session_state.show_confirm_modal = True
                    else:
                        status_box.error(f"DRY-RUN con errores ❌ (código {exit_code})")
                        st.session_state.last_result = "error"
                    st.session_state.running = False
                else:
                    if st.session_state.already_conciliated_detected:
                        status_box.error("Las facturas cargadas ya se encuentran conciliadas en el sistema.")
                        st.session_state.last_result = "error"
                    elif exit_code == 0:
                        status_box.success("Proceso finalizado OK ✅")
                        st.session_state.last_result = "success"
                    else:
                        status_box.error(f"Proceso finalizado con errores ❌ (código {exit_code})")
                        st.session_state.last_result = "error"
                    st.session_state.running = False
                break

    # Si DRY-RUN finalizó OK, cargar resumen y mostrar modal de confirmación
    if st.session_state.get("phase") in ("dry_run", "confirm") and not st.session_state.get("running") and st.session_state.get("last_result") != "error":
        summary_path = OUT_DIR / "dry_run_summary.csv"
        if summary_path.exists() and not st.session_state.get("dry_run_summary_loaded"):
            try:
                df_sum = pd.read_csv(summary_path, dtype={"NIT": str, "RAZON_SOCIAL": str})
                st.session_state["dry_run_df"] = df_sum
                st.session_state["dry_run_summary_loaded"] = True
            except Exception as exc:
                st.warning(f"No fue posible leer el resumen de dry-run: {exc}")

        df_sum = st.session_state.get("dry_run_df")
        has_rows = bool(df_sum is not None and not df_sum.empty)
        if has_rows and st.session_state.get("show_confirm_modal"):
            render_confirmation_ui(df_sum)
        elif not has_rows:
            st.info("No hay filas nuevas para insertar (posiblemente ya conciliadas).")

    st.subheader("Descargar resultados")
    cols = st.columns(3)

    with cols[0]:
        file_to_download_button(
            "Descargar log_errores.csv",
            OUT_DIR / "log_errores.csv",
            color="red" if st.session_state.last_result == "error" else "neutral",
        )
    with cols[1]:
        file_to_download_button(
            "Descargar carga_exitosa_join.csv",
            OUT_DIR / "carga_exitosa_join.csv",
            color="green" if st.session_state.last_result == "success" else "neutral",
        )
    with cols[2]:
        file_to_download_button(
            "Descargar runtime.log",
            LOG_FILE,
            color="neutral",
        )

    with st.expander("Descargar actas de conciliación", expanded=False):
        nit_filter_raw = st.text_input(
            "Filtrar por NIT (opcional)",
            key="actas_nit_filter",
            placeholder="Solo números",
            help="Ingresa solo dígitos del NIT (sin guiones ni DV)",
        )
        nit_filter = "".join(ch for ch in (nit_filter_raw or "") if ch.isdigit())
        if (nit_filter_raw or "") and nit_filter_raw != nit_filter:
            st.warning("Solo números permitidos para NIT; se removieron caracteres no numéricos.")
        try:
            cfg, missing = get_db_config()
            if missing:
                st.warning("Faltan variables de entorno MySQL para listar actas.")
            else:
                import pymysql, os
                conn = pymysql.connect(
                    host=cfg["host"], port=cfg["port"], user=cfg["user"],
                    password=cfg["password"], database=cfg["database"],
                    cursorclass=pymysql.cursors.Cursor,
                )
                try:
                    with conn.cursor() as cur:
                        if nit_filter:
                            cur.execute("""
                              SELECT nit, file_name, file_path, valor_glosa,
                                     valor_aceptado_eps, valor_aceptado_ips, valor_ratificado,
                                     usuario, created_at
                              FROM conciliation_acta_files
                              WHERE nit = %s
                              ORDER BY created_at DESC
                              LIMIT 200
                            """, (nit_filter,))
                        else:
                            cur.execute("""
                              SELECT nit, file_name, file_path, valor_glosa,
                                     valor_aceptado_eps, valor_aceptado_ips, valor_ratificado,
                                     usuario, created_at
                              FROM conciliation_acta_files
                              ORDER BY created_at DESC
                              LIMIT 200
                            """)
                        rows = cur.fetchall()
                        cols = [d[0] for d in cur.description]
                        df_act = pd.DataFrame(rows, columns=cols)
                        if df_act.empty:
                            if nit_filter:
                                st.info("No se encontraron actas para el NIT ingresado.")
                            else:
                                st.info("Aún no hay actas registradas para descargar.")
                            return
                        # Construir columna de descarga dentro de la tabla usando data URLs
                        import base64
                        download_urls = []
                        for _, r in df_act.iterrows():
                            p = Path(str(r.get("file_path", "")))
                            if p.exists():
                                try:
                                    with open(p, "rb") as f:
                                        b64 = base64.b64encode(f.read()).decode()
                                    safe_name = str(r.get("file_name") or "acta.xlsx").replace(",", "_")
                                    url = (
                                        "data:application/"
                                        f"vnd.openxmlformats-officedocument.spreadsheetml.sheet;name={safe_name};base64," + b64
                                    )
                                except Exception:
                                    url = ""
                            else:
                                url = ""
                            download_urls.append(url)
                        # Forzar tipo texto para compatibilidad con LinkColumn
                        df_act["Descargar"] = pd.Series(download_urls, dtype=object)

                        # Mostrar como editor solo-lectura con columna Link
                        # Evitar duplicar "Descargar" y ocultar "file_path"
                        visible_cols = [c for c in df_act.columns if c not in ("file_path", "Descargar")] + ["Descargar"]
                        df_view = df_act[[c for c in visible_cols if c in df_act.columns]]
                        # Etiquetas amigables: reemplazar '_' por ' '
                        pretty_cols = {c: st.column_config.Column(c.replace("_", " ")) for c in df_view.columns if c != "Descargar"}
                        st.data_editor(
                            df_view,
                            width='stretch',
                            hide_index=True,
                            disabled=True,
                            column_config={
                                "Descargar": st.column_config.LinkColumn(
                                    "Descargar",
                                    display_text="⬇️ Descargar",
                                ),
                                **pretty_cols,
                            },
                        )
                finally:
                    conn.close()
        except Exception as ex:
            st.error(f"No fue posible listar actas: {ex}")

    with st.expander("🗑️ Eliminar registros de conciliación", expanded=False):
        st.caption("Sube un archivo Excel con los registros a eliminar. Solo se eliminarán registros con eps_ratified_value > 0")
        
        delete_file = st.file_uploader(
            "Selecciona archivo Excel con registros a eliminar",
            type=["xlsx", "xls"],
            key="delete_excel_upload",
            help="El archivo debe contener al menos la columna 'id' (puede ser en mayúsculas o minúsculas)"
        )
        
        st.session_state.setdefault("delete_validation_done", False)
        st.session_state.setdefault("delete_records_to_process", None)
        st.session_state.setdefault("delete_show_confirmation", False)
        
        if delete_file:
            col_val, col_clear = st.columns([1, 1])
            with col_val:
                if st.button("📋 Validar registros", type="primary", use_container_width=True):
                    try:
                        df_delete = pd.read_excel(delete_file)
                        
                        df_delete.columns = [col.lower() for col in df_delete.columns]
                        
                        if 'id' not in df_delete.columns:
                            st.error("❌ El archivo debe contener la columna 'id'")
                        else:
                            ids_to_check = df_delete['id'].dropna().astype(str).tolist()
                            
                            if not ids_to_check:
                                st.error("❌ No se encontraron IDs válidos en el archivo")
                            else:
                                st.info(f"🔍 Validando {len(ids_to_check)} registros...")
                                
                                cfg, missing = get_db_config()
                                if missing:
                                    st.error("Faltan variables de entorno MySQL: " + ", ".join(missing))
                                else:
                                    import pymysql
                                    conn = pymysql.connect(
                                        host=cfg["host"],
                                        port=cfg["port"],
                                        user=cfg["user"],
                                        password=cfg["password"],
                                        database=cfg["database"],
                                        cursorclass=pymysql.cursors.DictCursor,
                                    )
                                    
                                    validation_results = []
                                    
                                    try:
                                        with conn.cursor() as cursor:
                                            for audit_id in ids_to_check:
                                                cursor.execute("""
                                                    SELECT 
                                                        cr.id,
                                                        cr.auditory_final_report_id,
                                                        cr.eps_ratified_value,
                                                        afr.nit,
                                                        afr.razon_social,
                                                        afr.numero_factura
                                                    FROM conciliation_results cr
                                                    LEFT JOIN auditory_final_reports afr 
                                                        ON cr.auditory_final_report_id = afr.id
                                                    WHERE cr.auditory_final_report_id = %s
                                                """, (audit_id,))
                                                
                                                row = cursor.fetchone()
                                                
                                                if row:
                                                    eps_ratified = float(row.get('eps_ratified_value', 0) or 0)
                                                    can_delete = eps_ratified > 0
                                                    
                                                    validation_results.append({
                                                        'ID Auditoría': audit_id,
                                                        'ID Conciliación': row.get('id'),
                                                        'NIT': row.get('nit', 'N/A'),
                                                        'Razón Social': row.get('razon_social', 'N/A'),
                                                        'Factura': row.get('numero_factura', 'N/A'),
                                                        'Valor Ratificado': eps_ratified,
                                                        'Estado': '✅ Puede eliminarse' if can_delete else '❌ NO puede eliminarse (valor ratificado = 0)',
                                                        'Eliminar': can_delete
                                                    })
                                                else:
                                                    validation_results.append({
                                                        'ID Auditoría': audit_id,
                                                        'ID Conciliación': None,
                                                        'NIT': 'N/A',
                                                        'Razón Social': 'N/A',
                                                        'Factura': 'N/A',
                                                        'Valor Ratificado': 0,
                                                        'Estado': '⚠️ No encontrado en conciliation_results',
                                                        'Eliminar': False
                                                    })
                                    finally:
                                        conn.close()
                                    
                                    df_validation = pd.DataFrame(validation_results)
                                    st.session_state["delete_records_to_process"] = df_validation
                                    st.session_state["delete_validation_done"] = True
                                    st.session_state["delete_show_confirmation"] = True
                                    rerun_app()
                    
                    except Exception as exc:
                        st.error(f"❌ Error al validar archivo: {exc}")
                        import traceback
                        st.code(traceback.format_exc())
            
            with col_clear:
                if st.button("🔄 Limpiar", use_container_width=True):
                    st.session_state["delete_validation_done"] = False
                    st.session_state["delete_records_to_process"] = None
                    st.session_state["delete_show_confirmation"] = False
                    rerun_app()
        
        if st.session_state.get("delete_show_confirmation") and st.session_state.get("delete_records_to_process") is not None:
            df_val = st.session_state["delete_records_to_process"]
            
            st.divider()
            st.subheader("📊 Resumen de validación")
            
            can_delete = df_val[df_val['Eliminar'] == True]
            cannot_delete = df_val[df_val['Eliminar'] == False]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total registros", len(df_val))
            with col_m2:
                st.metric("✅ Pueden eliminarse", len(can_delete), delta="OK", delta_color="normal")
            with col_m3:
                st.metric("❌ NO pueden eliminarse", len(cannot_delete), delta="Bloqueados", delta_color="inverse")
            
            st.dataframe(
                df_val,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID Auditoría": st.column_config.NumberColumn("ID Auditoría"),
                    "ID Conciliación": st.column_config.NumberColumn("ID Conciliación"),
                    "NIT": st.column_config.TextColumn("NIT"),
                    "Razón Social": st.column_config.TextColumn("Razón Social"),
                    "Factura": st.column_config.TextColumn("Factura"),
                    "Valor Ratificado": st.column_config.NumberColumn("Valor Ratificado", format="%.2f"),
                    "Estado": st.column_config.TextColumn("Estado"),
                    "Eliminar": st.column_config.CheckboxColumn("Eliminar", disabled=True)
                }
            )
            
            if len(cannot_delete) > 0:
                with st.expander("⚠️ Ver registros que NO pueden eliminarse", expanded=False):
                    st.dataframe(
                        cannot_delete[['ID Auditoría', 'NIT', 'Razón Social', 'Factura', 'Valor Ratificado', 'Estado']],
                        use_container_width=True,
                        hide_index=True
                    )
            
            if len(can_delete) > 0:
                st.warning(f"⚠️ **ATENCIÓN**: Estás a punto de eliminar {len(can_delete)} registro(s) de la tabla `conciliation_results`. Esta acción NO se puede deshacer.")
                
                confirm_text = st.text_input(
                    "Para confirmar, escribe 'ELIMINAR' en mayúsculas:",
                    key="delete_confirm_text",
                    placeholder="ELIMINAR"
                )
                
                if st.button("🗑️ CONFIRMAR ELIMINACIÓN", type="primary", disabled=(confirm_text != "ELIMINAR")):
                    if confirm_text == "ELIMINAR":
                        try:
                            cfg, missing = get_db_config()
                            if missing:
                                st.error("Faltan variables de entorno MySQL")
                            else:
                                import pymysql
                                conn = pymysql.connect(
                                    host=cfg["host"],
                                    port=cfg["port"],
                                    user=cfg["user"],
                                    password=cfg["password"],
                                    database=cfg["database"],
                                )
                                
                                deleted_count = 0
                                deletion_log = []
                                
                                try:
                                    with conn.cursor() as cursor:
                                        for _, row in can_delete.iterrows():
                                            conciliation_id = row['ID Conciliación']
                                            audit_id = row['ID Auditoría']
                                            
                                            cursor.execute(
                                                "DELETE FROM conciliation_results WHERE id = %s",
                                                (conciliation_id,)
                                            )
                                            
                                            if cursor.rowcount > 0:
                                                deleted_count += 1
                                                deletion_log.append(f"✅ Eliminado: ID Conciliación {conciliation_id} (Auditoría {audit_id}) - {row['Razón Social']} - Factura {row['Factura']}")
                                            else:
                                                deletion_log.append(f"⚠️ No se pudo eliminar: ID Conciliación {conciliation_id}")
                                        
                                        conn.commit()
                                    
                                    st.success(f"✅ Se eliminaron {deleted_count} registro(s) exitosamente")
                                    
                                    with st.expander("📋 Ver log de eliminación", expanded=True):
                                        for log_entry in deletion_log:
                                            st.text(log_entry)
                                    
                                    st.session_state["delete_validation_done"] = False
                                    st.session_state["delete_records_to_process"] = None
                                    st.session_state["delete_show_confirmation"] = False
                                    
                                except Exception as e:
                                    conn.rollback()
                                    st.error(f"❌ Error durante la eliminación: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                                finally:
                                    conn.close()
                        
                        except Exception as exc:
                            st.error(f"❌ Error al conectar con la base de datos: {exc}")
                    else:
                        st.error("Debes escribir 'ELIMINAR' para confirmar")
            else:
                st.info("ℹ️ No hay registros que puedan eliminarse")

    with st.expander("Generar sábana por %", expanded=False):
        pct_file = st.file_uploader(
            "Excel de conciliación",
            type=["xlsx"],
            key="pct_excel_upl",
        )
        col_ips, col_eps, col_rat = st.columns(3)
        pct_ips = col_ips.number_input(
            "% IPS",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key="pct_ips",
        )
        pct_eps = col_eps.number_input(
            "% EPS",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key="pct_eps",
        )
        pct_rat = col_rat.number_input(
            "% RAT",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key="pct_rat",
        )

        if st.button("Generar sábana", key="btn_gen_pct"):
            file_obj = pct_file
            ips = pct_ips
            eps = pct_eps
            rat = pct_rat
            if not file_obj:
                st.error("Sube un Excel primero.")
            elif abs((ips + eps + rat) - 100.0) > 0.001:
                st.error("Los porcentajes deben sumar 100%.")
            else:
                try:
                    df_out, resumen, csv_bytes = generate_percentage_sheet(file_obj, ips, eps, rat)
                    st.success("Sábana generada.")
                    st.dataframe(df_out.head(15), width='stretch')
                    st.code(resumen)
                    st.download_button(
                        "Descargar sábana por %",
                        data=csv_bytes,
                        file_name="sabana_porcentajes.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"No se pudo generar la sábana: {exc}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    user_email = require_login()

    ensure_dirs()

    default_menu = st.session_state.get("menu_option", "Cargar CSV")
    menu_options = ["Cargar CSV", "Resumen", "Reportes"]

    show_detail_option = bool(st.session_state.get("selected_nit"))
    if show_detail_option:
        menu_options.append("Detalle NIT")

    with st.sidebar:
        st.markdown(f"**Usuario:** {user_email}")
        if st.button("Cerrar sesión"):
            for key in ("auth_ok", "user_email", "login_error", "last_email"):
                st.session_state.pop(key, None)
            rerun_app()
        if default_menu not in menu_options:
            default_menu = "Resumen" if "Resumen" in menu_options else menu_options[0]
        index = menu_options.index(default_menu)
        menu_option = st.radio("Menú", options=menu_options, index=index)

    st.session_state["menu_option"] = menu_option

    if menu_option == "Resumen":
        render_summary_page()
    elif menu_option == "Detalle NIT":
        render_nit_detail_page()
    elif menu_option == "Reportes":
        render_reports_page()
    else:
        render_loader_page()


if __name__ == "__main__":
    main()
