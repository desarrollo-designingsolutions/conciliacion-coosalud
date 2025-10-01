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

APP_TITLE = "Cargar sabanas de conciliación"
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
        default=unique_nits,
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



def run_job(csv_path: Path, verbose: bool = True):
    cmd = [
        "python",
        "csv_conciliation_loader.py",
        "--csv", str(csv_path),
        "--out-dir", str(OUT_DIR),
    ]
    if verbose:
        cmd.append("--verbose")
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

    if start_btn:
        if uploaded is None:
            st.error("Primero sube un CSV.")
        else:
            dest = IN_DIR / (uploaded.name or "archivo.csv")
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Archivo guardado: {dest}")
            st.session_state.proc = run_job(dest, verbose=verbose)
            st.session_state.running = True
            st.session_state.log_offset = 0
            st.session_state.last_result = "neutral"
            st.session_state.already_conciliated_detected = False

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
        nit_filter = st.text_input("Filtrar por NIT (opcional)")
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
                        import pandas as pd
                        df_act = pd.DataFrame(rows, columns=cols)
                        # Construir columna de descarga dentro de la tabla usando data URLs
                        from pathlib import Path
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
                        df_act["Descargar"] = download_urls

                        # Mostrar como editor solo-lectura con columna Link
                        # Evitar duplicar "Descargar" y ocultar "file_path"
                        visible_cols = [c for c in df_act.columns if c not in ("file_path", "Descargar")] + ["Descargar"]
                        df_view = df_act[[c for c in visible_cols if c in df_act.columns]]
                        st.data_editor(
                            df_view,
                            width='stretch',
                            hide_index=True,
                            disabled=True,
                            column_config={
                                "Descargar": st.column_config.LinkColumn(
                                    "Descargar",
                                    display_text="⬇️ Descargar",
                                )
                            },
                        )
                finally:
                    conn.close()
        except Exception as ex:
            st.error(f"No fue posible listar actas: {ex}")

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
    menu_options = ["Cargar CSV", "Resumen"]

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
    else:
        render_loader_page()


if __name__ == "__main__":
    main()
