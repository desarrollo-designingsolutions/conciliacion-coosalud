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


def generate_percentage_sheet(excel_file, perc_ips: float, perc_eps: float, perc_rat: float):
    try:
        df = pd.read_excel(excel_file, dtype=str)
    except Exception as exc:
        raise ValueError(f"No fue posible leer el Excel: {exc}") from exc

    expected_headers = EXPECTED_HEADERS
    if list(df.columns) != expected_headers:
        raise ValueError("Las columnas del Excel no coinciden con la especificación.")

    df = df.fillna("")
    try:
        glosa_values = [parse_decimal_value(val).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) for val in df["VALOR_GLOSA"].tolist()]
    except Exception as exc:
        raise ValueError(f"VALOR_GLOSA contiene valores inválidos: {exc}")

    total_glosa = sum(glosa_values, Decimal("0"))

    suma_pct = Decimal(str(perc_ips + perc_eps + perc_rat))
    if abs(suma_pct - Decimal("100")) > Decimal("0.001"):
        raise ValueError("La suma de los porcentajes debe ser 100%.")

    objetivo_ips = (total_glosa * Decimal(str(perc_ips)) / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    objetivo_eps = (total_glosa * Decimal(str(perc_eps)) / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    objetivo_rat = (total_glosa * Decimal(str(perc_rat)) / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    n = len(df)
    ips_vals = [Decimal("0")] * n
    eps_vals = [Decimal("0")] * n
    rat_vals = [Decimal("0")] * n

    def remaining_capacity(idx: int) -> Decimal:
        return max(glosa_values[idx] - (ips_vals[idx] + eps_vals[idx] + rat_vals[idx]), Decimal("0"))

    def assign_phase(values: list[Decimal], objetivo: Decimal):
        restante = objetivo
        for idx in range(n):
            if restante <= Decimal("0"):
                break
            cap = remaining_capacity(idx)
            if cap <= Decimal("0"):
                continue
            asignar = min(cap, restante)
            asignar = asignar.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            values[idx] += asignar
            restante = (restante - asignar).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return restante

    restante_ips = assign_phase(ips_vals, objetivo_ips)
    restante_eps = assign_phase(eps_vals, objetivo_eps)
    restante_rat = assign_phase(rat_vals, objetivo_rat)

    def adjust(values: list[Decimal], objetivo: Decimal):
        diff = (objetivo - sum(values, Decimal("0"))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if diff == Decimal("0"):
            return
        if diff > Decimal("0"):
            for idx in range(n):
                if diff <= Decimal("0"):
                    break
                cap = remaining_capacity(idx)
                if cap <= Decimal("0"):
                    continue
                ajuste = min(cap, diff)
                ajuste = ajuste.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if ajuste <= Decimal("0"):
                    continue
                values[idx] += ajuste
                diff = (diff - ajuste).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            diff = abs(diff)
            for idx in range(n - 1, -1, -1):
                if diff <= Decimal("0"):
                    break
                disponible = values[idx]
                if disponible <= Decimal("0"):
                    continue
                ajuste = min(disponible, diff)
                ajuste = ajuste.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if ajuste <= Decimal("0"):
                    continue
                values[idx] -= ajuste
                diff = (diff - ajuste).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    adjust(ips_vals, objetivo_ips)
    adjust(eps_vals, objetivo_eps)
    adjust(rat_vals, objetivo_rat)

    df_out = df.copy()
    df_out["VALOR_ACEPTADO_POR_IPS"] = [format_decimal_value(v) for v in ips_vals]
    df_out["VALOR_ACEPTADO_POR_EPS"] = [format_decimal_value(v) for v in eps_vals]
    df_out["VALOR_RATIFICADO_EPS"] = [format_decimal_value(v) for v in rat_vals]

    nuevas_obs = []
    for glosa, ips, eps, rat, original in zip(
        glosa_values,
        ips_vals,
        eps_vals,
        rat_vals,
        df.get("OBSERVACIONES", pd.Series([""] * n))
    ):
        ips_v = ips.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        eps_v = eps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        rat_v = rat.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        glosa_v = glosa.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if ips_v == glosa_v and eps_v == Decimal("0") and rat_v == Decimal("0"):
            nuevas_obs.append("EL PRESTADOR ACEPTA EL VALOR")
        elif ips_v == glosa_v:
            nuevas_obs.append("SE LEVANTA GLOSA, EL PRESTADOR PRESENTA LOS SOPORTES NECESARIOS")
        elif ips_v > Decimal("0") and eps_v > Decimal("0") and rat_v == Decimal("0"):
            nuevas_obs.append(
                "SE LEVANTA GLOSA PARCIAL, EL PRESTADOR ADJUNTA LOS SOPORTES NECESARIOS PARA LA GLOSA APLICADA Y ADICIONAN EL CONTRATO Y ANEXOS DEL MISMO"
            )
        elif rat_v == glosa_v and ips_v == Decimal("0") and eps_v == Decimal("0"):
            nuevas_obs.append("SE RATIFICA LA GLOSA")
        elif ips_v > Decimal("0") and rat_v > Decimal("0"):
            nuevas_obs.append(
                "SE LEVANTA GLOSA PARCIAL, EL PRESTADOR PRESENTA LOS SOPORTES NECESARIOS, SE RATIFICA GLOSA PARCIAL"
            )
        elif eps_v > Decimal("0") and rat_v > Decimal("0"):
            nuevas_obs.append("EPS ACEPTA GLOSA PARCIAL, SE RATIFICA GLOSA PARCIAL")
        else:
            nuevas_obs.append(str(original))

    df_out["OBSERVACIONES"] = nuevas_obs

    numeric_cols = [
        "VALOR_GLOSA",
        "VALOR_TOTAL_SERVICIO",
        "VALOR_ACEPTADO_POR_IPS",
        "VALOR_ACEPTADO_POR_EPS",
        "VALOR_RATIFICADO_EPS",
    ]
    for col in numeric_cols:
        if col in df_out.columns:
            df_out[col] = [format_decimal_value(parse_decimal_value(val)) for val in df_out[col]]

    csv_buffer = StringIO()
    df_out.to_csv(
        csv_buffer,
        index=False,
        sep=";",
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    resumen = {
        "total_glosa": format_decimal_value(total_glosa),
        "objetivo_ips": format_decimal_value(objetivo_ips),
        "objetivo_eps": format_decimal_value(objetivo_eps),
        "objetivo_rat": format_decimal_value(objetivo_rat),
        "asignado_ips": format_decimal_value(sum(ips_vals, Decimal("0"))),
        "asignado_eps": format_decimal_value(sum(eps_vals, Decimal("0"))),
        "asignado_rat": format_decimal_value(sum(rat_vals, Decimal("0"))),
    }

    return df_out, csv_buffer.getvalue().encode("utf-8"), resumen

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
    st.text_area(
        "Comentario (opcional)",
        key=comentario_key,
        height=150,
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Guardar", key=save_key):
            estado = st.session_state.get(estado_key)
            if not estado:
                st.warning("Debes seleccionar un estado.")
            else:
                comentario = st.session_state.get(comentario_key, "")
                user_email = st.session_state.get("user_email", "")
                try:
                    insert_nit_state(cfg, nit, estado, comentario, user_email)
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

    st.session_state.pop("summary_table_editor", None)

    editor_result = st.data_editor(
        df_view,
        use_container_width=True,
        hide_index=True,
        disabled=[col for col in df_view.columns if col != "Acciones"],
        column_config={
            "Acciones": st.column_config.CheckboxColumn(
                "➕",
                help="Agregar estado al NIT",
                default=False,
            )
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
        row = filtered.loc[idx]
        st.session_state["selected_nit"] = str(row.get("nit", ""))
        st.session_state["selected_razon"] = str(row.get("razon_social", ""))
        st.session_state.show_state_modal = True
        rerun_app()

    selected_nit = st.session_state.get("selected_nit")
    selected_razon = st.session_state.get("selected_razon")

    if st.session_state.get("show_state_modal") and selected_nit:
        if hasattr(st, "modal"):
            with st.modal("Agregar estado al NIT"):
                try:
                    cfg_modal, missing_modal = get_db_config()
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if missing_modal:
                        st.error(
                            "Faltan variables de entorno para la conexión MySQL: "
                            + ", ".join(missing_modal)
                        )
                    else:
                        render_state_form(cfg_modal, selected_nit, selected_razon, prefix="modal")
        else:
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
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


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

    log_box = st.empty()
    status_box = st.empty()

    with st.expander("Generar sábana por %"):
        excel_percentage = st.file_uploader("Excel", type=["xlsx"], key="percentage_excel")
        col_ips, col_eps, col_rat = st.columns(3)
        perc_ips = col_ips.number_input("% IPS", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="perc_ips")
        perc_eps = col_eps.number_input("% EPS", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="perc_eps")
        perc_rat = col_rat.number_input("% RAT", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="perc_rat")
        if st.button("Generar sábana", key="generate_percentage_button"):
            if excel_percentage is None:
                st.error("Debes subir un Excel de conciliación.")
            else:
                try:
                    _, csv_bytes, resumen = generate_percentage_sheet(
                        excel_percentage,
                        perc_ips,
                        perc_eps,
                        perc_rat,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.session_state["percentage_csv"] = csv_bytes
                    st.session_state["percentage_summary"] = resumen
                    st.success("Sábana generada correctamente.")
        if st.session_state.get("percentage_csv"):
            resumen = st.session_state.get("percentage_summary", {})
            if resumen:
                st.markdown(
                    "\n".join(
                        [
                            f"**{k.replace('_', ' ').title()}:** {v}"
                            for k, v in resumen.items()
                        ]
                    )
                )
            st.download_button(
                "Descargar sábana por %",
                data=st.session_state["percentage_csv"],
                file_name="sabana_porcentajes.csv",
                mime="text/csv",
                key="download_percentage_csv",
            )

    if st.session_state.running and st.session_state.proc:
        status_box.info("Proceso en ejecución… leyendo logs en tiempo real.")
        for _ in range(2000):
            if st.session_state.proc.poll() is None:
                st.session_state.log_offset, new_text = tail_file(LOG_FILE, st.session_state.log_offset)
                if new_text:
                    log_box.code(new_text, language="log")
                else:
                    if st.session_state.proc.stdout:
                        line = st.session_state.proc.stdout.readline()
                        if line:
                            log_box.code(line, language="log")
                time.sleep(0.3)
            else:
                st.session_state.log_offset, new_text = tail_file(LOG_FILE, st.session_state.log_offset)
                if new_text:
                    log_box.code(new_text, language="log")
                exit_code = st.session_state.proc.returncode
                if exit_code == 0:
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
            value=90.0,
            step=0.1,
            key="pct_ips",
        )
        pct_eps = col_eps.number_input(
            "% EPS",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            key="pct_eps",
        )
        pct_rat = col_rat.number_input(
            "% RAT",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            key="pct_rat",
        )

        if st.button("Generar sábana", key="btn_gen_pct"):
            if pct_file is None:
                st.error("Sube un Excel primero.")
            else:
                total_pct = pct_ips + pct_eps + pct_rat
                if abs(total_pct - 100.0) > 0.001:
                    st.error("Los porcentajes deben sumar 100%.")
                else:
                    try:
                        df_pct, resumen = generate_percentage_sheet(
                            pct_file,
                            pct_ips,
                            pct_eps,
                            pct_rat,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Archivo válido. Pendiente de cálculos.")
                        st.text(resumen)
                        st.dataframe(df_pct.head(10), use_container_width=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    user_email = require_login()

    ensure_dirs()

    default_menu = st.session_state.get("menu_option", "Cargar CSV")
    menu_options = ["Cargar CSV", "Resumen"]

    show_detail_option = (
        not hasattr(st, "modal")
        and st.session_state.get("selected_nit")
    )
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
