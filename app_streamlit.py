# app_streamlit.py
import os
import time
import base64
import subprocess
from pathlib import Path
import streamlit as st

APP_TITLE = "Cargar sabanas de conciliación"
IN_DIR = Path("/data/in")
OUT_DIR = Path("/data/out")
LOG_FILE = OUT_DIR / "runtime.log"

# ------------------------- Utils -------------------------

def ensure_dirs():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def tail_file(path: Path, from_bytes=0, max_lines=4000):
    """Devuelve (nuevo_offset, texto_nuevo) para append incremental del log."""
    if not path.exists():
        return from_bytes, ""
    with open(path, "rb") as f:
        f.seek(from_bytes)
        chunk = f.read()
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()[-max_lines:]  # limitar líneas
    return from_bytes + len(chunk), "\n".join(lines)

def run_job(csv_path: Path, verbose: bool = True):
    """Lanza el proceso como subproceso y retorna el Popen."""
    cmd = [
        "python",
        "csv_conciliation_loader.py",
        "--csv", str(csv_path),
        "--out-dir", str(OUT_DIR),
    ]
    if verbose:
        cmd.append("--verbose")
    # limpiar log previo
    try:
        if LOG_FILE.exists():
            LOG_FILE.unlink()
    except Exception:
        pass
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def file_to_download_button(label: str, path: Path, color: str = "neutral"):
    """Genera un botón HTML con color (neutral, red, green)."""
    if not path.exists():
        return
    mime = "text/csv" if path.suffix.lower() == ".csv" else "text/plain"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    colors = {
        "red":   "#dc2626",  # rojo
        "green": "#16a34a",  # verde
        "neutral": "#374151",  # gris
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

# ------------------------- App -------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Sube tu CSV, mira el progreso en vivo y descarga los resultados.")

    ensure_dirs()

    uploaded = st.file_uploader("Sube el archivo CSV", type=["csv"])
    col1, col2 = st.columns([1,1])
    with col1:
        verbose = st.checkbox("Verbose", value=True)
    with col2:
        start_btn = st.button("Iniciar proceso")

    # Estado
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("proc", None)
    st.session_state.setdefault("log_offset", 0)
    st.session_state.setdefault("last_result", "neutral")  
    # valores posibles: neutral, error, success

    # Iniciar proceso
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
            st.session_state.last_result = "neutral"  # reset al iniciar

    log_box = st.empty()
    status_box = st.empty()

    # Loop de logs
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
                # dump final
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

    # Descargas
    st.subheader("Descargar resultados")
    cols = st.columns(3)

    with cols[0]:
        file_to_download_button(
            "Descargar log_errores.csv",
            OUT_DIR / "log_errores.csv",
            color="red" if st.session_state.last_result == "error" else "neutral"
        )
    with cols[1]:
        file_to_download_button(
            "Descargar carga_exitosa_join.csv",
            OUT_DIR / "carga_exitosa_join.csv",
            color="green" if st.session_state.last_result == "success" else "neutral"
        )
    with cols[2]:
        file_to_download_button(
            "Descargar runtime.log",
            LOG_FILE,
            color="neutral"
        )

if __name__ == "__main__":
    main()