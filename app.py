"""
app.py – Netflora Online
Dashboard Streamlit para detecção de espécies vegetais em ortofotos de drone.
Desenvolvido com base no projeto Netflora da Embrapa.

Layout sem barra lateral, simples e didático.
"""

import gc
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import streamlit as st

# ==================== CONSTANTES ====================

APP_ROOT = Path(__file__).resolve().parent
WORKDIR = APP_ROOT / "workdir"
NETFLORA_DIR = WORKDIR / "netflora_src"
TILES_DIR = WORKDIR / "tiles"
COORDS_CSV = WORKDIR / "tile_coords.csv"
RUNS_DIR = WORKDIR / "runs"
LOCKS_DIR = WORKDIR / "locks"
INFERENCE_LOCK_FILE = LOCKS_DIR / "inference.lock"
DEFAULT_ORTHO = APP_ROOT / "ortofoto" / "ortofoto_exemplo1_corte.tif"
DEFAULT_WEIGHTS = WORKDIR / "model_weights.pt"
DEFAULT_NETFLORA_ZIP = "https://github.com/NetFlora/Netflora/archive/refs/heads/main.zip"
DEFAULT_WEIGHTS_URL = "https://github.com/NetFlora/Netflora/releases/download/Assets/PMFS_Embrapa00.pt"
# groups.json local (committed to git) — always available without downloading netflora_src
GROUPS_JSON = APP_ROOT / "json" / "groups.json"


def _sanitize_filename(filename: str) -> str:
    safe = "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in ("-", "_", "."))
    return safe.strip("._") or "upload.tif"


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def _get_session_root() -> Path:
    session_root = WORKDIR / "sessions" / _get_session_id()
    session_root.mkdir(parents=True, exist_ok=True)
    return session_root


def _new_unique_run_name() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"online_{ts}_{uuid.uuid4().hex[:8]}"


@contextmanager
def acquire_inference_lock(
    lock_path: Path,
    wait_timeout: int = 1800,
    stale_seconds: int = 7200,
    poll_seconds: float = 1.0,
) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    deadline = time.time() + wait_timeout
    waiting_note_shown = False

    while True:
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            owner = f"session={_get_session_id()} pid={os.getpid()} ts={int(time.time())}"
            os.write(lock_fd, owner.encode("utf-8"))
            os.close(lock_fd)
            lock_fd = None
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_seconds:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue

            if time.time() >= deadline:
                raise TimeoutError("Tempo de espera excedido na fila de inferência. Tente novamente em instantes.")

            if not waiting_note_shown:
                st.info("Outro usuário está executando detecção agora. Sua execução está em fila e iniciará automaticamente.")
                waiting_note_shown = True
            time.sleep(poll_seconds)

    try:
        yield
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        lock_path.unlink(missing_ok=True)

# ==================== TRANSLATIONS ====================

TRANSLATIONS: dict = {
    "en": {
        "title": "🌿 Netflora – Plant Species Detection with AI",
        "intro": (
            "**Project developed by Embrapa** for automated forest inventory using drones and artificial intelligence. "
            "The tool analyzes aerial orthophotos and automatically identifies the location of plant species of interest, "
            "such as Açaí, Brazil Nut, Palms, and other Amazon rainforest species."
        ),
        "how_it_works": "## 📖 How does it work?",
        "step1": "📷 <strong>1. Orthophoto</strong><br>Georeferenced aerial image (.tif) captured by drone.",
        "step2": "✂️ <strong>2. Tiles</strong><br>The image is split into crops to facilitate model analysis.",
        "step3": "🤖 <strong>3. Detection</strong><br>A deep learning model identifies species in each crop.",
        "step4": "🗺️ <strong>4. Result</strong><br>Detections are georeferenced and displayed on an interactive map.",
        "how_desc": (
            "To enable this solution, Embrapa structured the project using Python and the YOLO (You Only Look Once) "
            "object detection model, widely used in computer vision for automatic element identification in images. "
            "The original proposal was developed for cloud execution via Google Colab/Notebook, allowing orthophoto "
            "processing with AI support without requiring robust local infrastructure.\n\n"
            "From this base, the code was adapted for local execution, expanding flexibility and enabling greater "
            "data control. Additionally, this public-access page was implemented, where anyone can upload an "
            "orthophoto and receive the analysis results in a practical and accessible way.\n\n"
            "After processing, the page provides the generated files for export, especially the Shapefile format "
            "widely used in geographic information systems (GIS). This allows results to be integrated into "
            "topographic analysis, mapping and territorial planning tools, facilitating the technical use of the "
            "information obtained."
        ),
        "ortho_section": "## 📁 Input Orthophoto",
        "ortho_radio": "Select the orthophoto source:",
        "ortho_opt_example": "Use example orthophoto",
        "ortho_opt_upload": "Upload my orthophoto (.tif)",
        "ortho_radio_help": "Choose between the example orthophoto included in the project or upload your own GeoTIFF file.",
        "ortho_using": "✅ Using: `ortofoto_exemplo1_corte.tif`",
        "ortho_not_found": "Example orthophoto not found at `ortofoto/ortofoto_exemplo1_corte.tif`.",
        "ortho_upload_label": "Upload your GeoTIFF orthophoto:",
        "ortho_upload_help": "The file must be a GeoTIFF with a defined geographic reference system (CRS).",
        "ortho_received": "✅ File received: `{name}`",
        "ortho_waiting": "Waiting for orthophoto upload.",
        "config_section": "## ⚙️ Detection Settings",
        "algo_label": "🌱 Detection algorithm:",
        "algo_help": "Choose the set of species the model should detect.",
        "conf_label": "🎯 Minimum confidence:",
        "conf_help": "Detections below this confidence will be ignored. Higher values = fewer but more precise detections.",
        "advanced_expander": "🔧 Advanced Settings",
        "tile_size_label": "Tile size (px):",
        "tile_size_help": "Resolution of crops sent to the model.",
        "overlap_label": "Overlap (px):",
        "overlap_help": "Overlap between adjacent tiles to avoid missing detections at edges.",
        "max_tiles_label": "Maximum tiles:",
        "max_tiles_help": "Tile limit for demonstration. Increase to process larger images.",
        "device_label": "Device:",
        "device_help": "'cpu' for processor; '0' for NVIDIA GPU (requires CUDA).",
        "netflora_url_label": "Netflora repository URL (zip):",
        "weights_url_label": "Model weights URL (.pt):",
        "first_run_info": (
            "ℹ️ **First run:** the detection model (~135 MB) and Netflora code "
            "will be downloaded automatically when you click *Run Detection*. "
            "This may take a few minutes depending on your connection."
        ),
        "run_button": "🔍 Run Detection",
        "run_warning": "⚠️ Upload an orthophoto before running.",
        "pipeline_status": "🔄 Running detection pipeline...",
        "step_prep_netflora": "**1/6** Preparing Netflora code...",
        "step_prep_netflora_err": "❌ Failed to prepare Netflora.",
        "step_check_weights": "**2/6** Checking model weights...",
        "step_check_weights_err": "❌ Failed to prepare model weights.",
        "step_gen_tiles": "**3/6** Generating orthophoto tiles...",
        "step_gen_tiles_err": "❌ Failed to generate tiles.",
        "step_gen_tiles_done": "**3/6** {count} tiles generated.",
        "step_no_tiles_err": "❌ No valid tiles were generated.",
        "step_no_tiles_msg": "No valid tiles were generated. Check that the orthophoto is valid.",
        "step_run_detect": "**4/6** Running detection model...",
        "step_detect_err": "❌ Detection failed.",
        "step_detect_err_msg": "The detect.py execution failed.",
        "step_error_log": "View error log",
        "step_process_results": "**5/6** Processing results...",
        "step_gen_map": "**6/6** Generating interactive map...",
        "pipeline_done": "✅ Pipeline completed successfully!",
        "results_section": "## 📊 Detection Results",
        "metric_tiles": "🔲 Tiles processed",
        "metric_plants": "🌱 Plants detected",
        "metric_species": "🏷️ Species identified",
        "map_section": "### 🗺️ Interactive Map",
        "map_caption": (
            "The map displays georeferenced detections. "
            "Hover over markers to see the species and detection confidence. "
            "Use the controls in the upper right corner to toggle map layers."
        ),
        "map_unavailable": "Map not available for this result.",
        "table_section": "### 📋 Detection Table",
        "table_empty": "No detections found with current parameters. Try reducing the minimum confidence.",
        "col_tile": "Tile",
        "col_species": "Species",
        "col_confidence": "Confidence",
        "export_section": "### 💾 Export Results",
        "export_caption": "The ZIP file contains: interactive map (.html), detection spreadsheet (.csv) and georeferenced shapefile (.shp and auxiliary files).",
        "export_button": "📦 Download results ZIP",
        "clear_button": "🗑️ Clear results",
        "export_error": "Error generating export: {error}",
        "tile_expander": "🔬 View detections by tile",
        "tile_select": "Choose a tile to inspect:",
        "tile_no_tiles": "No tiles available.",
        "tile_orig_caption": "**Original**",
        "tile_det_caption": "**With detections**",
        "tile_no_det": "No detections in this tile",
        "lang_selector_label": "🌐 Language / Idioma",
    },
    "pt": {
        "title": "🌿 Netflora – Detecção de Espécies Vegetais com IA",
        "intro": (
            "**Projeto desenvolvido pela Embrapa** para o inventário florestal automatizado com uso de drones e inteligência artificial. "
            "A ferramenta analisa ortofotos aéreas e identifica automaticamente a localização de espécies vegetais de interesse, "
            "como Açaí, Castanheira, Palmeiras e outras espécies da floresta amazônica."
        ),
        "how_it_works": "## 📖 Como funciona?",
        "step1": "📷 <strong>1. Ortofoto</strong><br>Imagem aérea georreferenciada (.tif) capturada por drone.",
        "step2": "✂️ <strong>2. Tiles</strong><br>A imagem é dividida em recortes para facilitar a análise pelo modelo.",
        "step3": "🤖 <strong>3. Detecção</strong><br>Um modelo de deep learning identifica as espécies em cada recorte.",
        "step4": "🗺️ <strong>4. Resultado</strong><br>As detecções são georreferenciadas e exibidas em um mapa interativo.",
        "how_desc": (
            "Para viabilizar esta solução, a Embrapa estruturou o projeto com base em Python e no modelo de detecção de objetos YOLO (You Only Look Once), "
            "amplamente utilizado em visão computacional para identificação automática de elementos em imagens. "
            "A proposta original foi desenvolvida para execução em ambiente de nuvem, por meio do Google Colab/Notebook, "
            "permitindo o processamento de ortofotos aéreas com apoio de inteligência artificial sem exigir infraestrutura local robusta.\n\n"
            "A partir dessa base, o código foi adaptado para execução local, ampliando a flexibilidade de uso e possibilitando maior controle sobre os dados. "
            "Além disso, foi implementada essa página de acesso público, onde qualquer pessoa pode enviar uma ortofoto e receber o resultado da análise de forma prática e acessível.\n\n"
            "Após o processamento, a página disponibiliza os arquivos gerados para exportação, incluindo especialmente o Shapefile, "
            "formato amplamente utilizado em sistemas de informação geográfica (GIS). Isso permite que os resultados sejam integrados a "
            "ferramentas de análise topográfica, mapeamento e planejamento territorial, facilitando o aproveitamento técnico das informações obtidas."
        ),
        "ortho_section": "## 📁 Ortofoto de entrada",
        "ortho_radio": "Selecione a fonte da ortofoto:",
        "ortho_opt_example": "Usar ortofoto de exemplo",
        "ortho_opt_upload": "Enviar minha ortofoto (.tif)",
        "ortho_radio_help": "Escolha entre usar a ortofoto de exemplo incluída no projeto ou enviar o seu próprio arquivo GeoTIFF.",
        "ortho_using": "✅ Usando: `ortofoto_exemplo1_corte.tif`",
        "ortho_not_found": "Ortofoto de exemplo não encontrada em `ortofoto/ortofoto_exemplo1_corte.tif`.",
        "ortho_upload_label": "Envie sua ortofoto GeoTIFF:",
        "ortho_upload_help": "O arquivo deve ser um GeoTIFF com sistema de referência geográfico definido (CRS).",
        "ortho_received": "✅ Arquivo recebido: `{name}`",
        "ortho_waiting": "Aguardando upload da ortofoto.",
        "config_section": "## ⚙️ Configurações de detecção",
        "algo_label": "🌱 Algoritmo de detecção:",
        "algo_help": "Escolha o conjunto de espécies que o modelo deve detectar.",
        "conf_label": "🎯 Confiança mínima:",
        "conf_help": "Detecções com confiança abaixo desse valor serão ignoradas. Valores mais altos = menos detecções, mas mais precisas.",
        "advanced_expander": "🔧 Configurações avançadas",
        "tile_size_label": "Tamanho do tile (px):",
        "tile_size_help": "Resolução dos recortes enviados ao modelo.",
        "overlap_label": "Sobreposição (px):",
        "overlap_help": "Sobreposição entre tiles adjacentes para evitar perda de detecções nas bordas.",
        "max_tiles_label": "Máximo de tiles:",
        "max_tiles_help": "Limite de tiles para demonstração. Aumente para processar imagens maiores.",
        "device_label": "Dispositivo:",
        "device_help": "'cpu' para processador; '0' para GPU NVIDIA (requer CUDA).",
        "netflora_url_label": "URL do repositório Netflora (zip):",
        "weights_url_label": "URL dos pesos do modelo (.pt):",
        "first_run_info": (
            "ℹ️ **Primeira execução:** o modelo de detecção (~135 MB) e o código do Netflora "
            "serão baixados automaticamente ao clicar em *Executar Detecção*. "
            "Isso pode levar alguns minutos dependendo da conexão."
        ),
        "run_button": "🔍 Executar Detecção",
        "run_warning": "⚠️ Envie uma ortofoto antes de executar.",
        "pipeline_status": "🔄 Executando pipeline de detecção...",
        "step_prep_netflora": "**1/6** Preparando código do Netflora...",
        "step_prep_netflora_err": "❌ Falha ao preparar Netflora.",
        "step_check_weights": "**2/6** Verificando pesos do modelo...",
        "step_check_weights_err": "❌ Falha ao preparar pesos do modelo.",
        "step_gen_tiles": "**3/6** Gerando tiles da ortofoto...",
        "step_gen_tiles_err": "❌ Falha ao gerar tiles.",
        "step_gen_tiles_done": "**3/6** {count} tiles gerados.",
        "step_no_tiles_err": "❌ Nenhum tile válido foi gerado.",
        "step_no_tiles_msg": "Nenhum tile válido foi gerado. Verifique se a ortofoto é válida.",
        "step_run_detect": "**4/6** Executando modelo de detecção...",
        "step_detect_err": "❌ Falha na detecção.",
        "step_detect_err_msg": "A execução do detect.py falhou.",
        "step_error_log": "Ver log de erro",
        "step_process_results": "**5/6** Processando resultados...",
        "step_gen_map": "**6/6** Gerando mapa interativo...",
        "pipeline_done": "✅ Pipeline concluído com sucesso!",
        "results_section": "## 📊 Resultados da detecção",
        "metric_tiles": "🔲 Tiles processados",
        "metric_plants": "🌱 Plantas detectadas",
        "metric_species": "🏷️ Espécies identificadas",
        "map_section": "### 🗺️ Mapa interativo",
        "map_caption": (
            "O mapa exibe as detecções georreferenciadas. "
            "Passe o mouse sobre os marcadores para ver a espécie e a confiança da detecção. "
            "Use os controles no canto superior direito para alternar as camadas do mapa."
        ),
        "map_unavailable": "Mapa não disponível para este resultado.",
        "table_section": "### 📋 Tabela de detecções",
        "table_empty": "Nenhuma detecção encontrada com os parâmetros atuais. Tente reduzir a confiança mínima.",
        "col_tile": "Tile",
        "col_species": "Espécie",
        "col_confidence": "Confiança",
        "export_section": "### 💾 Exportar resultados",
        "export_caption": "O arquivo ZIP contém: mapa interativo (.html), planilha de detecções (.csv) e shapefile georreferenciado (.shp e arquivos auxiliares).",
        "export_button": "📦 Baixar ZIP de resultados",
        "clear_button": "🗑️ Limpar resultados",
        "export_error": "Erro ao gerar exportação: {error}",
        "tile_expander": "🔬 Visualizar detecções por tile",
        "tile_select": "Escolha um tile para inspecionar:",
        "tile_no_tiles": "Nenhum tile disponível.",
        "tile_orig_caption": "**Original**",
        "tile_det_caption": "**Com detecções**",
        "tile_no_det": "Sem detecções neste tile",
        "lang_selector_label": "🌐 Language / Idioma",
    },
}

# ==================== CONFIGURAÇÃO DA PÁGINA ====================

st.set_page_config(
    page_title="Netflora – Detecção de Plantas com IA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS: oculta a sidebar, ajusta o layout e força tema claro
st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.main .block-container {
    max-width: 900px;
    margin: 0 auto;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1 { color: #1a4d1a; }
h2 { color: #2d6a2d; border-bottom: 2px solid #a8d5a2; padding-bottom: 0.3rem; }
h3 { color: #2d6a2d; }
.stButton > button[kind="primary"] {
    background-color: #2d6a2d;
    color: white;
    border-radius: 8px;
    font-size: 1.1rem;
    padding: 0.6rem 2rem;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1a4d1a;
}
.step-box {
    background: #f0f7f0;
    border-left: 4px solid #2d6a2d;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.5rem;
    color: #1a1a1a;
}
.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: #f0f7f0;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.footer-section {
    text-align: center;
    font-size: 14px;
    line-height: 1.8;
    color: #444;
}
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .steps-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .stButton > button[kind="primary"] {
        font-size: 1rem;
        padding: 0.5rem 1rem;
    }
}
@media (max-width: 480px) {
    .steps-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================== CARREGAMENTO LAZY DO PIPELINE ====================

@st.cache_resource(show_spinner=False)
def get_pipeline():
    from pipeline import (
        DetectionConfig,
        build_detection_polygons_wgs84,
        build_detection_table,
        build_map,
        build_map_html,
        render_map_html,
        draw_tile_detections,
        ensure_netflora_repo,
        ensure_weights_file,
        export_results_zip,
        generate_tiles,
        get_available_algorithms,
        get_class_name_map,
        run_netflora_detect,
    )
    return {
        "DetectionConfig": DetectionConfig,
        "build_detection_polygons_wgs84": build_detection_polygons_wgs84,
        "build_detection_table": build_detection_table,
        "build_map": build_map,
        "build_map_html": build_map_html,
        "render_map_html": render_map_html,
        "draw_tile_detections": draw_tile_detections,
        "ensure_netflora_repo": ensure_netflora_repo,
        "ensure_weights_file": ensure_weights_file,
        "export_results_zip": export_results_zip,
        "generate_tiles": generate_tiles,
        "get_available_algorithms": get_available_algorithms,
        "get_class_name_map": get_class_name_map,
        "run_netflora_detect": run_netflora_detect,
    }


@st.cache_data(show_spinner=False)
def cached_algorithms(groups_json_str: str) -> list:
    from pipeline import get_available_algorithms
    return get_available_algorithms(Path(groups_json_str))


# ==================== SESSION STATE ====================

if "last_run" not in st.session_state:
    st.session_state.last_run = None
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "scroll_to_results" not in st.session_state:
    st.session_state.scroll_to_results = False

# ==================== LANGUAGE SELECTOR ====================

_, _lang_col = st.columns([4, 2])
with _lang_col:
    _lang_choice = st.selectbox(
        TRANSLATIONS["en"]["lang_selector_label"],
        options=["🌎 English", "🇧🇷 Português"],
        index=0 if st.session_state.lang == "en" else 1,
        label_visibility="collapsed",
    )
    _lang_new = "en" if _lang_choice == "🌎 English" else "pt"
    if _lang_new != st.session_state.lang:
        st.session_state.lang = _lang_new
        st.rerun()

t = TRANSLATIONS[st.session_state.lang]

# ==================== TÍTULO ====================

st.markdown(f"# {t['title']}")
st.markdown(t["intro"])

st.divider()

# ==================== HOW IT WORKS ====================

st.markdown(t["how_it_works"])

st.markdown(
    f'<div class="steps-grid">'
    f'<div class="step-box">{t["step1"]}</div>'
    f'<div class="step-box">{t["step2"]}</div>'
    f'<div class="step-box">{t["step3"]}</div>'
    f'<div class="step-box">{t["step4"]}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(t["how_desc"])

st.divider()

# ==================== INPUT ORTHOPHOTO ====================

st.markdown(t["ortho_section"])

ortho_option = st.radio(
    t["ortho_radio"],
    options=[t["ortho_opt_example"], t["ortho_opt_upload"]],
    horizontal=True,
    help=t["ortho_radio_help"],
)

ortho_path = None
uploaded_ortho_path = None

if ortho_option == t["ortho_opt_example"]:
    if DEFAULT_ORTHO.exists():
        st.success(t["ortho_using"])
        ortho_path = DEFAULT_ORTHO
    else:
        st.error(t["ortho_not_found"])
else:
    uploaded = st.file_uploader(
        t["ortho_upload_label"],
        type=["tif", "tiff"],
        help=t["ortho_upload_help"],
    )
    if uploaded is not None:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        session_inputs_dir = _get_session_root() / "inputs"
        safe_name = _sanitize_filename(uploaded.name)
        unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}_{safe_name}"
        uploaded_ortho_path = session_inputs_dir / unique_name
        uploaded_ortho_path.parent.mkdir(parents=True, exist_ok=True)
        with uploaded_ortho_path.open("wb") as f:
            f.write(uploaded.getbuffer())
        st.success(t["ortho_received"].format(name=uploaded.name))
        ortho_path = uploaded_ortho_path
    else:
        st.info(t["ortho_waiting"])

st.divider()

# ==================== DETECTION SETTINGS ====================

st.markdown(t["config_section"])

# Load available algorithms — uses local groups.json copy (always available)
try:
    algorithms = cached_algorithms(str(GROUPS_JSON))
except Exception:
    algorithms = ["Açaí", "Ambiental", "Castanheira", "Ecológico", "PFNMs", "PMFS", "Palmeiras"]

col_alg, col_conf = st.columns(2)

with col_alg:
    algorithm = st.selectbox(
        t["algo_label"],
        options=algorithms,
        index=algorithms.index("Palmeiras") if "Palmeiras" in algorithms else 0,
        help=t["algo_help"],
    )

with col_conf:
    conf_thres = st.slider(
        t["conf_label"],
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        format="%.2f",
        help=t["conf_help"],
    )

with st.expander(t["advanced_expander"]):
    col_a, col_b = st.columns(2)
    with col_a:
        tile_size = st.select_slider(
            t["tile_size_label"],
            options=[640, 768, 1024, 1280, 1536, 2048],
            value=1536,
            help=t["tile_size_help"],
        )
    with col_b:
        overlap = st.slider(
            t["overlap_label"],
            min_value=0,
            max_value=min(512, tile_size - 1),
            value=min(256, tile_size - 1),
            step=32,
            help=t["overlap_help"],
        )
    col_c, col_d = st.columns(2)
    with col_c:
        max_tiles = st.number_input(
            t["max_tiles_label"],
            min_value=1,
            max_value=2000,
            value=120,
            help=t["max_tiles_help"],
        )
    with col_d:
        device = st.selectbox(
            t["device_label"],
            options=["cpu", "0"],
            index=0,
            help=t["device_help"],
        )

    netflora_zip_url = st.text_input(
        t["netflora_url_label"],
        value=DEFAULT_NETFLORA_ZIP,
    )
    weights_url = st.text_input(
        t["weights_url_label"],
        value=DEFAULT_WEIGHTS_URL,
    )

st.divider()

# ==================== FIRST RUN WARNING ====================

if not DEFAULT_WEIGHTS.exists() or not NETFLORA_DIR.exists():
    st.info(t["first_run_info"], icon="⏬")

# ==================== RUN BUTTON ====================

run_disabled = ortho_path is None
run_col, _ = st.columns([2, 2])
with run_col:
    run_pipeline = st.button(
        t["run_button"],
        type="primary",
        disabled=run_disabled,
        width="stretch",
    )

if run_disabled and ortho_option == t["ortho_opt_upload"]:
    st.warning(t["run_warning"])

# ==================== PIPELINE EXECUTION ====================

if run_pipeline and ortho_path is not None:
    # Free memory from the previous run (map_html can be hundreds of MB due to
    # the base64-encoded ortho overlay embedded inline by folium.ImageOverlay).
    # Keeping the old session state alive while building the new one doubles/triples
    # peak RAM and causes a silent OOM kill → the generic "Oh no" crash page.
    st.session_state.last_run = None
    gc.collect()  # force-release previous run's large objects before pipeline starts

    fn = get_pipeline()
    WORKDIR.mkdir(parents=True, exist_ok=True)
    session_root = _get_session_root()
    run_name = _new_unique_run_name()
    run_workspace = session_root / "runs" / run_name
    run_tiles_dir = run_workspace / "tiles"
    run_coords_csv = run_workspace / "tile_coords.csv"
    run_project_dir = run_workspace / "detections"
    run_workspace.mkdir(parents=True, exist_ok=True)

    with st.status(t["pipeline_status"], expanded=True) as status:
        _progress = st.progress(0)

        st.write(t["step_prep_netflora"])
        try:
            netflora_root = fn["ensure_netflora_repo"](NETFLORA_DIR, netflora_zip_url)
        except Exception as e:
            status.update(label=t["step_prep_netflora_err"], state="error")
            st.error(str(e))
            st.stop()
        _progress.progress(1 / 6)

        # Use local groups.json (committed to git) as primary; fall back to downloaded copy
        groups_json = GROUPS_JSON if GROUPS_JSON.exists() else netflora_root / "json" / "groups.json"

        st.write(t["step_check_weights"])
        try:
            weights_file = fn["ensure_weights_file"](DEFAULT_WEIGHTS, weights_url)
        except Exception as e:
            status.update(label=t["step_check_weights_err"], state="error")
            st.error(str(e))
            st.stop()
        _progress.progress(2 / 6)

        st.write(t["step_gen_tiles"])
        try:
            tile_count = fn["generate_tiles"](
                ortho_path=ortho_path,
                output_dir=run_tiles_dir,
                coords_csv=run_coords_csv,
                tile_size=tile_size,
                overlap=overlap,
                max_tiles=int(max_tiles),
            )
        except Exception as e:
            status.update(label=t["step_gen_tiles_err"], state="error")
            st.error(str(e))
            st.stop()
        _progress.progress(3 / 6)

        if tile_count == 0:
            status.update(label=t["step_no_tiles_err"], state="error")
            st.error(t["step_no_tiles_msg"])
            st.stop()

        st.write(t["step_gen_tiles_done"].format(count=tile_count))

        st.write(t["step_run_detect"])
        try:
            with acquire_inference_lock(INFERENCE_LOCK_FILE):
                detect_result = fn["run_netflora_detect"](
                    config=fn["DetectionConfig"](
                        repo_root=netflora_root,
                        weights_path=weights_file,
                        source_dir=run_tiles_dir,
                        img_size=tile_size,
                        conf_thres=conf_thres,
                        device=device,
                        project_dir=run_project_dir,
                        run_name=run_name,
                    )
                )
        except TimeoutError as e:
            status.update(label=t["step_detect_err"], state="error")
            st.error(str(e))
            st.stop()
        _progress.progress(4 / 6)

        if detect_result.returncode != 0:
            status.update(label=t["step_detect_err"], state="error")
            st.error(t["step_detect_err_msg"])
            with st.expander(t["step_error_log"]):
                st.code(detect_result.stderr or detect_result.stdout)
            st.stop()

        labels_dir = run_project_dir / run_name / "labels"

        st.write(t["step_process_results"])
        class_name_map = fn["get_class_name_map"](groups_json, algorithm)
        results_df = fn["build_detection_table"](
            labels_dir=labels_dir,
            coords_csv=run_coords_csv,
            class_name_map=class_name_map,
        )
        polygons_df = fn["build_detection_polygons_wgs84"](results_df, run_coords_csv)
        _progress.progress(5 / 6)

        st.write(t["step_gen_map"])
        # Build the folium Map object for display, then render HTML to disk for export.
        # Storing map_obj (not the rendered HTML string) in session state keeps memory low.
        map_obj = fn["build_map"](polygons_df, ortho_path)
        map_html_path = run_workspace / "map.html"
        map_html_path.write_text(fn["render_map_html"](map_obj), encoding="utf-8")
        _progress.progress(6 / 6)

        st.session_state.last_run = {
            "run_name": run_name,
            "tile_count": tile_count,
            "results_df": results_df,
            "polygons_df": polygons_df,
            "tile_dir": run_tiles_dir,
            "tile_files": sorted(run_tiles_dir.glob("*.jpg")),
            "labels_dir": labels_dir,
            "class_name_map": class_name_map,
            "ortho_path": ortho_path,
            "algorithm": algorithm,
            "map_obj": map_obj,
            "map_html_path": map_html_path,
        }

        status.update(label=t["pipeline_done"], state="complete")
    st.session_state.scroll_to_results = True
    st.rerun()

# ==================== RESULTS DISPLAY ====================

if st.session_state.last_run is not None:
    run_data = st.session_state.last_run
    fn = get_pipeline()

    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
    if st.session_state.get("scroll_to_results"):
        st.html(
            '<script>document.getElementById("results-anchor")'
            '.scrollIntoView({behavior:"smooth"});</script>'
        )
        st.session_state.scroll_to_results = False

    st.divider()
    st.markdown(t["results_section"])

    n_detections = len(run_data["results_df"])
    n_classes = run_data["results_df"]["class_name"].nunique() if not run_data["results_df"].empty else 0

    m1, m2, m3 = st.columns(3)
    m1.metric(t["metric_tiles"], run_data["tile_count"])
    m2.metric(t["metric_plants"], n_detections)
    m3.metric(t["metric_species"], n_classes)

    st.divider()

    st.markdown(t["map_section"])
    st.caption(t["map_caption"])

    _map_obj = run_data.get("map_obj")
    if _map_obj is not None:
        from streamlit_folium import st_folium
        st_folium(_map_obj, height=500, returned_objects=[])
    else:
        st.info(t["map_unavailable"])

    st.divider()

    st.markdown(t["table_section"])

    results_df = run_data["results_df"]
    if results_df.empty:
        st.info(t["table_empty"])
    else:
        display_df = results_df[["filename", "class_name", "confidence"]].copy()
        display_df.columns = [t["col_tile"], t["col_species"], t["col_confidence"]]
        if "confidence" in results_df.columns:
            display_df[t["col_confidence"]] = display_df[t["col_confidence"]].map(
                lambda x: f"{x:.2%}" if x is not None else "—"
            )
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.divider()

    st.markdown(t["export_section"])
    st.caption(t["export_caption"])

    export_col, clear_col = st.columns([1, 1])

    with export_col:
        try:
            _exp_map_path = run_data.get("map_html_path")
            _exp_map_html = Path(_exp_map_path).read_text(encoding="utf-8") if _exp_map_path and Path(_exp_map_path).exists() else ""
            zip_bytes = fn["export_results_zip"](run_data, _exp_map_html)
            st.download_button(
                label=t["export_button"],
                data=zip_bytes,
                file_name=f"netflora_{run_data['run_name']}.zip",
                mime="application/zip",
                width="stretch",
            )
        except Exception as e:
            st.error(t["export_error"].format(error=e))

    with clear_col:
        if st.button(t["clear_button"], width="stretch"):
            st.session_state.last_run = None
            st.rerun()

    st.divider()
    with st.expander(t["tile_expander"]):
        tile_files = run_data.get("tile_files", [])
        tile_dir = run_data.get("tile_dir", TILES_DIR)
        labels_dir = run_data.get("labels_dir")
        class_name_map = run_data.get("class_name_map", {})

        if not tile_files:
            st.info(t["tile_no_tiles"])
        else:
            selected_tile = st.selectbox(
                t["tile_select"],
                [p.name for p in tile_files],
                key=f"tile_sel_{run_data['run_name']}",
            )
            from PIL import Image
            tile_path = Path(tile_dir) / selected_tile
            label_path = labels_dir / f"{tile_path.stem}.txt" if labels_dir else None

            if label_path and label_path.exists():
                annotated = fn["draw_tile_detections"](tile_path, label_path, class_name_map)
                col_orig, col_det = st.columns(2)
                with col_orig:
                    st.caption(t["tile_orig_caption"])
                    st.image(Image.open(tile_path), width="stretch")
                with col_det:
                    st.caption(t["tile_det_caption"])
                    st.image(annotated, width="stretch")
            else:
                st.image(Image.open(tile_path), caption=t["tile_no_det"], width="stretch")


# ==================== FOOTER ====================

st.divider()
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; line-height: 2;">
        <strong>👤 Autor do Site:</strong> Matheus Bissoli<br>
        🌐 <a href="https://matheusflb.github.io/" target="_blank">Site pessoal</a> &nbsp;|&nbsp;
        💼 <a href="https://www.linkedin.com/in/matheusbissoli/" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
        💻 <a href="https://github.com/MatheusFLB/" target="_blank">GitHub</a> &nbsp;|&nbsp;
        🧑‍💻 <a href="https://github.com/MatheusFLB/netflora-online/" target="_blank">Código-fonte</a>
    </div>
    """,
    unsafe_allow_html=True,
)