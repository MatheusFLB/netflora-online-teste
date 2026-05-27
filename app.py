"""
app.py – Netflora Online
Dashboard Streamlit para detecção de espécies vegetais em ortofotos de drone.
Desenvolvido com base no projeto Netflora da Embrapa.

Layout sem barra lateral, simples e didático.
"""

from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit import components

# ==================== CONSTANTES ====================

APP_ROOT = Path(__file__).resolve().parent
WORKDIR = APP_ROOT / "workdir"
NETFLORA_DIR = WORKDIR / "netflora_src"
TILES_DIR = WORKDIR / "tiles"
COORDS_CSV = WORKDIR / "tile_coords.csv"
RUNS_DIR = WORKDIR / "runs"
DEFAULT_ORTHO = APP_ROOT / "ortofoto" / "ortofoto_exemplo1_corte.tif"
DEFAULT_WEIGHTS = WORKDIR / "model_weights.pt"
DEFAULT_NETFLORA_ZIP = "https://github.com/NetFlora/Netflora/archive/refs/heads/main.zip"
DEFAULT_WEIGHTS_URL = "https://github.com/NetFlora/Netflora/releases/download/Assets/PMFS_Embrapa00.pt"
# groups.json local (committed to git) — always available without downloading netflora_src
GROUPS_JSON = APP_ROOT / "json" / "groups.json"

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
.metric-card {
    background: #f0f7f0;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
footer-section {
    text-align: center;
    font-size: 14px;
    line-height: 1.8;
    color: #444;
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
        build_map_html,
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
        "build_map_html": build_map_html,
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

# ==================== TÍTULO ====================

st.markdown("# 🌿 Netflora – Detecção de Espécies Vegetais com IA")
st.markdown(
    """
    **Projeto desenvolvido pela Embrapa** para o inventário florestal automatizado com uso de drones e inteligência artificial.
    A ferramenta analisa ortofotos aéreas e identifica automaticamente a localização de espécies vegetais de interesse,
    como Açaí, Castanheira, Palmeiras e outras espécies da floresta amazônica.
    """
)

st.divider()

# ==================== COMO FUNCIONA ====================

st.markdown("## 📖 Como funciona?")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="step-box">📷 <strong>1. Ortofoto</strong><br>Imagem aérea georreferenciada (.tif) capturada por drone.</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="step-box">✂️ <strong>2. Tiles</strong><br>A imagem é dividida em recortes para facilitar a análise pelo modelo.</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="step-box">🤖 <strong>3. Detecção</strong><br>Um modelo de deep learning identifica as espécies em cada recorte.</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="step-box">🗺️ <strong>4. Resultado</strong><br>As detecções são georreferenciadas e exibidas em um mapa interativo.</div>', unsafe_allow_html=True)

st.markdown(
    """
    Para viabilizar esta solução, a Embrapa estruturou o projeto com base em Python e no modelo de detecção de objetos YOLO (You Only Look Once), amplamente utilizado em visão computacional para identificação automática de elementos em imagens. A proposta original foi desenvolvida para execução em ambiente de nuvem, por meio do Google Colab/Notebook, permitindo o processamento de ortofotos aéreas com apoio de inteligência artificial sem exigir infraestrutura local robusta.\n
    A partir dessa base, o código foi adaptado para execução local, ampliando a flexibilidade de uso e possibilitando maior controle sobre os dados. Além disso, foi implementada essa página de acesso público, onde qualquer pessoa pode enviar uma ortofoto e receber o resultado da análise de forma prática e acessível.\n
    Após o processamento, a página disponibiliza os arquivos gerados para exportação, incluindo especialmente o Shapefile, formato amplamente utilizado em sistemas de informação geográfica (GIS). Isso permite que os resultados sejam integrados a ferramentas de análise topográfica, mapeamento e planejamento territorial, facilitando o aproveitamento técnico das informações obtidas.\n
    """
)

st.divider()

# ==================== ENTRADA DA ORTOFOTO ====================

st.markdown("## 📁 Ortofoto de entrada")

ortho_option = st.radio(
    "Selecione a fonte da ortofoto:",
    options=["Usar ortofoto de exemplo", "Enviar minha ortofoto (.tif)"],
    horizontal=True,
    help="Escolha entre usar a ortofoto de exemplo incluída no projeto ou enviar o seu próprio arquivo GeoTIFF.",
)

ortho_path = None
uploaded_ortho_path = None

if ortho_option == "Usar ortofoto de exemplo":
    if DEFAULT_ORTHO.exists():
        st.success(f"✅ Usando: `ortofoto_exemplo1_corte.tif`")
        ortho_path = DEFAULT_ORTHO
    else:
        st.error("Ortofoto de exemplo não encontrada em `ortofoto/ortofoto_exemplo1_corte.tif`.")
else:
    uploaded = st.file_uploader(
        "Envie sua ortofoto GeoTIFF:",
        type=["tif", "tiff"],
        help="O arquivo deve ser um GeoTIFF com sistema de referência geográfico definido (CRS).",
    )
    if uploaded is not None:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        uploaded_ortho_path = WORKDIR / "inputs" / uploaded.name
        uploaded_ortho_path.parent.mkdir(parents=True, exist_ok=True)
        with uploaded_ortho_path.open("wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Arquivo recebido: `{uploaded.name}`")
        ortho_path = uploaded_ortho_path
    else:
        st.info("Aguardando upload da ortofoto.")

st.divider()

# ==================== CONFIGURAÇÕES DE DETECÇÃO ====================

st.markdown("## ⚙️ Configurações de detecção")

# Carregar algoritmos disponíveis — usa cópia local do groups.json (sempre disponível)
try:
    algorithms = cached_algorithms(str(GROUPS_JSON))
except Exception:
    algorithms = ["Açaí", "Ambiental", "Castanheira", "Ecológico", "PFNMs", "PMFS", "Palmeiras"]

col_alg, col_conf = st.columns(2)

with col_alg:
    algorithm = st.selectbox(
        "🌱 Algoritmo de detecção:",
        options=algorithms,
        index=algorithms.index("Palmeiras") if "Palmeiras" in algorithms else 0,
        help="Escolha o conjunto de espécies que o modelo deve detectar.",
    )

with col_conf:
    conf_thres = st.slider(
        "🎯 Confiança mínima:",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        format="%.2f",
        help="Detecções com confiança abaixo desse valor serão ignoradas. Valores mais altos = menos detecções, mas mais precisas.",
    )

# Configurações avançadas
with st.expander("🔧 Configurações avançadas"):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        tile_size = st.select_slider(
            "Tamanho do tile (px):",
            options=[640, 768, 1024, 1280, 1536, 2048],
            value=1536,
            help="Resolução dos recortes enviados ao modelo.",
        )
    with col_b:
        overlap = st.slider(
            "Sobreposição (px):",
            min_value=0,
            max_value=min(512, tile_size - 1),
            value=min(256, tile_size - 1),
            step=32,
            help="Sobreposição entre tiles adjacentes para evitar perda de detecções nas bordas.",
        )
    with col_c:
        max_tiles = st.number_input(
            "Máximo de tiles:",
            min_value=1,
            max_value=2000,
            value=120,
            help="Limite de tiles para demonstração. Aumente para processar imagens maiores.",
        )
    with col_d:
        device = st.selectbox(
            "Dispositivo:",
            options=["cpu", "0"],
            index=0,
            help="'cpu' para processador; '0' para GPU NVIDIA (requer CUDA).",
        )

    netflora_zip_url = st.text_input(
        "URL do repositório Netflora (zip):",
        value=DEFAULT_NETFLORA_ZIP,
    )
    weights_url = st.text_input(
        "URL dos pesos do modelo (.pt):",
        value=DEFAULT_WEIGHTS_URL,
    )

st.divider()

# ==================== AVISO DE PRIMEIRA EXECUÇÃO ====================

if not DEFAULT_WEIGHTS.exists() or not NETFLORA_DIR.exists():
    st.info(
        "ℹ️ **Primeira execução:** o modelo de detecção (~135 MB) e o código do Netflora "
        "serão baixados automaticamente ao clicar em *Executar Detecção*. "
        "Isso pode levar alguns minutos dependendo da conexão.",
        icon="⏬",
    )

# ==================== BOTÃO DE EXECUÇÃO ====================

run_disabled = ortho_path is None
run_col, _ = st.columns([1, 3])
with run_col:
    run_pipeline = st.button(
        "🔍 Executar Detecção",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    )

if run_disabled and ortho_option == "Enviar minha ortofoto (.tif)":
    st.warning("⚠️ Envie uma ortofoto antes de executar.")

# ==================== EXECUÇÃO DO PIPELINE ====================

if run_pipeline and ortho_path is not None:
    fn = get_pipeline()
    WORKDIR.mkdir(parents=True, exist_ok=True)

    with st.status("🔄 Executando pipeline de detecção...", expanded=True) as status:

        st.write("**1/6** Preparando código do Netflora...")
        try:
            netflora_root = fn["ensure_netflora_repo"](NETFLORA_DIR, netflora_zip_url)
        except Exception as e:
            status.update(label="❌ Falha ao preparar Netflora.", state="error")
            st.error(str(e))
            st.stop()

        # Use local groups.json (committed to git) as primary; fall back to downloaded copy
        groups_json = GROUPS_JSON if GROUPS_JSON.exists() else netflora_root / "json" / "groups.json"

        st.write("**2/6** Verificando pesos do modelo...")
        try:
            weights_file = fn["ensure_weights_file"](DEFAULT_WEIGHTS, weights_url)
        except Exception as e:
            status.update(label="❌ Falha ao preparar pesos do modelo.", state="error")
            st.error(str(e))
            st.stop()

        st.write("**3/6** Gerando tiles da ortofoto...")
        try:
            tile_count = fn["generate_tiles"](
                ortho_path=ortho_path,
                output_dir=TILES_DIR,
                coords_csv=COORDS_CSV,
                tile_size=tile_size,
                overlap=overlap,
                max_tiles=int(max_tiles),
            )
        except Exception as e:
            status.update(label="❌ Falha ao gerar tiles.", state="error")
            st.error(str(e))
            st.stop()

        if tile_count == 0:
            status.update(label="❌ Nenhum tile válido foi gerado.", state="error")
            st.error("Nenhum tile válido foi gerado. Verifique se a ortofoto é válida.")
            st.stop()

        st.write(f"**3/6** {tile_count} tiles gerados.")

        run_name = f"online_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        st.write("**4/6** Executando modelo de detecção...")
        detect_result = fn["run_netflora_detect"](
            config=fn["DetectionConfig"](
                repo_root=netflora_root,
                weights_path=weights_file,
                source_dir=TILES_DIR,
                img_size=tile_size,
                conf_thres=conf_thres,
                device=device,
                project_dir=RUNS_DIR,
                run_name=run_name,
            )
        )

        if detect_result.returncode != 0:
            status.update(label="❌ Falha na detecção.", state="error")
            st.error("A execução do detect.py falhou.")
            with st.expander("Ver log de erro"):
                st.code(detect_result.stderr or detect_result.stdout)
            st.stop()

        labels_dir = RUNS_DIR / run_name / "labels"

        st.write("**5/6** Processando resultados...")
        class_name_map = fn["get_class_name_map"](groups_json, algorithm)
        results_df = fn["build_detection_table"](
            labels_dir=labels_dir,
            coords_csv=COORDS_CSV,
            class_name_map=class_name_map,
        )
        polygons_df = fn["build_detection_polygons_wgs84"](results_df, COORDS_CSV)

        st.write("**6/6** Gerando mapa interativo...")
        map_html = fn["build_map_html"](polygons_df, ortho_path)

        st.session_state.last_run = {
            "run_name": run_name,
            "tile_count": tile_count,
            "results_df": results_df,
            "polygons_df": polygons_df,
            "tile_files": sorted(TILES_DIR.glob("*.jpg")),
            "labels_dir": labels_dir,
            "class_name_map": class_name_map,
            "ortho_path": ortho_path,
            "algorithm": algorithm,
            "map_html": map_html,
        }

        status.update(label="✅ Pipeline concluído com sucesso!", state="complete")
    st.rerun()

# ==================== EXIBIÇÃO DE RESULTADOS ====================

if st.session_state.last_run is not None:
    run_data = st.session_state.last_run
    fn = get_pipeline()

    st.divider()
    st.markdown("## 📊 Resultados da detecção")

    # Métricas resumidas
    n_detections = len(run_data["results_df"])
    n_classes = run_data["results_df"]["class_name"].nunique() if not run_data["results_df"].empty else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("🔲 Tiles processados", run_data["tile_count"])
    m2.metric("🌱 Plantas detectadas", n_detections)
    m3.metric("🏷️ Espécies identificadas", n_classes)

    st.divider()

    # Mapa interativo
    st.markdown("### 🗺️ Mapa interativo")
    st.caption(
        "O mapa exibe as detecções georreferenciadas. "
        "Passe o mouse sobre os marcadores para ver a espécie e a confiança da detecção. "
        "Use os controles no canto superior direito para alternar as camadas do mapa."
    )

    if run_data.get("map_html"):
        responsive_map = f"""
        <script>
        (function() {{
            function updateHeight() {{
                var vw = window.innerWidth;
                var targetHeight = vw < 768 ? Math.max(Math.round(vw * 1.4), 480) : 800;
                window.parent.postMessage({{
                    isStreamlitMessage: true,
                    type: "streamlit:setFrameHeight",
                    height: targetHeight
                }}, "*");
            }}
            updateHeight();
            window.addEventListener("resize", updateHeight);
        }})();
        </script>
        <div style="
            width: 100%;
            height: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            overflow: hidden;
            margin-bottom: 1rem;
        ">
            {run_data['map_html']}
        </div>
        """
        components.v1.html(responsive_map, height=800, scrolling=False)
    else:
        st.info("Mapa não disponível para este resultado.")

    st.divider()

    # Tabela de resultados
    st.markdown("### 📋 Tabela de detecções")

    results_df = run_data["results_df"]
    if results_df.empty:
        st.info("Nenhuma detecção encontrada com os parâmetros atuais. Tente reduzir a confiança mínima.")
    else:
        # Tabela formatada
        display_df = results_df[["filename", "class_name", "confidence"]].copy()
        display_df.columns = ["Tile", "Espécie", "Confiança"]
        if "confidence" in results_df.columns:
            display_df["Confiança"] = display_df["Confiança"].map(
                lambda x: f"{x:.2%}" if x is not None else "—"
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # Exportação
    st.markdown("### 💾 Exportar resultados")
    st.caption("O arquivo ZIP contém: mapa interativo (.html), planilha de detecções (.csv) e shapefile georreferenciado (.shp e arquivos auxiliares).")

    export_col, clear_col = st.columns([1, 1])

    with export_col:
        try:
            zip_bytes = fn["export_results_zip"](run_data, run_data.get("map_html", ""))
            st.download_button(
                label="📦 Baixar ZIP de resultados",
                data=zip_bytes,
                file_name=f"netflora_{run_data['run_name']}.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Erro ao gerar exportação: {e}")

    with clear_col:
        if st.button("🗑️ Limpar resultados", use_container_width=True):
            st.session_state.last_run = None
            st.rerun()

    # Visualização por tile (expansível)
    st.divider()
    with st.expander("🔬 Visualizar detecções por tile"):
        tile_files = run_data.get("tile_files", [])
        labels_dir = run_data.get("labels_dir")
        class_name_map = run_data.get("class_name_map", {})

        if not tile_files:
            st.info("Nenhum tile disponível.")
        else:
            selected_tile = st.selectbox(
                "Escolha um tile para inspecionar:",
                [p.name for p in tile_files],
                key=f"tile_sel_{run_data['run_name']}",
            )
            from PIL import Image
            tile_path = TILES_DIR / selected_tile
            label_path = labels_dir / f"{tile_path.stem}.txt" if labels_dir else None

            if label_path and label_path.exists():
                annotated = fn["draw_tile_detections"](tile_path, label_path, class_name_map)
                col_orig, col_det = st.columns(2)
                with col_orig:
                    st.caption("**Original**")
                    st.image(Image.open(tile_path), use_container_width=True)
                with col_det:
                    st.caption("**Com detecções**")
                    st.image(annotated, use_container_width=True)
            else:
                st.image(Image.open(tile_path), caption="Sem detecções neste tile", use_container_width=True)


# ==================== RODAPÉ ====================

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