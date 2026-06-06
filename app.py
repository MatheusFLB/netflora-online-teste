from __future__ import annotations

import datetime as dt
import gc
import os
import time
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import streamlit as st

import pipeline

ROOT = Path(__file__).parent
WORKDIR = ROOT / "workdir"
NETFLORA_DIR = WORKDIR / "netflora_src"
WEIGHTS_DIR = WORKDIR / "weights"
GROUPS_JSON = ROOT / "json" / "groups.json"
ORTO_DIR = ROOT / "ortofoto"
RUNS_DIR = WORKDIR / "sessions"
LOCKS_DIR = WORKDIR / "locks"
INFERENCE_LOCK_FILE = LOCKS_DIR / "inference.lock"
BOOTSTRAP_LOCK_FILE = LOCKS_DIR / "bootstrap.lock"
NETFLORA_ZIP_URL = "https://github.com/NetFlora/Netflora/archive/refs/heads/main.zip"
DEFAULT_ALGORITHM = "Palmeiras"
DEFAULT_WEIGHTS_URL = "https://github.com/NetFlora/Netflora/releases/download/Assets/PALMEIRAS_Embrapa00.pt"
CLI_SESSION_ID = f"cli_{uuid.uuid4().hex[:8]}"

ALGORITHM_WEIGHTS_URLS = {
    "acai": "https://github.com/NetFlora/Netflora/releases/download/Assets/ACAI_Embrapa00.pt",
    "palmeiras": "https://github.com/NetFlora/Netflora/releases/download/Assets/PALMEIRAS_Embrapa00.pt",
    "pmfs": "https://github.com/NetFlora/Netflora/releases/download/Assets/PMFS_Embrapa00.pt",
    "pfnms": "https://github.com/NetFlora/Netflora/releases/download/Assets/NM_Embrapa00.pt",
    "castanheira": None,
    "ecologico": None,
    "ambiental": None,
}


@st.cache_data
def load_algorithms() -> List[str]:
    return pipeline.get_available_algorithms(GROUPS_JSON)


def list_local_orthophotos() -> List[Path]:
    if not ORTO_DIR.exists():
        return []

    tif_files = [
        p
        for p in ORTO_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    return sorted(
        tif_files,
        key=lambda p: str(p.relative_to(ORTO_DIR)).replace("\\", "/").casefold(),
    )


def get_local_ortho_label(path: Path) -> str:
    return str(path.relative_to(ORTO_DIR)).replace("\\", "/")


def sanitize_filename(filename: str) -> str:
    safe = "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in ("-", "_", "."))
    return safe.strip("._") or "upload.tif"


def get_session_id() -> str:
    try:
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = uuid.uuid4().hex
        return str(st.session_state["session_id"])
    except Exception:
        return CLI_SESSION_ID


def get_session_root() -> Path:
    session_root = RUNS_DIR / get_session_id()
    session_root.mkdir(parents=True, exist_ok=True)
    return session_root


@contextmanager
def acquire_execution_lock(
    lock_path: Path,
    wait_timeout: int = 1800,
    stale_seconds: int = 7200,
    poll_seconds: float = 1.0,
    waiting_message: str = (
        "Outro usuario esta processando agora. Sua execucao entrou na fila e "
        "iniciara automaticamente."
    ),
    timeout_message: str = (
        "Tempo de espera excedido na fila de processamento. "
        "Tente novamente em instantes."
    ),
) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd: Optional[int] = None
    deadline = time.time() + wait_timeout
    waiting_note_shown = False

    while True:
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            owner = f"session={get_session_id()} pid={os.getpid()} ts={int(time.time())}"
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
                raise TimeoutError(timeout_message)

            if not waiting_note_shown:
                try:
                    st.info(waiting_message)
                except Exception:
                    pass
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


def get_default_algorithm_index(algorithms: List[str]) -> int:
    if not algorithms:
        return 0

    for index, name in enumerate(algorithms):
        if name.casefold() == DEFAULT_ALGORITHM.casefold():
            return index
    return 0


def normalize_algorithm_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.strip().casefold()


def get_default_weights_url(algorithm: str) -> Optional[str]:
    key = normalize_algorithm_name(algorithm)
    return ALGORITHM_WEIGHTS_URLS.get(key, DEFAULT_WEIGHTS_URL)


def get_weights_target_path(algorithm: str) -> Path:
    key = normalize_algorithm_name(algorithm)
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in key).strip("_") or "default"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_DIR / f"{safe_key}_model_weights.pt"


def save_upload(upload: st.runtime.uploaded_file_manager.UploadedFile, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(upload.getbuffer())
    return target


def build_run_id() -> str:
    ts = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def resolve_orthophoto(
    source_mode: str,
    local_files: List[Path],
    local_choice: Optional[str],
    upload: Optional[st.runtime.uploaded_file_manager.UploadedFile],
    ortho_url: str,
) -> Path:
    if source_mode == "local":
        if not local_files:
            raise FileNotFoundError(
                "Nenhuma ortofoto local foi encontrada em ortofoto/. "
                "No local orthophoto was found in ortofoto/."
            )

        local_lookup = {get_local_ortho_label(file_path): file_path for file_path in local_files}
        if local_choice:
            selected_path = local_lookup.get(local_choice)
            if selected_path is not None:
                return selected_path
            raise FileNotFoundError(
                "A ortofoto local selecionada nao foi encontrada. "
                "Selected local orthophoto was not found."
            )

        return local_files[0]

    if source_mode == "upload":
        if upload is None:
            raise ValueError(
                "Envie um arquivo .tif/.tiff. "
                "Please upload a .tif/.tiff file."
            )

        safe_name = sanitize_filename(upload.name)
        unique_name = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}_{safe_name}"
        target_path = get_session_root() / "inputs" / unique_name
        return save_upload(upload, target_path)

    if not ortho_url.strip():
        raise ValueError(
            "Informe uma URL valida para ortofoto. "
            "Please provide a valid orthophoto URL."
        )

    target_path = get_session_root() / "inputs" / "ortofoto_download.tif"
    return pipeline.ensure_ortho_file(None, ortho_url.strip(), target_path)


def run_detection(
    algorithm: str,
    ortho_path: Path,
    conf_thres: float,
    tile_size: int,
    overlap: int,
    max_tiles: int,
    img_size: int,
    weights_url: str,
    step_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def step(message: str) -> None:
        if step_callback is not None:
            step_callback(message)

    step("1/6 Preparando repositorio Netflora... / Preparing Netflora repository...")
    with acquire_execution_lock(
        BOOTSTRAP_LOCK_FILE,
        waiting_message=(
            "Outro usuario esta preparando o ambiente inicial. Sua execucao entrou na fila. "
            "Another user is preparing the environment. Your run was queued."
        ),
        timeout_message=(
            "Tempo de espera excedido para preparar ambiente. "
            "Preparation queue timeout exceeded."
        ),
    ):
        pipeline.ensure_netflora_repo(NETFLORA_DIR, NETFLORA_ZIP_URL)

    weights_path_target = get_weights_target_path(algorithm)
    default_weights_url = get_default_weights_url(algorithm)
    resolved_weights_url = (weights_url or "").strip() or (default_weights_url or "")
    if not resolved_weights_url and not weights_path_target.exists():
        raise FileNotFoundError(
            "Nao existe URL padrao de pesos para este algoritmo e nenhum arquivo local foi encontrado. "
            "There is no default weights URL for this algorithm and no local weights file was found."
        )

    step("2/6 Verificando pesos do modelo... / Checking model weights...")
    with acquire_execution_lock(
        BOOTSTRAP_LOCK_FILE,
        waiting_message=(
            "Outro usuario esta preparando pesos do modelo. Sua execucao entrou na fila. "
            "Another user is preparing model weights. Your run was queued."
        ),
        timeout_message=(
            "Tempo de espera excedido para preparar pesos. "
            "Weights preparation queue timeout exceeded."
        ),
    ):
        weights_path = pipeline.ensure_weights_file(
            weights_path_target,
            resolved_weights_url or None,
        )

    run_id = build_run_id()
    run_dir = get_session_root() / "runs" / run_id
    tiles_dir = run_dir / "tiles"
    coords_csv = run_dir / "tile_coords.csv"
    project_dir = run_dir / "detections"

    with acquire_execution_lock(
        INFERENCE_LOCK_FILE,
        waiting_message=(
            "Outro usuario esta processando uma ortofoto. Sua execucao entrou na fila e iniciara automaticamente. "
            "Another user is processing an orthophoto. Your run was queued and will start automatically."
        ),
        timeout_message=(
            "Tempo de espera excedido na fila de processamento. "
            "Processing queue timeout exceeded."
        ),
    ):
        step("3/6 Gerando tiles da ortofoto... / Generating orthophoto tiles...")
        tile_count = pipeline.generate_tiles(
            ortho_path=ortho_path,
            output_dir=tiles_dir,
            coords_csv=coords_csv,
            tile_size=tile_size,
            overlap=overlap,
            max_tiles=max_tiles,
        )

        if tile_count == 0:
            raise RuntimeError("Nenhum tile valido foi gerado. Verifique a ortofoto.")

        step("4/6 Executando deteccao... / Running detection...")
        config = pipeline.DetectionConfig(
            repo_root=NETFLORA_DIR,
            weights_path=weights_path,
            source_dir=tiles_dir,
            img_size=img_size,
            conf_thres=conf_thres,
            device="cpu",
            project_dir=project_dir,
            run_name=run_id,
        )

        result = pipeline.run_netflora_detect(config)

        step("5/6 Processando resultados... / Processing results...")
        labels_dir = project_dir / run_id / "labels"
        class_map = pipeline.get_class_name_map(GROUPS_JSON, algorithm)
        results_df = pipeline.build_detection_table(labels_dir, coords_csv, class_map)
        polygons_df = pipeline.build_detection_polygons_wgs84(results_df, coords_csv)

        step("6/6 Gerando mapa e exportacao... / Building map and exports...")
        map_obj = pipeline.build_map(polygons_df, ortho_path)
        map_html = pipeline.render_map_html(map_obj)
        zip_bytes = pipeline.export_results_zip(
            {"results_df": results_df, "polygons_df": polygons_df},
            map_html,
        )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "ortho_path": str(ortho_path),
        "tiles_dir": str(tiles_dir),
        "labels_dir": str(labels_dir),
        "class_map": class_map,
        "weights_path": str(weights_path),
        "weights_url_used": resolved_weights_url,
        "tile_count": tile_count,
        "results_df": results_df,
        "polygons_df": polygons_df,
        "map_html": map_html,
        "zip_bytes": zip_bytes,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def main() -> None:
    st.set_page_config(page_title="Netflora Online - Demo", page_icon="🌿", layout="centered")

    if "show_tiles" not in st.session_state:
        st.session_state["show_tiles"] = False
    if "run_data" not in st.session_state:
        st.session_state["run_data"] = None

    st.markdown(
        """
        <style>
        .app-shell {
            max-width: 920px;
            margin: 0 auto;
        }
        .app-hero {
            text-align: center;
            padding: 0.5rem 0 1rem;
        }
        .app-hero h1 {
            margin-bottom: 0;
        }
        .stButton > button[kind="primary"] {
            background-color: #c62828;
            border: 1px solid #c62828;
            color: #ffffff;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #a61f1f;
            border-color: #a61f1f;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="app-hero">
          <h1>Netflora Online</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not GROUPS_JSON.exists():
        st.error(
            "Arquivo de grupos nao encontrado. Verifique a pasta json/. "
            "Groups file not found. Please check the json/ folder."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    algorithms = load_algorithms()
    local_orthos = list_local_orthophotos()
    local_ortho_names = [get_local_ortho_label(path) for path in local_orthos]

    if not algorithms:
        st.error(
            "Nenhum algoritmo foi encontrado no groups.json. "
            "No algorithms were found in groups.json."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    source_labels = {
        "local": "Arquivo local do projeto / Project local file",
        "upload": "Upload de arquivo / File upload",
        "url": "URL da ortofoto / Orthophoto URL",
    }

    algorithm = st.selectbox(
        "Algoritmo / Algorithm",
        options=algorithms,
        index=get_default_algorithm_index(algorithms),
    )

    source_mode = st.radio(
        "Fonte da ortofoto / Orthophoto source",
        options=["local", "upload", "url"],
        format_func=lambda value: source_labels[value],
        index=0,
    )

    local_choice: Optional[str] = None
    ortho_upload: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None
    ortho_url = ""

    if source_mode == "local":
        if local_ortho_names:
            local_choice = st.selectbox(
                "Ortofoto local / Local orthophoto",
                options=local_ortho_names,
                index=0,
            )
            st.caption(
                "Padrao: ortofoto local do projeto. "
                "Default: project local orthophoto."
            )
        else:
            st.warning(
                "Nenhuma ortofoto local encontrada em ortofoto/. "
                "No local orthophoto found in ortofoto/."
            )
    elif source_mode == "upload":
        ortho_upload = st.file_uploader(
            "Ortofoto (.tif/.tiff) / Orthophoto (.tif/.tiff)",
            type=["tif", "tiff"],
        )
    else:
        ortho_url = st.text_input(
            "URL da ortofoto / Orthophoto URL",
            value="",
        )

    with st.expander("Ajustes avancados / Advanced settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            tile_size = st.number_input("Tamanho do tile (px) / Tile size (px)", 512, 4096, 1536, step=128)
            overlap = st.number_input("Sobreposicao (px) / Overlap (px)", 0, 1024, 256, step=64)
            max_tiles = st.number_input("Limite de tiles / Max tiles", 10, 500, 120, step=10)
        with col2:
            img_size = st.number_input("Tamanho de inferencia (px) / Inference size (px)", 320, 2048, 1536, step=64)
            conf_thres = st.slider("Confianca minima / Min confidence", 0.05, 0.70, 0.25, 0.01)

            default_weights_url = get_default_weights_url(algorithm) or ""
            weights_url = st.text_input(
                "URL opcional dos pesos / Optional weights URL",
                value=default_weights_url,
                key=f"weights_url_{normalize_algorithm_name(algorithm)}",
                help=(
                    "Se vazio, usa automaticamente a URL padrao do algoritmo. "
                    "If empty, the algorithm default weights URL is used automatically when available."
                ),
            )

            if not default_weights_url:
                st.info(
                    "Este algoritmo nao possui URL padrao de pesos. "
                    "Informe manualmente uma URL de .pt ou deixe um arquivo local pronto em workdir/weights/."
                )

    btn_left, btn_center, btn_right = st.columns([1, 2, 1])
    with btn_center:
        submitted = st.button(
            "Executar deteccao / Run detection",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if overlap >= tile_size:
            st.error(
                "A sobreposicao precisa ser menor que o tamanho do tile. "
                "Overlap must be smaller than tile size."
            )
        else:
            try:
                st.session_state["run_data"] = None
                gc.collect()

                ortho_path = resolve_orthophoto(
                    source_mode=source_mode,
                    local_files=local_orthos,
                    local_choice=local_choice,
                    upload=ortho_upload,
                    ortho_url=ortho_url,
                )

                with st.status(
                    "Executando pipeline de deteccao... / Running detection pipeline...",
                    expanded=True,
                ) as status:
                    run_data = run_detection(
                        algorithm=algorithm,
                        ortho_path=ortho_path,
                        conf_thres=conf_thres,
                        tile_size=tile_size,
                        overlap=overlap,
                        max_tiles=max_tiles,
                        img_size=img_size,
                        weights_url=weights_url,
                        step_callback=st.write,
                    )
                    status.update(
                        label=(
                            "Pipeline concluido com avisos. / Pipeline completed with warnings."
                            if run_data.get("returncode", 0) != 0
                            else "Pipeline concluido com sucesso. / Pipeline completed successfully."
                        ),
                        state="complete",
                    )

                st.session_state["run_data"] = run_data
                st.session_state["show_tiles"] = False
            except Exception as exc:
                st.error(
                    "Falha ao executar a deteccao. Verifique os dados e tente novamente. "
                    "Detection failed. Please check the data and try again."
                )
                with st.expander("Detalhes do erro / Error details"):
                    st.write(str(exc))

    run_data = st.session_state.get("run_data")
    if run_data:
        st.divider()
        st.subheader("Resultados / Results")

        # 1) Mapa
        st.components.v1.html(run_data["map_html"], height=520, scrolling=False)

        # 2) Botoes de download
        results_df = run_data["results_df"]
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        map_html_bytes = run_data["map_html"].encode("utf-8")

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                label="Baixar mapa HTML / Download map HTML",
                data=map_html_bytes,
                file_name=f"mapa_{run_data['run_id']}.html",
                mime="text/html",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="Baixar CSV / Download CSV",
                data=csv_bytes,
                file_name=f"deteccoes_{run_data['run_id']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col3:
            st.download_button(
                label="Baixar pacote ZIP / Download ZIP",
                data=run_data["zip_bytes"],
                file_name=f"netflora_resultados_{run_data['run_id']}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        # 3) Planilha de resultados
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info(
                "Nenhuma deteccao encontrada para os parametros atuais. "
                "No detections found for current parameters."
            )

        # 4) Botao para exibir tiles detectados
        toggle_label = (
            "Ocultar tiles detectados / Hide detected tiles"
            if st.session_state.get("show_tiles", False)
            else "Exibir tiles detectados / Show detected tiles"
        )
        if st.button(toggle_label):
            st.session_state["show_tiles"] = not st.session_state.get("show_tiles", False)

        if st.session_state.get("show_tiles", False):
            tiles_dir = Path(run_data["tiles_dir"])
            labels_dir = Path(run_data["labels_dir"])
            class_map = run_data["class_map"]
            tile_files = sorted(tiles_dir.glob("*.jpg"))

            if not tile_files:
                st.info("Nenhum tile disponivel. / No tiles available.")
            else:
                tile_names = [path.name for path in tile_files]
                selected_tile_name = st.selectbox(
                    "Escolha um tile para inspecionar / Choose a tile to inspect",
                    options=tile_names,
                    key="tile_select_name",
                )

                tile_path = tiles_dir / selected_tile_name
                label_path = labels_dir / f"{tile_path.stem}.txt"
                detected_image = pipeline.draw_tile_detections(tile_path, label_path, class_map)

                col_original, col_detected = st.columns(2)
                with col_original:
                    st.image(str(tile_path), caption="Original / Original")
                with col_detected:
                    st.image(detected_image, caption="Com deteccoes / With detections")

        if run_data["returncode"] != 0:
            st.warning(
                "A deteccao retornou avisos. Veja os logs abaixo se precisar de detalhes. "
                "Detection returned warnings. Check logs below if needed."
            )
            with st.expander("Logs do processamento / Processing logs"):
                if run_data["stderr"]:
                    st.code(run_data["stderr"])
                if run_data["stdout"]:
                    st.code(run_data["stdout"])

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
