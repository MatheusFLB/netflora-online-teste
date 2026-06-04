from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List

import streamlit as st

import pipeline

ROOT = Path(__file__).parent
WORKDIR = ROOT / "workdir"
NETFLORA_DIR = WORKDIR / "netflora_src"
WEIGHTS_PATH = WORKDIR / "model_weights.pt"
GROUPS_JSON = ROOT / "json" / "groups.json"
ORTO_DIR = ROOT / "ortofoto"
RUNS_DIR = WORKDIR / "local_runs"
NETFLORA_ZIP_URL = "https://github.com/NetFlora/Netflora/archive/refs/heads/main.zip"


@st.cache_data
def load_algorithms() -> List[str]:
    return pipeline.get_available_algorithms(GROUPS_JSON)


def save_upload(upload: st.runtime.uploaded_file_manager.UploadedFile, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(upload.getbuffer())
    return target


def build_run_id() -> str:
    return dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def run_detection(
    algorithm: str,
    ortho_path: Path,
    conf_thres: float,
    tile_size: int,
    overlap: int,
    max_tiles: int,
    img_size: int,
    weights_url: str,
) -> Dict[str, object]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline.ensure_netflora_repo(NETFLORA_DIR, NETFLORA_ZIP_URL)
    weights_path = pipeline.ensure_weights_file(WEIGHTS_PATH, weights_url or None)

    run_id = build_run_id()
    run_dir = RUNS_DIR / run_id
    tiles_dir = run_dir / "tiles"
    coords_csv = run_dir / "tile_coords.csv"
    project_dir = run_dir / "detections"

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

    labels_dir = project_dir / run_id / "labels"
    class_map = pipeline.get_class_name_map(GROUPS_JSON, algorithm)
    results_df = pipeline.build_detection_table(labels_dir, coords_csv, class_map)
    polygons_df = pipeline.build_detection_polygons_wgs84(results_df, coords_csv)

    map_obj = pipeline.build_map(polygons_df, ortho_path)
    map_html = pipeline.render_map_html(map_obj)
    zip_bytes = pipeline.export_results_zip(
        {"results_df": results_df, "polygons_df": polygons_df},
        map_html,
    )

    preview_tiles = sorted(tiles_dir.glob("*.jpg"))[:4]

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "tile_count": tile_count,
        "results_df": results_df,
        "polygons_df": polygons_df,
        "map_html": map_html,
        "zip_bytes": zip_bytes,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "preview_tiles": preview_tiles,
    }


def main() -> None:
    st.set_page_config(page_title="Netflora Online - Demo", page_icon="🌿", layout="centered")

    st.markdown(
        """
        <style>
        .app-shell {
            max-width: 920px;
            margin: 0 auto;
        }
        .app-hero {
            text-align: center;
            padding: 0.5rem 0 1.5rem;
        }
        .app-hero h1 {
            margin-bottom: 0.25rem;
        }
        .app-note {
            color: #526062;
            font-size: 0.95rem;
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
          <p class="app-note">
            Demo simplificada para executar deteccoes em ortofotos.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not GROUPS_JSON.exists():
        st.error("Arquivo de grupos nao encontrado. Verifique a pasta json/.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    algorithms = load_algorithms()

    with st.form("run_form"):
        algorithm = st.selectbox("Algoritmo", options=algorithms)
        ortho_upload = st.file_uploader("Ortofoto (.tif/.tiff)", type=["tif", "tiff"])
        ortho_url = st.text_input("Ou URL da ortofoto (opcional)")

        with st.expander("Ajustes avancados", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                tile_size = st.number_input("Tamanho do tile (px)", 512, 4096, 1536, step=128)
                overlap = st.number_input("Sobreposicao (px)", 0, 1024, 256, step=64)
                max_tiles = st.number_input("Limite de tiles", 10, 500, 120, step=10)
            with col2:
                img_size = st.number_input("Tamanho de inferencia (px)", 320, 2048, 1536, step=64)
                conf_thres = st.slider("Confianca minima", 0.05, 0.70, 0.25, 0.01)
                weights_url = st.text_input("URL opcional dos pesos", value="")

        submitted = st.form_submit_button("Executar deteccao")

    if submitted:
        if overlap >= tile_size:
            st.error("A sobreposicao precisa ser menor que o tamanho do tile.")
        elif not ortho_upload and not ortho_url:
            st.error("Envie uma ortofoto ou informe uma URL valida.")
        else:
            try:
                if ortho_upload:
                    ortho_path = save_upload(ortho_upload, ORTO_DIR / "ortofoto_upload.tif")
                else:
                    ortho_path = pipeline.ensure_ortho_file(None, ortho_url, ORTO_DIR / "ortofoto_download.tif")

                with st.spinner("Processando. Isso pode levar alguns minutos..."):
                    run_data = run_detection(
                        algorithm=algorithm,
                        ortho_path=ortho_path,
                        conf_thres=conf_thres,
                        tile_size=tile_size,
                        overlap=overlap,
                        max_tiles=max_tiles,
                        img_size=img_size,
                        weights_url=weights_url,
                    )

                st.session_state["run_data"] = run_data
            except Exception as exc:
                st.error("Falha ao executar a deteccao. Verifique os dados e tente novamente.")
                with st.expander("Detalhes do erro"):
                    st.write(str(exc))

    run_data = st.session_state.get("run_data")
    if run_data:
        st.divider()
        st.subheader("Resultados")
        st.write(f"Tiles gerados: {run_data['tile_count']}")
        st.write(f"Deteccoes encontradas: {len(run_data['results_df'])}")

        if run_data["preview_tiles"]:
            st.image([str(p) for p in run_data["preview_tiles"]], caption=[p.name for p in run_data["preview_tiles"]])

        st.components.v1.html(run_data["map_html"], height=520, scrolling=False)

        if not run_data["results_df"].empty:
            st.dataframe(run_data["results_df"], use_container_width=True)

            csv_bytes = run_data["results_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Baixar CSV",
                data=csv_bytes,
                file_name="deteccoes.csv",
                mime="text/csv",
            )

        st.download_button(
            label="Baixar pacote (mapa + CSV)",
            data=run_data["zip_bytes"],
            file_name=f"netflora_resultados_{run_data['run_id']}.zip",
            mime="application/zip",
        )

        if run_data["returncode"] != 0:
            st.warning("A deteccao retornou avisos. Veja os logs abaixo se precisar de detalhes.")
            with st.expander("Logs do processamento"):
                if run_data["stderr"]:
                    st.code(run_data["stderr"])
                if run_data["stdout"]:
                    st.code(run_data["stdout"])

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
