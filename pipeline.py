"""
pipeline.py – Netflora Online
Backend de detecção de espécies vegetais para o app Streamlit.
Baseado no pipeline da pasta cultivando, com suporte a exportação de shapefile e mapa HTML.
"""

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import folium
import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window


# ==================== CONFIG ====================

@dataclass
class DetectionConfig:
    repo_root: Path
    weights_path: Path
    source_dir: Path
    img_size: int
    conf_thres: float
    device: str
    project_dir: Path
    run_name: str


# ==================== DOWNLOAD / SETUP ====================

def download_file(target_path: Path, url: str, timeout: int = 300) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with target_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return target_path


def ensure_netflora_repo(target_dir: Path, zip_url: str, timeout: int = 180) -> Path:
    detect_file = target_dir / "detect.py"
    if detect_file.exists():
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir.parent / "netflora_main.zip"

    response = requests.get(zip_url, stream=True, timeout=timeout)
    response.raise_for_status()
    with zip_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    extract_root = target_dir.parent / "netflora_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    candidates = [p for p in extract_root.rglob("detect.py") if p.is_file()]
    if not candidates:
        raise RuntimeError("Não foi possível localizar detect.py no pacote baixado do Netflora.")

    source_root = candidates[0].parent
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(source_root), str(target_dir))

    if zip_path.exists():
        zip_path.unlink()
    if extract_root.exists():
        shutil.rmtree(extract_root)

    return target_dir


def ensure_weights_file(weights_path: Path, weights_url: Optional[str], timeout: int = 300) -> Path:
    # Minimum expected size for a valid .pt checkpoint (~10 MB)
    MIN_WEIGHTS_BYTES = 10 * 1024 * 1024

    if weights_path.exists():
        if weights_path.stat().st_size >= MIN_WEIGHTS_BYTES:
            return weights_path
        # File exists but is too small — likely a corrupted/partial download
        weights_path.unlink()

    if not weights_url:
        raise FileNotFoundError("Arquivo de pesos não encontrado e nenhuma URL foi informada.")
    return download_file(weights_path, weights_url, timeout=timeout)


def ensure_ortho_file(local_path: Optional[Path], ortho_url: Optional[str], target_path: Path) -> Path:
    if local_path and local_path.exists():
        return local_path
    if not ortho_url:
        raise FileNotFoundError("Ortofoto não encontrada localmente e nenhuma URL foi informada.")
    return download_file(target_path, ortho_url, timeout=600)


# ==================== ALGORITMOS ====================

def get_available_algorithms(groups_json: Path) -> list:
    with groups_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data.get("categories", {}).keys())


def get_class_name_map(groups_json: Path, algorithm: str) -> Dict[int, str]:
    with groups_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    species_dict = data.get("species_dict", {})
    category = data.get("categories", {}).get(algorithm, [])
    out: Dict[int, str] = {}

    for item in category:
        class_id = int(item["class_id"])
        specie_code = item["specie"]
        out[class_id] = species_dict.get(specie_code, {}).get("common_name", f"class_{class_id}")

    return out


# ==================== TILES ====================

def generate_tiles(
    ortho_path: Path,
    output_dir: Path,
    coords_csv: Path,
    tile_size: int,
    overlap: int,
    max_tiles: int,
) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_csv.parent.mkdir(parents=True, exist_ok=True)

    step = max(tile_size - overlap, 1)
    tile_counter = 0

    with rasterio.open(ortho_path) as src, coords_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "minX", "minY", "maxX", "maxY", "crs"])

        for i in range(0, src.height, step):
            for j in range(0, src.width, step):
                if tile_counter >= max_tiles:
                    return tile_counter

                w = min(tile_size, src.width - j)
                h = min(tile_size, src.height - i)
                window = Window(j, i, w, h)

                tile = src.read(window=window)
                if tile.size == 0:
                    continue

                rgb = tile[:3] if tile.shape[0] >= 3 else tile
                if not rgb.any():
                    continue

                arr = rgb.transpose(1, 2, 0)
                img = Image.fromarray(arr)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                tile_name = f"tile_{tile_counter}.jpg"
                img.save(output_dir / tile_name, "JPEG", quality=95)

                t = src.window_transform(window)
                bounds = array_bounds(h, w, t)
                writer.writerow([tile_name, bounds[0], bounds[1], bounds[2], bounds[3], str(src.crs)])

                tile_counter += 1

    return tile_counter


# ==================== DETECÇÃO ====================

def run_netflora_detect(config: DetectionConfig) -> subprocess.CompletedProcess:
    config.project_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(config.repo_root / "detect.py"),
        "--weights", str(config.weights_path),
        "--source", str(config.source_dir),
        "--img-size", str(config.img_size),
        "--conf-thres", str(config.conf_thres),
        "--device", str(config.device),
        "--project", str(config.project_dir),
        "--name", config.run_name,
        "--exist-ok",
        # Note: this detect.py defines --save-txt / --save-conf as string args
        # with default='save-txt' / default='save-conf' (always truthy), so
        # labels and confidence scores are saved without passing any extra flags.
    ]

    return subprocess.run(
        cmd,
        cwd=str(config.repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


# ==================== RESULTADOS ====================

def build_detection_table(
    labels_dir: Path,
    coords_csv: Path,
    class_name_map: Dict[int, str],
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=["filename", "class_id", "class_name", "confidence", "bb_xmin", "bb_ymin", "bb_xmax", "bb_ymax"]
    )

    if not labels_dir.exists() or not coords_csv.exists():
        return empty

    coords = pd.read_csv(coords_csv)
    rows = []

    for txt_file in sorted(labels_dir.glob("*.txt")):
        filename = f"{txt_file.stem}.jpg"
        match = coords[coords["filename"] == filename]
        if match.empty:
            continue

        c = match.iloc[0]
        min_x, min_y, max_x, max_y = c["minX"], c["minY"], c["maxX"], c["maxY"]
        utm_width = max_x - min_x
        utm_height = max_y - min_y

        with txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                cse_x = float(parts[1])
                cse_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                conf = float(parts[5]) if len(parts) >= 6 else None

                bb_xcenter = min_x + cse_x * utm_width
                bb_ycenter = max_y - cse_y * utm_height
                bb_xmin = bb_xcenter - (width * utm_width / 2)
                bb_ymin = bb_ycenter - (height * utm_height / 2)
                bb_xmax = bb_xmin + width * utm_width
                bb_ymax = bb_ymin + height * utm_height

                rows.append({
                    "filename": filename,
                    "class_id": class_id,
                    "class_name": class_name_map.get(class_id, f"class_{class_id}"),
                    "confidence": conf,
                    "bb_xmin": bb_xmin,
                    "bb_ymin": bb_ymin,
                    "bb_xmax": bb_xmax,
                    "bb_ymax": bb_ymax,
                })

    if not rows:
        return empty

    return pd.DataFrame(rows).sort_values(by=["filename", "class_id"]).reset_index(drop=True)


def build_detection_polygons_wgs84(results_df: pd.DataFrame, coords_csv: Path) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["polygon", "class_name", "confidence"])

    if results_df.empty or not coords_csv.exists():
        return empty

    coords = pd.read_csv(coords_csv)
    if coords.empty:
        return empty

    crs_text = str(coords.iloc[0].get("crs", ""))
    if not crs_text or crs_text == "nan":
        return empty

    src_crs = CRS.from_string(crs_text)
    dst_crs = CRS.from_epsg(4326)
    rows = []

    for _, row in results_df.iterrows():
        xs = [row["bb_xmin"], row["bb_xmax"], row["bb_xmax"], row["bb_xmin"], row["bb_xmin"]]
        ys = [row["bb_ymin"], row["bb_ymin"], row["bb_ymax"], row["bb_ymax"], row["bb_ymin"]]

        lons, lats = transform(src_crs, dst_crs, xs, ys)
        polygon = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]

        rows.append({
            "polygon": polygon,
            "class_name": row.get("class_name", "desconhecido"),
            "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else None,
        })

    return pd.DataFrame(rows)


# ==================== VISUALIZAÇÃO ====================

def draw_tile_detections(tile_path: Path, label_path: Path, class_name_map: Dict[int, str]) -> Image.Image:
    image = Image.open(tile_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    if not label_path.exists():
        return image

    width, height = image.size

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            cse_x = float(parts[1])
            cse_y = float(parts[2])
            bbox_w = float(parts[3])
            bbox_h = float(parts[4])
            conf = float(parts[5]) if len(parts) >= 6 else None

            x_center = cse_x * width
            y_center = cse_y * height
            px_w = bbox_w * width
            px_h = bbox_h * height

            x1 = x_center - px_w / 2
            y1 = y_center - px_h / 2
            x2 = x_center + px_w / 2
            y2 = y_center + px_h / 2

            draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)

            label = class_name_map.get(class_id, f"class_{class_id}")
            if conf is not None:
                label = f"{label} {conf:.2f}"
            draw.text((x1 + 3, max(y1 - 14, 0)), label, fill="yellow")

    return image


def build_ortho_preview(ortho_path: Path, max_side: int = 1800):
    """Returns (rgb_array, bounds_latlon) for ortofoto overlay on folium map."""
    with rasterio.open(ortho_path) as src:
        bands = [1, 2, 3] if src.count >= 3 else [1]
        scale = max(src.width, src.height) / max_side if max(src.width, src.height) > max_side else 1.0
        out_h = max(int(src.height / scale), 1)
        out_w = max(int(src.width / scale), 1)

        arr = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.bilinear)

        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)

        arr = arr.transpose(1, 2, 0).astype(np.float32)
        p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
        if p98 > p2:
            arr = (arr - p2) / (p98 - p2)
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)

        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        bounds = [[south, west], [north, east]]

    return arr, bounds


_RESPONSIVE_INJECT = """\
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
html, body { margin: 0 !important; padding: 0 !important; height: 100% !important; overflow: hidden; }
.leaflet-control-attribution { display: none !important; }
</style>
<script>
(function () {
    function resize() {
        var vw = window.innerWidth;
        var h = vw < 480  ? Math.round(vw * 1.6) :
                vw < 768  ? Math.round(vw * 1.4) :
                vw < 1024 ? 620 : 800;
        h = Math.max(h, 420);
        window.parent.postMessage(
            { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: h },
            "*"
        );
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", resize);
    } else {
        resize();
    }
    window.addEventListener("resize", resize);
})();
</script>
"""


def _render_map(m: "folium.Map") -> str:
    """Return a full, self-contained HTML document for a folium map.

    Uses get_root().render() so there is no nested iframe — the Leaflet map
    fills the single Streamlit component iframe directly.  Responsive JS is
    injected to tell Streamlit how tall to make that iframe based on viewport.
    """
    html = m.get_root().render()
    # Inject viewport + responsive CSS/JS right after <head>
    html = html.replace("<head>", "<head>\n" + _RESPONSIVE_INJECT, 1)
    return html


def build_map_html(polygons_df: pd.DataFrame, ortho_path: Optional[Path] = None) -> str:
    """Build a folium map with detection polygons and optional ortho overlay."""
    if polygons_df.empty:
        # Return a default map centered on Brazil
        m = folium.Map(location=[-15.0, -55.0], zoom_start=4, tiles=None, max_zoom=23)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satélite",
            max_zoom=23,
            max_native_zoom=23,
            show=True,
        ).add_to(m)
        return _render_map(m)

    all_lats, all_lons = [], []
    for _, row in polygons_df.iterrows():
        for lon, lat in row["polygon"]:
            all_lats.append(lat)
            all_lons.append(lon)

    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None, control_scale=True, max_zoom=23)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite",
        max_zoom=23,
        max_native_zoom=23,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="OpenStreetMap",
        max_zoom=23,
        max_native_zoom=19,
        show=False,
    ).add_to(m)

    # Optional: overlay ortho image
    if ortho_path and ortho_path.exists():
        try:
            ortho_arr, ortho_bounds = build_ortho_preview(ortho_path)
            folium.raster_layers.ImageOverlay(
                image=ortho_arr,
                bounds=ortho_bounds,
                opacity=0.75,
                interactive=True,
                cross_origin=False,
                name="Ortofoto",
            ).add_to(m)
        except Exception:
            pass

    # Add detection polygons
    fg = folium.FeatureGroup(name="Detecções")
    for _, row in polygons_df.iterrows():
        polygon_latlon = [[pt[1], pt[0]] for pt in row["polygon"]]
        label = row.get("class_name", "desconhecido")
        conf = row.get("confidence")
        tooltip = f"Espécie: {label} | Confiança: {conf:.2f}" if conf is not None else f"Espécie: {label}"

        folium.Polygon(
            locations=polygon_latlon,
            color="#FFD700",
            weight=2,
            fill=True,
            fill_color="#FF4500",
            fill_opacity=0.4,
            tooltip=tooltip,
            popup=folium.Popup(tooltip, max_width=300),
        ).add_to(fg)

    fg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    return _render_map(m)


# ==================== EXPORTAÇÃO ====================

def export_results_zip(run_data: dict, map_html: str) -> bytes:
    """Creates in-memory zip with: mapa.html, deteccoes.csv, shapefile/."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Map HTML
        zf.writestr("mapa.html", map_html.encode("utf-8"))

        # 2. CSV
        results_df = run_data["results_df"]
        if not results_df.empty:
            zf.writestr("deteccoes.csv", results_df.to_csv(index=False))

        # 3. Shapefile via geopandas
        polygons_df = run_data["polygons_df"]
        if not polygons_df.empty:
            try:
                import geopandas as gpd
                from shapely.geometry import Polygon as ShapelyPolygon

                geometries = [ShapelyPolygon(row["polygon"]) for _, row in polygons_df.iterrows()]
                gdf = gpd.GeoDataFrame(
                    polygons_df[["class_name", "confidence"]].reset_index(drop=True),
                    geometry=geometries,
                    crs="EPSG:4326",
                )

                with tempfile.TemporaryDirectory() as tmp_dir:
                    shp_path = Path(tmp_dir) / "deteccoes.shp"
                    gdf.to_file(shp_path)
                    for f in Path(tmp_dir).iterdir():
                        zf.write(f, f"shapefile/{f.name}")
            except ImportError:
                # geopandas not available, skip shapefile
                pass

    buffer.seek(0)
    return buffer.read()
