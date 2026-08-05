"""
WaPOR v3 NPP Explorer — Time Series, Seasonal Total & Last-Date Map
=====================================================================
Upload a field boundary (zipped Shapefile, .kml, or .kmz — one or more
polygons), pick a date range, and get:
  - A monthly NPP time series (spatial mean within each polygon)
  - The total seasonal NPP (sum of monthly means over the chosen period)
  - A map of the most recent month's NPP, clipped to each polygon

Data source: WaPOR v3, Level 3, Monthly NPP ("L3-NPP-M"), via the official
FAO `wapordl` package. This ALWAYS resolves to the current, corrected
dataset through FAO's own catalog — it does not rely on a hand-picked
Google Cloud Storage path, which avoids accidentally using a stale/
pre-correction copy of the data. NO API key, login, or credentials needed.

Units: gC/m²/month (grams of carbon per square metre per month).
Scale factor (0.001) and no-data value (-9999) are applied automatically
by this script when reading each GeoTIFF.

SETUP:
    pip install streamlit wapordl rasterio pyshp pyproj pandas matplotlib numpy --break-system-packages

RUN:
    streamlit run npp_wapor_app.py
"""

import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import date

import streamlit as st
import shapefile          # pyshp
from pyproj import CRS, Transformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rasterio
from rasterio.mask import mask as rio_mask
from wapordl import wapor_map

# ─────────────────────────────────────────────────────────────────────────────
# Boundary loading (Shapefile / KML / KMZ, one or more polygons)
# ─────────────────────────────────────────────────────────────────────────────

def _ring_to_lonlat(points, transformer=None):
    if transformer:
        points = [transformer.transform(x, y) for x, y in points]
    if points[0] != points[-1]:
        points.append(points[0])
    return [[float(x), float(y)] for x, y in points]


def load_all_shapefile_geometries(tmpdir):
    shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
    if not shp_files:
        raise ValueError("No .shp file found inside the zip.")
    shp_path = os.path.join(tmpdir, shp_files[0])

    transformer = None
    prj_path = shp_path.replace(".shp", ".prj")
    if os.path.exists(prj_path):
        with open(prj_path) as f:
            src_crs = CRS.from_wkt(f.read())
        if not src_crs.equals(CRS.from_epsg(4326)):
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    sf = shapefile.Reader(shp_path)
    results = []
    for i, shape_rec in enumerate(sf.shapeRecords()):
        shp = shape_rec.shape
        rec = shape_rec.record

        poly_name = None
        for field_idx, field_info in enumerate(sf.fields[1:]):
            field_name = field_info[0].upper()
            if field_name in ("NAME", "ID", "FID", "LABEL", "POLYNAME", "FIELD"):
                val = rec[field_idx]
                if val:
                    poly_name = str(val)
                    break
        if poly_name is None:
            poly_name = f"Polygon {i + 1}"

        if shp.shapeType not in (5, 15, 25):  # Polygon, PolygonZ, PolygonM
            continue

        if shp.parts:
            start = shp.parts[0]
            end = shp.parts[1] if len(shp.parts) > 1 else len(shp.points)
        else:
            start, end = 0, len(shp.points)

        ring = _ring_to_lonlat(list(shp.points[start:end]), transformer)
        results.append({"name": poly_name, "geometry": {"type": "Polygon", "coordinates": [ring]}})

    if not results:
        raise ValueError("No polygon features found in the Shapefile.")
    return results


def _strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _iter_tag(root, local_name):
    for el in root.iter():
        if _strip_ns(el.tag) == local_name:
            yield el


def _parse_kml_coords(text):
    coords = []
    for token in text.strip().split():
        parts = token.strip().split(",")
        try:
            coords.append([float(parts[0]), float(parts[1])])
        except (IndexError, ValueError):
            continue
    if len(coords) >= 2 and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def load_all_kml_geometries(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    def _placemark_name(element):
        el = element
        while el is not None:
            if _strip_ns(el.tag) == "Placemark":
                for child in el:
                    if _strip_ns(child.tag) == "name" and child.text:
                        return child.text.strip()
                return None
            el = parent_map.get(el)
        return None

    results, poly_idx = [], 0
    for poly_el in _iter_tag(root, "Polygon"):
        poly_idx += 1
        coords_el = None
        for outer in _iter_tag(poly_el, "outerBoundaryIs"):
            for ring in _iter_tag(outer, "LinearRing"):
                for c in _iter_tag(ring, "coordinates"):
                    coords_el = c
                    break
                if coords_el is not None:
                    break
            if coords_el is not None:
                break
        if coords_el is None:
            for c in _iter_tag(poly_el, "coordinates"):
                coords_el = c
                break
        if coords_el is None or not coords_el.text:
            continue
        ring = _parse_kml_coords(coords_el.text)
        if len(ring) < 4:
            continue
        name = _placemark_name(poly_el) or f"Polygon {poly_idx}"
        results.append({"name": name, "geometry": {"type": "Polygon", "coordinates": [ring]}})

    if not results:
        raise ValueError("No polygon coordinates found in this KML file.")
    return results


def load_boundary(uploaded_file):
    filename = uploaded_file.name.lower()
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = os.path.join(tmpdir, filename)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if filename.endswith(".zip"):
            with zipfile.ZipFile(saved_path) as z:
                z.extractall(tmpdir)
            return load_all_shapefile_geometries(tmpdir)
        elif filename.endswith(".kmz"):
            with zipfile.ZipFile(saved_path) as z:
                z.extractall(tmpdir)
            kml_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".kml")]
            if not kml_files:
                raise ValueError("No .kml file found inside the KMZ.")
            return load_all_kml_geometries(kml_files[0])
        elif filename.endswith(".kml"):
            return load_all_kml_geometries(saved_path)
        else:
            raise ValueError("Upload a .zip (Shapefile), .kml, or .kmz file.")


# ─────────────────────────────────────────────────────────────────────────────
# WaPOR v3 NPP retrieval — via the official `wapordl` package (no credentials)
# ─────────────────────────────────────────────────────────────────────────────

DATE_PATTERNS = [
    re.compile(r"(\d{4})[-_.](\d{2})[-_.](\d{2})"),  # YYYY-MM-DD / YYYY.MM.DD
    re.compile(r"(\d{4})[-_.](\d{2})(?!\d)"),          # YYYY-MM
    re.compile(r"(\d{4})(\d{2})(\d{2})"),              # YYYYMMDD
]


def _date_from_filename(fname):
    """Try several common patterns to pull a date out of a WaPOR filename."""
    base = os.path.basename(fname)
    for pat in DATE_PATTERNS:
        m = pat.search(base)
        if m:
            groups = m.groups()
            try:
                if len(groups) == 3:
                    return f"{groups[0]}-{groups[1]}-{groups[2]}"
                else:
                    return f"{groups[0]}-{groups[1]}-01"
            except Exception:
                continue
    return None  # caller falls back to filename order


def polygon_bbox(geometry):
    """[minlon, minlat, maxlon, maxlat] from a GeoJSON Polygon."""
    coords = geometry["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def fetch_npp_series(geometry, start_date, end_date, work_dir):
    """
    Downloads monthly WaPOR v3 NPP GeoTIFFs covering the polygon's bounding
    box for the given period (via wapordl, always the current/corrected
    dataset), clips each one to the EXACT polygon shape, and returns:
        df            — DataFrame with columns [date, npp_mean]
        last_raster   — (masked_array, transform) for the most recent date
        last_date_val — date/label string for that most recent raster
        tif_paths     — list of downloaded filenames, for diagnostics
    """
    bbox = polygon_bbox(geometry)

    tif_paths = wapor_map(
        bbox, "L3-NPP-M", [start_date, end_date], work_dir, extension=".tif"
    )
    if isinstance(tif_paths, str):
        tif_paths = [tif_paths]
    tif_paths = sorted(tif_paths)

    rows = []
    last_raster = None
    last_date_val = None

    for path in tif_paths:
        with rasterio.open(path) as src:
            scale = src.scales[0] if src.scales and src.scales[0] else 0.001
            offset = src.offsets[0] if src.offsets and src.offsets[0] else 0.0
            nodata = src.nodata if src.nodata is not None else -9999

            try:
                clipped, transform = rio_mask(src, [geometry], crop=True, nodata=nodata)
            except ValueError:
                continue  # polygon does not overlap this raster at all

            band = clipped[0].astype("float64")
            band[band == nodata] = np.nan
            band = band * scale + offset  # apply WaPOR's scale factor explicitly

            valid = band[~np.isnan(band)]
            if valid.size == 0:
                continue

            date_str = _date_from_filename(path)
            rows.append({"date": date_str or path, "npp_mean": float(np.mean(valid))})

            last_raster = (band, transform)
            last_date_val = date_str or os.path.basename(path)

    df = pd.DataFrame(rows)
    if not df.empty and df["date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

    return df, last_raster, last_date_val, tif_paths


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="WaPOR NPP Explorer", layout="wide")
st.title("🌾 WaPOR v3 NPP Explorer")
st.write(
    "Upload a field boundary (zipped Shapefile, KML, or KMZ — one or more polygons). "
    "Get a monthly Net Primary Production (NPP) time series, the total seasonal NPP, "
    "and a map of the most recent month, per polygon."
)

with st.expander("ℹ️ About this data"):
    st.markdown("""
- **Source:** WaPOR v3, Level 3, Monthly NPP (`L3-NPP-M`) — 20 m resolution, Northern Egypt coverage.
- **Access:** the official FAO `wapordl` package, which always resolves to the current,
  corrected dataset via FAO's catalog — no API key or login required, and no risk of
  accidentally pulling a stale pre-correction copy from a hand-picked storage path.
- **Units:** gC/m²/month.
- **Total seasonal NPP** = sum of the monthly spatial-mean values across your chosen date range.
- Scale factor (0.001) and no-data value (-9999) are applied automatically by this app.
""")

uploaded_file = st.file_uploader("Upload field boundary (zipped Shapefile, .kml, or .kmz)", type=["zip", "kml", "kmz"])

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", date(2024, 1, 1))
with col2:
    end_date = st.date_input("End date", date(2024, 12, 31))

if st.button("Run Analysis", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a boundary file first.")
        st.stop()

    with st.spinner("Reading boundary file..."):
        try:
            polygons = load_boundary(uploaded_file)
        except Exception as e:
            st.error(f"Could not read boundary file: {e}")
            st.stop()

    st.success(f"Found **{len(polygons)}** polygon(s).")

    work_dir = tempfile.mkdtemp()

    for poly in polygons:
        st.subheader(f"📍 {poly['name']}")

        with st.spinner(f"Downloading WaPOR NPP data for {poly['name']}..."):
            try:
                df, last_raster, last_date_val, tif_paths = fetch_npp_series(
                    poly["geometry"],
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    work_dir,
                )
            except Exception as e:
                st.error(f"⚠️ Could not fetch NPP data for {poly['name']}: {e}")
                continue

        if df.empty:
            st.warning("No NPP data found for this polygon/date range.")
            continue

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(df["date"], df["npp_mean"], marker="o", color="#1a9850")
        ax.set_xlabel("Date")
        ax.set_ylabel("NPP (gC/m²/month)")
        ax.set_title(f"Monthly NPP — {poly['name']}")
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        total_seasonal_npp = df["npp_mean"].sum()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Total seasonal NPP", f"{total_seasonal_npp:.1f} gC/m²")
            st.metric("Months with data", len(df))
        with col_b:
            display_df = df.copy()
            if pd.api.types.is_datetime64_any_dtype(display_df["date"]):
                display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df = display_df.rename(columns={"npp_mean": "NPP (gC/m²/month)"})
            st.dataframe(display_df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇️ Download {poly['name']} NPP CSV", csv,
                            f"npp_{poly['name'].replace(' ', '_')}.csv", "text/csv",
                            key=f"dl_{poly['name']}")

        if last_raster is not None:
            band, transform = last_raster
            st.markdown(f"**🗺️ NPP map — most recent date ({last_date_val})**")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            im = ax2.imshow(band, cmap="YlGn")
            ax2.set_title(f"{poly['name']} — {last_date_val}")
            ax2.set_xticks([]); ax2.set_yticks([])
            plt.colorbar(im, ax=ax2, label="gC/m²/month", fraction=0.046, pad=0.04)
            st.pyplot(fig2)

        with st.expander(f"🔧 Diagnostics — files downloaded for {poly['name']}"):
            st.write("If dates in the chart/table look wrong, check these actual filenames:")
            for p in tif_paths:
                st.text(os.path.basename(p))
