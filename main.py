"""
Cologne Isochrone Flood-Fill Map
=================================
Generates a presentation-quality raster isochrone map of Cologne.
Travel times are computed from a single starting point using per-road-type
speeds capped at 80 km/h, then flood-filled across the entire map via
nearest-neighbour interpolation.

Usage:
    pip install -r requirements.txt
    python main.py
Output:
    cologne_isochrone.png  (300 DPI)
"""

import os
import sys
import threading
import time
import warnings
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import osmnx as ox
import requests
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

STARTING_WAY_ID = 879335519          # OSM way – starting point for isochrones
BUFFER_M        = 10_000             # metres to extend beyond Cologne boundary
GRID_M          = 250                # raster cell size in metres
SMOOTH_SIGMA    = 2.0                # gaussian smoothing (grid cells)
OUTPUT_FILE     = "cologne_isochrone.png"
DPI             = 300
CACHE_DIR       = "cache"
GRAPH_CACHE     = os.path.join(CACHE_DIR, "cologne_network.graphml")

# Max speeds by OSM highway type (km/h) — capped at 80 km/h
EMERGENCY_SPEEDS = {
    "motorway":       80,
    "motorway_link":  80,
    "trunk":          80,
    "trunk_link":     80,
    "primary":        80,
    "primary_link":   70,
    "secondary":      70,
    "secondary_link": 60,
    "tertiary":       60,
    "tertiary_link":  50,
    "unclassified":   50,
    "residential":    50,
    "living_street":  30,
    "service":        30,
    "road":           50,
}
FALLBACK_SPEED = 50   # km/h for unmapped types


# ─────────────────────────────────────────────────────────────────────────────
# Progress indicator
# ─────────────────────────────────────────────────────────────────────────────

class ProgressPrinter:
    """Context manager that prints elapsed time every 5 s during a slow step."""

    def __init__(self, label: str):
        self.label   = label
        self._stop   = threading.Event()
        self._start  = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._start = time.monotonic()
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(5):
            elapsed = time.monotonic() - self._start
            print(f"    … {self.label} ({elapsed:.0f}s elapsed)", flush=True)

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        elapsed = time.monotonic() - self._start
        print(f"    done in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Get starting-point coordinates
# ─────────────────────────────────────────────────────────────────────────────

def get_way_centroid(way_id: int) -> tuple[float, float]:
    """Return (lat, lon) centroid of an OSM way via the Overpass API."""
    print(f"[1/5] Fetching OSM way {way_id} …")
    url   = "https://overpass-api.de/api/interpreter"
    query = f"[out:json];way({way_id});out center;"
    try:
        r = requests.get(url, params={"data": query}, timeout=30)
        r.raise_for_status()
        elem   = r.json()["elements"][0]
        centre = elem["center"]
        tags   = elem.get("tags", {})
        name   = tags.get("name", tags.get("addr:street", f"way/{way_id}"))
        print(f"    → '{name}'  lat={centre['lat']:.6f}  lon={centre['lon']:.6f}")
        return centre["lat"], centre["lon"]
    except Exception as exc:
        sys.exit(f"ERROR: could not fetch way {way_id}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Download Cologne admin boundary & buffered road network
# ─────────────────────────────────────────────────────────────────────────────

def get_cologne_boundary() -> gpd.GeoDataFrame:
    print("[2/5] Fetching Cologne admin boundary …")
    gdf = ox.geocode_to_gdf({"city": "Cologne", "country": "Germany"})
    return gdf.to_crs("EPSG:4326")


def download_network(cologne_gdf: gpd.GeoDataFrame):
    """Download (or load from cache) the driving network for Cologne + BUFFER_M metres."""
    cologne_proj = cologne_gdf.to_crs("EPSG:25832")
    buffered_m   = cologne_proj.geometry.iloc[0].buffer(BUFFER_M)
    buffered_wgs = (
        gpd.GeoDataFrame(geometry=[buffered_m], crs="EPSG:25832")
        .to_crs("EPSG:4326")
        .geometry.iloc[0]
    )

    if os.path.exists(GRAPH_CACHE):
        print(f"[3/5] Loading cached road network ({GRAPH_CACHE}) …")
        G = ox.load_graphml(GRAPH_CACHE)
    else:
        print(f"[3/5] Downloading road network (Cologne + {BUFFER_M/1000:.0f} km buffer) …")
        with ProgressPrinter("downloading"):
            G = ox.graph_from_polygon(buffered_wgs, network_type="drive", retain_all=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        ox.save_graphml(G, GRAPH_CACHE)
        print(f"    cached → {GRAPH_CACHE}")

    print(f"    → {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, buffered_wgs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Assign emergency-vehicle travel-time weights
# ─────────────────────────────────────────────────────────────────────────────

def _parse_maxspeed(raw) -> float | None:
    """Try to extract a numeric speed (km/h) from a maxspeed tag value."""
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0]
    raw = str(raw).strip().lower()
    # Common text tokens
    if raw in ("none", "signals", "variable", "walk"):
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


def assign_travel_times(G) -> None:
    """Add a `travel_time` attribute (seconds) to every edge in-place."""
    for _, _, data in G.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]

        maxspeed_raw   = data.get("maxspeed")
        maxspeed_km    = _parse_maxspeed(maxspeed_raw)
        road_speed     = EMERGENCY_SPEEDS.get(highway, FALLBACK_SPEED)

        if maxspeed_km:
            # Use the higher of tagged speed and road-type table, capped at 80 km/h
            speed = min(max(maxspeed_km, road_speed), 80)
        else:
            speed = road_speed

        length_m             = data.get("length", 1.0)
        data["travel_time"]  = length_m / (speed / 3.6)   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Dijkstra from the starting node
# ─────────────────────────────────────────────────────────────────────────────

def compute_travel_times(G, start_lat: float, start_lon: float) -> dict:
    """Return {node_id: travel_time_seconds} for all reachable nodes."""
    start_node = ox.nearest_nodes(G, start_lon, start_lat)
    G_undir    = G.to_undirected()
    print("[4/5] Running Dijkstra from starting node …")
    times      = nx.single_source_dijkstra_path_length(
                     G_undir, start_node, weight="travel_time")
    print(f"    → {len(times):,} nodes reached")
    return times


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Rasterise & render
# ─────────────────────────────────────────────────────────────────────────────

def _lat_deg_per_m(lat_deg: float) -> float:
    """Approximate latitude degrees per metre at the given latitude."""
    return 1.0 / 111_320.0


def _lon_deg_per_m(lat_deg: float) -> float:
    """Approximate longitude degrees per metre at the given latitude."""
    return 1.0 / (111_320.0 * np.cos(np.radians(lat_deg)))


def rasterise(G, times: dict, buffered_poly):
    """Interpolate travel times (minutes) onto a regular lon/lat grid."""
    # Collect (lon, lat, time_min) for all reached nodes
    lons, lats, tmins = [], [], []
    for node_id, t_sec in times.items():
        nd = G.nodes[node_id]
        lons.append(nd["x"])
        lats.append(nd["y"])
        tmins.append(t_sec / 60.0)

    lons  = np.array(lons)
    lats  = np.array(lats)
    tmins = np.array(tmins)

    # Build regular grid
    minx, miny, maxx, maxy = buffered_poly.bounds
    mid_lat   = (miny + maxy) / 2.0
    dlon      = _lon_deg_per_m(mid_lat) * GRID_M
    dlat      = _lat_deg_per_m(mid_lat) * GRID_M

    grid_lon  = np.arange(minx, maxx, dlon)
    grid_lat  = np.arange(miny, maxy, dlat)
    glon, glat = np.meshgrid(grid_lon, grid_lat)

    print(f"[5/5] Rasterising onto {glon.shape[1]}×{glon.shape[0]} grid …")
    points    = np.column_stack([lons, lats])
    grid_t    = griddata(points, tmins, (glon, glat), method="nearest")
    grid_t    = gaussian_filter(grid_t.astype(float), sigma=SMOOTH_SIGMA)

    return glon, glat, grid_t


def render(G, cologne_gdf, glon, glat, grid_t,
           start_lat, start_lon, times):
    """Produce the final map image."""
    import contextily as ctx

    # Clip colour scale at 99th percentile to avoid outliers bleaching the map
    vmax = np.percentile(grid_t, 99)
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(14, 14), dpi=DPI)

    # ── Raster layer ──────────────────────────────────────────────────────────
    cmap = plt.cm.plasma_r   # dark-purple/blue = close, yellow = far
    im = ax.pcolormesh(
        glon, glat, grid_t,
        cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.72, shading="auto", zorder=2,
    )

    # ── Basemap (CartoDB Positron – subtle, good contrast with plasma) ────────
    try:
        ctx.add_basemap(
            ax, crs="EPSG:4326",
            source=ctx.providers.CartoDB.Positron,
            zorder=1,
        )
    except Exception as exc:
        print(f"    (basemap unavailable: {exc})")

    # ── Cologne boundary ──────────────────────────────────────────────────────
    cologne_gdf.boundary.plot(ax=ax, color="white", linewidth=1.8, zorder=4)

    # ── Starting point marker ─────────────────────────────────────────────────
    ax.plot(start_lon, start_lat, marker="*", color="white",
            markersize=18, markeredgecolor="black", markeredgewidth=0.8,
            zorder=5, label="Starting point")

    # ── Isochrone contour lines (every 10 min) ────────────────────────────────
    levels = np.arange(10, vmax, 10)
    if len(levels):
        cs = ax.contour(
            glon, glat, grid_t,
            levels=levels, colors="white", linewidths=0.4, alpha=0.4, zorder=3,
        )
        ax.clabel(cs, fmt="%d min", fontsize=6, colors="white", inline=True)

    # ── Colour bar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.01, aspect=30)
    cbar.set_label(
        "Drive time (minutes) · emergency vehicle speed",
        color="white", fontsize=10,
    )
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("white")

    # ── Titles & aesthetics ───────────────────────────────────────────────────
    ax.set_title(
        "Köln — Isochrone Map\n(Emergency Vehicle Driving Time)",
        color="white", fontsize=15, pad=12,
    )
    ax.set_xlabel("Longitude", color="white", fontsize=8)
    ax.set_ylabel("Latitude",  color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    bg = "#1a1a2e"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {OUTPUT_FILE}")

    # Print quick stats
    t_arr = np.array(list(times.values())) / 60
    print(f"Max travel time in network : {t_arr.max():.1f} min")
    print(f"Colour scale cap (99th pct): {vmax:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_lat, start_lon = get_way_centroid(STARTING_WAY_ID)

    cologne_gdf           = get_cologne_boundary()
    G, buffered_poly      = download_network(cologne_gdf)
    assign_travel_times(G)

    times                 = compute_travel_times(G, start_lat, start_lon)
    glon, glat, grid_t    = rasterise(G, times, buffered_poly)

    render(G, cologne_gdf, glon, glat, grid_t,
           start_lat, start_lon, times)
