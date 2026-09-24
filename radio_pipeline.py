"""
Chaîne d'extraction des features radio — port fidèle de geo.py, relief.py,
Rx_heights.py, angles.py et visibility.py en un module unique appelable
point par point.

Ordre d'origine respecté :  geo → relief → Rx_heights → angles → visibility

Toute position de récepteur produit exactement le même vecteur de features
qu'une mesure réelle, puisque c'est le même code qui l'a produit.
"""

import math
import numpy as np
import rasterio
from shapely.geometry import LineString
from pyproj import Transformer
from geopy.distance import geodesic

# ── Constantes (identiques aux scripts d'origine) ───────────────────────────
C_LIGHT      = 3e8
STEP_METERS  = 30
LOCAL_RADIUS = 50
RX_EXTENSION = 50
N_SAMPLES    = 500
RX_ANTENNA_HEIGHT = 10

to_utm   = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
to_wgs84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)


# ============================================================
# geo.py — distance géodésique
# ============================================================
def compute_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters


# ============================================================
# relief.py — profil de terrain
# ============================================================
def sample_dem_along_line(dem, line, step=STEP_METERS):
    length = line.length
    distances = np.arange(0, length, step)
    distances = np.append(distances, length)
    coords = [(p.x, p.y) for p in (line.interpolate(d) for d in distances)]
    altitudes = np.array([v[0] for v in dem.sample(coords)], dtype=float)
    altitudes[altitudes <= 0] = np.nan          # eau / nodata FABDEM
    nans = np.isnan(altitudes)
    if nans.any() and not nans.all():
        idx = np.arange(len(altitudes))
        altitudes[nans] = np.interp(idx[nans], idx[~nans], altitudes[~nans])
    return distances, altitudes


def mean_slope(distances, altitudes):
    if len(distances) < 2:
        return np.nan
    dz, dx = np.diff(altitudes), np.diff(distances)
    valid = dx > 0
    if not np.any(valid):
        return np.nan
    return np.nanmean(dz[valid] / dx[valid])


def roughness(altitudes):
    if np.sum(~np.isnan(altitudes)) < 2:
        return np.nan
    return np.nanstd(altitudes)


def terrain_features(dem, tx_lat, tx_lon, rx_lat, rx_lon, distance):
    """Rx_alt, pentes et rugosités — relief.py, à l'identique."""
    Tx_X, Tx_Y = to_utm.transform(tx_lon, tx_lat)
    Rx_X, Rx_Y = to_utm.transform(rx_lon, rx_lat)

    line = LineString([(Tx_X, Tx_Y), (Rx_X, Rx_Y)])
    dist, alt = sample_dem_along_line(dem, line)
    Rx_alt          = alt[-1]
    Slope_Tx_Rx     = mean_slope(dist, alt)
    Roughness_Tx_Rx = roughness(alt)

    factor   = (distance + RX_EXTENSION) / distance
    Rx_ext_X = Tx_X + factor * (Rx_X - Tx_X)
    Rx_ext_Y = Tx_Y + factor * (Rx_Y - Tx_Y)
    line_ext = LineString([(Tx_X, Tx_Y), (Rx_ext_X, Rx_ext_Y)])
    dist2, alt2 = sample_dem_along_line(dem, line_ext)
    Slope_Tx_Rx_50m     = mean_slope(dist2, alt2)
    Roughness_Tx_Rx_50m = roughness(alt2)

    mask       = (dist2 >= (distance - LOCAL_RADIUS)) & (dist2 <= (distance + LOCAL_RADIUS))
    local_dist = dist2[mask] - distance
    local_alt  = alt2[mask]
    Slope_Local_Rx_50m     = mean_slope(local_dist, local_alt)
    Roughness_Local_Rx_50m = roughness(local_alt)

    return dict(Rx_alt=Rx_alt,
                Slope_Tx_Rx=Slope_Tx_Rx, Roughness_Tx_Rx=Roughness_Tx_Rx,
                Slope_Tx_Rx_50m=Slope_Tx_Rx_50m, Roughness_Tx_Rx_50m=Roughness_Tx_Rx_50m,
                Slope_Local_Rx_50m=Slope_Local_Rx_50m,
                Roughness_Local_Rx_50m=Roughness_Local_Rx_50m)


# ============================================================
# angles.py — azimut et tilt
# ============================================================
def calculate_azimuth(lat1, lon1, lat2, lon2):
    lat1, lat2 = map(math.radians, [lat1, lat2])
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_tilt(tx_height, rx_height, distance):
    if distance <= 0:
        return 0.0
    return math.degrees(math.atan((tx_height - rx_height) / distance))


# ============================================================
# visibility.py — Fresnel, LOS, obstacles
# ============================================================
def load_raster(path):
    src = rasterio.open(path)
    return src.read(1), src.transform


def get_value(raster, transform, x, y):
    col, row = ~transform * (x, y)
    col, row = int(col), int(row)
    if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
        val = raster[row, col]
        return 0.0 if (np.isnan(val) or val < 0) else float(val)
    return 0.0


def fresnel_radius(d_tx, d_rx, freq_hz):
    lam = C_LIGHT / freq_hz
    d_total = d_tx + d_rx
    return 0.0 if d_total == 0 else np.sqrt(lam * d_tx * d_rx / d_total)


def visibility_features(tx_lat, tx_lon, rx_lat, rx_lon, tx_height, rx_height,
                        distance, freq_mhz, city,
                        dsm, tr_dsm, buildings, tr_bld, fabdem, tr_fab):
    """LOS, First_*_m, Fresnel_*, *_near_Rx, Environment — visibility.py."""
    tx_utm_x, tx_utm_y = to_utm.transform(tx_lon, tx_lat)
    rx_utm_x, rx_utm_y = to_utm.transform(rx_lon, rx_lat)
    freq_hz = freq_mhz * 1e6

    fracs = np.linspace(0, 1, N_SAMPLES)
    dists = fracs * distance

    fresnel_buildings = fresnel_trees = 0
    first_building_m = first_tree_m = -1.0
    los_itu = True

    for i, frac in enumerate(fracs):
        if frac == 0.0 or frac == 1.0:
            continue
        utm_x = tx_utm_x + frac * (rx_utm_x - tx_utm_x)
        utm_y = tx_utm_y + frac * (rx_utm_y - tx_utm_y)
        lon_i, lat_i = to_wgs84.transform(utm_x, utm_y)

        d_tx_i = dists[i]
        d_rx_i = distance - d_tx_i

        h_terrain = get_value(fabdem,    tr_fab, utm_x, utm_y)
        h_dsm     = get_value(dsm,       tr_dsm, lon_i, lat_i)
        is_bld    = get_value(buildings, tr_bld, lon_i, lat_i)

        h_obs_rel = max(h_dsm - h_terrain, 0.0)
        h_obs_abs = h_terrain + h_obs_rel
        h_los_abs = tx_height + frac * (rx_height - tx_height)
        clearance = h_los_abs - h_obs_abs
        r_fres    = fresnel_radius(d_tx_i, d_rx_i, freq_hz)

        if is_bld == 1 and h_obs_rel > 1.0 and r_fres > 0 and clearance < r_fres:
            fresnel_buildings += 1
            if first_building_m == -1.0:
                first_building_m = d_tx_i
            los_itu = False

        if is_bld == 0 and h_obs_rel > 2.0 and r_fres > 0 and clearance < r_fres:
            fresnel_trees += 1
            if first_tree_m == -1.0:
                first_tree_m = d_tx_i
            los_itu = False

    buildings_near_rx = trees_near_rx = 0
    seen = set()
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    radii  = np.linspace(5, 50, 10)
    for r in radii:
        for a in angles:
            xb = rx_utm_x + r * np.cos(a)
            yb = rx_utm_y + r * np.sin(a)
            lon_b, lat_b = to_wgs84.transform(xb, yb)
            col_b = int((lon_b - tr_bld.c) / tr_bld.a)
            row_b = int((lat_b - tr_bld.f) / tr_bld.e)
            if (col_b, row_b) in seen:
                continue
            seen.add((col_b, row_b))
            is_bld_rx = get_value(buildings, tr_bld, lon_b, lat_b)
            h_ter_rx  = get_value(fabdem,    tr_fab, xb, yb)
            h_dsm_rx  = get_value(dsm,       tr_dsm, lon_b, lat_b)
            h_obs_rx  = max(h_dsm_rx - h_ter_rx, 0.0)
            if is_bld_rx == 1 and h_obs_rx > 1.0:
                buildings_near_rx += 1
            elif is_bld_rx == 0 and h_obs_rx > 2.0:
                trees_near_rx += 1

    return dict(LOS=1 if los_itu else 0,
                First_building_m=first_building_m, First_tree_m=first_tree_m,
                Fresnel_buildings=fresnel_buildings, Fresnel_trees=fresnel_trees,
                Buildings_near_Rx=buildings_near_rx, Trees_near_Rx=trees_near_rx,
                Environment="urban" if str(city).lower() == "cotonou" else "suburban")


# ============================================================
# Chaîne complète pour une position de récepteur
# ============================================================
def compute_all_features(rx_lat, rx_lon, tx, rasters):
    """
    tx      : dict — Tx_lat, Tx_lon, Tx_alt, Tx_antenna_height, Tx_power, Freq, City, Tx_id
    rasters : dict — 'dem' (dataset FABDEM ouvert), et par ville
              'dsm','tr_dsm','bld','tr_bld','fabdem','tr_fab'
    Renvoie le dictionnaire complet des features, dans l'ordre de la chaîne.
    """
    city = str(tx['City']).lower()
    R = rasters[city]

    # geo.py
    distance = compute_distance(tx['Tx_lat'], tx['Tx_lon'], rx_lat, rx_lon)

    # relief.py
    ter = terrain_features(rasters['dem'], tx['Tx_lat'], tx['Tx_lon'],
                           rx_lat, rx_lon, distance)

    # Rx_heights.py
    rx_height = ter['Rx_alt'] + RX_ANTENNA_HEIGHT

    # angles.py
    tx_height = tx['Tx_alt'] + tx['Tx_antenna_height']
    azimut = calculate_azimuth(tx['Tx_lat'], tx['Tx_lon'], rx_lat, rx_lon)
    tilt   = calculate_tilt(tx_height, rx_height, distance)

    # visibility.py
    vis = visibility_features(tx['Tx_lat'], tx['Tx_lon'], rx_lat, rx_lon,
                              tx_height, rx_height, distance, tx['Freq'], tx['City'],
                              R['dsm'], R['tr_dsm'], R['bld'], R['tr_bld'],
                              rasters['fabdem'], rasters['tr_fab'])

    out = dict(Rx_lat=rx_lat, Rx_lon=rx_lon,
               Tx_id=tx['Tx_id'], City=tx['City'],
               Tx_lat=tx['Tx_lat'], Tx_lon=tx['Tx_lon'],
               Distance=distance, Tx_power=tx['Tx_power'],
               Tx_alt=tx['Tx_alt'], Tx_antenna_height=tx['Tx_antenna_height'],
               Tx_height=tx_height, Freq=tx['Freq'],
               Rx_antenna_height=RX_ANTENNA_HEIGHT, Rx_height=rx_height,
               Azimut_Tx_Rx=azimut, Tilt_Tx_Rx=tilt)
    out.update(ter)
    out.update(vis)
    return out
