"""routing/graph_builder.py

Bakhmut의 실제 도로망(OpenStreetMap)을 NetworkX 그래프로 가져옵니다.

작동:
  1. 첫 실행: OSM 에서 Bakhmut 도시 도로망을 다운로드 (1~3분)
  2. data/bakhmut_osm.pkl 에 캐시 -> 이후 즉시 로드
  3. 노드 좌표(lat/lon) <-> 우리 격자(row/col) 매핑 헬퍼 제공

Phase 5 의 시민 앱이 이 그래프 위에서 라우팅을 합니다.
"""

import pickle
from pathlib import Path

from config import settings

# Bakhmut 도시 중심부 bbox: (left=west, bottom=south, right=east, top=north) — osmnx 2.x 형식
# 대략 7km x 7km, 도시 도로망이 빽빽한 영역
BAKHMUT_BBOX = (37.94, 48.56, 38.04, 48.62)

# 시민이 갈 수 있는 대피소 5곳 (Bakhmut 도시 내 분산 배치)
# 발표 시 "병원/학교/지하주차장 등 실제 안전 시설로 매핑 가능 (future work)"
SHELTERS_LATLON = [
    ("북부 대피소", (48.610, 37.990)),
    ("남부 대피소", (48.570, 37.990)),
    ("동부 대피소", (48.590, 38.030)),
    ("서부 대피소", (48.590, 37.960)),
    ("중앙 대피소", (48.595, 38.000)),
]

OSM_CACHE = Path("data/bakhmut_osm.pkl")


def load_or_build_bakhmut_graph(bbox=BAKHMUT_BBOX, network_type: str = "drive"):
    """OSM Bakhmut 도로망을 캐시에서 로드하거나 새로 다운로드.

    OSM(Overpass API) 가 일시 차단/장애일 경우, 격자형 합성 그래프로 자동 폴백.
    합성도 같은 인터페이스(x/y 속성)를 가지므로 라우팅 코드는 그대로 동작합니다.
    """
    if OSM_CACHE.exists():
        with open(OSM_CACHE, "rb") as f:
            G = pickle.load(f)
        kind = G.graph.get("source", "osm")
        print(f"[Graph] 캐시 사용 ({kind}): {len(G.nodes)} 노드, {len(G.edges)} 간선")
        return G

    print(f"[OSM] Bakhmut 도로망 다운로드 중... (Overpass API, 30초~1분)")
    try:
        import osmnx as ox
        # osmnx 2.x: bbox = (left=west, bottom=south, right=east, top=north)
        G = ox.graph_from_bbox(bbox=bbox, network_type=network_type)
        G.graph["source"] = "osm"
        print(f"[OSM] 다운로드 성공: {len(G.nodes)} 노드, {len(G.edges)} 간선")
    except Exception as e:
        print(f"[OSM] 다운로드 실패 ({type(e).__name__}: {str(e)[:80]})")
        print(f"[Graph] 합성 격자 그래프로 폴백 (Overpass 복구되면 다음번에 실제 OSM 사용)")
        G = _build_synthetic_graph(bbox)

    OSM_CACHE.parent.mkdir(exist_ok=True)
    with open(OSM_CACHE, "wb") as f:
        pickle.dump(G, f)
    print(f"[Graph] 캐시 저장 -> {OSM_CACHE}")
    return G


def _build_synthetic_graph(bbox, grid_n: int = 30):
    """OSM 없을 때 사용할 합성 도로망. Bakhmut 도시 영역에 격자형 도로를 깔아 만듦.

    각 노드: (i, j) 격자 좌표, x=경도, y=위도 속성을 가짐 (osmnx 호환).
    간선: 양방향, length 속성은 실제 미터 거리.
    """
    import math
    import networkx as nx
    west, south, east, north = bbox
    G = nx.MultiDiGraph()
    G.graph["source"] = "synthetic"
    G.graph["crs"] = "EPSG:4326"

    for i in range(grid_n):
        for j in range(grid_n):
            lon = west + (east - west) * i / (grid_n - 1)
            lat = south + (north - south) * j / (grid_n - 1)
            G.add_node((i, j), x=lon, y=lat)

    def meters(lat1, lon1, lat2, lon2):
        dlat = (lat1 - lat2) * 111000
        dlon = (lon1 - lon2) * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
        return math.hypot(dlat, dlon)

    for i in range(grid_n):
        for j in range(grid_n):
            for di, dj in [(1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_n and 0 <= nj < grid_n:
                    a, b = (i, j), (ni, nj)
                    d = meters(G.nodes[a]["y"], G.nodes[a]["x"],
                               G.nodes[b]["y"], G.nodes[b]["x"])
                    G.add_edge(a, b, key=0, length=d)
                    G.add_edge(b, a, key=0, length=d)
    return G


def node_latlon(G, node_id):
    """OSM 노드 ID -> (lat, lon). osmnx 관례: y=위도, x=경도."""
    return G.nodes[node_id]["y"], G.nodes[node_id]["x"]


def latlon_to_cell(lat: float, lon: float, bbox):
    """(lat, lon) -> 격자 (row, col). bbox = (lat_min, lat_max, lon_min, lon_max) (ACLED 형식)."""
    N = settings.GRID_SIZE
    lat_min, lat_max, lon_min, lon_max = bbox
    row = int((lat_max - lat) / (lat_max - lat_min + 1e-9) * (N - 1))
    col = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * (N - 1))
    return min(max(row, 0), N - 1), min(max(col, 0), N - 1)


def find_nearest_node(G, lat: float, lon: float):
    """(lat, lon)에 가장 가까운 노드 ID 반환.

    osmnx.distance.nearest_nodes 가 합성(synthetic) 그래프의 튜플 노드ID에서
    오류를 내므로, 두 그래프 모두에서 안전하게 동작하는 간단한 직접 구현 사용.
    """
    import math
    best = None
    best_d = float("inf")
    for n, data in G.nodes(data=True):
        ny = data.get("y")
        nx = data.get("x")
        if ny is None or nx is None:
            continue
        # 제곱거리만 비교하면 충분 (sqrt 생략)
        dlat = (lat - ny) * 111000.0
        dlon = (lon - nx) * 111000.0 * math.cos(math.radians((lat + ny) / 2))
        d = dlat * dlat + dlon * dlon
        if d < best_d:
            best_d = d
            best = n
    return best


AMENITY_CACHE = Path("data/bakhmut_amenities.pkl")


def fetch_osm_amenities(bbox=BAKHMUT_BBOX):
    """Bakhmut 시내의 학교/병원/지역센터/진료소 등 '대피 가능 시설'을 OSM에서 가져옴.
    한 번 받으면 디스크 캐시(data/bakhmut_amenities.pkl) 사용. 실패 시 None.
    """
    if AMENITY_CACHE.exists():
        try:
            with open(AMENITY_CACHE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        import osmnx as ox
        print("[Amenity] Bakhmut 학교/병원 등 OSM 조회 중...")
        gdf = ox.features_from_bbox(bbox=bbox, tags={
            "amenity": ["school", "hospital", "community_centre", "clinic"]
        })
        result = []
        for _, row in gdf.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            if geom.geom_type == "Point":
                lat, lon = float(geom.y), float(geom.x)
            else:
                c = geom.centroid
                lat, lon = float(c.y), float(c.x)
            result.append({
                "name": str(row.get("name", "(이름없음)")),
                "amenity": str(row.get("amenity", "?")),
                "lat": lat, "lon": lon,
            })
        AMENITY_CACHE.parent.mkdir(exist_ok=True)
        with open(AMENITY_CACHE, "wb") as f:
            pickle.dump(result, f)
        print(f"[Amenity] {len(result)}곳 캐시 저장 -> {AMENITY_CACHE}")
        return result
    except Exception as e:
        print(f"[Amenity] OSM 조회 실패 ({type(e).__name__}): hardcoded 대피소 사용")
        return None


def pick_spread_shelters(amenities, n: int = 5, bbox=BAKHMUT_BBOX):
    """amenities 중 5곳을 지리적으로 분산되게 고른다 (북·남·동·서·중앙)."""
    if not amenities or len(amenities) < n:
        return None
    bb_w, bb_s, bb_e, bb_n = bbox
    cx, cy = (bb_w + bb_e) / 2, (bb_s + bb_n) / 2
    targets = [
        ("북부", (cx, bb_n - 0.005)),
        ("남부", (cx, bb_s + 0.005)),
        ("동부", (bb_e - 0.005, cy)),
        ("서부", (bb_w + 0.005, cy)),
        ("중앙", (cx, cy)),
    ][:n]
    used = set()
    out = []
    for kor, (tlon, tlat) in targets:
        best = None
        best_d = float("inf")
        for i, a in enumerate(amenities):
            if i in used:
                continue
            d = (a["lat"] - tlat) ** 2 + (a["lon"] - tlon) ** 2
            if d < best_d:
                best_d, best = d, i
        if best is None:
            continue
        a = amenities[best]
        used.add(best)
        raw = a.get("name", "")
        if raw and raw not in ("(이름없음)", "nan", "None") and not raw.startswith("nan"):
            display = f"{kor} - {raw[:18]}"
        else:
            amenity_kr = {"hospital": "병원", "school": "학교",
                          "clinic": "진료소", "community_centre": "주민센터"}.get(
                a.get("amenity", ""), "시설")
            display = f"{kor} - {amenity_kr}"
        out.append({"name": display, "lat": a["lat"], "lon": a["lon"],
                    "amenity": a["amenity"], "raw_name": raw})
    return out if len(out) == n else None


def shelters_to_osm_nodes(G, use_real: bool = True):
    """대피소 위치 -> 그래프 노드 ID.

    use_real=True 면 실제 OSM 학교/병원에서 5곳 자동 선정 (지리적 분산).
    실패 시 SHELTERS_LATLON (하드코드) 으로 폴백.
    """
    if use_real:
        amenities = fetch_osm_amenities()
        picked = pick_spread_shelters(amenities) if amenities else None
        if picked:
            print(f"[Shelter] 실제 OSM 시설 {len(picked)}곳 사용:")
            out = []
            for sh in picked:
                node = find_nearest_node(G, sh["lat"], sh["lon"])
                out.append({"name": sh["name"], "lat": sh["lat"],
                            "lon": sh["lon"], "node": node})
                print(f"          - {sh['name']}")
            return out

    print("[Shelter] hardcoded SHELTERS_LATLON 사용 (폴백)")
    out = []
    for name, (lat, lon) in SHELTERS_LATLON:
        node = find_nearest_node(G, lat, lon)
        out.append({"name": name, "lat": lat, "lon": lon, "node": node})
    return out
