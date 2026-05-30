"""routing/risk_astar.py

위험 가중치를 더한 A* 알고리즘 — 시민의 이동 능력에 맞춤화.

핵심 수식:
  edge_cost = (length / speed_mps) * (1 + alpha * avg_risk)
              └─ 통과 시간(초) ─┘   └─ 위험 가중치 ─┘

  -> "시간 × 위험" 즉 **위험에 노출되는 시간**을 최소화하는 라우팅.
     - 느린 사람일수록 거리 페널티가 자연스럽게 커짐 -> 짧은 경로 선호
     - 위험 회피적 사람은 alpha 가 커서 위험한 칸을 더 강하게 회피

추가 제약:
  - max_distance_m : 각 프로필별 도달 한계 거리. 이보다 먼 대피소는 후보에서 제외
  - 도로 폐쇄(closed_edges) : 해당 간선 비용 = 무한대

사용자 분류는 UNHCR/적십자 대피 가이드에서 사용하는 '이동 능력' 기준:
  - 건강한 성인 도보 / 어린이 동반 / 노약자·이동약자 / 차량 이동
"""

import math
import networkx as nx

from routing.graph_builder import node_latlon, latlon_to_cell, find_nearest_node


# 실제 시민 분류 (속도와 한계 거리는 보행/이동 연구 데이터 기반)
USER_PROFILES = {
    "healthy_adult": {
        "label": "건강한 성인 (도보)",
        "speed_mps": 1.4,        # 일반 보행 속도
        "max_distance_m": 8000,  # 8km — 도시 규모 대피 한계
        "alpha": 6.0,
    },
    "with_kids": {
        "label": "어린이 동반 (도보)",
        "speed_mps": 0.9,        # 어린이와 함께면 절반 정도 속도
        "max_distance_m": 3500,  # 3.5km — 어린이 체력 한계 (~65분 도보)
        "alpha": 8.0,
    },
    "elderly_or_disabled": {
        "label": "노약자/이동약자",
        "speed_mps": 0.8,        # 가장 느림
        "max_distance_m": 1500,  # 1.5km — 휠체어/보조구 보행 한계 (~31분, FEMA 기준)
        "alpha": 10.0,
    },
    "by_vehicle": {
        "label": "차량 이동",
        "speed_mps": 8.0,        # 약 28 km/h (도심 대피 평균)
        "max_distance_m": 30000, # 사실상 무제한
        "alpha": 3.0,            # 빠르게 통과 -> 위험 가중치 작아도 됨
    },
}

# 하위 호환: 기존 코드가 FAMILY_PROFILES 를 쓸 수도 있어서 별칭 유지
FAMILY_PROFILES = USER_PROFILES


def _haversine_meters(lat1, lon1, lat2, lon2):
    """두 점 간 대략적 직선 거리(미터). A* 휴리스틱 + 거리 계산용."""
    dlat = (lat1 - lat2) * 111000
    dlon = (lon1 - lon2) * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


def _edge_length(edge_data):
    """MultiDiGraph 간선 데이터에서 length 추출 (없으면 1.0)."""
    if not edge_data:
        return 1.0
    if isinstance(edge_data, dict) and edge_data:
        first = next(iter(edge_data.values()))
        if isinstance(first, dict):
            return float(first.get("length", 1.0))
    return 1.0


def make_risk_weight_function(G, risk_map, bbox, speed_mps=1.4, alpha=8.0,
                              closed_edges=None):
    """A* 가중치 함수. cost = (length/speed) * (1 + alpha * avg_risk)."""
    closed = set(closed_edges or [])

    def weight(u, v, data):
        if (u, v) in closed or (v, u) in closed:
            return float("inf")
        length = float(data.get("length", 1.0))
        travel_time = length / speed_mps
        lat_u, lon_u = node_latlon(G, u)
        lat_v, lon_v = node_latlon(G, v)
        r_u, c_u = latlon_to_cell(lat_u, lon_u, bbox)
        r_v, c_v = latlon_to_cell(lat_v, lon_v, bbox)
        risk = (float(risk_map[r_u, c_u]) + float(risk_map[r_v, c_v])) / 2.0
        return travel_time * (1.0 + alpha * risk)

    return weight


def make_heuristic(G, speed_mps=1.4):
    """A* 휴리스틱: 직선거리 / 속도 = 최소 가능 통과 시간."""
    def h(u, v):
        lat_u, lon_u = node_latlon(G, u)
        lat_v, lon_v = node_latlon(G, v)
        return _haversine_meters(lat_u, lon_u, lat_v, lon_v) / speed_mps
    return h


def compute_route(G, start_node, goal_node, risk_map, bbox,
                  speed_mps=1.4, alpha=8.0, closed_edges=None):
    """A* 로 시간·위험 가중 경로 계산."""
    weight_fn = make_risk_weight_function(G, risk_map, bbox,
                                          speed_mps=speed_mps, alpha=alpha,
                                          closed_edges=closed_edges)
    h = make_heuristic(G, speed_mps=speed_mps)
    try:
        return nx.astar_path(G, start_node, goal_node, heuristic=h, weight=weight_fn)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _path_distance_m(G, path):
    """경로의 총 실거리(미터)."""
    total = 0.0
    for u, v in zip(path, path[1:]):
        total += _edge_length(G.get_edge_data(u, v))
    return total


def find_best_shelter(G, start_node, shelter_nodes_info, risk_map, bbox,
                      speed_mps=1.4, alpha=8.0, max_distance_m=8000,
                      closed_edges=None):
    """후보 대피소들 중 (1) 최대 거리 안에 있고 (2) 시간·위험 비용이 최소인 곳 선택.

    반환: (best_shelter_info, best_weighted_cost, candidates_summary)
    """
    weight_fn = make_risk_weight_function(G, risk_map, bbox,
                                          speed_mps=speed_mps, alpha=alpha,
                                          closed_edges=closed_edges)
    best = None
    best_cost = float("inf")
    summary = []
    for s in shelter_nodes_info:
        try:
            # 먼저 실거리(미터) 체크 — 너무 멀면 후보 제외
            raw_path = nx.shortest_path(G, start_node, s["node"], weight="length")
            raw_distance = _path_distance_m(G, raw_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            summary.append({"name": s["name"], "ok": False, "reason": "도달 불가"})
            continue

        if raw_distance > max_distance_m:
            summary.append({"name": s["name"], "ok": False,
                            "distance_m": raw_distance,
                            "reason": f"체력 한계 초과 ({raw_distance:.0f}m > {max_distance_m}m)"})
            continue

        try:
            cost = nx.shortest_path_length(G, start_node, s["node"], weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            summary.append({"name": s["name"], "ok": False, "reason": "위험 가중 경로 없음"})
            continue

        summary.append({"name": s["name"], "ok": True,
                        "distance_m": raw_distance, "weighted_cost_s": cost})
        if cost < best_cost:
            best_cost = cost
            best = s
    return best, best_cost, summary


def route_for_civilian(G, start_latlon, shelter_nodes_info, risk_map, bbox,
                       profile_key="healthy_adult", closed_edges=None):
    """시민 위치(lat,lon) + 사용자 분류 -> 최적 대피소 + 경로 + 통계 + 후보 요약."""
    if profile_key not in USER_PROFILES:
        raise ValueError(f"알 수 없는 분류: {profile_key}")
    profile = USER_PROFILES[profile_key]

    start_node = find_nearest_node(G, start_latlon[0], start_latlon[1])

    shelter, weighted_cost, summary = find_best_shelter(
        G, start_node, shelter_nodes_info, risk_map, bbox,
        speed_mps=profile["speed_mps"],
        alpha=profile["alpha"],
        max_distance_m=profile["max_distance_m"],
        closed_edges=closed_edges)
    if shelter is None:
        return {
            "profile_key": profile_key,
            "profile_label": profile["label"],
            "shelter": None,
            "candidates_summary": summary,
            "reason": "도달 가능한 대피소 없음",
            "max_distance_m": profile["max_distance_m"],
            "speed_mps": profile["speed_mps"],
            "alpha": profile["alpha"],
        }

    path = compute_route(G, start_node, shelter["node"], risk_map, bbox,
                         speed_mps=profile["speed_mps"],
                         alpha=profile["alpha"],
                         closed_edges=closed_edges)
    if path is None:
        return {
            "profile_key": profile_key,
            "profile_label": profile["label"],
            "shelter": shelter,
            "candidates_summary": summary,
            "reason": "경로 계산 실패",
        }

    # 경로 통계
    coords = [node_latlon(G, n) for n in path]
    total_meters = 0.0
    weighted_risk_sum = 0.0
    for u, v in zip(path, path[1:]):
        length = _edge_length(G.get_edge_data(u, v))
        total_meters += length
        lat_u, lon_u = node_latlon(G, u)
        lat_v, lon_v = node_latlon(G, v)
        r_u, c_u = latlon_to_cell(lat_u, lon_u, bbox)
        r_v, c_v = latlon_to_cell(lat_v, lon_v, bbox)
        risk = (float(risk_map[r_u, c_u]) + float(risk_map[r_v, c_v])) / 2.0
        weighted_risk_sum += risk * length
    avg_risk = weighted_risk_sum / total_meters if total_meters > 0 else 0.0
    travel_time_s = total_meters / profile["speed_mps"]

    return {
        "profile_key": profile_key,
        "profile_label": profile["label"],
        "path_nodes": path,
        "path_coords": coords,
        "start_node": start_node,
        "start_latlon": start_latlon,
        "shelter": shelter,
        "weighted_cost_s": weighted_cost,
        "total_meters": total_meters,
        "travel_time_s": travel_time_s,
        "travel_time_min": travel_time_s / 60.0,
        "avg_risk": avg_risk,
        "speed_mps": profile["speed_mps"],
        "max_distance_m": profile["max_distance_m"],
        "alpha": profile["alpha"],
        "candidates_summary": summary,
    }
