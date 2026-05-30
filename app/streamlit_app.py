"""app/streamlit_app.py

🐝 Hive Evac — Bakhmut 시민 대피 앱 (모바일 친화 + 실시간 시연).

레이아웃:
  [좌] 시스템 뷰: 드론 + 위험 지도가 실시간으로 변하는 모습 (3초마다 자동 진행)
  [우] 시민 뷰  : 깔끔한 OSM 지도 위에 추천 경로, 대피소, 위험 영역
  [하] 통계 + PDF 다운로드

위치 입력:
  - 주소/장소 검색 (Nominatim 지오코딩)
  - 지도의 GPS 버튼 (브라우저 위치 정보)
  - 지도 클릭
  - 위도/경도 직접 입력 (고급)

실행: streamlit run app/streamlit_app.py
"""

# Streamlit이 app/ 디렉토리를 기준으로 실행하므로 프로젝트 루트를 sys.path 에 추가
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import io

import folium
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw
from folium.plugins import HeatMap, LocateControl
from streamlit_folium import st_folium

from ai.bee_voting import BeeVoting
from app.print_mode import generate_route_pdf
from hazard_modules.conflict import conflict_module
from routing.graph_builder import (BAKHMUT_BBOX, load_or_build_bakhmut_graph,
                                    shelters_to_osm_nodes)
from routing.risk_astar import route_for_civilian, USER_PROFILES
from simulation.run import build_world
from simulation.visualize import compose_rgb

# ===========================================================
# 페이지 설정
# ===========================================================
st.set_page_config(
    page_title="Hive Evac — Bakhmut",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================
# 세계 한 번만 초기화 (Streamlit singleton 캐시)
# ===========================================================
@st.cache_resource(show_spinner="🐝 시뮬레이션 초기화 중... (한 번만, 약 30초)")
def init_world():
    city, _, field, swarm, bee, bbox = build_world(
        conflict_module, use_news=True, seed=42)
    # 초기 60 스텝 진행 -> 뭐라도 보이게
    for _ in range(60):
        field.step()
        swarm.step(field.field, bee)
    G = load_or_build_bakhmut_graph()
    shelter_info = shelters_to_osm_nodes(G)
    return {
        "city": city, "field": field, "swarm": swarm, "bee": bee,
        "bbox": bbox, "G": G, "shelters": shelter_info,
    }


@st.cache_resource(show_spinner="📊 비교용 (뉴스 없는) 세계 초기화 중...")
def init_world_nonews():
    """3분할 비교용: 같은 시드, 같은 ACLED·CA, 다만 LLM 뉴스 융합만 OFF."""
    city, _, field, swarm, bee, bbox = build_world(
        conflict_module, use_news=False, seed=42)
    for _ in range(60):
        field.step()
        swarm.step(field.field, bee)
    return {"city": city, "field": field, "swarm": swarm, "bee": bee, "bbox": bbox}


def maybe_advance_worlds(world_main, world_alt=None, min_interval_s: float = 2.5):
    """3초마다 두 세계를 5 스텝 진행 (auto_play 켜졌을 때만).
    여러 fragment 에서 호출돼도 시간 게이트로 한 번만 진행됨 (idempotent)."""
    import time
    if not ss.auto_play:
        return False
    now = time.time()
    last = ss.get("_last_advance_t", 0)
    if now - last < min_interval_s:
        return False
    ss._last_advance_t = now
    for _ in range(5):
        world_main["field"].step()
        world_main["swarm"].step(world_main["field"].field, world_main["bee"])
        if world_alt is not None:
            world_alt["field"].step()
            world_alt["swarm"].step(world_alt["field"].field, world_alt["bee"])
    # 활성 경로 재계산 (위험 지도가 변했으니)
    if ss.route and ss.route.get("path_coords"):
        ss.route = route_for_civilian(
            world_main["G"], (ss.lat, ss.lon), world_main["shelters"],
            world_main["bee"].confirmed_risk(), world_main["bbox"],
            profile_key=ss.get("profile_key", "healthy_adult"))
    return True


def render_compare_panel(world, field_array, drones=None, scale: int = 3) -> Image.Image:
    """3분할 비교용 단일 패널 이미지."""
    rgb = compose_rgb(world["city"], field_array, conflict_module)
    N = rgb.shape[0]
    img = Image.fromarray(rgb).resize((N * scale, N * scale), Image.NEAREST)
    if drones:
        draw = ImageDraw.Draw(img, mode="RGBA")
        for d in drones:
            x = d.pos[1] * scale
            y = d.pos[0] * scale
            v = d.vision * scale
            draw.ellipse((x - v, y - v, x + v, y + v),
                         outline=(60, 130, 255, 200), width=1)
            r_dot = max(3, scale)
            draw.ellipse((x - r_dot, y - r_dot, x + r_dot, y + r_dot),
                         fill=(20, 60, 230, 255))
    return img


def _recall_at(field_obj, bee, thresh=0.15):
    true_cells = field_obj.field.max(axis=2) > thresh
    conf_cells = bee.confirmed_mask().any(axis=2)
    nt = int(true_cells.sum())
    if nt == 0:
        return 0.0
    tp = int((true_cells & conf_cells).sum())
    return tp / nt


# ===========================================================
# 시스템 뷰 (PIL — 빠른 픽셀 렌더링, 3초마다 갱신해도 부담 없음)
# ===========================================================
def render_system_image(world, scale: int = 3, box_bounds=None) -> Image.Image:
    """드론 + 위험 지도 전체 뷰. 시민 앱이 보여주는 영역을 박스로 강조.

    box_bounds : (west, south, east, north) tuple. 오른쪽 OSM 지도의
                현재 표시 영역. None 이면 BAKHMUT_BBOX 사용 (초기값).
    """
    from routing.graph_builder import latlon_to_cell

    rgb = compose_rgb(world["city"], world["field"].field, conflict_module)
    bbox = world["bbox"]
    lat_min, lat_max, lon_min, lon_max = bbox
    N = rgb.shape[0]

    img = Image.fromarray(rgb).resize((N * scale, N * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img, mode="RGBA")

    # 1) 드론 (전체 격자)
    for d in world["swarm"].drones:
        x = d.pos[1] * scale
        y = d.pos[0] * scale
        v = d.vision * scale
        draw.ellipse((x - v, y - v, x + v, y + v),
                     outline=(60, 130, 255, 200), width=1)
        r_dot = max(3, scale)
        draw.ellipse((x - r_dot, y - r_dot, x + r_dot, y + r_dot),
                     fill=(20, 60, 230, 255))

    # 2) "시민 앱 뷰" 박스 (드론 위에 그려서 가려지지 않게)
    bb = box_bounds if box_bounds else BAKHMUT_BBOX
    bb_w, bb_s, bb_e, bb_n = bb
    # 격자 범위로 클램프 (지도가 광역으로 줌아웃됐을 때 박스가 이미지 밖으로 안 나가도록)
    bb_w_c = max(lon_min, min(lon_max, bb_w))
    bb_e_c = max(lon_min, min(lon_max, bb_e))
    bb_s_c = max(lat_min, min(lat_max, bb_s))
    bb_n_c = max(lat_min, min(lat_max, bb_n))
    r_nw, c_nw = latlon_to_cell(bb_n_c, bb_w_c, bbox)
    r_se, c_se = latlon_to_cell(bb_s_c, bb_e_c, bbox)
    x1, y1 = c_nw * scale, r_nw * scale
    x2, y2 = c_se * scale, r_se * scale

    # 강한 시각 효과: 흰 후광 + 형광 magenta 두꺼운 테두리 + 옅은 채움
    # (위험의 빨강·노란 도로·회색 건물 사이에서 가장 잘 보이는 조합)
    draw.rectangle((x1 - 3, y1 - 3, x2 + 3, y2 + 3),
                   outline=(255, 255, 255, 230), width=2)   # 흰 후광
    draw.rectangle((x1, y1, x2, y2),
                   fill=(255, 20, 200, 50),                 # 옅은 magenta 채움
                   outline=(255, 0, 180, 255), width=5)     # 진한 magenta 테두리
    return img


# ===========================================================
# 주소 -> 좌표 (Nominatim, Bakhmut 영역으로 제한)
# ===========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_bakhmut(query: str):
    if not query.strip():
        return None
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{query}, Bakhmut, Ukraine",
                "format": "json",
                "limit": 1,
                "viewbox": "37.94,48.62,38.04,48.56",   # left,top,right,bottom
                "bounded": 1,
            },
            headers={"User-Agent": "hive-evac-kcf/0.1"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display_name": data[0]["display_name"],
            }
    except Exception as e:
        return {"error": str(e)}
    return None


# ===========================================================
# 시민용 Folium 지도
# ===========================================================
def render_civilian_map(world, route, current_lat, current_lon,
                         center=None, zoom=14):
    bbox = world["bbox"]
    bb = BAKHMUT_BBOX  # (west, south, east, north)
    if center is None:
        center = [(bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2]
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )
    # GPS 버튼 (브라우저 위치 정보)
    LocateControl(auto_start=False,
                  position="topleft",
                  strings={"title": "내 위치 찾기 (GPS)"}).add_to(m)

    # 위험 히트맵 (간단/옅게)
    risk_map = world["bee"].confirmed_risk()
    N = risk_map.shape[0]
    lat_min, lat_max, lon_min, lon_max = bbox
    heat_points = []
    for r in range(0, N, 3):
        for c in range(0, N, 3):
            v = float(risk_map[r, c])
            if v > 0.3:
                lat = lat_max - r / (N - 1) * (lat_max - lat_min)
                lon = lon_min + c / (N - 1) * (lon_max - lon_min)
                heat_points.append([lat, lon, v])
    if heat_points:
        HeatMap(heat_points, radius=12, blur=18, min_opacity=0.35,
                gradient={0.3: "yellow", 0.6: "orange", 0.9: "red"}).add_to(m)

    # 대피소 마커
    for s in world["shelters"]:
        folium.Marker(
            location=[s["lat"], s["lon"]],
            popup=f"🏥 {s['name']}",
            tooltip=s["name"],
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(m)

    # 현재 위치
    folium.Marker(
        location=[current_lat, current_lon],
        popup="📍 현재 위치",
        tooltip="현재 위치",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)

    # 경로
    if route and route.get("path_coords"):
        folium.PolyLine(
            locations=route["path_coords"],
            weight=6, color="#00aaff", opacity=0.9,
            tooltip=f"{route['total_meters']:.0f}m, "
                    f"{route['travel_time_min']:.1f}분",
        ).add_to(m)
        sh = route["shelter"]
        folium.CircleMarker(
            location=[sh["lat"], sh["lon"]],
            radius=14, color="#16a34a", fill=True, fill_opacity=0.4,
            tooltip=f"→ {sh['name']}",
        ).add_to(m)

    return m


# ===========================================================
# 본격 시작
# ===========================================================
world = init_world()
world_no_news = init_world_nonews()    # tab 2 비교용 (뉴스 OFF 동일 세계)
ss = st.session_state
if "lat" not in ss:
    ss.lat = 48.605
if "lon" not in ss:
    ss.lon = 37.978
if "route" not in ss:
    ss.route = None

# 자동 진행 컨트롤
if "auto_play" not in ss:
    ss.auto_play = False

# ===========================================================
# 시스템 뷰 fragment — 3초마다 이 부분만 새로고침.
# 지도(오른쪽)는 fragment 밖이라 사용자 드래그/줌이 보존됩니다.
# ===========================================================
@st.fragment(run_every="3s")
def system_view_panel():
    maybe_advance_worlds(world, world_no_news)   # 두 세계 모두 한 번에 진행

    st.subheader("🛰️ 시스템 뷰 — 드론 + 위험")
    status = "🟢 자동 진행 중" if ss.auto_play else "⏸️ 정지 (헤더 토글로 시작)"
    st.caption(f"**파란 점** = 드론 · **파란 원** = 시야 · **빨강** = 확인된 위험 · "
               f"💗 **분홍 박스 = 오른쪽 시민 앱이 현재 보고 있는 영역** (지도 확대·이동 시 자동 변경) · "
               f"step **{world['field'].step_count}** · {status}")
    sys_img = render_system_image(world, box_bounds=ss.get("map_bounds"))
    st.image(sys_img, use_container_width=True)
    # 미니 통계 (fragment 안이라 자동 갱신됨)
    if ss.route and ss.route.get("shelter") and ss.route.get("path_coords"):
        r = ss.route
        c1, c2, c3 = st.columns(3)
        c1.metric("거리", f"{r['total_meters']:.0f} m")
        c2.metric("시간", f"{r['travel_time_min']:.1f} 분")
        c3.metric("평균 위험", f"{r['avg_risk']:.2f}")


@st.fragment(run_every="3s")
def three_panel_view():
    """탭 2: 정답 | 드론만 | 드론+뉴스 — 알고리즘이 일하는 모습을 보여주는 데모."""
    maybe_advance_worlds(world, world_no_news)

    st.subheader("🛰️ 3분할 시스템 시연 — 알고리즘 동작 비교")
    st.caption("같은 시각, 같은 위험. 가운데(드론만) vs 오른쪽(드론+뉴스) 의 채워짐 차이가 "
               "**LLM 뉴스 융합의 가치**입니다. 자동 진행을 켜면 실시간 변화 관찰 가능.")

    rec_no = _recall_at(world_no_news["field"], world_no_news["bee"])
    rec_yes = _recall_at(world["field"], world["bee"])
    gap = (rec_yes - rec_no) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**① 실제 위험 (정답)**")
        st.caption("드론·시스템과 무관한 시뮬레이션 ground truth")
        img1 = render_compare_panel(world, world["field"].field, drones=None)
        st.image(img1, use_container_width=True)
    with c2:
        st.markdown(f"**② 드론만** — 재현율 **{rec_no:.2f}**")
        st.caption("Phase 2: 드론 정찰 + 꿀벌 투표만 사용")
        img2 = render_compare_panel(world_no_news,
                                     world_no_news["bee"].confirmed_field(),
                                     drones=world_no_news["swarm"].drones)
        st.image(img2, use_container_width=True)
    with c3:
        sign = "+" if gap >= 0 else ""
        st.markdown(f"**③ 드론 + 뉴스** — 재현율 **{rec_yes:.2f}** "
                    f"({sign}{gap:.1f}%p)")
        st.caption("Phase 3: LLM 뉴스 융합 추가")
        img3 = render_compare_panel(world,
                                     world["bee"].confirmed_field(),
                                     drones=world["swarm"].drones)
        st.image(img3, use_container_width=True)

    st.caption(f"step **{world['field'].step_count}** · "
               f"드론 {len(world['swarm'].drones)}대 정찰 중")


# ===========================================================
# 헤더 + 자동 진행 토글
# ===========================================================
st.title("🐝 Hive Evac — Bakhmut 시민 대피")

h1, h2, h3 = st.columns([2, 1, 1])
h1.caption(f"⏩ **60배 가속 데모** · 1 step = 1 sim minute · "
           f"드론 {len(world['swarm'].drones)}대 정찰 중")
new_auto = h2.toggle("▶️ 자동 진행 (3s/5step)",
                     value=ss.auto_play, key="auto_play_toggle")
if new_auto != ss.auto_play:
    ss.auto_play = new_auto
    st.rerun()
h3.caption("위험 = 꿀벌 투표 확인됨 · 경로 = 위험 가중 A*")

# ===========================================================
# 탭 메뉴: [📱 시민 앱] / [🛰️ 3분할 시연]
# ===========================================================
tab1, tab2 = st.tabs(["📱 시민 앱", "🛰️ 3분할 시스템 시연"])

with tab1:
    sys_col, civ_col = st.columns([1, 1])

    with sys_col:
        system_view_panel()

    with civ_col:
        st.subheader("📱 시민 뷰 — 안전 경로")
        st.caption("**파란 핀** = 내 위치 · **녹색 집** = 대피소 · **하늘색 선** = 추천 경로")
        fmap = render_civilian_map(
            world, ss.route, ss.lat, ss.lon,
            center=ss.get("map_center"),
            zoom=ss.get("map_zoom", 14),
        )
        map_state = st_folium(
            fmap, height=520, width=None,
            returned_objects=["last_clicked", "center", "zoom", "bounds"],
            key="civ_map",
        )

        # 드래그/줌·표시 영역 상태 저장
        # bounds 는 시스템 뷰의 분홍 박스가 따라가는 기준
        if map_state:
            if map_state.get("center"):
                ss.map_center = [map_state["center"]["lat"],
                                  map_state["center"]["lng"]]
            if map_state.get("zoom") is not None:
                ss.map_zoom = int(map_state["zoom"])
            b = map_state.get("bounds")
            if b and b.get("_southWest") and b.get("_northEast"):
                ss.map_bounds = (
                    b["_southWest"]["lng"], b["_southWest"]["lat"],
                    b["_northEast"]["lng"], b["_northEast"]["lat"],
                )

        # 지도 클릭 -> 위치 업데이트
        if map_state and map_state.get("last_clicked"):
            clk = map_state["last_clicked"]
            nlat, nlon = float(clk["lat"]), float(clk["lng"])
            if (BAKHMUT_BBOX[1] <= nlat <= BAKHMUT_BBOX[3]
                    and BAKHMUT_BBOX[0] <= nlon <= BAKHMUT_BBOX[2]):
                if (abs(nlat - ss.lat) > 1e-5 or abs(nlon - ss.lon) > 1e-5):
                    ss.lat = nlat
                    ss.lon = nlon
                    st.rerun()

    # ----- 탭 1 하단: 경로 통계 + PDF -----
    st.divider()
    st.subheader("📊 경로 정보")
    _route = ss.route
    if _route is None:
        st.info("👈 사이드바에서 분류·위치를 정한 뒤 **\"안전 경로 계산\"** 을 누르세요.")
    elif _route.get("shelter") is None or not _route.get("path_coords"):
        st.error(
            f"⚠️ **{_route.get('profile_label')}** 분류로는 "
            f"근처 {_route.get('max_distance_m', 0)}m 안에 도달할 대피소가 없습니다."
        )
        st.markdown("### 🚨 시스템 권고: **차량 지원 / 이웃 도움 필요**")
        st.caption("이 메시지가 가족·적십자·이웃에게 자동 통보되는 기능은 향후 추가 가능.")
    else:
        sh = _route["shelter"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏥 대피소", sh["name"])
        c2.metric("📏 거리", f"{_route['total_meters']:.0f} m")
        c3.metric("⏱️ 예상 시간", f"{_route['travel_time_min']:.1f} 분")
        c4.metric("⚠️ 평균 위험도", f"{_route['avg_risk']:.2f}")
        st.markdown("### 🖨️ 통신 두절 대비")
        st.caption("핸드폰 꺼져도 종이 한 장으로 안내 가능 (QR + 한글 경로)")
        _pdf = generate_route_pdf(_route, ss.lat, ss.lon)
        st.download_button(
            "📥 안전 경로 PDF 다운로드",
            data=_pdf,
            file_name=f"hive_evac_{sh['name']}.pdf",
            mime="application/pdf",
            type="primary",
        )

with tab2:
    three_panel_view()

# ===========================================================
# 사이드바: 시민 입력
# ===========================================================
with st.sidebar:
    st.header("⚙️ 시민 정보")

    profile_key = st.selectbox(
        "이동 능력 분류",
        list(USER_PROFILES.keys()),
        format_func=lambda k: USER_PROFILES[k]["label"],
        key="profile_key",
    )
    p = USER_PROFILES[profile_key]
    st.caption(
        f"속도 {p['speed_mps']} m/s · 최대 {p['max_distance_m']/1000:.1f} km · "
        f"α={p['alpha']}"
    )

    st.divider()
    st.subheader("📍 내 위치")

    addr_input = st.text_input(
        "주소/장소로 검색",
        placeholder="예: 시청, 학교, 병원 이름",
        help="Bakhmut 영역 내의 장소만 검색됩니다.",
    )
    col_a, col_b = st.columns([1, 1])
    if col_a.button("🔍 검색", use_container_width=True):
        if addr_input.strip():
            with st.spinner("주소 변환 중..."):
                result = geocode_bakhmut(addr_input)
            if result and "error" not in result:
                ss.lat = result["lat"]
                ss.lon = result["lon"]
                st.success(f"✓ {result['display_name'][:50]}")
                st.rerun()
            elif result and "error" in result:
                st.error(f"검색 실패: {result['error']}")
            else:
                st.warning(f"'{addr_input}' 위치를 찾을 수 없습니다.")
    col_b.caption("📡 지도 좌측 상단 GPS 버튼도 사용 가능")

    st.caption(f"현재 위치: **{ss.lat:.4f}, {ss.lon:.4f}**")

    with st.expander("⚙️ 좌표 직접 입력 (고급)"):
        new_lat = st.number_input("위도", value=float(ss.lat),
                                   min_value=48.56, max_value=48.62,
                                   step=0.001, format="%.4f")
        new_lon = st.number_input("경도", value=float(ss.lon),
                                   min_value=37.94, max_value=38.04,
                                   step=0.001, format="%.4f")
        if abs(new_lat - ss.lat) > 1e-6 or abs(new_lon - ss.lon) > 1e-6:
            ss.lat = new_lat
            ss.lon = new_lon

    st.divider()
    if st.button("🧭 안전 경로 계산", type="primary", use_container_width=True):
        ss.route = route_for_civilian(
            world["G"], (ss.lat, ss.lon), world["shelters"],
            world["bee"].confirmed_risk(), world["bbox"],
            profile_key=profile_key)
        st.rerun()

# 푸터
with st.expander("ℹ️ 어떻게 작동하나요?"):
    st.markdown("""
    이 화면에서 보고 있는 것:

    1. **드론 군집** (왼쪽, 파란 점) — 10대가 도시를 정찰하며 위험을 탐지
    2. **꿀벌 투표** — 여러 보고를 종합해 오탐 거름 (정밀도 ~0.87)
    3. **LLM 뉴스 융합** — Anthropic Claude Haiku 4.5 가 영문 뉴스에서 위험 추출 (재현율 +30%)
    4. **위험 가중 A*** — 시민의 이동 능력에 맞춘 가장 안전한 경로
       (비용 = 거리/속도 × (1+α×위험), 도달 한계 거리 제약)
    5. **자동 진행 ON** 으로 두면 3초마다 시뮬레이션이 5 스텝씩 진행 — 위험·경로가 변하는 모습 관찰 가능.

    **시뮬레이션 시간 스케일**: 1 step = 1 분 (현장 1분당 시뮬 1스텝).
    자동 진행 시 3초마다 5스텝 = **60배 가속**.
    """)

with st.expander("📚 출처 (Data sources & References)"):
    st.markdown("""
    **데이터 출처**
    - **ACLED** (Armed Conflict Location & Event Data Project) — Bakhmut 사건 5,879건 (2022.08–2023.05). 출처: [acleddata.com](https://acleddata.com)
    - **NewsAPI** — 영문 실시간 뉴스. [newsapi.org](https://newsapi.org)
    - **OpenStreetMap** — Bakhmut 도로망 + 학교·병원 시설 (실제 좌표). [openstreetmap.org](https://openstreetmap.org)

    **알고리즘 출처**
    - **Seeley, T.D.** (2010). *Honeybee Democracy*. Princeton University Press. → quorum sensing 개념
    - **Karafyllidis, I. & Thanailakis, A.** (1997). "A model for predicting forest fire spreading using cellular automata." *Ecological Modelling* 99: 87–97. → CA 확산 모델
    - **Anthropic Claude API** (Haiku 4.5) — 뉴스 위험 추출

    **시민 분류 파라미터 출처** (속도 · 거리 한계)
    - Knoblauch, Pietrucha & Nitzburg (1996). "Field studies of pedestrian walking speed and start-up time." *Transp. Res. Record* 1538. → 성인 1.2–1.5 m/s
    - Bohannon, R.W. (1997). "Comfortable and maximum walking speed of adults aged 20–79 years." *Age & Ageing* 26(1): 15–19. → 노년층 0.8–1.2 m/s
    - Daamen & Hoogendoorn (2007). "Free speed distributions for pedestrian traffic." *TRB Annual Meeting* → 가족 그룹 평균
    - **FEMA P-361** (2021). *Safe Rooms for Tornadoes and Hurricanes* → 노약자 도달 거리 가이드
    - Japan Cabinet Office (2017). *Tsunami Evacuation Guidelines* → 시민별 권장 보행 거리
    - **NCHRP Report 740** (2013). *Urban Evacuation Modeling* → 차량 대피 평균 속도
    """)
