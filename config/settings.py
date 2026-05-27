"""config/settings.py

시스템 전체에서 함께 쓰는 기본 설정값을 한곳에 모았습니다.
숫자를 바꾸고 싶을 때 코드 여기저기를 찾을 필요 없이, 이 파일만 고치면 됩니다.
"""

# --- 도시 격자(grid) 설정 ---
GRID_SIZE = 200        # 도시를 200 x 200 칸으로 나눕니다.
CELL_PIXELS = 4        # 한 칸을 화면에서 4 x 4 픽셀로 그립니다.

# 창 크기는 위 값에서 자동으로 계산됩니다. (200 * 4 = 800)
WINDOW_WIDTH = GRID_SIZE * CELL_PIXELS
WINDOW_HEIGHT = GRID_SIZE * CELL_PIXELS

# --- 칸(cell) 종류 --- (Phase 1에서 사용)
CELL_EMPTY = 0      # 공터
CELL_BUILDING = 1   # 건물
CELL_ROAD = 2       # 도로
CELL_SHELTER = 3    # 대피소

CELL_COLORS = {
    CELL_EMPTY: (250, 250, 250),     # 거의 흰색 (공터)
    CELL_BUILDING: (210, 212, 218),  # 옅은 회색 (위험 색이 위에서 잘 보이도록)
    CELL_ROAD: (232, 226, 190),      # 옅은 노란색 (도로)
    CELL_SHELTER: (40, 175, 90),     # 초록색 (대피소 — 또렷하게)
}

# --- 드론 설정 --- (Phase 2에서 사용)
NUM_DRONES = 10            # 드론 대수
DRONE_VISION_RADIUS = 10   # 드론 시야 반경 (칸 단위)
DRONE_BATTERY_DRAIN = 1.0  # 1 스텝(=1분)당 배터리 1% 감소

# --- Bee voting (꿀벌 투표) 설정 --- (Phase 2에서 사용)
DRONE_CONFIDENCE = 0.1     # 드론 1대가 위험을 보고할 때의 신뢰도
NEWS_CONFIDENCE = 0.3      # 뉴스(LLM)가 보고할 때의 신뢰도 (뉴스를 더 신뢰함)
QUORUM_THRESHOLD = 0.5     # 합산 신뢰도가 이 값을 넘어야 '확인된 위험'으로 인정

# --- 시뮬레이션 ---
TIME_SCALE = 10            # 시간 가속 배율 (실제보다 10배 빠르게)
FPS = 30                   # 화면 갱신 속도 (초당 프레임)
