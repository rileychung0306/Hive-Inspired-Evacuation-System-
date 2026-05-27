"""simulation/city.py

가상 Bakhmut 도시를 200x200 격자로 만듭니다.
각 칸은 공터/건물/도로/대피소 중 하나입니다.

지금은 도로와 건물을 규칙적으로 만들어 넣습니다(절차적 생성).
Phase 4에서 실제 OpenStreetMap의 Bakhmut 도로망으로 교체할 예정입니다.
"""

import numpy as np
from config import settings


def build_city(seed: int = 42):
    """도시 격자와 대피소 위치 목록을 만들어 돌려줍니다.

    seed를 고정하면 매번 똑같은 도시가 나옵니다(발표·실험 재현성).
    """
    N = settings.GRID_SIZE
    rng = np.random.default_rng(seed)
    grid = np.full((N, N), settings.CELL_EMPTY, dtype=np.int8)

    # 1) 도로: 일정 간격(block)으로 가로/세로 길을 깐다
    block = 16
    grid[::block, :] = settings.CELL_ROAD
    grid[:, ::block] = settings.CELL_ROAD

    # 2) 건물: 도로가 아닌 칸을, 중심에 가까울수록 더 빽빽하게 채운다
    rows, cols = np.indices((N, N))
    cy, cx = N / 2, N / 2
    dist = np.hypot(rows - cy, cols - cx) / (N / 2)     # 중심 0 ~ 가장자리 1
    prob = np.clip(0.75 * (1 - dist), 0, 1)             # 중심 0.75 -> 가장자리 0
    building_mask = (rng.random((N, N)) < prob) & (grid == settings.CELL_EMPTY)
    grid[building_mask] = settings.CELL_BUILDING

    # 3) 대피소: 가장자리 쪽 안전한 지점 몇 곳 (Phase 4 라우팅의 목적지)
    shelters = [(N - 8, 8), (8, N - 8), (N - 8, N - 8), (8, 8), (N // 2, 6)]
    for (r, c) in shelters:
        grid[r - 1:r + 2, c - 1:c + 2] = settings.CELL_SHELTER

    return grid, shelters
