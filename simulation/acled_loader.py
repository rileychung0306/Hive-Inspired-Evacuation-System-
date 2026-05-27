"""simulation/acled_loader.py

실제 ACLED Bakhmut 사건 데이터(data/acled_bakhmut.csv)를 읽어서,
200x200 격자 위의 '초기 위험 지도'로 바꿔줍니다.

핵심 아이디어:
  - 각 사건에는 위도(latitude)/경도(longitude)가 있습니다 -> 격자의 한 칸으로 변환
  - ACLED의 사건 종류(sub_event_type)를 우리 5가지 위험 이름으로 연결
  - 같은 칸에 사건이 많을수록 그 칸의 초기 위험도가 높아짐
이렇게 하면 시뮬레이션이 "지어낸 시나리오"가 아니라 "실제 Bakhmut 사건"에서 출발합니다.
"""

import os
import numpy as np
import pandas as pd

from config import settings
from hazard_modules.base import HazardModule

# ACLED의 sub_event_type -> 우리 위험 종류 이름
SUBEVENT_TO_HAZARD = {
    "Shelling/artillery/missile attack": "shelling",
    "Armed clash": "enemy_forces",
    "Air/drone strike": "enemy_drone",
    "Remote explosive/landmine/IED": "landmine",
    "Non-state actor overtakes territory": "enemy_forces",
    "Government regains territory": "enemy_forces",
}

ACLED_CSV = os.path.join("data", "acled_bakhmut.csv")


def _blur3(a, passes=1):
    """3x3 평균으로 살짝 번지게 — 점 하나가 작은 얼룩이 되어 화면에 잘 보입니다."""
    for _ in range(passes):
        p = np.pad(a, 1, mode="constant")
        a = (p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
             p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
             p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
    return a


def load_acled_initial(hazard: HazardModule, csv_path: str = ACLED_CSV):
    """ACLED CSV -> (GRID, GRID, K) 모양의 초기 위험 배열(값 0~1) 반환.

    파일이 없으면(예: 아직 데이터 못 받았을 때) 0으로 채운 빈 지도를 돌려줍니다.
    """
    N = settings.GRID_SIZE
    names = hazard.hazard_names
    K = len(names)
    idx = {name: i for i, name in enumerate(names)}
    field = np.zeros((N, N, K), dtype=np.float32)

    if not os.path.exists(csv_path):
        print(f"[ACLED] {csv_path} 가 없어 빈 초기 지도를 사용합니다. (mock 모드)")
        return field, None

    df = pd.read_csv(csv_path).dropna(subset=["latitude", "longitude"])
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()

    def to_cell(lat, lon):
        # 위도가 클수록 북쪽 -> 윗줄(row 0). 경도가 클수록 오른쪽(col 큰 값).
        row = int((lat_max - lat) / (lat_max - lat_min + 1e-9) * (N - 1))
        col = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * (N - 1))
        return min(max(row, 0), N - 1), min(max(col, 0), N - 1)

    used = 0
    for lat, lon, sub, fat in zip(df["latitude"], df["longitude"],
                                  df["sub_event_type"], df["fatalities"].fillna(0)):
        hname = SUBEVENT_TO_HAZARD.get(sub)
        if hname is None or hname not in idx:
            continue
        r, c = to_cell(lat, lon)
        weight = 1.0 + min(float(fat), 20.0) / 10.0   # 사상자 많으면 가중치를 더 줌
        field[r, c, idx[hname]] += weight
        # 사망자가 난 포격 자리에는 '파괴된 건물'도 함께 씨앗으로 둠
        if hname == "shelling" and float(fat) > 0 and "destroyed_building" in idx:
            field[r, c, idx["destroyed_building"]] += 0.5
        used += 1

    # 각 위험 층을 0~1로 정규화(상위 95% 값을 1로) 후 살짝 번지게
    for k in range(K):
        layer = field[:, :, k]
        nz = layer[layer > 0]
        if nz.size:
            scale = np.percentile(nz, 95)
            if scale > 0:
                layer = np.clip(layer / scale, 0, 1)
        layer = _blur3(layer, passes=1)
        if layer.max() > 0:
            layer = layer / layer.max()      # 가장 뜨거운 칸을 1.0으로 (씨앗을 또렷하게)
        field[:, :, k] = np.clip(layer, 0, 1)

    bbox = (float(lat_min), float(lat_max), float(lon_min), float(lon_max))
    print(f"[ACLED] {used}개 사건을 {N}x{N} 격자에 반영했습니다 "
          f"(위도 {lat_min:.3f}~{lat_max:.3f}, 경도 {lon_min:.3f}~{lon_max:.3f}).")
    return field, bbox
