"""ai/news_fusion.py

LLM 이 뽑아낸 위험 목록을 꿀벌 투표(bee) 신뢰도 지도에 합칩니다.

작동:
  1. (lat, lon)을 ACLED bounding box 기준으로 격자 칸 (row, col) 로 변환
  2. 위험 종류를 hazard_module 의 위험 인덱스로 매핑
  3. severity(1~10)에 따라 신뢰도 가중치 조정
  4. 뉴스는 한 점이 아니라 '주변 영역'에 영향 -> 3x3 패치로 적용

설계 이유:
  - 뉴스는 드론 1대보다 신뢰도 높음 (NEWS_CONFIDENCE=0.3 > DRONE_CONFIDENCE=0.1)
  - 뉴스는 정확한 칸이 아닌 지역 단위 정보 -> 작은 패치로 번지게
  - bbox 밖 위험(예: 키이우)은 무시
"""

import numpy as np
from config import settings


def _build_index(hazard_module):
    return {name: i for i, name in enumerate(hazard_module.hazard_names)}


def _to_cell(lat, lon, bbox, N):
    """위/경도 -> (row, col). bbox 밖이면 None."""
    lat_min, lat_max, lon_min, lon_max = bbox
    row = int((lat_max - lat) / (lat_max - lat_min + 1e-9) * (N - 1))
    col = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * (N - 1))
    if 0 <= row < N and 0 <= col < N:
        return row, col
    return None


def apply_news_to_bee(news_hazards, bee, hazard_module, bbox):
    """뉴스에서 추출된 위험을 꿀벌 신뢰도 지도에 반영.

    반환: (적용된 위험 수, 격자 영역 밖이라 건너뛴 수)
    """
    if bbox is None or not news_hazards:
        return 0, 0

    idx = _build_index(hazard_module)
    N = settings.GRID_SIZE
    applied = 0
    skipped = 0

    for h in news_hazards:
        htype = h.get("type")
        if htype not in idx:
            skipped += 1
            continue
        lat, lon = h.get("lat"), h.get("lon")
        if lat is None or lon is None:
            skipped += 1
            continue
        cell = _to_cell(float(lat), float(lon), bbox, N)
        if cell is None:
            skipped += 1
            continue   # Bakhmut bbox 밖 -> 우리 시뮬레이션 영역 아님

        sev = float(h.get("severity", 5)) / 10.0          # 0.1 ~ 1.0
        base_conf = settings.NEWS_CONFIDENCE * (0.5 + sev) # severity 가중

        r, c = cell
        k = idx[htype]
        # 뉴스는 영역 단위 -> 3x3 패치 (중심 full, 이웃 0.5)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    falloff = 1.0 if (dr == 0 and dc == 0) else 0.5
                    bee.report(rr, cc, k, confidence=base_conf * falloff)
        applied += 1

    return applied, skipped
