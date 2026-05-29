"""hazard_modules/conflict.py

분쟁 지역(예: 우크라이나 Bakhmut)에서 다루는 5가지 위험을 정의합니다.
base.py의 HazardModule 형식을 그대로 따릅니다.

나중에 산불 버전을 만들고 싶으면 이 파일을 본떠 wildfire.py를 만들면 됩니다.
시스템의 나머지 부분은 전혀 바꾸지 않아도 됩니다.
"""

from hazard_modules.base import HazardModule, HazardSpec


class ConflictModule(HazardModule):
    """분쟁(conflict) 재난 모듈 — 현재 우리가 사용하는 기본 모듈."""

    name = "conflict"

    # 5가지 위험. 숫자(확률, 감쇠 등)는 Phase 1에서 시뮬레이션을 보며 조정합니다.
    hazards = [
        # 적군: 넓게, 비교적 천천히 번지고 천천히 줄어듦
        HazardSpec("enemy_forces", "적군", (200, 0, 0),
                   spread_prob=0.22, neighborhood=8, decay_per_step=0.06),
        # 포격: 빠르게 8방향으로 번지지만 금방 사그라듦
        HazardSpec("shelling", "포격", (255, 120, 0),
                   spread_prob=0.50, neighborhood=8, decay_per_step=0.30),
        # 파괴된 건물: 거의 그대로 남아 있음 (정적)
        HazardSpec("destroyed_building", "파괴된 건물", (120, 120, 120),
                   static=True),
        # 적 드론: 위치가 계속 움직임 (이동형)
        HazardSpec("enemy_drone", "적 드론", (180, 0, 180),
                   mobile=True),
        # 지뢰: 정적이고 사라지지 않으며, 가끔 새로 생김
        HazardSpec("landmine", "지뢰", (90, 50, 10),
                   static=True, spawn_rate=0.001),
    ]

    # Phase 3에서 NewsAPI 뉴스 검색에 사용할 키워드
    news_keywords = ["Bakhmut", "Ukraine", "shelling", "evacuation", "Donbas"]

    # Phase 3에서 GPT에게 줄 지시문(프롬프트). JSON으로만 답하도록 요구합니다.
    llm_prompt = """You are a conflict-zone hazard analyst focused on Bakhmut and the Donetsk region of Ukraine.
Read the news article and extract hazard information. Respond with ONLY valid JSON in this exact schema:

{
  "hazards": [
    {
      "type": "shelling | enemy_forces | destroyed_building | enemy_drone | landmine",
      "location_description": "specific location text (e.g. 'central Bakhmut', 'M-03 highway near Bakhmut')",
      "lat": 48.59,
      "lon": 38.00,
      "severity": 7,
      "time": "approximate time or date if known"
    }
  ],
  "road_closures": ["road names that are reportedly closed"],
  "civilian_advisory": "one-line advisory for civilians"
}

Rules:
- severity: integer 1-10 (10 = most dangerous).
- lat/lon: estimate the approximate coordinates of the location. If unknown, use Bakhmut center (48.59, 38.00).
- ONLY include hazards in or near the Bakhmut / Donetsk Oblast area (roughly lat 48.0-49.0, lon 37.5-39.5).
- If the article has no relevant hazard info, return "hazards": [].
- Output JSON only — no extra prose, no markdown fences."""


# 다른 파일에서 'from hazard_modules.conflict import conflict_module' 로 바로 쓰도록 준비
conflict_module = ConflictModule()
