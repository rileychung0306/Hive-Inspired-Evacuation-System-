"""ai/bee_voting.py

꿀벌 투표(Bee Voting) — 우리 프로젝트의 핵심 알고리즘.

문제: 드론 하나의 보고는 못 믿습니다 (놓치거나 잘못 보기도 함).
해결: 여러 보고가 '같은 칸의 같은 위험'에 쌓여 임계값(quorum)을 넘어야 '확인된 위험'으로 인정.
      오래된 정보는 매 스텝 조금씩 잊혀집니다(decay) -> 한 번뿐인 오탐은 사라지고,
      진짜 위험은 반복 관측되며 살아남습니다.

이것이 꿀벌의 집단 의사결정(Seeley의 quorum sensing)을 단순화한 버전입니다.
  - 드론 보고: 신뢰도 +0.1
  - 뉴스(LLM) 보고: 신뢰도 +0.3 (뉴스를 더 신뢰) — Phase 3에서 연결
  - 합산 신뢰도 >= 0.5 이면 '확인됨'
"""

import numpy as np
from config import settings


class BeeVoting:
    """위험에 대한 군집의 '신뢰도 지도'를 관리하는 객체."""

    def __init__(self, hazard_module):
        N = settings.GRID_SIZE
        self.K = hazard_module.num_hazards
        self.confidence = np.zeros((N, N, self.K), dtype=np.float32)
        self.threshold = settings.QUORUM_THRESHOLD
        self.decay = settings.CONFIDENCE_DECAY

    def report(self, r, c, k, confidence=None):
        """한 건의 보고를 신뢰도 지도에 더합니다 (드론=0.1, 뉴스=0.3)."""
        conf = settings.DRONE_CONFIDENCE if confidence is None else confidence
        self.confidence[r, c, k] = min(1.5, self.confidence[r, c, k] + conf)

    def decay_step(self):
        """매 스텝 신뢰도를 조금씩 낮춤 — 오래된/한 번뿐인 정보는 사라지게."""
        self.confidence *= (1.0 - self.decay)

    def confirmed_mask(self):
        """확인된(신뢰도 >= 임계값) 위험 위치 (불리언 배열, N x N x K)."""
        return self.confidence >= self.threshold

    def confirmed_field(self):
        """확인된 위험만 남긴 강도 지도 (N x N x K, 0~1). 시각화/라우팅용."""
        conf = np.where(self.confidence >= self.threshold, self.confidence, 0.0)
        return np.clip(conf, 0.0, 1.0)

    def confirmed_risk(self):
        """각 칸의 확인된 전체 위험도 = 확인된 위험 중 최댓값 (라우팅 입력)."""
        return self.confirmed_field().max(axis=2)
