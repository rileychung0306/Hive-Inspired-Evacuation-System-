"""simulation/cellular_automata.py

위험이 시간에 따라 '산불처럼' 번지고 사그라드는 모델입니다 (셀룰러 오토마타).
격자의 각 칸이 매 스텝마다 '이웃 칸'을 보고 자기 위험도를 바꿉니다.

위험 종류마다 움직임이 다릅니다 (성질은 hazard_modules/conflict.py 의 숫자로 정해짐):
  - 번지는 위험 (적군/포격): 이웃의 강한 위험이 확률적으로 옮겨붙고, 시간이 지나면 감쇠
  - 이동형 (적 드론):        위치가 매 스텝 무작위로 조금씩 이동
  - 정적 (파괴된 건물/지뢰): 거의 변하지 않음 (지뢰는 가끔 새로 생김)
"""

import numpy as np
from config import settings


def _neighbor_max(a, neighborhood=8):
    """각 칸에서 '이웃 칸들의 최댓값'을 구합니다. (가장자리는 0으로 채움)"""
    p = np.pad(a, 1, mode="constant")
    H, W = a.shape
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighborhood == 8:
        shifts += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    m = np.zeros_like(a)
    for dy, dx in shifts:
        m = np.maximum(m, p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W])
    return m


class HazardField:
    """200x200x5 위험 지도를 들고, 매 스텝 위험을 갱신하는 객체."""

    def __init__(self, hazard_module, initial=None, seed: int = 0):
        self.hazard = hazard_module
        self.specs = hazard_module.hazards          # 위험 종류 목록 (층 순서)
        self.K = len(self.specs)
        N = settings.GRID_SIZE
        if initial is not None:
            self.field = initial.astype(np.float32).copy()
        else:
            self.field = np.zeros((N, N, self.K), dtype=np.float32)
        # 초기 위험(=분쟁 핫스폿)을 기억: 활성 전선이라 계속 다시 타오르게 함
        self.sources = self.field.copy()
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    def step(self):
        """시간을 1칸 진행 — 모든 위험을 한 번씩 갱신합니다."""
        for k, spec in enumerate(self.specs):
            layer = self.field[:, :, k]

            if spec.mobile:
                # 이동형(적 드론): 무작위로 조금 이동 + 약하게 감쇠
                dy = int(self.rng.integers(-3, 4))
                dx = int(self.rng.integers(-3, 4))
                layer = np.roll(np.roll(layer, dy, axis=0), dx, axis=1) * 0.98

            elif spec.static:
                # 정적: 그대로 둔다. 단, 지뢰(spawn_rate>0)는 가끔 전선 근처에 새로 생김
                if spec.spawn_rate > 0 and self.rng.random() < 0.3:
                    front = self.field.sum(axis=2)         # 위험이 있는 '전선' 부근
                    cand = np.argwhere(front > 0.3)
                    if len(cand):
                        r, c = cand[self.rng.integers(len(cand))]
                        layer[r, c] = 1.0

            else:
                # 번지는 위험: 이웃의 강한 값이 spread_prob 확률로 옮겨붙음
                nmax = _neighbor_max(layer, spec.neighborhood)
                ignite = (self.rng.random(layer.shape) < spec.spread_prob) & (nmax > 0.05)
                layer = np.where(ignite, np.maximum(layer, nmax * 0.9), layer)
                # 시간이 지나면 감쇠
                layer = layer * (1.0 - spec.decay_per_step)

            # 활성 분쟁: 정적이 아닌 위험은 원래 핫스폿(전선)이 계속 다시 타오름
            if not spec.static:
                src = self.sources[:, :, k]
                pulse = float(self.rng.uniform(0.5, 1.0))
                reignite = (src > 0.2) & (self.rng.random(layer.shape) < 0.5)
                layer = np.where(reignite, np.maximum(layer, src * pulse), layer)

            self.field[:, :, k] = np.clip(layer, 0.0, 1.0)

        self.step_count += 1
        return self.field

    def total_risk(self):
        """각 칸의 전체 위험도 = 5가지 중 가장 큰 값. (라우팅/시각화에 사용)"""
        return self.field.max(axis=2)
