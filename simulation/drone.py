"""simulation/drone.py

정찰 드론과 드론 군집(swarm).

비유: 벌집에서 나온 정찰벌들. 각자 다른 구역으로 흩어져 위험을 살피고,
배터리가 부족하면 본부로 돌아가 충전합니다.

중요한 점: 드론 하나하나는 '완벽하지 않습니다'.
  - 위험을 놓칠 수도 있고(DETECT_PROB), 가끔 잘못 보기도 합니다(FALSE_DETECT_PROB).
그래서 여러 드론의 보고를 모아 '꿀벌 투표'로 거르는 것이 핵심입니다 (ai/bee_voting.py).
"""

import numpy as np
from config import settings


class Drone:
    """정찰 드론 한 대."""

    def __init__(self, drone_id, home_base, region, rng):
        self.id = drone_id
        self.home = np.array(home_base, dtype=float)
        self.pos = self.home.copy()
        self.region = region          # 담당 구역 (r0, r1, c0, c1)
        self.vision = settings.DRONE_VISION_RADIUS
        self.battery = 100.0
        self.rng = rng
        self.returning = False
        self.target = self._new_target()

    def _new_target(self, recruit=None):
        """다음 순찰 지점을 정합니다.

        recruit(=다른 드론들이 발견한 위험 신뢰도 지도)가 주어지면,
        DRONE_RECRUIT_PROB 확률로 '위험이 발견된 칸'으로 이끌립니다.
        이것이 꿀벌의 waggle dance(춤으로 동료를 좋은 곳으로 부르기)에 해당합니다.
        """
        r0, r1, c0, c1 = self.region
        if recruit is not None and self.rng.random() < settings.DRONE_RECRUIT_PROB:
            region = recruit[r0:r1, c0:c1]
            flat = region.ravel().astype(np.float64)
            total = flat.sum()
            if total > 0:
                i = int(self.rng.choice(len(flat), p=flat / total))
                lr, lc = divmod(i, region.shape[1])
                return np.array([r0 + lr, c0 + lc], dtype=float)
        return np.array([self.rng.uniform(r0, r1), self.rng.uniform(c0, c1)])

    def move(self, recruit=None):
        """담당 구역 안을 순찰. 배터리가 낮으면 본부로 복귀해 충전."""
        N = settings.GRID_SIZE
        if self.battery < 20:
            self.returning = True
        if self.battery >= 95:
            self.returning = False

        goal = self.home if self.returning else self.target
        delta = goal - self.pos
        dist = float(np.hypot(delta[0], delta[1]))

        if dist < 2.0:
            if self.returning:
                self.battery = min(100.0, self.battery + 20.0)   # 충전
            else:
                self.target = self._new_target(recruit)          # 새 순찰 지점
        else:
            self.pos = np.clip(self.pos + delta / dist * settings.DRONE_SPEED, 0, N - 1)

        self.battery -= settings.DRONE_BATTERY_DRAIN

    def scan(self, field):
        """시야 안의 위험을 (확률적으로) 탐지해 [(row, col, hazard_index), ...] 반환."""
        N = settings.GRID_SIZE
        r, c = int(round(self.pos[0])), int(round(self.pos[1]))
        v = self.vision
        r0, r1 = max(0, r - v), min(N, r + v + 1)
        c0, c1 = max(0, c - v), min(N, c + v + 1)

        window = field[r0:r1, c0:c1, :]
        reports = []
        # 1) 진짜 위험: 임계값을 넘는 칸을 DETECT_PROB 확률로 탐지
        for lr, lc, k in np.argwhere(window > settings.DETECT_THRESHOLD):
            if self.rng.random() < settings.DETECT_PROB:
                reports.append((r0 + int(lr), c0 + int(lc), int(k)))
        # 2) 오탐: 가끔 시야 안의 엉뚱한 칸을 위험이라고 잘못 보고
        if self.rng.random() < settings.FALSE_DETECT_PROB:
            fr = int(self.rng.integers(r0, r1))
            fc = int(self.rng.integers(c0, c1))
            fk = int(self.rng.integers(field.shape[2]))
            reports.append((fr, fc, fk))
        return reports


class DroneSwarm:
    """드론 여러 대를 만들고, 도시를 나눠 맡겨 한 곳에 몰리지 않게 합니다."""

    def __init__(self, hazard_module, num=None, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        N = settings.GRID_SIZE
        num = num or settings.NUM_DRONES

        # 도시를 rows x cols 구역으로 나누고 각 구역에 드론 1대씩
        cols = int(np.ceil(np.sqrt(num)))
        rows = int(np.ceil(num / cols))
        self.drones = []
        i = 0
        for ri in range(rows):
            for ci in range(cols):
                if i >= num:
                    break
                r0, r1 = ri * N // rows, (ri + 1) * N // rows
                c0, c1 = ci * N // cols, (ci + 1) * N // cols
                home = ((r0 + r1) // 2, (c0 + c1) // 2)
                self.drones.append(Drone(i, home, (r0, r1, c0, c1), self.rng))
                i += 1

    def step(self, field, bee):
        """모든 드론을 한 스텝 움직이고, 본 것을 꿀벌 투표(bee)에 보고."""
        recruit = bee.confidence.sum(axis=2)   # 지금까지 발견된 위험 (모집 신호)
        for d in self.drones:
            d.move(recruit)
            for (r, c, k) in d.scan(field):
                bee.report(r, c, k)
        bee.decay_step()
