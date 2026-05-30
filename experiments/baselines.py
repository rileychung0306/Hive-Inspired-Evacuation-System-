"""experiments/baselines.py

Phase 6 #6: 우리 알고리즘 vs 실제 드론 군집/탐색 알고리즘 비교.

비교 대상 (모두 같은 격자·같은 ACLED 위험·같은 드론 수=10·같은 bee voting):
  1) Random walk           — 무작위 보행 (null baseline)
                              Viswanathan et al. 1999, Nature
  2) Lawn-mower (Boustro.) — 지그재그 격자 스캔 (상용 드론 표준)
                              Choset 2001, Annals of Math & AI
  3) Frontier-based        — 미탐색 영역 경계로 이동 (로보틱스 표준)
                              Yamauchi 1997, IEEE CIRA
  4) Waggle-dance + Quorum — 우리 시스템 (꿀벌 분산 의사결정)
                              Seeley 2010, Honeybee Democracy

측정: step 별 재현율 (recall) 곡선. 모든 방법이 같은 bee voting 으로 보고하므로
      차이는 오직 "어디로 가서 무엇을 보는가" 에서만 옴.

실행: python experiments/baselines.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import matplotlib
import numpy as np

from ai.bee_voting import BeeVoting
from config import settings
from hazard_modules.conflict import conflict_module
from simulation.acled_loader import load_acled_initial
from simulation.cellular_automata import HazardField
from simulation.city import build_city
from simulation.drone import Drone, DroneSwarm

OUT_PNG = "frames/phase6_baselines.png"
N_STEPS = 150
MEASURE_EVERY = 5


# =============================================================
# 베이스라인 드론 군집 (3가지 전략)
# =============================================================
class BaselineSwarm:
    """3개 베이스라인을 한 클래스로 처리 (drone.scan() 은 그대로 재사용)."""

    def __init__(self, hazard_module, strategy: str, num: int = None,
                 seed: int = 1):
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        N = settings.GRID_SIZE
        num = num or settings.NUM_DRONES

        # Drone 객체 생성 — region 은 strategy 별로 따로 설정
        self.drones = []
        for i in range(num):
            # 임의 초기 위치
            home = (int(self.rng.integers(0, N)), int(self.rng.integers(0, N)))
            d = Drone(drone_id=i, home_base=home,
                      region=(0, N, 0, N), rng=self.rng)
            self.drones.append(d)

        if strategy == "lawnmower":
            self._setup_lawnmower()
        elif strategy == "frontier":
            self.seen = np.zeros((N, N), dtype=bool)

    # ----- 1) Random walk -----
    def _move_random(self):
        N = settings.GRID_SIZE
        s = settings.DRONE_SPEED
        for d in self.drones:
            d.pos[0] = np.clip(d.pos[0] + self.rng.uniform(-1, 1) * s, 0, N - 1)
            d.pos[1] = np.clip(d.pos[1] + self.rng.uniform(-1, 1) * s, 0, N - 1)
            d.battery = max(20, d.battery - settings.DRONE_BATTERY_DRAIN)

    # ----- 2) Lawn-mower -----
    def _setup_lawnmower(self):
        """드론 i 에게 가로 띠 하나씩 배정, 좌→우 스캔."""
        N = settings.GRID_SIZE
        num = len(self.drones)
        strip_h = max(1, N // num)
        for i, d in enumerate(self.drones):
            r0 = i * strip_h
            r1 = min(N, (i + 1) * strip_h)
            d._strip = (r0, r1)
            d._dir = 1
            d.pos = np.array([float((r0 + r1) / 2), 0.0])

    def _move_lawnmower(self):
        N = settings.GRID_SIZE
        s = settings.DRONE_SPEED
        for d in self.drones:
            d.pos[1] += d._dir * s
            if d.pos[1] >= N - 1:
                d.pos[1] = N - 1
                d._dir = -1
                # 띠 내에서 위·아래 약간 흔들기 (커버 효과 ↑)
                d.pos[0] += 2.0
                if d.pos[0] >= d._strip[1] - 1:
                    d.pos[0] = d._strip[0]
            elif d.pos[1] <= 0:
                d.pos[1] = 0
                d._dir = 1
                d.pos[0] += 2.0
                if d.pos[0] >= d._strip[1] - 1:
                    d.pos[0] = d._strip[0]
            d.battery = max(20, d.battery - settings.DRONE_BATTERY_DRAIN)

    # ----- 3) Frontier-based -----
    def _mark_seen(self):
        N = settings.GRID_SIZE
        for d in self.drones:
            r = int(round(d.pos[0])); c = int(round(d.pos[1]))
            v = d.vision
            self.seen[max(0, r - v):min(N, r + v + 1),
                       max(0, c - v):min(N, c + v + 1)] = True

    def _move_frontier(self):
        N = settings.GRID_SIZE
        s = settings.DRONE_SPEED
        self._mark_seen()
        unseen = np.argwhere(~self.seen)
        if len(unseen) == 0:
            self._move_random()
            return
        # 미탐색 셀에서 일부만 샘플 (전체 거리 계산 느림)
        k = min(500, len(unseen))
        sample_idx = self.rng.choice(len(unseen), size=k, replace=False)
        candidates = unseen[sample_idx]
        for d in self.drones:
            pos = np.array([d.pos[0], d.pos[1]])
            dists = np.sum((candidates - pos) ** 2, axis=1)
            nearest = candidates[np.argmin(dists)]
            delta = nearest - pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                d.pos = np.clip(d.pos + delta / norm * s, 0, N - 1)
            d.battery = max(20, d.battery - settings.DRONE_BATTERY_DRAIN)

    # ----- 공통: step (motion + scan + report) -----
    def step(self, field, bee):
        if self.strategy == "random":
            self._move_random()
        elif self.strategy == "lawnmower":
            self._move_lawnmower()
        elif self.strategy == "frontier":
            self._move_frontier()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        for d in self.drones:
            for (r, c, k) in d.scan(field):
                bee.report(r, c, k)
        bee.decay_step()


# =============================================================
# 한 알고리즘 실험 실행
# =============================================================
def run_method(strategy: str, n_steps: int = N_STEPS):
    print(f"\n[{strategy}] running {n_steps} steps...")
    city, _ = build_city(seed=42)
    initial, bbox = load_acled_initial(conflict_module)
    field = HazardField(conflict_module, initial=initial)
    bee = BeeVoting(conflict_module)
    if strategy == "waggle":
        swarm = DroneSwarm(conflict_module)
    else:
        swarm = BaselineSwarm(conflict_module, strategy=strategy)

    N = settings.GRID_SIZE
    discovered = np.zeros((N, N), dtype=bool)   # 어떤 드론이라도 시야에 한 번이라도 들어온 칸

    history = []
    for step in range(n_steps):
        field.step()
        swarm.step(field.field, bee)
        # 시야에 들어온 셀 누적 (탐색 측정용)
        for d in swarm.drones:
            r = int(round(d.pos[0])); c = int(round(d.pos[1]))
            v = d.vision
            discovered[max(0, r - v):min(N, r + v + 1),
                       max(0, c - v):min(N, c + v + 1)] = True

        if step % MEASURE_EVERY == 0 or step == n_steps - 1:
            true_cells = field.field.max(axis=2) > 0.15
            conf_cells = bee.confirmed_mask().any(axis=2)
            nt = int(true_cells.sum())
            tp = int((true_cells & conf_cells).sum())
            tp_disc = int((true_cells & discovered).sum())
            nc = int(conf_cells.sum())
            recall = tp / nt if nt else 0.0
            discovery_rate = tp_disc / nt if nt else 0.0
            precision = tp / nc if nc else 0.0
            history.append({"step": step, "recall": recall,
                            "discovery": discovery_rate,
                            "precision": precision, "n_true": nt,
                            "n_conf": nc, "tp": tp})

    final = history[-1]
    print(f"  final: discovery {final['discovery']:.3f}  "
          f"recall {final['recall']:.3f}  "
          f"precision {final['precision']:.3f}")
    return history


# =============================================================
# 4개 모두 실행 + 비교 그래프
# =============================================================
def main():
    print("=" * 60)
    print(" Phase 6 #6: 베이스라인 알고리즘 비교")
    print("=" * 60)

    methods = {
        "random":     "① Random walk",
        "lawnmower":  "② Lawn-mower",
        "frontier":   "③ Frontier-based",
        "waggle":     "④ Waggle-dance + Quorum (우리)",
    }

    results = {}
    for key in methods:
        results[key] = run_method(key, n_steps=N_STEPS)

    # 결과 표
    print("\n" + "=" * 60)
    print(" Step 150 최종 비교")
    print("=" * 60)
    print(f"  {'알고리즘':37s}  {'탐색':>6s}  {'확인':>6s}  {'정밀도':>6s}")
    for key, label in methods.items():
        f = results[key][-1]
        print(f"  {label:37s}  {f['discovery']*100:5.1f}%  "
              f"{f['recall']*100:5.1f}%  {f['precision']*100:5.1f}%")

    # 그래프 (2-panel: 탐색 / 확인)
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"random": "#9ca3af", "lawnmower": "#10b981",
              "frontier": "#3b82f6", "waggle": "#dc2626"}

    for key, label in methods.items():
        h = results[key]
        steps = [x["step"] for x in h]
        disc = [x["discovery"] * 100 for x in h]
        recalls = [x["recall"] * 100 for x in h]
        lw = 3.0 if key == "waggle" else 1.7
        ms = 5 if key == "waggle" else 3
        ax1.plot(steps, disc, "-o", color=colors[key], label=label,
                 lw=lw, markersize=ms)
        ax2.plot(steps, recalls, "-o", color=colors[key], label=label,
                 lw=lw, markersize=ms)

    ax1.set_title("탐색 효율 (Discovery Rate)\n"
                  "= 시야에 한 번이라도 들어온 위험 칸의 비율", fontsize=12)
    ax1.set_xlabel("시뮬레이션 step"); ax1.set_ylabel("탐색 비율 (%)")
    ax1.legend(fontsize=10, loc="lower right"); ax1.grid(alpha=0.3)

    ax2.set_title("확인 효율 (Confirmed Recall)\n"
                  "= bee voting quorum(0.5) 넘긴 위험 칸의 비율", fontsize=12)
    ax2.set_xlabel("시뮬레이션 step"); ax2.set_ylabel("확인 비율 (%)")
    ax2.legend(fontsize=10, loc="best"); ax2.grid(alpha=0.3)

    fig.suptitle(
        "Phase 6 — 우리 알고리즘 vs 실제 드론 탐색 알고리즘 비교\n"
        "(같은 격자·같은 ACLED 위험·같은 드론 10대·같은 bee voting)",
        fontsize=14)
    os.makedirs("frames", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
