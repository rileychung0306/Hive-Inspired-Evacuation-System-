"""simulation/run.py

실행기: 도시 + ACLED 초기 위험 + 위험 확산(셀룰러 오토마타)
        + 드론 군집(Phase 2) + 꿀벌 투표(Phase 2).

실행 방법:
  실시간 창으로 보기:    python -m simulation.run        (T: 실제/확인 위험 전환, ESC: 종료)
  PNG 스냅샷 저장(검증):  python -m simulation.run --snapshots
"""

import sys

from hazard_modules.conflict import conflict_module
from simulation.city import build_city
from simulation.acled_loader import load_acled_initial
from simulation.cellular_automata import HazardField
from simulation.drone import DroneSwarm
from simulation import visualize
from ai.bee_voting import BeeVoting


def build_world(hazard_module=conflict_module, seed: int = 42):
    """도시 + 초기 위험 + 위험 확산 + 드론 군집 + 꿀벌 투표를 한 번에 준비."""
    city, shelters = build_city(seed)
    initial, bbox = load_acled_initial(hazard_module)
    field = HazardField(hazard_module, initial=initial)
    swarm = DroneSwarm(hazard_module)
    bee = BeeVoting(hazard_module)
    return city, shelters, field, swarm, bee


def main():
    hazard = conflict_module
    city, shelters, field, swarm, bee = build_world(hazard)

    if "--snapshots" in sys.argv:
        for target in [0, 30, 80, 150]:
            while field.step_count < target:
                field.step()
                swarm.step(field.field, bee)
            path = f"frames/phase2_step{field.step_count:03d}.png"
            visualize.save_snapshot_swarm(
                city, field.field, bee, swarm.drones, hazard, path,
                title=f"드론 군집 정찰 — step {field.step_count}")
            print(f"저장됨: {path}")
    else:
        visualize.run_pygame(city, field, swarm, bee, hazard, steps_per_frame=1)


if __name__ == "__main__":
    main()
