"""simulation/run.py

Phase 1 실행기: 도시 + ACLED 초기 위험 + 셀룰러 오토마타(위험 확산).

실행 방법:
  실시간 창으로 보기:    python -m simulation.run
  PNG 스냅샷 저장(검증):  python -m simulation.run --snapshots
"""

import sys

from hazard_modules.conflict import conflict_module
from simulation.city import build_city
from simulation.acled_loader import load_acled_initial
from simulation.cellular_automata import HazardField
from simulation import visualize


def build_world(hazard_module=conflict_module, seed: int = 42):
    """도시 + 초기 위험 지도 + 위험 확산 객체를 한 번에 준비."""
    city, shelters = build_city(seed)
    initial, bbox = load_acled_initial(hazard_module)
    field = HazardField(hazard_module, initial=initial)
    return city, shelters, field


def main():
    hazard = conflict_module
    city, shelters, field = build_world(hazard)

    if "--snapshots" in sys.argv:
        # 시간 흐름에 따른 위험 확산을 PNG 여러 장으로 저장
        for target in [0, 20, 60, 120]:
            while field.step_count < target:
                field.step()
            path = f"frames/phase1_step{field.step_count:03d}.png"
            visualize.save_snapshot(city, field.field, hazard, path,
                                    title=f"Bakhmut 위험 확산 — step {field.step_count}")
            print(f"저장됨: {path}")
    else:
        visualize.run_pygame(city, field, hazard, steps_per_frame=1)


if __name__ == "__main__":
    main()
