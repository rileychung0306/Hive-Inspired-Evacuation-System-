"""main.py

전체 시스템을 실행하는 입구(entry point)입니다.
단계가 진행되며 드론 -> 뉴스 -> 라우팅이 차례로 여기에 연결됩니다.

지금(Phase 1)은 도시 + 5가지 위험 확산 시뮬레이션을 실시간 창으로 보여줍니다.

실행 방법 (가상환경 켠 상태):
    python main.py                      # 실시간 창으로 보기
    python -m simulation.run --snapshots  # PNG 스냅샷으로 저장
"""

from simulation.run import main as run_simulation


if __name__ == "__main__":
    run_simulation()
