"""simulation/run.py

실행기: 도시 + ACLED 초기 위험 + 위험 확산(CA)
        + 드론 군집(Phase 2) + 꿀벌 투표(Phase 2)
        + LLM 뉴스 융합(Phase 3).

실행 방법:
  실시간 창(단일 뷰):         python -m simulation.run        (T: 실제/확인 위험 전환, ESC: 종료)
  실시간 창(3분할 비교):       python -m simulation.run --compare
  PNG 스냅샷 저장(검증):       python -m simulation.run --snapshots
  3분할 비교 PNG:             python -m simulation.run --compare --snapshots
  뉴스 캐시 강제 갱신:         python -m simulation.run --refresh-news
  뉴스 융합 끄기(비교용):       python -m simulation.run --no-news
"""

import sys

from hazard_modules.conflict import conflict_module
from simulation.city import build_city
from simulation.acled_loader import load_acled_initial
from simulation.cellular_automata import HazardField
from simulation.drone import DroneSwarm
from simulation import visualize
from ai.bee_voting import BeeVoting


def build_world(hazard_module=conflict_module, seed: int = 42,
                use_news: bool = True, refresh_news: bool = False):
    """도시 + 초기 위험 + 확산 + 드론 군집 + 꿀벌 + (선택)뉴스 융합."""
    city, shelters = build_city(seed)
    initial, bbox = load_acled_initial(hazard_module)
    field = HazardField(hazard_module, initial=initial)
    swarm = DroneSwarm(hazard_module)
    bee = BeeVoting(hazard_module)

    if use_news and bbox is not None:
        # 늦은 import: 뉴스 끌 때 OpenAI/requests 안 부르도록
        from ai.news_collector import fetch_news
        from ai.llm_analyzer import extract_hazards_from_all
        from ai.news_fusion import apply_news_to_bee

        articles = fetch_news(hazard_module.news_keywords,
                              force_refresh=refresh_news)
        if articles:
            news_hazards, closures = extract_hazards_from_all(articles, hazard_module)
            applied, skipped = apply_news_to_bee(news_hazards, bee,
                                                  hazard_module, bbox)
            print(f"[News Fusion] 기사 {len(articles)}건 -> "
                  f"위험 {len(news_hazards)}개 추출 -> "
                  f"격자 반영 {applied}개 (영역 밖/형식오류 {skipped}개 건너뜀)")
            if closures:
                print(f"[News Fusion] 도로 폐쇄 보고: {[c['road'] for c in closures[:5]]}")

    return city, shelters, field, swarm, bee, bbox


def main():
    hazard = conflict_module
    args = sys.argv
    use_news = "--no-news" not in args
    refresh_news = "--refresh-news" in args
    snapshots = "--snapshots" in args
    compare = "--compare" in args

    if compare:
        # 두 세계를 같은 시드로 만들어 같은 도시/같은 위험 위에서 비교
        city, shelters, field_a, swarm_a, bee_a, _ = build_world(
            hazard, use_news=False, refresh_news=refresh_news, seed=42)
        _, _, field_b, swarm_b, bee_b, _ = build_world(
            hazard, use_news=True, refresh_news=False, seed=42)

        if snapshots:
            for target in [0, 30, 80, 150]:
                while field_a.step_count < target:
                    field_a.step(); swarm_a.step(field_a.field, bee_a)
                    field_b.step(); swarm_b.step(field_b.field, bee_b)
                path = f"frames/phase3_compare_step{field_a.step_count:03d}.png"
                visualize.save_snapshot_compare(
                    city, field_a, swarm_a, bee_a,
                    field_b, swarm_b, bee_b, hazard, path,
                    title=f"3분할 비교 — step {field_a.step_count}")
                print(f"저장됨: {path}")
        else:
            visualize.run_pygame_compare(
                city, field_a, swarm_a, bee_a,
                field_b, swarm_b, bee_b, hazard)
        return

    city, shelters, field, swarm, bee, bbox = build_world(
        hazard, use_news=use_news, refresh_news=refresh_news)

    if snapshots:
        tag = "" if use_news else "_nonews"
        for target in [0, 30, 80, 150]:
            while field.step_count < target:
                field.step()
                swarm.step(field.field, bee)
            path = f"frames/phase3_step{field.step_count:03d}{tag}.png"
            visualize.save_snapshot_swarm(
                city, field.field, bee, swarm.drones, hazard, path,
                title=f"Phase 3 — step {field.step_count}"
                      + (" (with news fusion)" if use_news else " (no news)"))
            print(f"저장됨: {path}")
    else:
        visualize.run_pygame(city, field, swarm, bee, hazard, steps_per_frame=1)


if __name__ == "__main__":
    main()
