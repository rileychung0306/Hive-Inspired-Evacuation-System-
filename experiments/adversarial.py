"""experiments/adversarial.py

Phase 6 #3: Adversarial robustness — 가짜 위치 뉴스를 거부하는지 검증.

심사위원 질문: "분쟁 지역에서 적이 가짜 뉴스로 시민을 위험지로 유도하면?
                LLM 이 가짜 뉴스를 받아들이면 어떻게 합니까?"

실험:
  - 실제 위험과 무관한 5개 위치에 '최대 심각도'(severity 10) 가짜 뉴스 주입
  - 시뮬레이션 150 스텝 진행
  - 측정:
    (a) 가짜 위치가 confirmed 됐는가? -> 거부율
    (b) 실제 위험은 여전히 발견되는가? -> 실제 recall 유지율

이론: news_confidence(0.3) < quorum_threshold(0.5) 이므로,
       드론 보고와 일치하지 않는 뉴스 단독으로는 confirm 불가.
       이중 검증 → 의도적 misinformation 자동 거부.

실행: python experiments/adversarial.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import matplotlib

from ai.bee_voting import BeeVoting
from ai.news_fusion import apply_news_to_bee
from config import settings
from hazard_modules.conflict import conflict_module
from simulation.acled_loader import load_acled_initial
from simulation.cellular_automata import HazardField
from simulation.city import build_city
from simulation.drone import DroneSwarm


# 실제 ACLED 위험은 Bakhmut 중심부(48.59, 38.00) 주변 5km 내에 분포.
# 가짜 뉴스는 의도적으로 그 영역 멀리 떨어진 곳을 지목.
FAKE_NEWS_INJECTION = [
    {"type": "shelling",     "lat": 48.90, "lon": 38.95, "severity": 10,
     "location_description": "[FAKE] 격자 북동쪽 외곽 (실제 위험 없음)"},
    {"type": "enemy_forces", "lat": 48.28, "lon": 38.85, "severity": 10,
     "location_description": "[FAKE] 격자 남동쪽 외곽"},
    {"type": "shelling",     "lat": 48.91, "lon": 38.00, "severity": 10,
     "location_description": "[FAKE] 격자 정북단"},
    {"type": "enemy_drone",  "lat": 48.30, "lon": 38.00, "severity": 10,
     "location_description": "[FAKE] 격자 정남단"},
    {"type": "shelling",     "lat": 48.60, "lon": 38.95, "severity": 10,
     "location_description": "[FAKE] 도시 동쪽 멀리"},
]

OUT_PNG = "frames/phase6_adversarial.png"
N_STEPS = 150


def check_fakes(bee, bbox, fakes):
    """가짜 위치(±1 칸 패치)가 confirmed 됐는지 확인."""
    N = settings.GRID_SIZE
    lat_min, lat_max, lon_min, lon_max = bbox
    hazard_idx = {n: i for i, n in enumerate(conflict_module.hazard_names)}
    confirmed = bee.confirmed_mask()
    out = []
    for f in fakes:
        row = int((lat_max - f["lat"]) / (lat_max - lat_min + 1e-9) * (N - 1))
        col = int((f["lon"] - lon_min) / (lon_max - lon_min + 1e-9) * (N - 1))
        if not (0 <= row < N and 0 <= col < N):
            out.append({**f, "row": row, "col": col, "out": True,
                        "confirmed": False})
            continue
        k = hazard_idx[f["type"]]
        # 패치 영역 (3x3) 중 하나라도 confirmed 면 트림 (보수적)
        r0, r1 = max(0, row - 1), min(N, row + 2)
        c0, c1 = max(0, col - 1), min(N, col + 2)
        any_conf = bool(confirmed[r0:r1, c0:c1, k].any())
        out.append({**f, "row": row, "col": col, "out": False,
                    "confirmed": any_conf})
    return out


def main():
    print("=" * 60)
    print(" Phase 6 #3: Adversarial 가짜 뉴스 거부 테스트")
    print("=" * 60)

    city, _ = build_city(seed=42)
    initial, bbox = load_acled_initial(conflict_module)
    field = HazardField(conflict_module, initial=initial)
    swarm = DroneSwarm(conflict_module)
    bee = BeeVoting(conflict_module)

    # 가짜 뉴스 주입 (실제 NewsAPI 결과 대신 의도적으로 만든 misinformation)
    print(f"\n주입: 의도적 가짜 위치 뉴스 {len(FAKE_NEWS_INJECTION)}건 (severity=10)")
    applied, skipped = apply_news_to_bee(FAKE_NEWS_INJECTION, bee, conflict_module, bbox)
    print(f"  apply_news_to_bee: applied={applied}, skipped={skipped}")
    for f in FAKE_NEWS_INJECTION:
        print(f"  - {f['type']:15s} ({f['lat']:.3f}, {f['lon']:.3f}) sev={f['severity']}  "
              f"{f['location_description']}")

    print(f"\n시뮬레이션 {N_STEPS} 스텝 진행 (드론은 정상 작동)...")
    for _ in range(N_STEPS):
        field.step()
        swarm.step(field.field, bee)

    # 가짜가 confirmed 됐는지 확인
    fake_status = check_fakes(bee, bbox, FAKE_NEWS_INJECTION)

    # 실제 recall 유지율
    true_cells = field.field.max(axis=2) > 0.15
    conf_cells = bee.confirmed_mask().any(axis=2)
    nt = int(true_cells.sum())
    tp = int((true_cells & conf_cells).sum())
    real_recall = tp / nt if nt else 0.0

    n_fake = len(fake_status)
    n_confirmed_fake = sum(1 for f in fake_status if f["confirmed"])
    rejection_rate = (n_fake - n_confirmed_fake) / n_fake * 100

    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    for f in fake_status:
        mark = "❌ CONFIRMED (BAD)" if f["confirmed"] else "✅ rejected"
        print(f"  {mark:22s}  {f['location_description']}")
    print(f"\n  가짜 거부율 (rejection rate):  {rejection_rate:5.1f}%  "
          f"({n_fake - n_confirmed_fake}/{n_fake} rejected)")
    print(f"  실제 위험 recall (유지율):       {real_recall * 100:5.1f}%")

    # 그래프
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    N = settings.GRID_SIZE
    conf_density = bee.confidence.max(axis=2)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(conf_density, cmap="Reds", vmin=0, vmax=1, alpha=0.85)
    cb = plt.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("신뢰도(confidence)")

    # 가짜 위치 표시
    for f in fake_status:
        if f["out"]:
            continue
        color = "red" if f["confirmed"] else "#22c55e"
        marker = "X" if f["confirmed"] else "o"
        ax.scatter(f["col"], f["row"], c=color, marker=marker, s=420,
                   edgecolors="black", linewidths=2.5,
                   label=("가짜 — confirmed (취약!)"
                          if f["confirmed"] else "가짜 — rejected (강건)"))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(),
                  loc="upper right", fontsize=11, framealpha=0.92)

    ax.set_title(
        f"Phase 6 — Adversarial 테스트: 의도적 가짜 뉴스 {n_fake}건 주입\n"
        f"거부율 {rejection_rate:.0f}%  ·  실제 위험 recall {real_recall * 100:.1f}% 유지",
        fontsize=13)
    ax.set_xlabel("격자 col (서 → 동)")
    ax.set_ylabel("격자 row (북 → 남)")

    os.makedirs("frames", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] saved -> {OUT_PNG}")

    print("\n" + "=" * 60)
    print(" 발표 요지")
    print("=" * 60)
    if n_confirmed_fake == 0:
        print(f"  '의도적 misinformation 5건 중 5건 모두 거부.'")
        print(f"  근거: news_confidence(0.3) < quorum_threshold(0.5).")
        print(f"        드론 보고 corroboration 없이는 confirm 불가.")
        print(f"  이중 검증으로 적의 정보 공격에 자동 강건.")
    else:
        print(f"  '5건 중 {n_confirmed_fake}건이 confirmed 됨 — 추가 강화 필요.'")


if __name__ == "__main__":
    main()
