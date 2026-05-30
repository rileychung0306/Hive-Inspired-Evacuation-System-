"""experiments/ablation.py

Phase 6 - 검증 실험 #2: Ablation study (각 컴포넌트 기여도 정량화).

심사위원 질문: "각 컴포넌트가 정말 필요한가요? 빼면 어떻게 되나요?"

5가지 시나리오 비교:
  1) baseline                — 모든 컴포넌트 ON
  2) recruitment OFF        — 드론 waggle-dance 비활성 (균등 임의 패트롤)
  3) decay OFF              — 신뢰도가 시간에 따라 잊혀지지 않음
  4) news OFF               — LLM 뉴스 융합 끔 (드론만)
  5) recruitment + news OFF — 기본 Phase 2 수준 (단순 분산 드론)

각 시나리오의 정밀도/재현율 막대그래프 -> KCF 슬라이드 한 장.

실행: python experiments/ablation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import matplotlib
import numpy as np

from ai.bee_voting import BeeVoting
from ai.news_collector import fetch_news
from ai.llm_analyzer import extract_hazards_from_all
from ai.news_fusion import apply_news_to_bee
from config import settings
from hazard_modules.conflict import conflict_module
from simulation.acled_loader import load_acled_initial
from simulation.cellular_automata import HazardField
from simulation.city import build_city
from simulation.drone import DroneSwarm

OUT_PNG = "frames/phase6_ablation.png"
THRESH = 0.15
N_STEPS = 150


def _metrics(field, bee, thresh=THRESH):
    true_cells = field.field.max(axis=2) > thresh
    conf_cells = bee.confirmed_mask().any(axis=2)
    nt = int(true_cells.sum())
    nc = int(conf_cells.sum())
    tp = int((true_cells & conf_cells).sum())
    return {
        "n_true": nt, "n_confirmed": nc, "tp": tp,
        "precision": tp / nc if nc else 0.0,
        "recall": tp / nt if nt else 0.0,
    }


def run_scenario(label, *, recruitment=True, decay=True, news=True,
                  steps=N_STEPS):
    print(f"\n[{label}]")
    saved_recruit = settings.DRONE_RECRUIT_PROB
    saved_decay = settings.CONFIDENCE_DECAY
    if not recruitment:
        settings.DRONE_RECRUIT_PROB = 0.0
    if not decay:
        settings.CONFIDENCE_DECAY = 0.0
    try:
        city, _ = build_city(seed=42)
        initial, bbox = load_acled_initial(conflict_module)
        field = HazardField(conflict_module, initial=initial)
        swarm = DroneSwarm(conflict_module)
        bee = BeeVoting(conflict_module)
        if news:
            try:
                arts = fetch_news(conflict_module.news_keywords)
                if arts:
                    hazards, _ = extract_hazards_from_all(arts, conflict_module)
                    apply_news_to_bee(hazards, bee, conflict_module, bbox)
            except Exception as e:
                print(f"  뉴스 무시: {e}")
        for _ in range(steps):
            field.step()
            swarm.step(field.field, bee)
        m = _metrics(field, bee)
        print(f"  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"(confirmed {m['n_confirmed']}, true {m['n_true']})")
        return {"label": label, **m}
    finally:
        settings.DRONE_RECRUIT_PROB = saved_recruit
        settings.CONFIDENCE_DECAY = saved_decay


def plot_bars(results):
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    recalls = [r["recall"] * 100 for r in results]
    precs = [r["precision"] * 100 for r in results]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.5))
    b1 = ax.bar(x - width / 2, precs, width, label="정밀도 (precision)",
                color="#3b82f6")
    b2 = ax.bar(x + width / 2, recalls, width, label="재현율 (recall)",
                color="#ef4444")
    ax.set_ylabel("성능 (%)", fontsize=12)
    ax.set_title("Phase 6 — Ablation: 각 컴포넌트가 기여하는 정도",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(precs), max(recalls)) * 1.18)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

    os.makedirs("frames", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] saved -> {OUT_PNG}")


def main():
    print("=" * 60)
    print(" Phase 6 #2: Ablation study")
    print("=" * 60)

    results = [
        run_scenario("① baseline (전체 ON)",
                     recruitment=True, decay=True, news=True),
        run_scenario("② recruitment OFF",
                     recruitment=False, decay=True, news=True),
        run_scenario("③ decay OFF",
                     recruitment=True, decay=False, news=True),
        run_scenario("④ news OFF",
                     recruitment=True, decay=True, news=False),
        run_scenario("⑤ recruit + news OFF (Phase 2 수준)",
                     recruitment=False, decay=True, news=False),
    ]

    print("\n" + "=" * 60)
    print(" 해석 (baseline 기준 Δ)")
    print("=" * 60)
    base = results[0]
    for r in results[1:]:
        d_p = (r["precision"] - base["precision"]) * 100
        d_r = (r["recall"] - base["recall"]) * 100
        impact = "↓ 큰 영향" if abs(d_r) > 3 else "→ 미미"
        print(f"  {r['label']:36s}  Δprec {d_p:+5.1f}%p, "
              f"Δrecall {d_r:+5.1f}%p  {impact}")

    plot_bars(results)


if __name__ == "__main__":
    main()
