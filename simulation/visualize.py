"""simulation/visualize.py

도시 + 위험 지도를 '그림'으로 만드는 코드입니다.
  - compose_rgb()  : 색 배열 계산 (Pygame/Matplotlib 둘 다 사용하는 공통 함수)
  - save_snapshot(): PNG 한 장으로 저장 (검증 + 발표 자료용, Matplotlib 사용)
  - run_pygame()   : 실시간 창으로 시뮬레이션을 보여줌 (데모용, Pygame 사용)
"""

import os
import numpy as np
from config import settings


def compose_rgb(city_grid, field, hazard_module):
    """도시 바탕색 위에 위험 색을 강도만큼 덧칠한 RGB 이미지(배열)를 만듭니다."""
    N = settings.GRID_SIZE
    rgb = np.zeros((N, N, 3), dtype=np.float32)

    # 1) 도시 바탕색 (공터/건물/도로/대피소)
    for cell_type, color in settings.CELL_COLORS.items():
        rgb[city_grid == cell_type] = color

    # 2) 위험 색을 강도(0~1)만큼 위에 덧칠
    for k, spec in enumerate(hazard_module.hazards):
        a = np.clip(field[:, :, k], 0, 1)[:, :, None]   # 투명도처럼 사용
        rgb = rgb * (1 - a) + np.array(spec.color, dtype=np.float32) * a

    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_snapshot(city_grid, field, hazard_module, path, title=None):
    """현재 상태를 PNG 한 장으로 저장 (범례 포함)."""
    import matplotlib
    matplotlib.use("Agg")                       # 창 없이 파일로만 저장
    matplotlib.rcParams["font.family"] = "AppleGothic"   # 한글 깨짐 방지(맥)
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rgb = compose_rgb(city_grid, field, hazard_module)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)

    handles = [Patch(facecolor=np.array(s.color) / 255, label=s.display_name)
               for s in hazard_module.hazards]
    handles.append(Patch(facecolor=np.array(settings.CELL_COLORS[settings.CELL_SHELTER]) / 255,
                         label="대피소"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_drones_pygame(screen, drones):
    """드론을 파란 점 + 반투명 시야 원으로 그립니다."""
    import pygame
    px = settings.CELL_PIXELS
    for d in drones:
        x, y = int(d.pos[1] * px), int(d.pos[0] * px)
        pygame.draw.circle(screen, (60, 130, 255), (x, y), int(d.vision * px), 1)
        pygame.draw.circle(screen, (10, 60, 230), (x, y), max(3, px), 0)


def run_pygame(city_grid, field_obj, swarm, bee, hazard_module,
               steps_per_frame=1, max_steps=None):
    """실시간 창으로 시뮬레이션 보기.
    화면에는 기본적으로 '드론 군집이 확인한 위험'(꿀벌 투표 결과)을 보여줍니다.
    키: T = 실제 위험/확인된 위험 전환,  ESC = 종료.
    """
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    show_truth = False

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_t:
                show_truth = not show_truth

        for _ in range(steps_per_frame):
            field_obj.step()
            swarm.step(field_obj.field, bee)

        field_to_show = field_obj.field if show_truth else bee.confirmed_field()
        rgb = compose_rgb(city_grid, field_to_show, hazard_module)
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))
        screen.blit(surf, (0, 0))
        _draw_drones_pygame(screen, swarm.drones)

        mode = "실제 위험(정답)" if show_truth else "드론이 확인한 위험"
        pygame.display.set_caption(f"Hive Evac Phase 2 — {mode}  (T:전환, ESC:종료)  step {field_obj.step_count}")
        pygame.display.flip()
        clock.tick(settings.FPS)

        if max_steps and field_obj.step_count >= max_steps:
            running = False

    pygame.quit()


def save_snapshot_swarm(city_grid, true_field, bee, drones, hazard_module, path, title=None):
    """두 칸 비교 그림: (왼쪽) 실제 위험  vs  (오른쪽) 드론 군집이 확인한 위험 + 드론."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Circle

    rgb_true = compose_rgb(city_grid, true_field, hazard_module)
    rgb_conf = compose_rgb(city_grid, bee.confirmed_field(), hazard_module)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].imshow(rgb_true)
    axes[0].set_title("실제 위험 (정답)")
    axes[1].imshow(rgb_conf)
    axes[1].set_title("드론 군집이 확인한 위험 (꿀벌 투표)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    for d in drones:
        axes[1].add_patch(Circle((d.pos[1], d.pos[0]), d.vision,
                                 fill=False, color="#3b82f6", lw=0.6, alpha=0.6))
        axes[1].plot(d.pos[1], d.pos[0], "o", color="#1e3aff", ms=4)

    handles = [Patch(facecolor=np.array(s.color) / 255, label=s.display_name)
               for s in hazard_module.hazards]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    if title:
        fig.suptitle(title)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
