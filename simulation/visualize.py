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


def _recall_at(field_obj, bee, thresh=0.15):
    """현재 시점의 재현율(recall) — 진짜 위험 중 군집이 확인한 비율."""
    true_cells = field_obj.field.max(axis=2) > thresh
    conf_cells = bee.confirmed_mask().any(axis=2)
    nt = int(true_cells.sum())
    if nt == 0:
        return 0.0
    tp = int((true_cells & conf_cells).sum())
    return tp / nt


def _draw_drones_on_panel(screen, drones, x_off, y_off, panel_cell_pixels):
    """드론을 패널 내부 좌표(x_off, y_off)에 맞춰 그림."""
    import pygame
    px = panel_cell_pixels
    for d in drones:
        x = int(x_off + d.pos[1] * px)
        y = int(y_off + d.pos[0] * px)
        pygame.draw.circle(screen, (60, 130, 255), (x, y), max(2, int(d.vision * px)), 1)
        pygame.draw.circle(screen, (10, 60, 230), (x, y), max(2, px), 0)


def run_pygame_compare(city_grid, field_a, swarm_a, bee_a,
                       field_b, swarm_b, bee_b, hazard_module,
                       steps_per_frame=1, panel_cell_pixels=2, max_steps=None):
    """3분할 라이브 비교 창:
       ① 실제 위험 (정답)  |  ② 드론만  |  ③ 드론 + 뉴스

    field_a/swarm_a/bee_a = 뉴스 OFF (Phase 2 비교용)
    field_b/swarm_b/bee_b = 뉴스 ON  (Phase 3)
    ESC 또는 창 닫기로 종료.
    """
    import pygame
    pygame.init()
    pygame.font.init()
    px = panel_cell_pixels
    panel_w = settings.GRID_SIZE * px
    gap = 10
    label_h = 30
    stat_h = 24
    win_w = 3 * panel_w + 2 * gap
    win_h = panel_w + label_h + stat_h + 6
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Hive Evac — 3분할 비교 (정답 vs 드론만 vs 드론+뉴스)")

    font_lbl = pygame.font.SysFont("applegothic", 16, bold=True)
    font_stat = pygame.font.SysFont("applegothic", 13)

    clock = pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

        for _ in range(steps_per_frame):
            field_a.step()
            swarm_a.step(field_a.field, bee_a)
            field_b.step()
            swarm_b.step(field_b.field, bee_b)

        screen.fill((245, 245, 245))

        # 세 패널의 RGB 만들기
        rgb_truth = compose_rgb(city_grid, field_b.field, hazard_module)
        rgb_no    = compose_rgb(city_grid, bee_a.confirmed_field(), hazard_module)
        rgb_with  = compose_rgb(city_grid, bee_b.confirmed_field(), hazard_module)

        for i, rgb in enumerate([rgb_truth, rgb_no, rgb_with]):
            surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (panel_w, panel_w))
            screen.blit(surf, (i * (panel_w + gap), label_h))

        # 드론 (패널 2, 3 위에만)
        _draw_drones_on_panel(screen, swarm_a.drones, panel_w + gap, label_h, px)
        _draw_drones_on_panel(screen, swarm_b.drones, 2 * (panel_w + gap), label_h, px)

        # 패널 라벨
        labels = ["① 실제 위험 (정답)", "② 드론만 (Phase 2)", "③ 드론 + 뉴스 (Phase 3)"]
        for i, lbl in enumerate(labels):
            text = font_lbl.render(lbl, True, (20, 20, 20))
            screen.blit(text, (i * (panel_w + gap) + 8, 6))

        # 라이브 수치: step + 재현율 비교
        rec_no  = _recall_at(field_a, bee_a)
        rec_yes = _recall_at(field_b, bee_b)
        gap_pct = (rec_yes - rec_no) * 100
        stat_y = label_h + panel_w + 4
        screen.blit(font_stat.render(f"step {field_a.step_count}", True, (50, 50, 50)),
                    (8, stat_y))
        screen.blit(font_stat.render(f"재현율(recall) = {rec_no:.2f}", True, (50, 50, 50)),
                    (panel_w + gap + 8, stat_y))
        sign = "+" if gap_pct >= 0 else ""
        screen.blit(font_stat.render(
            f"재현율(recall) = {rec_yes:.2f}    ← 뉴스 융합 효과 {sign}{gap_pct:.1f}%p",
            True, (200, 30, 30)),
            (2 * (panel_w + gap) + 8, stat_y))

        pygame.display.flip()
        clock.tick(settings.FPS)

        if max_steps and field_a.step_count >= max_steps:
            running = False

    pygame.quit()


def save_snapshot_compare(city_grid, field_a, swarm_a, bee_a,
                          field_b, swarm_b, bee_b, hazard_module,
                          path, title=None):
    """3분할 PNG: [실제 위험] | [드론만] | [드론+뉴스]. (Matplotlib, 발표용)"""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Circle

    rgb_truth = compose_rgb(city_grid, field_b.field, hazard_module)
    rgb_no    = compose_rgb(city_grid, bee_a.confirmed_field(), hazard_module)
    rgb_with  = compose_rgb(city_grid, bee_b.confirmed_field(), hazard_module)

    rec_no  = _recall_at(field_a, bee_a)
    rec_yes = _recall_at(field_b, bee_b)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    axes[0].imshow(rgb_truth); axes[0].set_title("① 실제 위험 (정답)", fontsize=13)
    axes[1].imshow(rgb_no);    axes[1].set_title(f"② 드론만 (Phase 2) — 재현율 {rec_no:.2f}", fontsize=13)
    axes[2].imshow(rgb_with);  axes[2].set_title(f"③ 드론 + 뉴스 (Phase 3) — 재현율 {rec_yes:.2f}", fontsize=13)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    for d in swarm_a.drones:
        axes[1].add_patch(Circle((d.pos[1], d.pos[0]), d.vision,
                                 fill=False, color="#3b82f6", lw=0.6, alpha=0.6))
        axes[1].plot(d.pos[1], d.pos[0], "o", color="#1e3aff", ms=4)
    for d in swarm_b.drones:
        axes[2].add_patch(Circle((d.pos[1], d.pos[0]), d.vision,
                                 fill=False, color="#3b82f6", lw=0.6, alpha=0.6))
        axes[2].plot(d.pos[1], d.pos[0], "o", color="#1e3aff", ms=4)

    handles = [Patch(facecolor=np.array(s.color) / 255, label=s.display_name)
               for s in hazard_module.hazards]
    axes[2].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    if title:
        fig.suptitle(title, fontsize=14)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def save_snapshot_routes(city_grid, risk_map, hazard_module, routes,
                          start_latlon, bbox, path, title=None):
    """가족 구성별 안전 경로 비교 그림 (Phase 4 발표용).

    경로 + 도시 + 위험을 보여줌. 4개 이상이면 2x2 격자, 그 외는 1행.
    경로가 잘 보이도록 도시 영역으로 확대(zoom)합니다.
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from routing.graph_builder import latlon_to_cell

    N = settings.GRID_SIZE
    # 도시 + 위험 합성 (위험은 빨강으로)
    rgb_bg = np.zeros((N, N, 3), dtype=np.float32)
    for cell_type, color in settings.CELL_COLORS.items():
        rgb_bg[city_grid == cell_type] = color
    a = np.clip(risk_map, 0, 1)[:, :, None]
    rgb_bg = rgb_bg * (1 - a * 0.8) + np.array([220, 50, 30], dtype=np.float32) * (a * 0.8)
    rgb_bg = np.clip(rgb_bg, 0, 255).astype(np.uint8)

    sr, sc = latlon_to_cell(start_latlon[0], start_latlon[1], bbox)

    # Zoom 영역: 출발점 + (도달 가능한) 대피소를 포함하도록, 여유 10칸
    rs = [sr]
    cs = [sc]
    for r in routes:
        if r.get("shelter") and r["shelter"].get("lat") is not None:
            shr, shc = latlon_to_cell(r["shelter"]["lat"], r["shelter"]["lon"], bbox)
            rs.append(shr); cs.append(shc)
        for lat, lon in r.get("path_coords", []) or []:
            rr, cc = latlon_to_cell(lat, lon, bbox)
            rs.append(rr); cs.append(cc)
    pad = 8
    r_min, r_max = max(0, min(rs) - pad), min(N - 1, max(rs) + pad)
    c_min, c_max = max(0, min(cs) - pad), min(N - 1, max(cs) + pad)

    n = len(routes)
    if n >= 3:
        nrows, ncols = 2, (n + 1) // 2
        figsize = (7 * ncols, 7 * nrows)
    else:
        nrows, ncols = 1, n
        figsize = (8 * n, 7)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, route in zip(axes, routes):
        ax.imshow(rgb_bg)
        ax.set_xlim(c_min, c_max)
        ax.set_ylim(r_max, r_min)  # imshow 의 y축이 위->아래라서 반전
        ax.set_xticks([]); ax.set_yticks([])

        # 도달 불가 케이스
        if not route.get("path_coords") or route.get("shelter") is None:
            ax.set_title(
                f"{route['profile_label']}\n"
                f"[경고] 도달 가능한 대피소 없음 "
                f"(체력 한계 {route.get('max_distance_m', 0)}m 초과)\n"
                f"시스템 권고: 차량 지원 / 이웃 도움 필요",
                fontsize=11, color="#b91c1c")
            ax.plot(sc, sr, "o", ms=15, color="#22c55e",
                    markeredgecolor="white", mew=2, label="출발 (대피 불가)", zorder=5)
            ax.legend(loc="lower right", fontsize=10, framealpha=0.92)
            continue

        time_str = (f"{route['travel_time_min']:.1f}분"
                    if "travel_time_min" in route else "")
        ax.set_title(
            f"{route['profile_label']}\n"
            f"→ {route['shelter']['name']}, "
            f"{route['total_meters']:.0f}m"
            + (f" / {time_str}" if time_str else "")
            + f", 평균 위험 {route['avg_risk']:.2f}",
            fontsize=12)

        coords_rc = [latlon_to_cell(lat, lon, bbox) for lat, lon in route["path_coords"]]
        rows = [rc[0] for rc in coords_rc]
        cols = [rc[1] for rc in coords_rc]
        ax.plot(cols, rows, "-", lw=4.0, color="#00aaff", alpha=0.95, label="경로")
        ax.plot(sc, sr, "o", ms=15, color="#22c55e",
                markeredgecolor="white", mew=2, label="출발", zorder=5)
        sh_r, sh_c = latlon_to_cell(route["shelter"]["lat"],
                                     route["shelter"]["lon"], bbox)
        ax.plot(sh_c, sh_r, "s", ms=16, color="#10b981",
                markeredgecolor="black", mew=2, label=f"대피소: {route['shelter']['name']}", zorder=5)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.92)

    # 사용 안 한 panel 끄기
    for ax in axes[len(routes):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


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
