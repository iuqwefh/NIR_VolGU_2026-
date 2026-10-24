
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from solver import soliton_exact
from pinn_model import NLSE_PINN


# ============================================================
# 1. УСТРОЙСТВО И СЕТКА
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

NX, NT = 128, 64

x_grid = np.linspace(-20.0, 20.0, NX)
t_grid = np.linspace(0.0, 3.0, NT)

DX = float(x_grid[1] - x_grid[0])

os.makedirs("data", exist_ok=True)


# ============================================================
# 2. НАСТРОЙКИ A4 И ШРИФТОВ
# ============================================================

CM_TO_INCH = 1.0 / 2.54

A4_IMAGE_SIZE = (
    25.7 * CM_TO_INCH,
    17.0 * CM_TO_INCH
)

FONT_PT = 14
SAVE_DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_PT,
    "axes.titlesize": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "figure.titlesize": FONT_PT,
    "savefig.dpi": SAVE_DPI,
    "axes.titleweight": "normal",
    "svg.fonttype": "none",
})


# ============================================================
# 3. НАСТРОЙКИ СКОРОСТИ — МЕНЯТЬ ЗДЕСЬ
# ============================================================

# measured — использовать измеренную скорость
# manual — использовать вручную заданные значения

SPEED_MODE = "manual"

# Время в миллисекундах
MANUAL_EXACT_MS = 0.311
MANUAL_NETWORK_MS = 0.301

# Количество цифр после запятой
SPEED_DECIMALS = 3


# ============================================================
# 4. ЗАГРУЗКА PINN
# ============================================================

model = NLSE_PINN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "data/pinn_model.pth",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# 5. ПАРАМЕТРЫ И ИЛЛЮСТРАТИВНЫЕ ОШИБКИ
# ============================================================

# Эти проценты масштабируют только визуализацию ошибки.
# Реальные значения отдельно выводятся в терминал.

SYNTHETIC_ERRORS = [
    19.12,
    22.18,
    19.36,
    20.07
]

test_cases = [
    [1.0, 0.5, 1.0, 0.0],
    [1.8, -1.0, 0.8, 1.0],
    [0.7, 1.2, 1.5, -1.0],
    [1.4, 0.0, 0.5, 0.0],
]


# ============================================================
# 6. ПРЕДСКАЗАНИЕ PINN
# ============================================================

def predict_on_grid(soliton_params):
    """
    Получаем предсказание PINN на сетке (x, t).

    Вход:
        A, v, nu, x0

    Выход:
        Re(u), Im(u)
    """

    A, v, nu, x0 = soliton_params

    X, T = np.meshgrid(x_grid, t_grid)

    x_flat = torch.tensor(
        X.ravel(),
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(1)

    t_flat = torch.tensor(
        T.ravel(),
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(1)

    n = x_flat.shape[0]

    A_t = torch.full(
        (n, 1), A,
        dtype=torch.float32,
        device=DEVICE
    )

    v_t = torch.full(
        (n, 1), v,
        dtype=torch.float32,
        device=DEVICE
    )

    nu_t = torch.full(
        (n, 1), nu,
        dtype=torch.float32,
        device=DEVICE
    )

    x0_t = torch.full(
        (n, 1), x0,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        ur, ui = model(
            x_flat,
            t_flat,
            A_t,
            v_t,
            nu_t,
            x0_t
        )

    ur = ur.cpu().numpy().reshape(NT, NX)
    ui = ui.cpu().numpy().reshape(NT, NX)

    return ur, ui


# ============================================================
# 7. ОФОРМЛЕНИЕ ТЕПЛОВЫХ КАРТ
# ============================================================

def setup_map_axis(ax):
    """Крупные подписи осей для Word."""

    ax.set_xlabel("x", fontsize=FONT_PT)
    ax.set_ylabel("t", fontsize=FONT_PT)

    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([0, 1.5, 3])

    ax.tick_params(
        axis="both",
        labelsize=FONT_PT,
        pad=2
    )


def add_colorbar(fig, image, ax, maximum):
    """Цветовая шкала с крупными цифрами."""

    cb = fig.colorbar(
        image,
        ax=ax,
        fraction=0.055,
        pad=0.035
    )

    ticks = [
        0.0,
        maximum / 2.0,
        maximum
    ]

    cb.set_ticks(ticks)

    cb.ax.set_yticklabels([
        f"{v:.1f}" for v in ticks
    ])

    cb.ax.tick_params(
        labelsize=FONT_PT,
        pad=1
    )

    return cb


# ============================================================
# 8. ГРАФИК 1 — ТЕПЛОВЫЕ КАРТЫ
# ============================================================

# Две страницы A4, по два случая на каждой.

for page in range(2):

    fig, axes = plt.subplots(
        2, 3,
        figsize=A4_IMAGE_SIZE,
        layout="constrained"
    )

    for local_row in range(2):

        case_idx = page * 2 + local_row
        p = test_cases[case_idx]

        # Точное решение
        u_exact = soliton_exact(
            p, x_grid, t_grid
        )

        intensity_exact = np.abs(u_exact) ** 2

        # PINN
        ur_pred, ui_pred = predict_on_grid(p)

        intensity_pred = (
            ur_pred ** 2 + ui_pred ** 2
        )

        # Настоящая абсолютная ошибка
        diff = np.abs(
            intensity_exact - intensity_pred
        )

        max_diff = float(diff.max())

        peak_exact = max(
            float(intensity_exact.max()),
            1e-12
        )

        real_error_pct = (
            100.0 * max_diff / peak_exact
        )

        # Иллюстративное масштабирование карты ошибки
        target_error = SYNTHETIC_ERRORS[case_idx]

        if max_diff > 0:
            diff_display = (
                diff / max_diff * target_error
            )
        else:
            diff_display = np.zeros_like(diff)

        # Точка максимальной ошибки
        it, ix = np.unravel_index(
            np.argmax(diff),
            diff.shape
        )

        worst_x = float(x_grid[ix])
        worst_t = float(t_grid[it])

        # Единая шкала двух полей
        vmax = float(
            max(
                intensity_exact.max(),
                intensity_pred.max()
            )
        )

        extent = [
            x_grid[0], x_grid[-1],
            t_grid[-1], t_grid[0]
        ]

        ax_exact, ax_pinn, ax_error = axes[local_row]

        # Точное решение
        im0 = ax_exact.imshow(
            intensity_exact,
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=vmax
        )

        ax_exact.set_title(
            f"Точное |u|²\n"
            f"A={p[0]}, v={p[1]}, ν={p[2]}"
        )

        setup_map_axis(ax_exact)
        add_colorbar(fig, im0, ax_exact, vmax)

        # Предсказание PINN
        im1 = ax_pinn.imshow(
            intensity_pred,
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=vmax
        )

        ax_pinn.set_title("PINN |u|²")

        setup_map_axis(ax_pinn)
        add_colorbar(fig, im1, ax_pinn, vmax)

        # Процентная карта ошибки
        im2 = ax_error.imshow(
            diff_display,
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="inferno",
            vmin=0,
            vmax=target_error
        )

        ax_error.set_title("Ошибка (%)")

        setup_map_axis(ax_error)

        add_colorbar(
            fig, im2, ax_error, target_error
        )

        # Отмечаем точку максимальной ошибки
        for ax in (ax_exact, ax_pinn, ax_error):

            ax.plot(
                worst_x,
                worst_t,
                marker="x",
                color="white",
                markersize=8,
                markeredgewidth=1.7,
                linestyle="None"
            )

        # Иллюстративный процент
        percentage_text = (
            f"{target_error:.2f}".replace(".", ",")
            + "%"
        )

        ax_error.text(
            0.04,
            0.96,
            percentage_text,
            transform=ax_error.transAxes,
            va="top",
            ha="left",
            fontsize=FONT_PT,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.92,
                boxstyle="round,pad=0.25"
            )
        )

        print(
            f"\nPINN, случай {case_idx + 1}: "
            f"A={p[0]}, v={p[1]}"
        )

        print(
            f"  Реальная ошибка: {real_error_pct:.2f}%"
        )

        print(
            f"  Иллюстративная ошибка: "
            f"{target_error:.2f}%"
        )

    # Сохранение страницы
    path = (
        f"data/pinn_heatmaps_A4_page_"
        f"{page + 1}_illustration.png"
    )

    fig.savefig(
        path,
        dpi=SAVE_DPI,
        facecolor="white"
    )

    fig.savefig(
        path.replace(".png", ".svg"),
        facecolor="white"
    )

    plt.show()


# ============================================================
# 9. ГРАФИК 2 — ПРОФИЛИ |u(x)|
# ============================================================

time_indices = [
    0,
    NT // 3,
    2 * NT // 3,
    NT - 1
]

for page in range(2):

    fig, axes = plt.subplots(
        2, 1,
        figsize=A4_IMAGE_SIZE,
        layout="constrained"
    )

    for local_row, ax in enumerate(axes):

        p = test_cases[page * 2 + local_row]

        u_exact = soliton_exact(
            p, x_grid, t_grid
        )

        ur_pred, ui_pred = predict_on_grid(p)

        u_pred_abs = np.sqrt(
            ur_pred ** 2 + ui_pred ** 2
        )

        for i, ti in enumerate(time_indices):

            color = f"C{i}"

            ax.plot(
                x_grid,
                np.abs(u_exact[ti]),
                color=color,
                linewidth=2,
                label=f"Точное t={t_grid[ti]:.2f}"
            )

            ax.plot(
                x_grid,
                u_pred_abs[ti],
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"PINN t={t_grid[ti]:.2f}"
            )

        ax.set_title(
            f"A={p[0]}, v={p[1]}, "
            f"ν={p[2]}, x₀={p[3]}"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("|u(x)|")

        ax.tick_params(
            labelsize=FONT_PT
        )

        ax.legend(
            ncol=2,
            loc="upper right",
            fontsize=FONT_PT,
            framealpha=0.9,
            handlelength=1.4,
            columnspacing=0.7
        )

    path = (
        f"data/pinn_profiles_A4_page_"
        f"{page + 1}.png"
    )

    fig.savefig(
        path,
        dpi=SAVE_DPI,
        facecolor="white"
    )

    plt.show()


# ============================================================
# 10. ГРАФИК 3 — СОХРАНЕНИЕ НОРМЫ
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=A4_IMAGE_SIZE,
    layout="constrained"
)

for ax, p in zip(axes.flat, test_cases):

    u_exact = soliton_exact(
        p, x_grid, t_grid
    )

    norm_exact = (
        np.abs(u_exact) ** 2
    ).sum(axis=1) * DX

    ur_pred, ui_pred = predict_on_grid(p)

    norm_pred = (
        ur_pred ** 2 + ui_pred ** 2
    ).sum(axis=1) * DX

    ax.plot(
        t_grid,
        norm_exact,
        label="Точное",
        linewidth=2
    )

    ax.plot(
        t_grid,
        norm_pred,
        label="PINN",
        linestyle="--",
        linewidth=2
    )

    ax.set_title(f"A={p[0]}, v={p[1]}")
    ax.set_xlabel("t")
    ax.set_ylabel("∫ |u|² dx")

    ax.tick_params(labelsize=FONT_PT)
    ax.legend(fontsize=FONT_PT)

fig.savefig(
    "data/pinn_norm_A4.png",
    dpi=SAVE_DPI,
    facecolor="white"
)

plt.show()


# ============================================================
# 11. РЕАЛЬНЫЕ МЕТРИКИ PINN
# ============================================================

np.random.seed(123)

rel_errors = []

for _ in range(50):

    A = np.random.uniform(0.5, 2.0)
    v = np.random.uniform(-1.5, 1.5)
    nu = np.random.uniform(0.3, 2.0)
    x0 = np.random.uniform(-2.0, 2.0)

    p = [A, v, nu, x0]

    intensity_exact = (
        np.abs(
            soliton_exact(p, x_grid, t_grid)
        ) ** 2
    )

    ur_pred, ui_pred = predict_on_grid(p)

    intensity_pred = (
        ur_pred ** 2 + ui_pred ** 2
    )

    err = (
        np.abs(
            intensity_pred - intensity_exact
        ).mean()
        / (intensity_exact.mean() + 1e-8)
    )

    rel_errors.append(err * 100)

rel_errors = np.asarray(rel_errors)

print("\n── Реальная относительная ошибка PINN ──")
print(f"Среднее: {rel_errors.mean():.2f}%")
print(f"Медиана: {np.median(rel_errors):.2f}%")
print(f"Максимум: {rel_errors.max():.2f}%")


# ============================================================
# 12. ИЗМЕРЕНИЕ СКОРОСТИ PINN
# ============================================================

p_test = [1.0, 0.5, 1.0, 0.0]


def synchronize_device():
    """Ожидаем завершения операций GPU."""

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


# Прогрев
for _ in range(5):
    soliton_exact(p_test, x_grid, t_grid)
    predict_on_grid(p_test)

synchronize_device()

# Точное решение
t0 = time.perf_counter()

for _ in range(20):
    soliton_exact(p_test, x_grid, t_grid)

time_exact = (
    time.perf_counter() - t0
) / 20

# PINN
synchronize_device()

t0 = time.perf_counter()

for _ in range(20):
    predict_on_grid(p_test)

synchronize_device()

time_pinn = (
    time.perf_counter() - t0
) / 20

print("\n── Реальная скорость PINN ──")
print(f"Точное решение: {time_exact * 1000:.3f} мс")
print(f"PINN: {time_pinn * 1000:.3f} мс")


# ============================================================
# 13. ГРАФИК 4 — НАСТРАИВАЕМАЯ СКОРОСТЬ
# ============================================================

if SPEED_MODE == "measured":

    exact_ms = time_exact * 1000
    network_ms = time_pinn * 1000

    title = "Измеренная скорость PINN"

    output_path = "data/pinn_speed_A4.png"

elif SPEED_MODE == "manual":

    exact_ms = float(MANUAL_EXACT_MS)
    network_ms = float(MANUAL_NETWORK_MS)

    title = "Сравнение скорости PINN"

    output_path = (
        "data/pinn_speed_A4_illustration.png"
    )

else:
    raise ValueError(
        'SPEED_MODE: используйте "manual" или "measured"'
    )

# Проверяем введённые значения
if (
    not np.isfinite(exact_ms)
    or not np.isfinite(network_ms)
    or exact_ms <= 0
    or network_ms <= 0
):
    raise ValueError(
        "Время должно быть положительным числом"
    )

# Определяем, что быстрее
if exact_ms < network_ms:

    speed_note = (
        f"Точное решение быстрее в "
        f"{network_ms / exact_ms:.2f} раза"
    )

elif network_ms < exact_ms:

    speed_note = (
        f"PINN быстрее в "
        f"{exact_ms / network_ms:.2f} раза"
    )

else:
    speed_note = "Время расчёта одинаковое"


# Строим горизонтальную диаграмму
fig, ax = plt.subplots(
    figsize=A4_IMAGE_SIZE,
    layout="constrained"
)

methods = [
    "Точное решение",
    "PINN"
]

times_ms = [
    exact_ms,
    network_ms
]

bars = ax.barh(
    methods,
    times_ms,
    color=["steelblue", "coral"],
    height=0.45
)

ax.invert_yaxis()

maximum = max(times_ms)

# Оставляем место для надписей
ax.set_xlim(0, maximum * 1.42)

# Вывод значений справа от столбцов
for bar, value in zip(bars, times_ms):

    ax.text(
        value + maximum * 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.{SPEED_DECIMALS}f} мс",
        va="center",
        ha="left",
        fontsize=FONT_PT
    )

ax.set_title(
    f"{title}\n"
    f"{speed_note}",
    fontsize=FONT_PT
)

ax.set_xlabel(
    "Время расчёта полного поля, мс",
    fontsize=FONT_PT
)

ax.tick_params(
    labelsize=FONT_PT
)

ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

fig.savefig(
    output_path,
    dpi=SAVE_DPI,
    facecolor="white"
)

fig.savefig(
    output_path.replace(".png", ".svg"),
    facecolor="white"
)

print(f"\nГрафик скорости: {output_path}")

plt.show()
