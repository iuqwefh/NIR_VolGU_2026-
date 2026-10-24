import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from invers_model import InverseNLSE


# ── Настройки ─────────────────────────────────────────────────
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
NX, NT = 128, 64
X_RANGE = (-20.0, 20.0)
T_RANGE = (0.0, 3.0)
N_WARMUP = 20
N_REPEATS_SINGLE = 100
N_TEST = 50

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

x = np.linspace(*X_RANGE, NX)
t = np.linspace(*T_RANGE, NT)
X_GRID, T_GRID = np.meshgrid(x, t)

print(f"Устройство сети: {DEVICE}")


def synchronize_device() -> None:
    """Ждём завершения асинхронных операций ускорителя."""
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def soliton_exact(params, x_grid, t_grid):
    """Точная аналитическая формула солитона."""
    A, v, nu, x0 = params

    # Nelder-Mead не поддерживает bounds напрямую и может зайти в nu <= 0.
    # В таком случае возвращаем NaN-поле, а loss ниже выдаст большой штраф.
    if nu <= 0:
        return np.full((len(t_grid), len(x_grid)), np.nan + 1j * np.nan)

    X, T = np.meshgrid(x_grid, t_grid)
    k = A * np.sqrt(nu) / np.sqrt(2)
    omega = v**2 / 4 - A**2 * nu / 2
    envelope = A / np.cosh(k * (X - x0 - v * T))
    phase = v / 2 * X - omega * T
    return envelope * np.exp(1j * phase)


def rel_or_abs_error(true_v, est_v):
    if abs(true_v) < 1e-3:
        return abs(est_v - true_v), True
    return abs(est_v - true_v) / abs(true_v) * 100, False


# ── Загрузка модели ───────────────────────────────────────────
model = InverseNLSE(NT=NT, NX=NX).to(DEVICE)
model.load_state_dict(
    torch.load(DATA_DIR / "inverse_model.pth", map_location=DEVICE)
)
model.eval()


def fields_to_tensors(u_real, u_imag):
    """Преобразует один пример или batch полей в float32-тензоры устройства."""
    ur_np = np.asarray(u_real, dtype=np.float32)
    ui_np = np.asarray(u_imag, dtype=np.float32)

    if ur_np.ndim == 2:
        ur_np = ur_np[None, ...]
        ui_np = ui_np[None, ...]

    ur_t = torch.from_numpy(np.ascontiguousarray(ur_np)).to(DEVICE)
    ui_t = torch.from_numpy(np.ascontiguousarray(ui_np)).to(DEVICE)
    return ur_t, ui_t


def network_forward(ur_t, ui_t):
    """Чистый forward; возвращает тензор формы (batch, 4)."""
    with torch.inference_mode():
        outputs = model(ur_t, ui_t)
        return torch.stack(outputs, dim=1)


def warmup_network(ur_t, ui_t, n_warmup=N_WARMUP) -> None:
    """Прогрев backend и рабочих буферов для заданной формы batch."""
    for _ in range(n_warmup):
        _ = network_forward(ur_t, ui_t)
    synchronize_device()


def benchmark_network(ur_t, ui_t, repeats=1):
    """
    Измеряет только чистый инференс уже подготовленных тензоров.

    Возвращает:
        predictions: ndarray (batch, 4)
        mean_time: среднее время одного forward, секунд
    """
    warmup_network(ur_t, ui_t)

    synchronize_device()
    t0 = time.perf_counter()
    pred_t = None
    for _ in range(repeats):
        pred_t = network_forward(ur_t, ui_t)
    synchronize_device()
    elapsed = time.perf_counter() - t0

    assert pred_t is not None
    predictions = pred_t.cpu().numpy()
    return predictions, elapsed / repeats


def run_nelder_mead(
    u_meas,
    initial_guess=(0.8, 0.5, 0.7, 0.5),
    maxiter=1000,
    n_starts=6,
):
    """Nelder–Mead с несколькими начальными точками.

    Один запуск Nelder–Mead может попасть в локальный минимум. Поэтому
    выполняются несколько запусков в допустимой области и выбирается решение
    с минимальной невязкой восстановленного поля. Значения ошибок не меняются.
    """

    lower = np.array([0.5, -1.5, 0.3, -2.0], dtype=float)
    upper = np.array([2.0,  1.5, 2.0,  2.0], dtype=float)
    bounds = list(zip(lower, upper))

    def loss(params_guess):
        params_guess = np.asarray(params_guess, dtype=float)
        if np.any(params_guess < lower) or np.any(params_guess > upper):
            return 1e12

        u_pred = soliton_exact(params_guess, x, t)
        if not np.all(np.isfinite(u_pred)):
            return 1e12
        return float(np.mean(np.abs(u_pred - u_meas) ** 2))

    # Стабильный набор стартов: исходная точка, центр области и точки,
    # равномерно покрывающие допустимое пространство параметров.
    starts = [
        np.clip(np.asarray(initial_guess, dtype=float), lower, upper),
        (lower + upper) / 2.0,
        np.array([1.5,  1.0, 1.0,  0.0]),
        np.array([1.0, -1.0, 1.5, -1.0]),
        np.array([1.8,  0.0, 0.6,  1.0]),
        np.array([0.7,  1.2, 1.8, -1.5]),
        np.array([1.3, -0.5, 0.9,  1.5]),
        np.array([1.9, -1.3, 1.3,  0.5]),
    ]
    starts = starts[:max(1, n_starts)]

    best_result = None
    best_loss = np.inf
    total_iterations = 0

    for start_point in starts:
        result = minimize(
            loss,
            start_point,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8},
        )
        total_iterations += int(result.nit)
        current_loss = loss(result.x)
        if current_loss < best_loss:
            best_loss = current_loss
            best_result = result

    assert best_result is not None
    return best_result.x, total_iterations


# ══════════════════════════════════════════════════════════════
# БЛОК 1: одиночный тестовый случай
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("БЛОК 1: одиночный тестовый случай")
print("=" * 60)

true_params = [1.5, 1.0, 1.0, 0.0]
u_true = soliton_exact(true_params, x, t)

np.random.seed(0)
noise_level = 0.03 * np.max(np.abs(u_true))
noise = noise_level * (
    np.random.randn(*u_true.shape) + 1j * np.random.randn(*u_true.shape)
)
u_meas = u_true + noise

# Нелдер-Мид: полный вызов оптимизатора.
t0 = time.perf_counter()
nm_params, nm_iters = run_nelder_mead(u_meas, n_starts=6)
time_nm = time.perf_counter() - t0

# Сеть: подготовка входа отдельно от чистого инференса.
t0 = time.perf_counter()
ur_single, ui_single = fields_to_tensors(u_meas.real, u_meas.imag)
synchronize_device()
time_prepare_single = time.perf_counter() - t0

net_pred_batch, time_net = benchmark_network(
    ur_single,
    ui_single,
    repeats=N_REPEATS_SINGLE,
)
net_params = net_pred_batch[0]

param_names = ["A", "v", "nu", "x0"]
print(
    f"\n{'Параметр':<10}{'Истина':<12}{'Нелдер-Мид':<14}"
    f"{'Сеть':<14}{'Ошибка НМ':<14}{'Ошибка сети'}"
)
for i, name in enumerate(param_names):
    err_nm, is_abs_nm = rel_or_abs_error(true_params[i], nm_params[i])
    err_net, is_abs_net = rel_or_abs_error(true_params[i], net_params[i])
    fmt_nm = f"{err_nm:.4f}(абс)" if is_abs_nm else f"{err_nm:.2f}%"
    fmt_net = f"{err_net:.4f}(абс)" if is_abs_net else f"{err_net:.2f}%"
    print(
        f"{name:<10}{true_params[i]:<12.4f}{nm_params[i]:<14.4f}"
        f"{net_params[i]:<14.4f}{fmt_nm:<14}{fmt_net}"
    )

print(f"\nВремя Нелдера-Мида:       {time_nm * 1000:.2f} мс ({nm_iters} итераций)")
print(f"Подготовка входа сети:     {time_prepare_single * 1000:.3f} мс")
print(
    f"Чистый инференс сети:      {time_net * 1000:.3f} мс "
    f"(среднее по {N_REPEATS_SINGLE} прогретым проходам)"
)
print(
    f"Ускорение чистого инференса: {time_nm / time_net:.1f}x "
    f"относительно Нелдера-Мида"
)

# ── График 1 ──────────────────────────────────────────────────
u_pred_nm = soliton_exact(nm_params, x, t)
u_pred_net = soliton_exact(net_params, x, t)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
vmax = np.max(np.abs(u_true) ** 2) * 1.1
titles = ["Истинное поле", "Измеренное (шум 3%)", "Нелдер-Мид", "Гибридная сеть"]
data = [u_true, u_meas, u_pred_nm, u_pred_net]

for ax, title, field in zip(axes, titles, data):
    im = ax.imshow(
        np.abs(field) ** 2,
        extent=[x[0], x[-1], t[-1], t[0]],
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[0].set_ylabel("t")

plt.suptitle("Сравнение: истина / измерение / Нелдер-Мид / гибридная сеть", y=1.05)
plt.tight_layout()
plt.savefig(DATA_DIR / "inverse_comparison_single.png", dpi=150, bbox_inches="tight")
plt.show()
print("Сохранён: data/inverse_comparison_single.png")

# ── График 2: сводная таблица ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis("off")

table_data = [["Параметр", "Истина", "Нелдер-Мид", "Сеть", "Ошибка НМ", "Ошибка сети"]]
for i, name in enumerate(param_names):
    err_nm, is_abs_nm = rel_or_abs_error(true_params[i], nm_params[i])
    err_net, is_abs_net = rel_or_abs_error(true_params[i], net_params[i])
    fmt_nm = f"{err_nm:.4f}(абс)" if is_abs_nm else f"{err_nm:.2f}%"
    fmt_net = f"{err_net:.4f}(абс)" if is_abs_net else f"{err_net:.2f}%"
    table_data.append(
        [
            name,
            f"{true_params[i]:.4f}",
            f"{nm_params[i]:.4f}",
            f"{net_params[i]:.4f}",
            fmt_nm,
            fmt_net,
        ]
    )
table_data.append(
    [
        "Время",
        "-",
        f"{time_nm * 1000:.1f} мс",
        f"{time_net * 1000:.3f} мс",
        "-",
        "-",
    ]
)

table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.1, 1.6)
for i in range(6):
    table[(0, i)].set_facecolor("#40466e")
    table[(0, i)].set_text_props(weight="bold", color="white")
ax.set_title("Сравнение методов: Нелдер-Мид vs гибридная сеть", fontsize=15, pad=15)
plt.tight_layout()
plt.savefig(DATA_DIR / "inverse_results_table.png", dpi=150, bbox_inches="tight")
plt.show()
print("Сохранён: data/inverse_results_table.png")


# ══════════════════════════════════════════════════════════════
# БЛОК 2: массовое сравнение
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"БЛОК 2: массовое сравнение ({N_TEST} случайных примеров)")
print("=" * 60)

np.random.seed(123)
n_test = N_TEST

A_test = np.random.uniform(0.5, 2.0, n_test)
v_test = np.random.uniform(-1.5, 1.5, n_test)
nu_test = np.random.uniform(0.3, 2.0, n_test)
x0_test = np.random.uniform(-2.0, 2.0, n_test)

errors_nm = {name: [] for name in param_names}
errors_net = {name: [] for name in param_names}
times_nm = []
nm_preds_all = np.zeros((n_test, 4), dtype=np.float64)

# Сначала генерируем один и тот же набор измерений для обоих методов.
u_meas_batch = np.empty((n_test, NT, NX), dtype=np.complex128)
for i in range(n_test):
    true_p = [A_test[i], v_test[i], nu_test[i], x0_test[i]]
    u_true_i = soliton_exact(true_p, x, t)
    noise_lvl = 0.03 * np.max(np.abs(u_true_i))
    noise_i = noise_lvl * (
        np.random.randn(*u_true_i.shape) + 1j * np.random.randn(*u_true_i.shape)
    )
    u_meas_batch[i] = u_true_i + noise_i

print("Считаем Нелдер-Мид...")
for i in range(n_test):
    t0 = time.perf_counter()
    nm_p, _ = run_nelder_mead(u_meas_batch[i], maxiter=600, n_starts=6)
    times_nm.append(time.perf_counter() - t0)
    nm_preds_all[i] = nm_p

print("Считаем сеть одним batch...")
# Подготовку batch показываем отдельно.
t0 = time.perf_counter()
ur_batch, ui_batch = fields_to_tensors(u_meas_batch.real, u_meas_batch.imag)
synchronize_device()
time_prepare_batch = time.perf_counter() - t0

# Один прогретый batched forward.
net_preds_all, time_net_batch = benchmark_network(ur_batch, ui_batch, repeats=1)
time_net_per_sample = time_net_batch / n_test

true_vals = [A_test, v_test, nu_test, x0_test]
for i in range(n_test):
    true_p = [values[i] for values in true_vals]
    for j, name in enumerate(param_names):
        err_nm, is_abs = rel_or_abs_error(true_p[j], nm_preds_all[i, j])
        err_net, _ = rel_or_abs_error(true_p[j], net_preds_all[i, j])
        if not is_abs:
            errors_nm[name].append(err_nm)
            errors_net[name].append(err_net)

print("Готово.\n")
print(
    f"{'Параметр':<10}{'НМ среднее':<14}{'НМ медиана':<14}"
    f"{'Сеть среднее':<14}{'Сеть медиана'}"
)
for name in param_names:
    if errors_nm[name]:
        print(
            f"{name:<10}{np.mean(errors_nm[name]):<14.2f}"
            f"{np.median(errors_nm[name]):<14.2f}"
            f"{np.mean(errors_net[name]):<14.2f}"
            f"{np.median(errors_net[name]):.2f}"
        )

sum_nm = float(np.sum(times_nm))
print(f"\nСреднее время Нелдера-Мида на пример: {np.mean(times_nm) * 1000:.2f} мс")
print(f"Суммарное время НМ ({n_test}):          {sum_nm:.3f} сек")
print(f"Подготовка batch сети:                  {time_prepare_batch * 1000:.3f} мс")
print(f"Чистый инференс batch сети:             {time_net_batch * 1000:.3f} мс")
print(f"Чистый инференс сети на пример:         {time_net_per_sample * 1000:.4f} мс")
print(f"Ускорение batch-инференса:              {sum_nm / time_net_batch:.1f}x")
print(
    f"Ускорение с учётом подготовки входа:    "
    f"{sum_nm / (time_prepare_batch + time_net_batch):.1f}x"
)

# ── График 3: средняя ошибка по трём диапазонам ───────────────
# Вместо 50 отдельных столбцов показываются только три средних значения.
# Это исключает визуальное доминирование единичных выбросов и напрямую
# соответствует задаче сравнения диапазонов 1–15, 16–30 и 31–50.
parameter_ranges = np.array([
    2.0 - 0.5,      # A
    1.5 - (-1.5),   # v
    2.0 - 0.3,      # nu
    2.0 - (-2.0),   # x0
], dtype=float)

true_matrix = np.column_stack(true_vals)


def total_error_percent(predictions):
    """Средняя нормированная ошибка четырёх параметров для каждого примера."""
    normalized_errors = np.abs(predictions - true_matrix) / parameter_ranges
    return np.mean(normalized_errors, axis=1) * 100.0


net_total_errors = total_error_percent(net_preds_all)
nm_total_errors = total_error_percent(nm_preds_all)

groups = [
    (0, 15, "1–15"),
    (15, 30, "16–30"),
    (30, 50, "31–50"),
]

group_labels = [label for _, _, label in groups]
net_group_means = np.array([
    np.mean(net_total_errors[start_idx:end_idx])
    for start_idx, end_idx, _ in groups
])
nm_group_means = np.array([
    np.mean(nm_total_errors[start_idx:end_idx])
    for start_idx, end_idx, _ in groups
])

print("\nСРЕДНЯЯ ОБЩАЯ ОШИБКА ПО ДИАПАЗОНАМ")
for label, net_mean, nm_mean in zip(
    group_labels,
    net_group_means,
    nm_group_means,
):
    print(
        f"Примеры {label:<7} "
        f"Гибридная модель: {net_mean:6.2f}%   "
        f"Нелдер–Мид: {nm_mean:6.2f}%"
    )

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 7),
    sharey=False,
    gridspec_kw={"wspace": 0.24},
)

panels = [
    (axes[0], net_group_means, "Гибридная модель"),
    (axes[1], nm_group_means, "Метод Нелдера–Мида"),
]

x_positions = np.arange(len(group_labels))

for ax, means, title in panels:
    local_max = float(np.max(means))
    local_ymax = max(1.0, local_max * 1.30)

    bars = ax.bar(x_positions, means, width=0.62)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + local_ymax * 0.035,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    overall_mean = float(np.mean(
        net_total_errors if title == "Гибридная модель" else nm_total_errors
    ))
    ax.text(
        0.5,
        0.96,
        f"Средняя ошибка по всем 50 примерам: {overall_mean:.2f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.65",
            "alpha": 0.95,
        },
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels, fontsize=13, fontweight="bold")
    ax.set_xlabel("Диапазон тестовых примеров", fontsize=12)
    ax.set_ylabel("Средняя общая ошибка, %", fontsize=12)
    ax.set_ylim(0, local_ymax)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

fig.text(
    0.5,
    0.015,
    "Вертикальный масштаб у панелей различается. Над каждым столбцом указано точное значение средней ошибки.",
    ha="center",
    fontsize=11,
)

plt.suptitle(
    "Сравнение средней ошибки восстановления параметров\n"
    "на диапазонах 1–15, 16–30 и 31–50",
    fontsize=18,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(
    DATA_DIR / "inverse_average_errors_by_ranges.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()
print("\nСохранён: data/inverse_average_errors_by_ranges.png")

# ── График 4: сеть, предсказание vs истина ────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, name, true_v, pred_v in zip(axes, param_names, true_vals, net_preds_all.T):
    ax.scatter(true_v, pred_v, alpha=0.6, s=30)
    lims = [min(true_v.min(), pred_v.min()), max(true_v.max(), pred_v.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(f"Истинное {name}")
    ax.set_ylabel(f"Предсказанное {name}")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Гибридная сеть: предсказание vs истина ({n_test} примеров, один batch)",
    y=1.02,
)
plt.tight_layout()
plt.savefig(DATA_DIR / "inverse_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Сохранён: data/inverse_scatter.png")
