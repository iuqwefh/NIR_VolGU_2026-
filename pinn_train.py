import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pinn_model import NLSE_PINN


# ─────────────────────────────────────────────────────────────
# Гиперпараметры обучения
# ─────────────────────────────────────────────────────────────

EPOCHS = 3000
LR = 1e-3
PATIENCE = 300

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Устройство: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# Диапазоны параметров солитона
# ─────────────────────────────────────────────────────────────

A_RANGE = (0.5, 2.0)
V_RANGE = (-1.5, 1.5)
NU_RANGE = (0.3, 2.0)
X0_RANGE = (-2.0, 2.0)

X_RANGE = (-20.0, 20.0)
T_RANGE = (0.0, 3.0)


# ─────────────────────────────────────────────────────────────
# Количество случайных точек
# ─────────────────────────────────────────────────────────────

# ВАЖНО:
# теперь для физической невязки используем 3000 случайных точек,
# а не 4000
N_COLLOCATION = 3000

N_INITIAL = 500
N_BOUNDARY = 500


# ─────────────────────────────────────────────────────────────
# Веса функции потерь
# ─────────────────────────────────────────────────────────────

W_INITIAL = 10.0

# boundary немного уменьшаем,
# чтобы он не конкурировал слишком сильно с physics loss
W_BOUNDARY = 2.0


def get_physics_weight(epoch):
    """
    Вес физической невязки постепенно увеличивается.

    В начале сети важнее научиться воспроизводить
    начальное состояние солитона.

    Затем всё сильнее заставляем сеть соблюдать НУШ.
    """

    if epoch <= 300:
        return 1.0

    elif epoch <= 700:
        # плавно увеличиваем вес от 1 до 5
        progress = (epoch - 300) / 400
        return 1.0 + 4.0 * progress

    else:
        return 5.0


# ─────────────────────────────────────────────────────────────
# Вспомогательная функция
# ─────────────────────────────────────────────────────────────

def sample_uniform(low, high, n, device):
    """
    Создаёт n случайных точек
    равномерно в диапазоне [low, high].
    """

    return (
        torch.rand(n, 1, device=device)
        * (high - low)
        + low
    )


# ─────────────────────────────────────────────────────────────
# Физическая невязка НУШ
# ─────────────────────────────────────────────────────────────

def physics_residual(model, x, t, A, v, nu, x0):
    """
    Считаем остаток нелинейного уравнения Шрёдингера:

        i*u_t + u_xx + nu*|u|^2*u = 0

    Если сеть идеально соблюдает физику,
    R_real и R_imag должны стремиться к нулю.
    """

    ur, ui = model(x, t, A, v, nu, x0)

    # ── производные по времени ──

    dur_dt = torch.autograd.grad(
        ur,
        t,
        grad_outputs=torch.ones_like(ur),
        create_graph=True,
        retain_graph=True
    )[0]

    dui_dt = torch.autograd.grad(
        ui,
        t,
        grad_outputs=torch.ones_like(ui),
        create_graph=True,
        retain_graph=True
    )[0]


    # ── первые производные по x ──

    dur_dx = torch.autograd.grad(
        ur,
        x,
        grad_outputs=torch.ones_like(ur),
        create_graph=True,
        retain_graph=True
    )[0]

    dui_dx = torch.autograd.grad(
        ui,
        x,
        grad_outputs=torch.ones_like(ui),
        create_graph=True,
        retain_graph=True
    )[0]


    # ── вторые производные по x ──

    d2ur_dx2 = torch.autograd.grad(
        dur_dx,
        x,
        grad_outputs=torch.ones_like(dur_dx),
        create_graph=True,
        retain_graph=True
    )[0]

    d2ui_dx2 = torch.autograd.grad(
        dui_dx,
        x,
        grad_outputs=torch.ones_like(dui_dx),
        create_graph=True,
        retain_graph=True
    )[0]


    # |u|^2 = Re(u)^2 + Im(u)^2

    intensity = ur ** 2 + ui ** 2


    # нелинейная часть:
    # nu * |u|^2 * u

    nl_real = nu * intensity * ur
    nl_imag = nu * intensity * ui


    # действительная часть невязки

    R_real = (
        -dui_dt
        + d2ur_dx2
        + nl_real
    )


    # мнимая часть невязки

    R_imag = (
        dur_dt
        + d2ui_dx2
        + nl_imag
    )

    return R_real, R_imag


# ─────────────────────────────────────────────────────────────
# Создание модели
# ─────────────────────────────────────────────────────────────

model = NLSE_PINN().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=50,
    factor=0.5
)

mse = nn.MSELoss()


# ─────────────────────────────────────────────────────────────
# История обучения
# ─────────────────────────────────────────────────────────────

history = {
    "physics": [],
    "initial": [],
    "boundary": [],
    "total": [],
    "physics_weight": []
}


best_loss = float("inf")
best_epoch = 0

epochs_without_improvement = 0


# ─────────────────────────────────────────────────────────────
# Обучение
# ─────────────────────────────────────────────────────────────

for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()


    # ─────────────────────────────────────────────────────────
    # Случайные параметры солитона
    # ─────────────────────────────────────────────────────────

    def sample_params(n):

        A = sample_uniform(
            *A_RANGE,
            n,
            DEVICE
        )

        v = sample_uniform(
            *V_RANGE,
            n,
            DEVICE
        )

        nu = sample_uniform(
            *NU_RANGE,
            n,
            DEVICE
        )

        x0 = sample_uniform(
            *X0_RANGE,
            n,
            DEVICE
        )

        return A, v, nu, x0


    # =========================================================
    # 1. PHYSICS LOSS
    # =========================================================

    # на каждой эпохе создаются НОВЫЕ 3000 случайных точек

    x_col = sample_uniform(
        *X_RANGE,
        N_COLLOCATION,
        DEVICE
    ).requires_grad_(True)

    t_col = sample_uniform(
        *T_RANGE,
        N_COLLOCATION,
        DEVICE
    ).requires_grad_(True)


    A_col, v_col, nu_col, x0_col = sample_params(
        N_COLLOCATION
    )


    R_real, R_imag = physics_residual(
        model,
        x_col,
        t_col,
        A_col,
        v_col,
        nu_col,
        x0_col
    )


    # хотим, чтобы обе части невязки были около нуля

    loss_physics = (
        (R_real ** 2).mean()
        +
        (R_imag ** 2).mean()
    )


    # =========================================================
    # 2. INITIAL CONDITION LOSS
    # =========================================================

    x_init = sample_uniform(
        *X_RANGE,
        N_INITIAL,
        DEVICE
    )

    # начальное условие задаётся при t = 0

    t_init = torch.zeros(
        N_INITIAL,
        1,
        device=DEVICE
    )


    A_init, v_init, nu_init, x0_init = sample_params(
        N_INITIAL
    )


    ur_init_pred, ui_init_pred = model(
        x_init,
        t_init,
        A_init,
        v_init,
        nu_init,
        x0_init
    )


    # коэффициент ширины солитона

    k_init = (
        A_init
        * torch.sqrt(nu_init)
        / torch.sqrt(
            torch.tensor(
                2.0,
                device=DEVICE
            )
        )
    )


    # точная огибающая солитона при t = 0

    envelope = (
        A_init
        /
        torch.cosh(
            k_init
            * (x_init - x0_init)
        )
    )


    # фаза при t = 0

    phase = (
        v_init
        / 2
        * x_init
    )


    # действительная и мнимая части точного решения

    ur_init_true = envelope * torch.cos(phase)

    ui_init_true = envelope * torch.sin(phase)


    loss_initial = (
        mse(
            ur_init_pred,
            ur_init_true
        )
        +
        mse(
            ui_init_pred,
            ui_init_true
        )
    )


    # =========================================================
    # 3. BOUNDARY LOSS
    # =========================================================

    t_bound = sample_uniform(
        *T_RANGE,
        N_BOUNDARY,
        DEVICE
    )


    A_bound, v_bound, nu_bound, x0_bound = sample_params(
        N_BOUNDARY
    )


    # левая граница x = -20

    x_left = torch.full(
        (N_BOUNDARY, 1),
        X_RANGE[0],
        device=DEVICE
    )


    # правая граница x = 20

    x_right = torch.full(
        (N_BOUNDARY, 1),
        X_RANGE[1],
        device=DEVICE
    )


    ur_left, ui_left = model(
        x_left,
        t_bound,
        A_bound,
        v_bound,
        nu_bound,
        x0_bound
    )


    ur_right, ui_right = model(
        x_right,
        t_bound,
        A_bound,
        v_bound,
        nu_bound,
        x0_bound
    )


    # далеко от центра солитона
    # решение должно стремиться к нулю

    loss_boundary = (
        (ur_left ** 2).mean()
        +
        (ui_left ** 2).mean()
        +
        (ur_right ** 2).mean()
        +
        (ui_right ** 2).mean()
    )


    # =========================================================
    # 4. ОБЩАЯ ФУНКЦИЯ ПОТЕРЬ
    # =========================================================

    W_PHYSICS = get_physics_weight(epoch)


    loss = (
        W_PHYSICS * loss_physics
        +
        W_INITIAL * loss_initial
        +
        W_BOUNDARY * loss_boundary
    )


    # считаем градиенты

    loss.backward()


    # обновляем веса нейросети

    optimizer.step()


    # scheduler следит за общим loss

    scheduler.step(
        loss.item()
    )


    # =========================================================
    # СОХРАНЯЕМ ИСТОРИЮ
    # =========================================================

    history["physics"].append(
        loss_physics.item()
    )

    history["initial"].append(
        loss_initial.item()
    )

    history["boundary"].append(
        loss_boundary.item()
    )

    history["total"].append(
        loss.item()
    )

    history["physics_weight"].append(
        W_PHYSICS
    )


    # =========================================================
    # СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
    # =========================================================

    if loss.item() < best_loss:

        best_loss = loss.item()

        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "data/pinn_model.pth"
        )

    else:

        epochs_without_improvement += 1


    # =========================================================
    # ВЫВОД В КОНСОЛЬ
    # =========================================================

    if epoch % 100 == 0:

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:4d} | "
            f"physics: {loss_physics.item():.6f} | "
            f"initial: {loss_initial.item():.6f} | "
            f"boundary: {loss_boundary.item():.6f} | "
            f"W_phys: {W_PHYSICS:.2f} | "
            f"total: {loss.item():.6f} | "
            f"lr: {current_lr:.2e}"
        )


    # =========================================================
    # EARLY STOPPING
    # =========================================================

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping на эпохе {epoch}. "
            f"Лучшая модель была на эпохе {best_epoch}, "
            f"loss={best_loss:.6f}"
        )

        break


print(
    f"\nЛучшая модель: эпоха {best_epoch}, "
    f"loss={best_loss:.6f}"
)


# ─────────────────────────────────────────────────────────────
# ГРАФИКИ ОБУЧЕНИЯ
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 4)
)


# physics loss

axes[0].plot(
    history["physics"]
)

axes[0].set_title(
    "Physics loss"
)

axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "MSE"
)

axes[0].set_yscale(
    "log"
)

axes[0].grid(
    alpha=0.3
)


# initial condition loss

axes[1].plot(
    history["initial"]
)

axes[1].set_title(
    "Initial condition loss"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "MSE"
)

axes[1].set_yscale(
    "log"
)

axes[1].grid(
    alpha=0.3
)


# boundary loss

axes[2].plot(
    history["boundary"]
)

axes[2].set_title(
    "Boundary condition loss"
)

axes[2].set_xlabel(
    "Epoch"
)

axes[2].set_ylabel(
    "MSE"
)

axes[2].set_yscale(
    "log"
)

axes[2].grid(
    alpha=0.3
)


# общий loss

axes[3].plot(
    history["total"]
)

axes[3].set_title(
    "Total loss"
)

axes[3].set_xlabel(
    "Epoch"
)

axes[3].set_ylabel(
    "Loss"
)

axes[3].set_yscale(
    "log"
)

axes[3].grid(
    alpha=0.3
)


plt.tight_layout()

plt.savefig(
    "data/pinn_training_curves.png",
    dpi=150
)

plt.show()


# ─────────────────────────────────────────────────────────────
# Отдельный график изменения веса physics loss
# ─────────────────────────────────────────────────────────────

plt.figure(
    figsize=(8, 4)
)

plt.plot(
    history["physics_weight"]
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "W physics"
)

plt.title(
    "Изменение веса физической невязки"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "data/pinn_physics_weight.png",
    dpi=150
)

plt.show()


print(
    "Обучение завершено."
)