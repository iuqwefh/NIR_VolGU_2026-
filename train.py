import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from model import NLSESurrogate

# ── Гиперпараметры ──────────────────────────────────────────
NT, NX     = 64, 128
BATCH_SIZE = 32
EPOCHS     = 250
LR         = 1e-3
PATIENCE   = 30
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Устройство: {DEVICE}")

# Веса компонент функции потерь
W_U      = 1.0   # основной MSE по Re(u), Im(u)
W_NORM   = 0.5   # штраф за нарушение сохранения нормы во времени
W_SMOOTH = 0.5   # штраф за "дрожание" между соседними точками сетки

# ── Загрузка данных ─────────────────────────────────────────
params = torch.tensor(np.load("data/params.npy"))   # (N, 4): [A, v, nu, x0]
u_real = torch.tensor(np.load("data/u_real.npy"))   # (N, NT, NX)
u_imag = torch.tensor(np.load("data/u_imag.npy"))   # (N, NT, NX)
x_grid = np.load("data/x_grid.npy")
t_grid = np.load("data/t_grid.npy")

DX = float(x_grid[1] - x_grid[0])  # шаг сетки по x, нужен для штрафа нормы

# Нормализация входных параметров
params_mean = params.mean(0)
params_std  = params.std(0)
params_norm = (params - params_mean) / params_std

dataset = TensorDataset(params_norm, u_real, u_imag)

n_train = int(0.8 * len(dataset))
n_val   = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ── Модель, оптимизатор, планировщик ───────────────────────
model     = NLSESurrogate(NT=NT, NX=NX).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)
mse = nn.MSELoss()


def norm_conservation_penalty(u_real_pred, u_imag_pred):
    """
    Физический штраф: для НУШ норма integral |u(x,t)|^2 dx
    должна быть ПОСТОЯННОЙ во времени (закон сохранения).

    Считаем норму на каждый момент времени t, затем штрафуем
    разброс этой нормы вдоль оси t -- она должна быть одним числом,
    а не меняться от кадра к кадру.
    """
    intensity = u_real_pred ** 2 + u_imag_pred ** 2     # (batch, NT, NX)
    norm_per_t = intensity.sum(dim=2) * DX               # (batch, NT)
    target = norm_per_t.mean(dim=1, keepdim=True)        # средняя норма по времени
    return mse(norm_per_t, target.expand_as(norm_per_t))


def smoothness_penalty(u_real_pred, u_imag_pred):
    """Штраф за дрожание между соседними точками по x (внутри каждого кадра t)."""
    diff_real = u_real_pred[:, :, 1:] - u_real_pred[:, :, :-1]
    diff_imag = u_imag_pred[:, :, 1:] - u_imag_pred[:, :, :-1]
    return (diff_real ** 2).mean() + (diff_imag ** 2).mean()


# ── Цикл обучения ───────────────────────────────────────────
history = {
    "train_u": [], "val_u": [],
    "train_norm": [], "val_norm": [],
    "train_smooth": [], "val_smooth": [],
    "train_total": [], "val_total": [],
}

best_val_loss = float("inf")
epochs_without_improvement = 0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):

    # --- Train ---
    model.train()
    sums = {"u": 0.0, "norm": 0.0, "smooth": 0.0, "total": 0.0}

    for X, ur_true, ui_true in train_loader:
        X, ur_true, ui_true = X.to(DEVICE), ur_true.to(DEVICE), ui_true.to(DEVICE)

        ur_pred, ui_pred = model(X)

        loss_u      = mse(ur_pred, ur_true) + mse(ui_pred, ui_true)
        loss_norm   = norm_conservation_penalty(ur_pred, ui_pred)
        loss_smooth = smoothness_penalty(ur_pred, ui_pred)

        loss = W_U * loss_u + W_NORM * loss_norm + W_SMOOTH * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sums["u"]      += loss_u.item()
        sums["norm"]   += loss_norm.item()
        sums["smooth"] += loss_smooth.item()
        sums["total"]  += loss.item()

    n_batches_train = len(train_loader)
    train_u      = sums["u"]      / n_batches_train
    train_norm   = sums["norm"]   / n_batches_train
    train_smooth = sums["smooth"] / n_batches_train
    train_tot    = sums["total"]  / n_batches_train

    # --- Validation ---
    model.eval()
    sums = {"u": 0.0, "norm": 0.0, "smooth": 0.0, "total": 0.0}

    with torch.no_grad():
        for X, ur_true, ui_true in val_loader:
            X, ur_true, ui_true = X.to(DEVICE), ur_true.to(DEVICE), ui_true.to(DEVICE)
            ur_pred, ui_pred = model(X)

            loss_u      = mse(ur_pred, ur_true) + mse(ui_pred, ui_true)
            loss_norm   = norm_conservation_penalty(ur_pred, ui_pred)
            loss_smooth = smoothness_penalty(ur_pred, ui_pred)
            loss = W_U * loss_u + W_NORM * loss_norm + W_SMOOTH * loss_smooth

            sums["u"]      += loss_u.item()
            sums["norm"]   += loss_norm.item()
            sums["smooth"] += loss_smooth.item()
            sums["total"]  += loss.item()

    n_batches_val = len(val_loader)
    val_u      = sums["u"]      / n_batches_val
    val_norm   = sums["norm"]   / n_batches_val
    val_smooth = sums["smooth"] / n_batches_val
    val_tot    = sums["total"]  / n_batches_val

    history["train_u"].append(train_u)
    history["val_u"].append(val_u)
    history["train_norm"].append(train_norm)
    history["val_norm"].append(val_norm)
    history["train_smooth"].append(train_smooth)
    history["val_smooth"].append(val_smooth)
    history["train_total"].append(train_tot)
    history["val_total"].append(val_tot)

    scheduler.step(val_tot)

    if val_tot < best_val_loss:
        best_val_loss = val_tot
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "data/model.pth")
    else:
        epochs_without_improvement += 1

    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | "
              f"u: {train_u:.4f}/{val_u:.4f} | "
              f"norm: {train_norm:.4f}/{val_norm:.4f} | "
              f"smooth: {train_smooth:.4f}/{val_smooth:.4f} | "
              f"lr: {current_lr:.2e}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping на эпохе {epoch} "
              f"(лучшая модель была на эпохе {best_epoch}, "
              f"val_loss={best_val_loss:.4f})")
        break

# ── Сохраняем нормировочные константы ───────────────────────
print(f"\nЛучшая модель: эпоха {best_epoch}, val_loss={best_val_loss:.4f}")
np.save("data/norm_params.npy", [params_mean.numpy(), params_std.numpy()])

# ── Графики сходимости ──────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

axes[0].plot(history["train_u"], label="Train")
axes[0].plot(history["val_u"],   label="Val")
axes[0].set_title("Loss: u(x,t) -- основной MSE")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
axes[0].legend(); axes[0].set_yscale("log")

axes[1].plot(history["train_norm"], label="Train")
axes[1].plot(history["val_norm"],   label="Val")
axes[1].set_title("Штраф сохранения нормы")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
axes[1].legend(); axes[1].set_yscale("log")

axes[2].plot(history["train_smooth"], label="Train")
axes[2].plot(history["val_smooth"],   label="Val")
axes[2].set_title("Штраф гладкости (anti-шум)")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("MSE")
axes[2].legend(); axes[2].set_yscale("log")

axes[3].plot(history["train_total"], label="Train")
axes[3].plot(history["val_total"],   label="Val")
axes[3].set_title("Total loss")
axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("MSE")
axes[3].legend(); axes[3].set_yscale("log")

plt.tight_layout()
plt.savefig("data/training_curves.png", dpi=150)
plt.show()
print("Обучение завершено. Графики сохранены.")