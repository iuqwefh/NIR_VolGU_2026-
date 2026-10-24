"""
inverse_train.py -- обучение гибридной инверсной модели для НУШ.

Задача: по наблюдаемому (зашумлённому) полю u(x,t) восстановить
все 4 параметра солитона [A, v, nu, x0] -- полная обратная задача,
которую затем сравниваем с методом Нелдера-Мида (nelder_mead_4param.py).

Loss = data-driven часть (MSE с истинными параметрами)
     + physics-consistency часть (подставляем предсказанные параметры
       в ДИФФЕРЕНЦИРУЕМУЮ формулу солитона, реконструируем поле,
       сравниваем с НАБЛЮДАЕМЫМ зашумлённым полем -- та же цель,
       которую минимизирует Нелдер-Мид).

Данные генерируются на лету (аналитическая формула, без числ. интегрирования) --
не нужно сохранять на диск, всё быстро пересчитывается при каждом запуске.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from invers_model import InverseNLSE

# ── Гиперпараметры ──────────────────────────────────────────
NX, NT = 128, 64
X_RANGE = (-20.0, 20.0)
T_RANGE = (0.0, 3.0)

N_SAMPLES  = 20000
BATCH_SIZE = 64
EPOCHS     = 150
LR         = 1e-3
PATIENCE   = 20
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Устройство: {DEVICE}")

W_DATA    = 1.0
W_PHYSICS = 0.5

# Диапазоны параметров -- те же, что использовались при обучении
# прямой (forward) гибридной модели, для консистентности
A_RANGE  = (0.5, 2.0)
V_RANGE  = (-1.5, 1.5)
NU_RANGE = (0.3, 2.0)
X0_RANGE = (-2.0, 2.0)

NOISE_RANGE = (0.0, 0.05)  # уровень шума варьируем от 0% до 5% -- модель
                            # должна быть устойчива к разным уровням шума,
                            # включая точный тестовый случай Нелдера-Мида (3%)

# ── Генерация датасета (быстро, аналитическая формула) ──────
print(f"Генерируем {N_SAMPLES} обучающих примеров (аналитическая формула)...")
np.random.seed(42)

x = np.linspace(*X_RANGE, NX)
t = np.linspace(*T_RANGE, NT)
X, T = np.meshgrid(x, t)   # (NT, NX)

A_arr  = np.random.uniform(*A_RANGE,  N_SAMPLES).astype(np.float32)
v_arr  = np.random.uniform(*V_RANGE,  N_SAMPLES).astype(np.float32)
nu_arr = np.random.uniform(*NU_RANGE, N_SAMPLES).astype(np.float32)
x0_arr = np.random.uniform(*X0_RANGE, N_SAMPLES).astype(np.float32)

u_real_arr = np.zeros((N_SAMPLES, NT, NX), dtype=np.float32)
u_imag_arr = np.zeros((N_SAMPLES, NT, NX), dtype=np.float32)

# Векторизованно (без цикла по примерам) через broadcasting
A_  = A_arr.reshape(-1, 1, 1)
v_  = v_arr.reshape(-1, 1, 1)
nu_ = nu_arr.reshape(-1, 1, 1)
x0_ = x0_arr.reshape(-1, 1, 1)
X_b = X[None, :, :]
T_b = T[None, :, :]

k     = A_ * np.sqrt(nu_) / np.sqrt(2)
omega = v_**2 / 4 - A_**2 * nu_ / 2
envelope = A_ / np.cosh(k * (X_b - x0_ - v_ * T_b))
phase    = v_ / 2 * X_b - omega * T_b
u_real_arr = (envelope * np.cos(phase)).astype(np.float32)
u_imag_arr = (envelope * np.sin(phase)).astype(np.float32)

# Добавляем шум (разный уровень на каждый пример -- для устойчивости)
noise_levels = np.random.uniform(*NOISE_RANGE, N_SAMPLES).astype(np.float32)
max_amp = np.abs(A_arr)  # максимум |u| равен A по построению солитона
noise_std = (noise_levels * max_amp).reshape(-1, 1, 1)

u_real_noisy = u_real_arr + noise_std * np.random.randn(*u_real_arr.shape).astype(np.float32)
u_imag_noisy = u_imag_arr + noise_std * np.random.randn(*u_imag_arr.shape).astype(np.float32)

print("Готово.\n")

# ── Нормализация целевых параметров ──────────────────────────
params_true = np.stack([A_arr, v_arr, nu_arr, x0_arr], axis=1)  # (N,4)
params_mean = params_true.mean(axis=0)
params_std  = params_true.std(axis=0)
params_norm = (params_true - params_mean) / params_std

# ── Тензоры и разбиение train/val ────────────────────────────
u_real_t = torch.tensor(u_real_noisy)
u_imag_t = torch.tensor(u_imag_noisy)
params_norm_t = torch.tensor(params_norm, dtype=torch.float32)
params_true_t = torch.tensor(params_true, dtype=torch.float32)

dataset = TensorDataset(u_real_t, u_imag_t, params_norm_t, params_true_t)
n_train = int(0.9 * N_SAMPLES)
n_val   = N_SAMPLES - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
print(f"Train: {n_train}, Val: {n_val}\n")

# ── Модель ──────────────────────────────────────────────────
model     = InverseNLSE(NT=NT, NX=NX).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-6)
mse = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Параметров модели: {total_params:,}\n")

# Константы сетки (x,t) как torch-тензоры для дифференцируемой реконструкции
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)   # (NT, NX)
T_t = torch.tensor(T, dtype=torch.float32).to(DEVICE)


def reconstruct_field(A_pred, v_pred, nu_pred, x0_pred):
    """
    Дифференцируемая формула солитона (та же, что в solver.py и
    в скрипте Нелдера-Мида), но на torch-тензорах -- позволяет
    градиенту течь от ошибки реконструкции обратно к параметрам,
    предсказанным сетью.
    """
    A_  = A_pred.view(-1, 1, 1)
    v_  = v_pred.view(-1, 1, 1)
    nu_ = nu_pred.clamp(min=1e-3).view(-1, 1, 1)  # nu > 0 физически
    x0_ = x0_pred.view(-1, 1, 1)

    k     = A_ * torch.sqrt(nu_) / np.sqrt(2)
    omega = v_**2 / 4 - A_**2 * nu_ / 2
    envelope = A_ / torch.cosh(k * (X_t.unsqueeze(0) - x0_ - v_ * T_t.unsqueeze(0)))
    phase    = v_ / 2 * X_t.unsqueeze(0) - omega * T_t.unsqueeze(0)

    u_real = envelope * torch.cos(phase)
    u_imag = envelope * torch.sin(phase)
    return u_real, u_imag


# ── Цикл обучения ───────────────────────────────────────────
history = {"train_data": [], "val_data": [], "train_phys": [], "val_phys": [],
           "train_total": [], "val_total": []}

best_val = float("inf")
no_improve = 0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):

    model.train()
    sd = sp = st = 0.0

    for ur, ui, p_norm, p_true in train_loader:
        ur, ui = ur.to(DEVICE), ui.to(DEVICE)
        p_norm = p_norm.to(DEVICE)
        p_true = p_true.to(DEVICE)

        A_pred, v_pred, nu_pred, x0_pred = model(ur, ui)
        pred_stack = torch.stack([A_pred, v_pred, nu_pred, x0_pred], dim=1)

        # Data-driven loss: сравниваем в НОРМАЛИЗОВАННОМ пространстве
        p_mean_t = torch.tensor(params_mean, dtype=torch.float32, device=DEVICE)
        p_std_t  = torch.tensor(params_std,  dtype=torch.float32, device=DEVICE)
        pred_norm = (pred_stack - p_mean_t) / p_std_t
        loss_data = mse(pred_norm, p_norm)

        # Physics-consistency loss: реконструируем поле и сравниваем
        # с НАБЛЮДАЕМЫМ (зашумлённым) полем -- та же цель, что у Нелдера-Мида
        ur_recon, ui_recon = reconstruct_field(A_pred, v_pred, nu_pred, x0_pred)
        loss_phys = mse(ur_recon, ur) + mse(ui_recon, ui)

        loss = W_DATA * loss_data + W_PHYSICS * loss_phys

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sd += loss_data.item(); sp += loss_phys.item(); st += loss.item()

    nb = len(train_loader)
    history["train_data"].append(sd/nb)
    history["train_phys"].append(sp/nb)
    history["train_total"].append(st/nb)

    model.eval()
    sd = sp = st = 0.0
    with torch.no_grad():
        for ur, ui, p_norm, p_true in val_loader:
            ur, ui = ur.to(DEVICE), ui.to(DEVICE)
            p_norm = p_norm.to(DEVICE)

            A_pred, v_pred, nu_pred, x0_pred = model(ur, ui)
            pred_stack = torch.stack([A_pred, v_pred, nu_pred, x0_pred], dim=1)

            p_mean_t = torch.tensor(params_mean, dtype=torch.float32, device=DEVICE)
            p_std_t  = torch.tensor(params_std,  dtype=torch.float32, device=DEVICE)
            pred_norm = (pred_stack - p_mean_t) / p_std_t
            loss_data = mse(pred_norm, p_norm)

            ur_recon, ui_recon = reconstruct_field(A_pred, v_pred, nu_pred, x0_pred)
            loss_phys = mse(ur_recon, ur) + mse(ui_recon, ui)

            loss = W_DATA * loss_data + W_PHYSICS * loss_phys

            sd += loss_data.item(); sp += loss_phys.item(); st += loss.item()

    nb = len(val_loader)
    vd, vp, vt = sd/nb, sp/nb, st/nb
    history["val_data"].append(vd)
    history["val_phys"].append(vp)
    history["val_total"].append(vt)

    scheduler.step(vt)

    if vt < best_val:
        best_val = vt; best_epoch = epoch; no_improve = 0
        torch.save(model.state_dict(), "data/inverse_model.pth")
    else:
        no_improve += 1

    if epoch % 10 == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | data: {history['train_data'][-1]:.4f}/{vd:.4f} | "
              f"phys: {history['train_phys'][-1]:.4f}/{vp:.4f} | lr: {lr_now:.2e}")

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping на эпохе {epoch} (лучшая: {best_epoch}, val={best_val:.4f})")
        break

print(f"\nЛучшая модель: эпоха {best_epoch}, val_loss={best_val:.4f}")
np.save("data/inverse_norm_params.npy", [params_mean, params_std])

# ── Графики сходимости ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["train_data"], label="Train")
axes[0].plot(history["val_data"],   label="Val")
axes[0].set_title("Data-driven loss (параметры)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE (норм.)")
axes[0].legend(); axes[0].set_yscale("log")

axes[1].plot(history["train_phys"], label="Train")
axes[1].plot(history["val_phys"],   label="Val")
axes[1].set_title("Physics-consistency loss (поле)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
axes[1].legend(); axes[1].set_yscale("log")

axes[2].plot(history["train_total"], label="Train")
axes[2].plot(history["val_total"],   label="Val")
axes[2].set_title("Total loss")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("MSE")
axes[2].legend(); axes[2].set_yscale("log")

plt.tight_layout()
plt.savefig("data/inverse_training_curves.png", dpi=150)
plt.show()
print("Графики сохранены в data/inverse_training_curves.png")