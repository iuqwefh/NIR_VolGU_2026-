import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from hybrid_model import NLSEHybrid


# -------------------- Воспроизводимость и устройство --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE_OVERRIDE = os.environ.get("HYBRID_DEVICE", "").strip().lower()
if DEVICE_OVERRIDE in {"cpu", "mps", "cuda"}:
    DEVICE = DEVICE_OVERRIDE
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Устройство: {DEVICE}")
os.makedirs("data", exist_ok=True)


# -------------------- Гиперпараметры --------------------
NT, NX = 64, 128
BATCH_SIZE = 32

# Этап 1: сеть сначала учится воспроизводить эталонные решения.
DATA_EPOCHS = 50
LR_DATA = 2e-4

# Этап 2: затем подключается физическая часть (PINN fine-tuning).
HYBRID_EPOCHS = 100
PHYSICS_RAMP_EPOCHS = 50
LR_HYBRID = 1e-4
PATIENCE = 30
MIN_DELTA = 1e-7
GRAD_CLIP = 1.0

# L_total = W_DATA*L_data + alpha*(W_PDE*L_PDE + W_NORM*L_norm).
# alpha плавно растёт от 0 до 1 в начале второго этапа.
W_DATA = 1.0
W_PDE = 0.01
W_NORM = 0.005


# -------------------- Данные без лишней копии ~1.3 ГБ в RAM --------------------
class HybridDataset(Dataset):
    def __init__(self, params_norm, params_raw, real_path, imag_path):
        self.params_norm = torch.from_numpy(params_norm.astype(np.float32, copy=False))
        self.params_raw = torch.from_numpy(params_raw.astype(np.float32, copy=False))
        self.u_real = np.load(real_path, mmap_mode="r")
        self.u_imag = np.load(imag_path, mmap_mode="r")

        n = len(self.params_raw)
        expected = (n, NT, NX)
        if self.u_real.shape != expected or self.u_imag.shape != expected:
            raise ValueError(
                f"Ожидались u_real/u_imag формы {expected}, получены "
                f"{self.u_real.shape} и {self.u_imag.shape}"
            )

    def __len__(self):
        return len(self.params_raw)

    def __getitem__(self, index):
        # Копируется только один пример, а не весь массив решений.
        ur = torch.from_numpy(np.array(self.u_real[index], dtype=np.float32, copy=True))
        ui = torch.from_numpy(np.array(self.u_imag[index], dtype=np.float32, copy=True))
        return self.params_norm[index], self.params_raw[index], ur, ui

    def validate_finite(self, chunk_size=256):
        """До обучения проверяет, что в датасете нет NaN/Inf."""
        for name, array in (("u_real", self.u_real), ("u_imag", self.u_imag)):
            min_value = np.inf
            max_value = -np.inf
            for start in range(0, len(array), chunk_size):
                stop = min(start + chunk_size, len(array))
                block = np.asarray(array[start:stop])
                finite = np.isfinite(block)
                if not finite.all():
                    local = np.argwhere(~finite)[0]
                    sample, t_index, x_index = (int(v) for v in local)
                    sample += start
                    value = array[sample, t_index, x_index]
                    raise ValueError(
                        f"В {name} найдено NaN/Inf: sample={sample}, "
                        f"t_index={t_index}, x_index={x_index}, value={value}. "
                        "Пересоздайте датасет исправленным hybrid_dataset.py."
                    )
                min_value = min(min_value, float(block.min()))
                max_value = max(max_value, float(block.max()))
            print(f"Проверка {name}: все значения конечны, диапазон [{min_value:.5g}, {max_value:.5g}]")


params_np = np.load("data/hybrid_params.npy").astype(np.float32, copy=False)
x_grid = np.load("data/hybrid_x_grid.npy")
t_grid = np.load("data/hybrid_t_grid.npy")

if params_np.ndim != 2 or params_np.shape[1] != 4:
    raise ValueError("hybrid_params.npy должен иметь форму (N, 4): [A, v, nu, x0]")
if len(x_grid) != NX or len(t_grid) != NT:
    raise ValueError(f"Ожидалась сетка NX={NX}, NT={NT}")

DX = float(x_grid[1] - x_grid[0])
DT = float(t_grid[1] - t_grid[0])
NU_COL = 2

params_mean = params_np.mean(axis=0)
params_std = params_np.std(axis=0)
params_std = np.where(params_std < 1e-12, 1.0, params_std).astype(np.float32)
params_norm_np = (params_np - params_mean) / params_std

dataset = HybridDataset(
    params_norm_np,
    params_np,
    "data/hybrid_u_real.npy",
    "data/hybrid_u_imag.npy",
)

if not np.isfinite(params_np).all() or not np.isfinite(params_norm_np).all():
    raise ValueError("В hybrid_params.npy или после нормализации параметров появились NaN/Inf")
print(
    "Проверка params: все значения конечны; std="
    + np.array2string(params_std, precision=6)
)
dataset.validate_finite()

# Фиксированный split 80/10/10. Test не участвует ни в обучении,
# ни в early stopping и позже используется в hybrid_test.py.
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(dataset))
n_train = int(0.80 * len(dataset))
n_val = int(0.10 * len(dataset))
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

np.savez(
    "data/hybrid_split_indices.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    seed=np.array(SEED),
)
np.save("data/hybrid_norm_params.npy", np.stack([params_mean, params_std]))

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": 0,
    "pin_memory": DEVICE == "cuda",
}
train_loader = DataLoader(
    Subset(dataset, train_idx.tolist()),
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    **loader_kwargs,
)
val_loader = DataLoader(Subset(dataset, val_idx.tolist()), shuffle=False, **loader_kwargs)

print(
    f"Разбиение: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}; "
    f"DX={DX:.6g}, DT={DT:.6g}"
)


# -------------------- Модель и компоненты loss --------------------
model = NLSEHybrid(NT=NT, NX=NX).to(DEVICE)
mse = nn.MSELoss()


def physics_residual_grid(u_real_pred, u_imag_pred, nu_batch):
    """Остаток R = i*u_t + u_xx + nu*|u|^2*u на внутренних узлах."""
    dur_dt = (u_real_pred[:, 2:, 1:-1] - u_real_pred[:, :-2, 1:-1]) / (2.0 * DT)
    dui_dt = (u_imag_pred[:, 2:, 1:-1] - u_imag_pred[:, :-2, 1:-1]) / (2.0 * DT)

    d2ur_dx2 = (
        u_real_pred[:, 1:-1, 2:]
        - 2.0 * u_real_pred[:, 1:-1, 1:-1]
        + u_real_pred[:, 1:-1, :-2]
    ) / (DX ** 2)
    d2ui_dx2 = (
        u_imag_pred[:, 1:-1, 2:]
        - 2.0 * u_imag_pred[:, 1:-1, 1:-1]
        + u_imag_pred[:, 1:-1, :-2]
    ) / (DX ** 2)

    ur = u_real_pred[:, 1:-1, 1:-1]
    ui = u_imag_pred[:, 1:-1, 1:-1]
    intensity = ur.square() + ui.square()
    nu = nu_batch.view(-1, 1, 1)

    residual_real = -dui_dt + d2ur_dx2 + nu * intensity * ur
    residual_imag = dur_dt + d2ui_dx2 + nu * intensity * ui
    return residual_real, residual_imag


def norm_conservation_penalty(u_real_pred, u_imag_pred):
    """Штрафует изменение интеграла |u|^2 относительно предсказания при t=0."""
    intensity = u_real_pred.square() + u_imag_pred.square()
    norm_per_t = intensity.sum(dim=2) * DX
    initial_norm = norm_per_t[:, :1].detach()
    return mse(norm_per_t, initial_norm.expand_as(norm_per_t))


def calculate_losses(x_norm, x_raw, ur_true, ui_true, physics_alpha):
    ur_pred, ui_pred = model(x_norm)

    data_loss = mse(ur_pred, ur_true) + mse(ui_pred, ui_true)
    if physics_alpha <= 0.0:
        # На первом этапе физическая часть не участвует в оптимизации,
        # поэтому не тратим время на производные по всей сетке.
        zero = data_loss.new_zeros(())
        return {
            "data": data_loss,
            "pde": zero,
            "norm": zero,
            "pinn": zero,
            "total": W_DATA * data_loss,
        }

    residual_real, residual_imag = physics_residual_grid(
        ur_pred, ui_pred, x_raw[:, NU_COL]
    )
    pde_loss = residual_real.square().mean() + residual_imag.square().mean()
    norm_loss = norm_conservation_penalty(ur_pred, ui_pred)

    # Отдельная PINN-компонента нужна и для оптимизации, и для честного графика.
    pinn_loss = W_PDE * pde_loss + W_NORM * norm_loss
    total_loss = W_DATA * data_loss + physics_alpha * pinn_loss

    return {
        "data": data_loss,
        "pde": pde_loss,
        "norm": norm_loss,
        "pinn": pinn_loss,
        "total": total_loss,
    }


def _tensor_status(name, tensor):
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return f"{name}=finite, range=[{tensor.min().item():.4g}, {tensor.max().item():.4g}]"
    return f"{name}=NON-FINITE, bad={int((~finite).sum().item())}"


def run_epoch(loader, optimizer, physics_alpha, epoch, stage):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("data", "pde", "norm", "pinn", "total")}
    n_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (x_norm, x_raw, ur_true, ui_true) in enumerate(loader, start=1):
            x_norm = x_norm.to(DEVICE, non_blocking=True)
            x_raw = x_raw.to(DEVICE, non_blocking=True)
            ur_true = ur_true.to(DEVICE, non_blocking=True)
            ui_true = ui_true.to(DEVICE, non_blocking=True)

            batch_tensors = {
                "x_norm": x_norm,
                "x_raw": x_raw,
                "ur_true": ur_true,
                "ui_true": ui_true,
            }
            bad_inputs = [
                name for name, tensor in batch_tensors.items()
                if not bool(torch.isfinite(tensor).all())
            ]
            if bad_inputs:
                details = "; ".join(
                    _tensor_status(name, batch_tensors[name]) for name in bad_inputs
                )
                raise ValueError(
                    f"NaN/Inf во входных данных: stage={stage}, epoch={epoch}, "
                    f"batch={batch_index}; {details}"
                )

            losses = calculate_losses(x_norm, x_raw, ur_true, ui_true, physics_alpha)

            if not bool(torch.isfinite(losses["total"])):
                nonfinite_parameters = [
                    name for name, parameter in model.named_parameters()
                    if not bool(torch.isfinite(parameter).all())
                ]
                loss_values = ", ".join(
                    f"{name}={float(value.detach().cpu())}"
                    for name, value in losses.items()
                )
                parameter_info = (
                    ", ".join(nonfinite_parameters[:5])
                    if nonfinite_parameters else "все параметры модели конечны"
                )
                raise FloatingPointError(
                    f"Loss стал NaN/Inf: stage={stage}, epoch={epoch}, "
                    f"batch={batch_index}, device={DEVICE}; {loss_values}; "
                    f"параметры: {parameter_info}. Если данные конечны, запустите "
                    f"с HYBRID_DEVICE=cpu для проверки MPS."
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if not bool(torch.isfinite(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Градиенты стали NaN/Inf: stage={stage}, epoch={epoch}, "
                        f"batch={batch_index}, loss={losses['total'].item():.6g}, "
                        f"device={DEVICE}. Попробуйте HYBRID_DEVICE=cpu."
                    )
                optimizer.step()

            batch_size = x_norm.shape[0]
            n_examples += batch_size
            for name in totals:
                totals[name] += float(losses[name].detach()) * batch_size

    return {name: value / n_examples for name, value in totals.items()}


history = {
    "epoch": [],
    "physics_alpha": [],
    "train_data": [], "val_data": [],
    "train_pde": [], "val_pde": [],
    "train_norm": [], "val_norm": [],
    "train_pinn": [], "val_pinn": [],
    "train_total": [], "val_total": [],
}


def record(epoch, alpha, train_metrics, val_metrics):
    history["epoch"].append(epoch)
    history["physics_alpha"].append(alpha)
    for name in ("data", "pde", "norm", "pinn", "total"):
        history[f"train_{name}"].append(train_metrics[name])
        history[f"val_{name}"].append(val_metrics[name])


def print_progress(stage, epoch, alpha, train_metrics, val_metrics, lr):
    print(
        f"[{stage}] epoch {epoch:3d} | alpha={alpha:.2f} | "
        f"data {train_metrics['data']:.4e}/{val_metrics['data']:.4e} | "
        f"PINN {train_metrics['pinn']:.4e}/{val_metrics['pinn']:.4e} | "
        f"total {train_metrics['total']:.4e}/{val_metrics['total']:.4e} | "
        f"lr={lr:.2e}"
    )


# -------------------- Этап 1: data-driven pretraining --------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR_DATA, eps=1e-7)
best_data_val = float("inf")
best_data_epoch = 0

for epoch in range(1, DATA_EPOCHS + 1):
    train_metrics = run_epoch(train_loader, optimizer, physics_alpha=0.0, epoch=epoch, stage="DATA/train")
    val_metrics = run_epoch(val_loader, None, physics_alpha=0.0, epoch=epoch, stage="DATA/val")
    record(epoch, 0.0, train_metrics, val_metrics)

    if val_metrics["data"] < best_data_val - MIN_DELTA:
        best_data_val = val_metrics["data"]
        best_data_epoch = epoch
        torch.save(model.state_dict(), "data/hybrid_model_pretrained.pth")

    if epoch == 1 or epoch % 10 == 0:
        print_progress(
            "DATA", epoch, 0.0, train_metrics, val_metrics,
            optimizer.param_groups[0]["lr"],
        )

print(
    f"Data-driven pretraining завершён: лучшая эпоха {best_data_epoch}, "
    f"val_data={best_data_val:.6e}"
)
model.load_state_dict(torch.load("data/hybrid_model_pretrained.pth", map_location=DEVICE))


# -------------------- Этап 2: PINN fine-tuning --------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR_HYBRID, eps=1e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
)

best_val_total = float("inf")
best_hybrid_epoch = 0
epochs_without_improvement = 0

for local_epoch in range(1, HYBRID_EPOCHS + 1):
    epoch = DATA_EPOCHS + local_epoch
    physics_alpha = min(1.0, local_epoch / PHYSICS_RAMP_EPOCHS)

    train_metrics = run_epoch(
        train_loader, optimizer, physics_alpha, epoch=epoch, stage="PINN/train"
    )
    val_metrics = run_epoch(
        val_loader, None, physics_alpha, epoch=epoch, stage="PINN/val"
    )
    record(epoch, physics_alpha, train_metrics, val_metrics)

    # Пока alpha меняется, значения total между эпохами несопоставимы.
    # Поэтому scheduler, best checkpoint и early stopping включаются только
    # после достижения полного PINN-loss (alpha == 1).
    if physics_alpha >= 1.0:
        scheduler.step(val_metrics["total"])
        if val_metrics["total"] < best_val_total - MIN_DELTA:
            best_val_total = val_metrics["total"]
            best_hybrid_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "data/hybrid_model.pth")
        else:
            epochs_without_improvement += 1

    if local_epoch == 1 or local_epoch % 10 == 0:
        print_progress(
            "PINN", epoch, physics_alpha, train_metrics, val_metrics,
            optimizer.param_groups[0]["lr"],
        )

    if physics_alpha >= 1.0 and epochs_without_improvement >= PATIENCE:
        print(
            f"Early stopping на эпохе {epoch}; лучшая гибридная модель была "
            f"на эпохе {best_hybrid_epoch}, val_total={best_val_total:.6e}"
        )
        break

if best_hybrid_epoch == 0:
    # Защита для случая, когда число hybrid-эпох меньше ramp-периода.
    best_hybrid_epoch = history["epoch"][-1]
    best_val_total = history["val_total"][-1]
    torch.save(model.state_dict(), "data/hybrid_model.pth")

print(
    f"Лучшая итоговая модель: эпоха {best_hybrid_epoch}, "
    f"val_total={best_val_total:.6e}"
)


# -------------------- История и отдельные графики для печати --------------------
history_np = {key: np.asarray(value) for key, value in history.items()}
np.savez("data/hybrid_training_history.npz", **history_np)
np.savez(
    "data/hybrid_training_info.npz",
    best_data_epoch=np.array(best_data_epoch),
    best_hybrid_epoch=np.array(best_hybrid_epoch),
    best_data_val=np.array(best_data_val),
    best_val_total=np.array(best_val_total),
    data_epochs=np.array(DATA_EPOCHS),
    physics_ramp_epochs=np.array(PHYSICS_RAMP_EPOCHS),
    weights=np.array([W_DATA, W_PDE, W_NORM], dtype=np.float32),
)

epochs = history_np["epoch"]
hybrid_mask = epochs > DATA_EPOCHS

# Размер 11.69 x 8.27 дюйма соответствует альбомному листу A4.
# PNG сохраняется с 300 DPI, а PDF остаётся векторным и подходит для печати.
PLOT_FIGSIZE = (11.69, 8.27)
PLOT_DPI = 300
TITLE_SIZE = 24
LABEL_SIZE = 21
TICK_SIZE = 17
LEGEND_SIZE = 18
LINE_WIDTH = 2.6


def save_loss_figure(
    x,
    train_values,
    val_values,
    title,
    ylabel,
    filename,
    color,
    vertical_lines=None,
):
    """Сохраняет один крупный график в PNG и PDF для вставки на A4."""
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        x,
        train_values,
        label="Train",
        color=color,
        linewidth=LINE_WIDTH,
    )
    ax.plot(
        x,
        val_values,
        label="Validation",
        color="tab:orange",
        linewidth=LINE_WIDTH,
        linestyle="--",
    )

    for line in vertical_lines or []:
        position, label, line_color = line
        ax.axvline(
            position,
            color=line_color,
            linewidth=2.0,
            linestyle=":",
            label=label,
        )

    ax.set_yscale("log")
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=18)
    ax.set_xlabel("Эпоха", fontsize=LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold", labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, width=1.5)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE - 2)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.9)
    ax.legend(fontsize=LEGEND_SIZE, frameon=True, loc="best")

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    fig.tight_layout(pad=1.5)
    fig.savefig(
        f"data/{filename}.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        f"data/{filename}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# 1. Data-driven loss показывается на всём обучении: так видно, не ухудшилась
# ли точность по данным после подключения физических ограничений.
save_loss_figure(
    epochs,
    history_np["train_data"],
    history_np["val_data"],
    "Data-driven loss: ошибка относительно эталонного решения",
    r"$L_{data}$ (MSE)",
    "loss_data_driven",
    "tab:blue",
    vertical_lines=[(DATA_EPOCHS, "Начало PINN", "black")],
)

# 2. PINN loss существует только на втором этапе обучения.
save_loss_figure(
    epochs[hybrid_mask],
    history_np["train_pinn"][hybrid_mask],
    history_np["val_pinn"][hybrid_mask],
    "PINN loss: физическая невязка и сохранение нормы",
    r"$L_{PINN}$",
    "loss_pinn_hybrid",
    "tab:blue",
)

# 3. Полная функция, которую фактически минимизирует оптимизатор.
save_loss_figure(
    epochs,
    history_np["train_total"],
    history_np["val_total"],
    "Total loss: общая функция потерь гибридной модели",
    r"$L_{total}$",
    "loss_total",
    "tab:blue",
    vertical_lines=[
        (DATA_EPOCHS, "Начало PINN", "black"),
        (best_hybrid_epoch, "Лучшая модель", "tab:red"),
    ],
)

print("Сохранено: data/loss_data_driven.png и .pdf")
print("Сохранено: data/loss_pinn_hybrid.png и .pdf")
print("Сохранено: data/loss_total.png и .pdf")
print("Сохранено: data/hybrid_training_history.npz")
