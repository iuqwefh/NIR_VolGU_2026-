import numpy as np
import os
from tqdm import tqdm
from solver import soliton_exact

# ── Сетка (x, t) -- та же, что у data-driven и PINN, для честного
# сравнения всех трёх подходов на одинаковых эталонных данных ───
NX, NT = 128, 64
X_RANGE = (-20.0, 20.0)
T_RANGE = (0.0, 3.0)

X_GRID = np.linspace(*X_RANGE, NX)
T_GRID = np.linspace(*T_RANGE, NT)


def generate_hybrid_dataset(n_samples=20000, seed=42):
    """
    Генерирует датасет для гибридной модели (data-driven + physics loss).

    Датасет крупнее, чем у чистого data-driven подхода (20000 против 3000) --
    цель: проверить, снижает ли увеличение датасета + physics-регуляризация
    итоговую ошибку модели на порядки, как просил научный руководитель.

    Сохраняет:
        params: (n_samples, 4)         -- [A, v, nu, x0]
        u_real: (n_samples, NT, NX)    -- действительная часть Re(u(x,t))
        u_imag: (n_samples, NT, NX)    -- мнимая часть Im(u(x,t))
    """
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(seed)

    params_list = []
    u_real_list = []
    u_imag_list = []

    print(f"Генерируем {n_samples} точных решений НУШ (для гибридной модели)...")
    for _ in tqdm(range(n_samples)):
        A = rng.uniform(0.5, 2.0)
        v = rng.uniform(-1.5, 1.5)
        nu = rng.uniform(0.3, 2.0)
        x0 = rng.uniform(-2.0, 2.0)

        u = soliton_exact([A, v, nu, x0], X_GRID, T_GRID)

        params_list.append([A, v, nu, x0])
        u_real_list.append(u.real)
        u_imag_list.append(u.imag)

    params = np.array(params_list, dtype=np.float32)
    u_real = np.array(u_real_list, dtype=np.float32)
    u_imag = np.array(u_imag_list, dtype=np.float32)

    np.save("data/hybrid_params.npy", params)
    np.save("data/hybrid_u_real.npy", u_real)
    np.save("data/hybrid_u_imag.npy", u_imag)
    np.save("data/hybrid_x_grid.npy", X_GRID)
    np.save("data/hybrid_t_grid.npy", T_GRID)

    print(f"Сохранено: {len(params)} образцов")
    print(f"  params: {params.shape}  -- [A, v, nu, x0]")
    print(f"  u_real: {u_real.shape}  -- (n_samples, NT, NX)")
    print(f"  u_imag: {u_imag.shape}")

    return params, u_real, u_imag


if __name__ == "__main__":
    generate_hybrid_dataset(n_samples=20000)
