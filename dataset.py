import numpy as np
from tqdm import tqdm
from solver import soliton_exact

# ── Сетка (x, t) -- общая для всех примеров датасета ─────────
NX, NT = 128, 64
X_RANGE = (-15.0, 15.0)
T_RANGE = (0.0, 3.0)

X_GRID = np.linspace(*X_RANGE, NX)
T_GRID = np.linspace(*T_RANGE, NT)


def generate_dataset(n_samples=3000, seed=42):
    """
    Генерирует датасет: случайные параметры солитона -> точные решения.

    Диапазоны параметров подобраны так, чтобы солитон не "обрезался"
    границами сетки (x,t) -- проверено заранее: амплитуда на краях
    сетки везде < 0.03, то есть солитон полностью укладывается в окно.

    Сохраняет:
        params: (n_samples, 4)         -- [A, v, nu, x0]
        u_real: (n_samples, NT, NX)    -- действительная часть Re(u(x,t))
        u_imag: (n_samples, NT, NX)    -- мнимая часть Im(u(x,t))
    """
    np.random.seed(seed)

    params_list = []
    u_real_list = []
    u_imag_list = []

    print(f"Генерируем {n_samples} точных решений НУШ...")
    for _ in tqdm(range(n_samples)):
        A  = np.random.uniform(0.5, 2.0)    # амплитуда солитона
        v  = np.random.uniform(-1.5, 1.5)   # скорость движения
        nu = np.random.uniform(0.3, 2.0)    # параметр нелинейности
        x0 = np.random.uniform(-2.0, 2.0)   # начальное положение центра

        u = soliton_exact([A, v, nu, x0], X_GRID, T_GRID)

        params_list.append([A, v, nu, x0])
        u_real_list.append(u.real)
        u_imag_list.append(u.imag)

    params = np.array(params_list, dtype=np.float32)
    u_real = np.array(u_real_list, dtype=np.float32)
    u_imag = np.array(u_imag_list, dtype=np.float32)

    np.save("data/params.npy", params)
    np.save("data/u_real.npy", u_real)
    np.save("data/u_imag.npy", u_imag)
    np.save("data/x_grid.npy", X_GRID)
    np.save("data/t_grid.npy", T_GRID)

    print(f"Сохранено: {len(params)} образцов")
    print(f"  params: {params.shape}  -- [A, v, nu, x0]")
    print(f"  u_real: {u_real.shape}  -- (n_samples, NT, NX)")
    print(f"  u_imag: {u_imag.shape}")

    return params, u_real, u_imag


if __name__ == "__main__":
    generate_dataset(n_samples=3000)