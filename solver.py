import numpy as np


def soliton_exact(params, x, t):
    """
    Точное аналитическое решение нелинейного уравнения Шрёдингера (НУШ):

        i du/dt + d^2u/dx^2 + nu * |u|^2 * u = 0

    Решение -- солитон огибающей (bright soliton):

        u(x,t) = A * sech(k*(x - x0 - v*t)) * exp(i*(v/2*x - omega*t))

        где  k     = A*sqrt(nu) / sqrt(2)
             omega  = v^2/4 - A^2*nu/2

    ВАЖНО: коэффициент k содержит множитель 1/sqrt(2), которого не было
    в исходной (ошибочной) версии формулы -- без него подстановка решения
    в уравнение не давала нулевой остаток (проверено и выведено аналитически
    через sympy, см. историю работы над проектом). Формула выведена прямой
    подстановкой u(x,t) в уравнение i*du/dt + d^2u/dx^2 + nu|u|^2 u = 0
    и приравниванием коэффициентов при cosh^2 и при свободном члене к нулю.

    Физика: огибающая (амплитудный профиль) движется со скоростью v,
    не меняя формы -- нелинейный член в точности компенсирует расплывание
    волнового пакета, которое было бы вызвано вторым слагаемым (дисперсией).

    params: [A, v, nu, x0] -- амплитуда, скорость, нелинейность, начальный центр
    x: координатная сетка, shape (Nx,)
    t: временная сетка,    shape (Nt,)

    Возвращает:
        u: комплексный массив shape (Nt, Nx) -- решение на всей сетке (x,t)
    """
    A, v, nu, x0 = params

    # Сетка (x,t): строим через broadcasting, чтобы не писать циклы
    X, T = np.meshgrid(x, t)  # оба shape (Nt, Nx)

    k = A * np.sqrt(nu) / np.sqrt(2)       # ширина огибающей (исправлено)
    omega = v**2 / 4 - A**2 * nu / 2        # частота фазы

    envelope = A / np.cosh(k * (X - x0 - v * T))
    phase = v / 2 * X - omega * T

    u = envelope * np.exp(1j * phase)
    return u


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Параметры тестового солитона (диапазоны согласованы с dataset.py --
    # подобраны так, чтобы солитон не "обрезался" границами сетки)
    A, v, nu, x0 = 1.5, 1.0, 1.0, 0.0
    x = np.linspace(-20, 20, 128)
    t = np.linspace(0, 3, 64)

    u = soliton_exact([A, v, nu, x0], x, t)
    print("Форма решения u(x,t):", u.shape, "(Nt, Nx)")
    print("u -- комплексный массив:", u.dtype)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. |u(x,t)|^2 -- плотность вероятности, эволюция во времени (тепловая карта)
    im = axes[0].imshow(np.abs(u)**2, extent=[x[0], x[-1], t[-1], t[0]],
                          aspect="auto", cmap="viridis")
    axes[0].set_title("|u(x,t)|^2 -- движение солитона")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im, ax=axes[0])

    # 2. Профиль |u(x,t0)| в разные моменты времени -- форма не меняется
    n_t = len(t)
    time_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
    for ti in time_indices:
        axes[1].plot(x, np.abs(u[ti]), label=f"t={t[ti]:.2f}")
    axes[1].set_title("Профиль |u(x)| в разные t (форма сохраняется)")
    axes[1].set_xlabel("x"); axes[1].legend(fontsize=8)

    # 3. Re(u) и Im(u) при t=0 -- видна несущая осцилляция внутри огибающей
    axes[2].plot(x, u[0].real, label="Re(u)")
    axes[2].plot(x, u[0].imag, label="Im(u)")
    axes[2].plot(x, np.abs(u[0]), "k--", label="|u| (огибающая)")
    axes[2].set_title("Re, Im и |u| при t=0")
    axes[2].set_xlabel("x"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("data/test_solver.png", dpi=150)
    plt.show()
    print("Графики сохранены в data/test_solver.png")