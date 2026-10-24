import torch
import torch.nn as nn


class NLSE_PINN(nn.Module):
    """
    Physics-Informed Neural Network для НУШ:

        i du/dt + d^2u/dx^2 + nu*|u|^2*u = 0

    В отличие от data-driven модели (model.py), эта сеть -- ТОЧЕЧНАЯ:
    принимает одну точку (x, t) и параметры солитона (A, v, nu, x0),
    выдаёт значение решения u именно в этой точке (Re и Im).

    Сеть, по сути, выступает как "формула" функции u(x,t,A,v,nu),
    параметризованная весами. Чтобы вся сетка (x,t) была покрыта,
    во время обучения мы просто прогоняем сеть много раз на разных
    точках (x,t) -- координаты тут ВХОД сети, а не индекс выходного
    массива, как было в data-driven подходе.

    Важно: x и t должны requires_grad=True снаружи (в train-коде),
    чтобы можно было взять автодифференцирование du/dt, d2u/dx2.

    Вход:  (batch, 6)  -- [x, t, A, v, nu, x0]
    Выход: (batch, 2)  -- [Re(u), Im(u)]
    """

    def __init__(self, n_input=6, hidden=128, n_layers=6):
        super().__init__()

        layers = [nn.Linear(n_input, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]  # выход: Re(u), Im(u)

        self.net = nn.Sequential(*layers)

    def forward(self, x, t, A, v, nu, x0):
        """
        Все аргументы -- тензоры формы (batch, 1).
        Склеиваем их в один вход (batch, 6) и прогоняем через сеть.
        """
        inp = torch.cat([x, t, A, v, nu, x0], dim=1)
        out = self.net(inp)
        u_real = out[:, 0:1]
        u_imag = out[:, 1:2]
        return u_real, u_imag


if __name__ == "__main__":
    model = NLSE_PINN()
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total:,}")

    batch = 16
    x  = torch.randn(batch, 1, requires_grad=True)
    t  = torch.randn(batch, 1, requires_grad=True)
    A  = torch.rand(batch, 1)
    v  = torch.randn(batch, 1)
    nu = torch.rand(batch, 1)
    x0 = torch.randn(batch, 1)

    u_real, u_imag = model(x, t, A, v, nu, x0)
    print(f"Выход Re(u): {u_real.shape}  (должно быть [16, 1])")
    print(f"Выход Im(u): {u_imag.shape}  (должно быть [16, 1])")