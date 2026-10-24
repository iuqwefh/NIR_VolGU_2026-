import torch
import torch.nn as nn


class NLSESurrogate(nn.Module):
    """
    Суррогатная модель для нелинейного уравнения Шрёдингера (НУШ):

        i du/dt + d^2u/dx^2 + nu*|u|^2*u = 0

    Задача: по параметрам солитона [A, v, nu, x0] предсказать решение
    u(x,t) на всей сетке (x,t) сразу -- то есть два изображения
    (Re(u) и Im(u)) размером (NT, NX).

    Архитектура: MLP-энкодер -> reshape в маленький "холст" ->
    Conv2d-декодер (ConvTranspose2d x3), который разворачивает холст
    до полного размера сетки.

    В отличие от прошлой задачи (волновые функции psi(x), вектор)
    здесь Conv2d используется по делу: сетка (x,t) физически
    двумерна, поэтому 2D-свёртка соответствует реальной геометрии
    задачи, а не привязана искусственно.

    Геометрия декодера (проверена аналитически):
        holst (8,16) --ConvTranspose2d(k=4,s=2,p=1)--> (16,32)
                      --ConvTranspose2d(k=4,s=2,p=1)--> (32,64)
                      --ConvTranspose2d(k=4,s=2,p=1)--> (64,128)  == (NT, NX)

    Вход:  (batch, 4)              -- [A, v, nu, x0]
    Выход: u_real, u_imag  каждый (batch, NT, NX)
    """

    def __init__(self, n_params=4, NT=64, NX=128, hidden=256,
                 holst_h=8, holst_w=16, conv_channels=64):
        super().__init__()
        self.NT, self.NX = NT, NX
        self.holst_h, self.holst_w = holst_h, holst_w
        self.conv_channels = conv_channels

        # MLP-энкодер: параметры солитона -> вектор признаков
        self.encoder = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Линейный слой -> "холст": маленькая 2D-карта признаков
        self.to_holst = nn.Sequential(
            nn.Linear(hidden, conv_channels * holst_h * holst_w),
            nn.Tanh(),
        )

        # Conv2d-декодер: holst (conv_channels, 8, 16) -> (2, 64, 128)
        # 2 выходных канала = Re(u) и Im(u) одновременно
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),  # финальное уточнение -> 2 канала
        )

    def forward(self, x):
        features = self.encoder(x)
        holst = self.to_holst(features)
        holst = holst.view(-1, self.conv_channels, self.holst_h, self.holst_w)

        out = self.conv_decoder(holst)        # (batch, 2, NT, NX)
        u_real = out[:, 0, :, :]               # (batch, NT, NX)
        u_imag = out[:, 1, :, :]               # (batch, NT, NX)

        return u_real, u_imag


if __name__ == "__main__":
    model = NLSESurrogate()
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total:,}")

    dummy = torch.randn(8, 4)
    u_real, u_imag = model(dummy)
    print(f"Выход Re(u): {u_real.shape}  (должно быть [8, 64, 128])")
    print(f"Выход Im(u): {u_imag.shape}  (должно быть [8, 64, 128])")