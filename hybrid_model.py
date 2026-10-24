import torch
import torch.nn as nn


class NLSEHybrid(nn.Module):
    """
    Гибридная суррогатная модель для НУШ: та же архитектура, что в
    чистом data-driven подходе (model.py, класс NLSESurrogate) --
    MLP-энкодер + Conv2d-декодер, выдающий всю сетку (x,t) за один проход.

    Отличие гибрида НЕ в архитектуре, а в функции потерь (см. hybrid_train.py):
    помимо обычного MSE с эталоном добавляется physics residual (как в PINN) --
    штраф за невыполнение самого уравнения НУШ. Поскольку сеть выдаёт сразу
    всю регулярную сетку (x,t), производные корректно считаются центральными
    конечными разностями по выходному тензору, а градиент этого штрафа по
    параметрам сети PyTorch вычисляет автоматически.

        i du/dt + d^2u/dx^2 + nu*|u|^2*u = 0

    Идея: data-driven даёт быстрый "грубый" ответ на всей сетке сразу,
    а physics-регуляризация дополнительно "подталкивает" предсказание
    в сторону физически согласованного решения, потенциально снижая
    итоговую ошибку без необходимости кратного увеличения датасета.

    Вход:  (batch, 4)              -- [A, v, nu, x0]
    Выход: u_real, u_imag  каждый (batch, NT, NX)
    """

    def __init__(self, n_params=4, NT=64, NX=128, hidden=256,
                 holst_h=8, holst_w=16, conv_channels=64):
        super().__init__()
        self.NT, self.NX = NT, NX
        self.holst_h, self.holst_w = holst_h, holst_w
        self.conv_channels = conv_channels

        self.encoder = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.to_holst = nn.Sequential(
            nn.Linear(hidden, conv_channels * holst_h * holst_w),
            nn.Tanh(),
        )

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        holst = self.to_holst(features)
        holst = holst.view(-1, self.conv_channels, self.holst_h, self.holst_w)

        out = self.conv_decoder(holst)
        if out.shape[-2:] != (self.NT, self.NX):
            raise RuntimeError(
                f"Декодер вернул сетку {tuple(out.shape[-2:])}, "
                f"ожидалась {(self.NT, self.NX)}"
            )
        u_real = out[:, 0, :, :]
        u_imag = out[:, 1, :, :]

        return u_real, u_imag


if __name__ == "__main__":
    model = NLSEHybrid()
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total:,}")

    dummy = torch.randn(8, 4)
    u_real, u_imag = model(dummy)
    print(f"Выход Re(u): {u_real.shape}  (должно быть [8, 64, 128])")
    print(f"Выход Im(u): {u_imag.shape}  (должно быть [8, 64, 128])")
