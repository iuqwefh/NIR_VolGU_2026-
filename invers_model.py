import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    """Свёрточный блок с понижением размерности вдвое (stride=2)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class InverseNLSE(nn.Module):
    """
    Гибридная (data-driven + physics-consistency) инверсная модель для НУШ.

    ПОЛНАЯ обратная задача: по наблюдаемому (зашумлённому) полю u(x,t)
    восстанавливаем ВСЕ 4 параметра [A, v, nu, x0] -- ни один не считается
    известным заранее (в отличие от урезанной 2-параметрической версии
    метода Нелдера-Мида, где v, x0 были зафиксированы).

    Архитектура: CNN-энкодер (4 свёрточных блока, stride=2) сжимает
    поле (2, 64, 128) -- каналы Re/Im -- в вектор признаков (4096),
    MLP-голова выдаёт (A, v, nu, x0).

    Геометрия энкодера (проверена аналитически):
        (2, 64, 128) -> (16, 32, 64) -> (32, 16, 32)
                      -> (64, 8, 16) -> (128, 4, 8)
        flatten: 128*4*8 = 4096

    Вход:  u_real, u_imag (batch, 64, 128) -- наблюдаемое поле
    Выход: A_pred, v_pred, nu_pred, x0_pred (batch,) каждый
    """

    def __init__(self, NT=64, NX=128, hidden=256):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock2D(2,  16),
            ConvBlock2D(16, 32),
            ConvBlock2D(32, 64),
            ConvBlock2D(64, 128),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, NT, NX)
            feat_dim = self.encoder(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),   # выход: [A, v, nu, x0]
        )

    def forward(self, u_real, u_imag):
        """
        u_real, u_imag: (batch, 64, 128)
        """
        x = torch.stack([u_real, u_imag], dim=1)   # (batch, 2, 64, 128)
        feat = self.encoder(x).flatten(1)          # (batch, 4096)

        out = self.head(feat)                        # (batch, 4)
        A_pred  = out[:, 0]
        v_pred  = out[:, 1]
        nu_pred = out[:, 2]
        x0_pred = out[:, 3]
        return A_pred, v_pred, nu_pred, x0_pred


if __name__ == "__main__":
    model = InverseNLSE()
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total:,}")

    batch = 8
    u_real = torch.randn(batch, 64, 128)
    u_imag = torch.randn(batch, 64, 128)

    A_pred, v_pred, nu_pred, x0_pred = model(u_real, u_imag)
    print(f"A_pred:  {A_pred.shape}  (должно быть [8])")
    print(f"v_pred:  {v_pred.shape}")
    print(f"nu_pred: {nu_pred.shape}")
    print(f"x0_pred: {x0_pred.shape}")