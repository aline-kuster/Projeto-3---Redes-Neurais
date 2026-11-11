"""VAE convolucional para imagens 28x28.

Encoder: duas convoluções com stride 2 (28 -> 14 -> 7), uma camada densa e
as cabeças mu e log(sigma^2). O decoder espelha o encoder com convoluções
transpostas. A topologia (número de filtros e largura da camada densa) é
um hiperparâmetro ajustado na busca.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# MODELO: encoder + reparametrização + decoder
class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int, channels: tuple[int, int] = (32, 64), hidden: int = 256):
        super().__init__()
        c1, c2 = channels
        self.latent_dim = latent_dim
        self.c2 = c2

        # ENCODER: imagem 28x28 -> mu e log(sigma^2)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, c1, 4, stride=2, padding=1), nn.BatchNorm2d(c1), nn.LeakyReLU(0.2),   # 14x14
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.BatchNorm2d(c2), nn.LeakyReLU(0.2),  # 7x7
            nn.Flatten(),
        )
        self.enc_fc = nn.Sequential(nn.Linear(c2 * 7 * 7, hidden), nn.LeakyReLU(0.2))
        self.fc_mu = nn.Linear(hidden, latent_dim)      # média
        self.fc_logvar = nn.Linear(hidden, latent_dim)  # log da variância

        # DECODER: z -> imagem 28x28 (espelho do encoder)
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, c2 * 7 * 7), nn.LeakyReLU(0.2),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.BatchNorm2d(c1), nn.LeakyReLU(0.2),  # 14x14
            nn.ConvTranspose2d(c1, 1, 4, stride=2, padding=1),                                          # 28x28
        )

    def encode(self, x):
        h = self.enc_fc(self.enc_conv(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode_logits(self, z):
        h = self.dec_fc(z).view(-1, self.c2, 7, 7)
        return self.dec_conv(h)

    # Sigmoide no final do decoder: pixels entre 0 e 1 (usado para gerar imagens)
    def decode(self, z):
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        # Truque da reparametrização: z = mu + sigma * epsilon
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode_logits(z), mu, logvar


# FUNÇÃO DE CUSTO: -ELBO = reconstrução + KL (por imagem)
def elbo_loss(logits, x, mu, logvar):
    """-ELBO por imagem = BCE somada nos 784 pixels + KL(q(z|x) || N(0, I)).

    Usa BCE com logits (numericamente mais estável que sigmoide + BCE).
    """
    n = x.size(0)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / n  # reconstrução
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n          # KL
    return rec + kl, rec, kl


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
