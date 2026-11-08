"""Análise dos VAEs treinados: reconstrução, geração e espaço latente.

Representação de cada imagem: a média mu do encoder (sem ruído de amostragem).
Projeções (PCA) são ajustadas nos códigos de TREINO e aplicadas ao TESTE.

Perguntas respondidas:
  1. Há formação de clusters?  -> sem usar rótulos: densidade (hexbin) e
     K-means com escolha de k pelo índice de silhueta; depois ARI/NMI contra
     os rótulos verdadeiros.
  2. Há separação dos rótulos? -> classificador linear (regressão logística)
     treinado nos códigos de treino e avaliado no teste, matriz de confusão
     e índice de Davies-Bouldin com os rótulos.
  3. Quanto da variância a projeção ilustra? -> variância explicada da PCA.

Uso: python src/analyze.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from data import CLASSES, load_splits
from model import ConvVAE, elbo_loss

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG = RESULTS / "figs"
COLORS = plt.get_cmap("tab10").colors
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})


def load_model(z):
    ck = torch.load(RESULTS / f"convvae_z{z}.pt", weights_only=False)
    m = ConvVAE(z, tuple(ck["config"]["channels"]), ck["config"]["hidden"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def encode(m, x, bs=2000):
    mus, lvs = [], []
    for i in range(0, len(x), bs):
        mu, lv = m.encode(x[i:i + bs])
        mus.append(mu); lvs.append(lv)
    return torch.cat(mus).numpy(), torch.cat(lvs).numpy()


@torch.no_grad()
def test_loss(m, x, bs=1000, reps=5):
    """-ELBO média no teste (média de 5 sorteios de z para reduzir ruído)."""
    torch.manual_seed(0)
    tot = np.zeros(3)
    for _ in range(reps):
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]
            logits, mu, lv = m(xb)
            tot += np.array([t.item() for t in elbo_loss(logits, xb, mu, lv)]) * len(xb)
    return dict(zip(["loss", "rec", "kl"], (tot / (reps * len(x))).tolist()))


@torch.no_grad()
def psnr(m, x):
    mu, _ = m.encode(x)
    rec = m.decode(mu)
    mse = ((rec - x) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).mean().item()


def scatter(ax, P, y, s=2, alpha=0.5):
    for c in range(10):
        k = y == c
        ax.scatter(P[k, 0], P[k, 1], s=s, alpha=alpha, color=COLORS[c], label=CLASSES[c], rasterized=True)
    for c in range(10):  # centróide de cada classe
        cx, cy = P[y == c].mean(0)
        ax.text(cx, cy, CLASSES[c], fontsize=8, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS[c], alpha=0.85))


def legend_below(fig):
    h = [plt.Line2D([], [], marker="o", ls="", color=COLORS[c], label=CLASSES[c]) for c in range(10)]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.11))


def cluster_metrics(Z_tr, Z_te, y_tr, y_te, label):
    """Métricas de agrupamento (sem rótulos) e de separação (com rótulos)."""
    # Clusters sem rótulos: k escolhido pela silhueta do K-means (k = 2..14), numa amostra.
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Z_te), 4000, replace=False)
    sil_k = {}
    for k in range(2, 15):
        km = KMeans(k, n_init=5, random_state=0).fit(Z_tr)
        sil_k[k] = float(silhouette_score(Z_te[sub], km.predict(Z_te[sub])))
    best_k = max(sil_k, key=sil_k.get)
    km10 = KMeans(10, n_init=10, random_state=0).fit(Z_tr)
    pred10 = km10.predict(Z_te)
    # Separação dos rótulos: classificador linear (treino -> teste).
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Z_tr, y_tr)
    yhat = clf.predict(Z_te)
    out = {
        "repr": label,
        "kmeans_silhouette_por_k": sil_k,
        "k_melhor_silhueta": best_k,
        "silhueta_k_melhor": sil_k[best_k],
        "ari_kmeans10": float(adjusted_rand_score(y_te, pred10)),
        "nmi_kmeans10": float(normalized_mutual_info_score(y_te, pred10)),
        "acc_logistica": float((yhat == y_te).mean()),
        "davies_bouldin_rotulos": float(davies_bouldin_score(Z_te, y_te)),
        "silhueta_rotulos": float(silhouette_score(Z_te[sub], y_te[sub])),
    }
    return out, confusion_matrix(y_te, yhat, normalize="true")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (x_tr, y_tr), (x_val, _), (x_te, y_te) = load_splits()
    y_tr, y_te = y_tr.numpy(), y_te.numpy()
    log = json.loads((RESULTS / "search.json").read_text())
    metrics = {"modelos": {}, "latente": []}
    models = {z: load_model(z) for z in (2, 8)}
    codes = {}

    # ---------- Treino: curvas e busca ----------
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, z in zip(axs, (2, 8)):
        h = log["final"][str(z)]["history"]
        ep = np.arange(1, len(h["val"]) + 1)
        ax.plot(ep, h["train"], label="treino")
        ax.plot(ep, h["val"], label="validação")
        b = log["final"][str(z)]["best_epoch"]
        ax.axvline(b, ls=":", c="gray", label=f"melhor época ({b})")
        lo = min(min(h["train"]), min(h["val"]))
        ax.set_ylim(lo - 3, lo + 40)
        ax.set(title=f"z = {z}", xlabel="Época", ylabel="-ELBO (nats/imagem)")
        ax.legend(frameon=False)
    fig.savefig(FIG / "curvas.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for ax, z in zip(axs, (2, 8)):
        tr = [t for t in log["search"] if t["latent_dim"] == z]
        tr = sorted(tr, key=lambda t: t["val_loss"])
        lab = [f"{t['channels'][0]}/{t['channels'][1]} | {t['hidden']} | {t['lr']:g} | {t['batch_size']}" for t in tr]
        ax.barh(range(len(tr)), [t["val_loss"] for t in tr],
                color=["#d62728" if i == 0 else "#9aa5b1" for i in range(len(tr))])
        ax.set_yticks(range(len(tr)), lab, fontsize=8)
        ax.invert_yaxis()
        v = [t["val_loss"] for t in tr]
        ax.set_xlim(min(v) - 2, max(v) + 2)
        ax.set(title=f"Busca z = {z}  (filtros | densa | lr | lote)", xlabel="-ELBO de validação")
    fig.tight_layout(); fig.savefig(FIG / "busca.png"); plt.close(fig)

    # ---------- Perdas e reconstrução no teste ----------
    for z, m in models.items():
        metrics["modelos"][z] = {**test_loss(m, x_te), "psnr": psnr(m, x_te),
                                 "params": log["final"][str(z)]["params"],
                                 "config": log["final"][str(z)]["config"],
                                 "best_epoch": log["final"][str(z)]["best_epoch"],
                                 "epochs_run": log["final"][str(z)]["epochs_run"],
                                 "val_loss": min(log["final"][str(z)]["history"]["val"])}
        mu_tr, _ = encode(m, x_tr)
        mu_te, lv_te = encode(m, x_te)
        codes[z] = (mu_tr, mu_te, lv_te)

    # Reconstruções: 2 exemplos por classe
    rng = np.random.default_rng(1)
    idx = np.concatenate([rng.choice(np.where(y_te == c)[0], 1, replace=False) for c in range(10)])
    with torch.no_grad():
        rows = [x_te[idx]] + [models[z].decode(models[z].encode(x_te[idx])[0]) for z in (2, 8)]
    fig, axs = plt.subplots(3, 10, figsize=(10, 3.2))
    for r, (imgs, name) in enumerate(zip(rows, ["original", "z = 2", "z = 8"])):
        for c in range(10):
            axs[r, c].imshow(imgs[c, 0], cmap="gray_r", vmin=0, vmax=1)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([]); axs[r, c].spines[:].set_visible(False)
            if r == 0: axs[r, c].set_title(CLASSES[c], fontsize=8)
        axs[r, 0].set_ylabel(name, fontsize=9)
    fig.savefig(FIG / "reconstrucoes.png"); plt.close(fig)

    # Amostras do prior
    torch.manual_seed(3)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, z in zip(axs, (2, 8)):
        with torch.no_grad():
            s = models[z].decode(torch.randn(64, z))
        grid = s.view(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
        ax.imshow(grid, cmap="gray_r"); ax.axis("off"); ax.set_title(f"z = {z}: amostras de N(0, I)")
    fig.savefig(FIG / "amostras.png"); plt.close(fig)

    # ---------- Latente 2D ----------
    mu_tr2, mu_te2, _ = codes[2]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    scatter(axs[0], mu_te2, y_te)
    axs[0].set(title="Latente 2D com rótulos (teste)", xlabel="$z_1$", ylabel="$z_2$")
    hb = axs[1].hexbin(mu_te2[:, 0], mu_te2[:, 1], gridsize=60, cmap="Greys", bins="log", mincnt=1)
    axs[1].set(title="Mesmo latente sem rótulos (densidade)", xlabel="$z_1$", ylabel="$z_2$")
    lim = np.percentile(np.abs(mu_te2), 99.5) * 1.05
    for ax in axs: ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    fig.colorbar(hb, ax=axs[1], fraction=0.046, label="nº de imagens (log)")
    legend_below(fig)
    fig.savefig(FIG / "latente_z2.png"); plt.close(fig)

    # Manifold do decoder (grade nos quantis da normal)
    n = 15
    q = norm.ppf(np.linspace(0.03, 0.97, n))
    zz = torch.tensor([[a, b] for b in q[::-1] for a in q], dtype=torch.float32)
    with torch.no_grad():
        im = models[2].decode(zz).view(n, n, 28, 28).permute(0, 2, 1, 3).reshape(n * 28, n * 28)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(im, cmap="gray_r", extent=[q[0], q[-1], q[0], q[-1]])
    ax.set(title="Imagens decodificadas numa grade do latente 2D", xlabel="$z_1$", ylabel="$z_2$")
    fig.savefig(FIG / "manifold_z2.png"); plt.close(fig)

    m2, cm2 = cluster_metrics(mu_tr2, mu_te2, y_tr, y_te, "z=2 (latente completo)")
    pca2 = PCA().fit(mu_tr2)
    m2["pca_var"] = pca2.explained_variance_ratio_.tolist()
    metrics["latente"].append(m2)

    # ---------- Latente 8D: PCA ----------
    mu_tr8, mu_te8, lv_te8 = codes[8]
    pca = PCA().fit(mu_tr8)
    evr = pca.explained_variance_ratio_
    P_te = pca.transform(mu_te8)
    P_tr = pca.transform(mu_tr8)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(np.arange(1, 9), evr, color="#4c72b0", label="por componente")
    ax.plot(np.arange(1, 9), np.cumsum(evr), "o-", c="k", label="acumulada")
    for i, v in enumerate(np.cumsum(evr)):
        ax.text(i + 1, v + 0.03, f"{100 * v:.0f}%", ha="center", fontsize=8)
    ax.set(xlabel="Componente principal", ylabel="Variância explicada", ylim=(0, 1.12),
           title="PCA do latente 8D (ajustada no treino)")
    ax.legend(frameon=False, loc="center right")
    fig.savefig(FIG / "pca_variancia.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    scatter(axs[0], P_te[:, :2], y_te)
    axs[0].set(title=f"PCA 8D→2D com rótulos ({100 * evr[:2].sum():.1f}% da variância)",
               xlabel=f"PC1 ({100 * evr[0]:.1f}%)", ylabel=f"PC2 ({100 * evr[1]:.1f}%)")
    hb = axs[1].hexbin(P_te[:, 0], P_te[:, 1], gridsize=60, cmap="Greys", bins="log", mincnt=1)
    axs[1].set(title="Mesma projeção sem rótulos (densidade)", xlabel="PC1", ylabel="PC2")
    fig.colorbar(hb, ax=axs[1], fraction=0.046, label="nº de imagens (log)")
    legend_below(fig)
    fig.savefig(FIG / "latente_z8_pca.png"); plt.close(fig)

    # Pares de componentes seguintes (PC3 x PC4) para mostrar o que a projeção 2D descarta
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (i, j) in zip(axs, [(0, 2), (2, 3)]):
        for c in range(10):
            k = y_te == c
            ax.scatter(P_te[k, i], P_te[k, j], s=2, alpha=0.5, color=COLORS[c], rasterized=True)
        ax.set(xlabel=f"PC{i + 1} ({100 * evr[i]:.1f}%)", ylabel=f"PC{j + 1} ({100 * evr[j]:.1f}%)")
    legend_below(fig)
    fig.savefig(FIG / "latente_z8_pca_outros.png"); plt.close(fig)

    m8, cm8 = cluster_metrics(mu_tr8, mu_te8, y_tr, y_te, "z=8 (latente completo)")
    m8["pca_var"] = evr.tolist()
    metrics["latente"].append(m8)
    m8p, cm8p = cluster_metrics(P_tr[:, :2], P_te[:, :2], y_tr, y_te, "z=8 (PCA, 2 componentes)")
    metrics["latente"].append(m8p)
    # Dimensões ativas: variância de mu entre imagens (unidade inativa ~ 0)
    metrics["var_mu_z8"] = mu_te8.var(0).tolist()
    metrics["var_mu_z2"] = mu_te2.var(0).tolist()

    # Silhueta do K-means por k (clusters sem rótulos)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for mm, st in zip(metrics["latente"], ["o-", "s-", "^--"]):
        ks = sorted(int(k) for k in mm["kmeans_silhouette_por_k"])
        ax.plot(ks, [mm["kmeans_silhouette_por_k"][k] for k in ks], st, label=mm["repr"], ms=4)
    ax.axvline(10, c="gray", ls=":", lw=1)
    ax.set(xlabel="k (nº de clusters do K-means)", ylabel="Silhueta", title="Qualidade dos clusters sem rótulos")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "kmeans_silhueta.png"); plt.close(fig)

    # Matrizes de confusão do classificador linear
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, cm, mm in zip(axs, [cm2, cm8, cm8p], metrics["latente"]):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.6 else "black")
        ax.set_xticks(range(10), CLASSES, rotation=60, fontsize=8)
        ax.set_yticks(range(10), CLASSES, fontsize=8)
        ax.set_title(f"{mm['repr']}\nacurácia {100 * mm['acc_logistica']:.1f}%", fontsize=10)
        ax.spines[:].set_visible(True)
    fig.tight_layout(); fig.savefig(FIG / "confusao.png"); plt.close(fig)

    # Interpolação no latente 8D entre pares de classes
    pairs = [(7, 9), (1, 3), (0, 6), (5, 8)]
    fig, axs = plt.subplots(len(pairs), 9, figsize=(9, 4.4))
    with torch.no_grad():
        for r, (a, b) in enumerate(pairs):
            za = torch.tensor(mu_te8[np.where(y_te == a)[0][0]])
            zb = torch.tensor(mu_te8[np.where(y_te == b)[0][0]])
            t = torch.linspace(0, 1, 9)[:, None]
            imgs = models[8].decode((1 - t) * za + t * zb)
            for c in range(9):
                axs[r, c].imshow(imgs[c, 0], cmap="gray_r"); axs[r, c].axis("off")
            axs[r, 0].set_title(CLASSES[a], fontsize=8, loc="left")
            axs[r, 8].set_title(CLASSES[b], fontsize=8, loc="right")
    fig.savefig(FIG / "interpolacao_z8.png"); plt.close(fig)

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in metrics.items() if k != "latente"}, indent=1, ensure_ascii=False))
    for mm in metrics["latente"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items() if k != "kmeans_silhouette_por_k"})


if __name__ == "__main__":
    main()
