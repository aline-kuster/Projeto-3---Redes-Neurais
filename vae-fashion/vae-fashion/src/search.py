"""Busca aleatória de hiperparâmetros do VAE convolucional, para z = 2 e z = 8.

Etapa 1 (busca): cada configuração sorteada do espaço SPACE é treinada por um
orçamento curto de épocas e avaliada pela -ELBO no conjunto de validação.
Etapa 2 (treino final): a melhor configuração de cada dimensão latente é
treinada de novo com mais épocas e parada antecipada (paciência na validação).

Uso: python src/search.py --trials 8 --search-epochs 10 --final-epochs 60 --patience 6
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch

from data import load_splits
from model import ConvVAE, count_params, elbo_loss

RESULTS = Path(__file__).resolve().parent.parent / "results"

# ESPAÇO DE BUSCA: valores possíveis de cada hiperparâmetro
SPACE = {
    "channels": [(16, 32), (32, 64)],   # filtros das 2 convoluções
    "hidden": [128, 256, 512],          # tamanho da camada densa
    "lr": [5e-4, 1e-3, 2e-3],           # taxa de aprendizado
    "batch_size": [64, 128],            # tamanho do lote
}


# Sorteia n combinações diferentes do espaço de busca (busca aleatória)
def sample_configs(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen, configs = set(), []
    while len(configs) < n:
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


# Calcula a perda média (-ELBO, reconstrução e KL) num conjunto, sem treinar
@torch.no_grad()
def evaluate(model, x, bs=1000):
    model.eval()
    tot = torch.zeros(3)
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        tot += torch.tensor([t.item() for t in _loss(model, xb)]) * len(xb)
    loss, rec, kl = (tot / len(x)).tolist()
    return {"loss": loss, "rec": rec, "kl": kl}


def _loss(model, xb):
    logits, mu, logvar = model(xb)
    return elbo_loss(logits, xb, mu, logvar)


# TREINO de um modelo com uma configuração
def train(cfg, latent_dim, x_tr, x_val, epochs, patience, seed):
    torch.manual_seed(seed)
    model = ConvVAE(latent_dim, tuple(cfg["channels"]), cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])  # otimizador Adam
    hist = {"train": [], "val": [], "val_rec": [], "val_kl": []}
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        # Uma época: passa por todas as imagens de treino, em lotes embaralhados
        model.train()
        perm = torch.randperm(len(x_tr))
        run = 0.0
        for i in range(0, len(x_tr), cfg["batch_size"]):
            xb = x_tr[perm[i:i + cfg["batch_size"]]]
            loss, _, _ = _loss(model, xb)
            opt.zero_grad()
            loss.backward()   # calcula os gradientes
            opt.step()        # atualiza os pesos
            run += loss.item() * len(xb)

        # Avalia na validação ao fim de cada época
        val = evaluate(model, x_val)
        hist["train"].append(run / len(x_tr))
        hist["val"].append(val["loss"])
        hist["val_rec"].append(val["rec"])
        hist["val_kl"].append(val["kl"])

        # PARADA ANTECIPADA: guarda a melhor época; para se passar "patience" épocas sem melhorar
        if val["loss"] < best - 1e-3:
            best, best_state, bad = val["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)  # volta para os pesos da melhor época
    return model, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--search-epochs", type=int, default=8)
    p.add_argument("--final-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    (x_tr, _), (x_val, _), _ = load_splits()
    log = {"space": {k: [list(v) if isinstance(v, tuple) else v for v in vals] for k, vals in SPACE.items()},
           "args": vars(a), "search": [], "final": {}}

    for z in (2, 8):
        # ETAPA 1 - BUSCA: treina cada combinação sorteada por poucas épocas
        configs = sample_configs(a.trials, seed=a.seed + z)
        trials = []
        for cfg in configs:
            t0 = time.time()
            model, hist = train(cfg, z, x_tr, x_val, a.search_epochs, patience=a.search_epochs, seed=a.seed)
            best_val = min(hist["val"])
            row = {"latent_dim": z, **cfg, "channels": list(cfg["channels"]), "val_loss": best_val,
                   "params": count_params(model), "seconds": time.time() - t0, "history": hist}
            trials.append(row)
            print(f"[busca] z={z} {cfg} -> val {best_val:.2f} ({row['seconds']:.0f}s)", flush=True)
        log["search"] += trials

        # ETAPA 2 - TREINO FINAL: a combinação com menor perda de validação é treinada de novo
        best = min(trials, key=lambda r: r["val_loss"])
        cfg = {k: best[k] for k in SPACE}
        print(f"[final] z={z} melhor configuração: {cfg}", flush=True)
        t0 = time.time()
        model, hist = train(cfg, z, x_tr, x_val, a.final_epochs, a.patience, seed=a.seed)
        best_ep = int(min(range(len(hist["val"])), key=lambda i: hist["val"][i])) + 1
        log["final"][str(z)] = {"config": cfg, "params": count_params(model), "epochs_run": len(hist["val"]),
                                "best_epoch": best_ep, "seconds": time.time() - t0, "history": hist}

        # Salva o modelo final e o histórico da busca
        torch.save({"latent_dim": z, "config": cfg, "state_dict": model.state_dict()}, RESULTS / f"convvae_z{z}.pt")
        print(f"[final] z={z} {len(hist['val'])} épocas, melhor {best_ep}: val {min(hist['val']):.2f}", flush=True)
        (RESULTS / "search.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
