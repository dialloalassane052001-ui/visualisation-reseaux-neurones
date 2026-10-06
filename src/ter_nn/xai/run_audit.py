"""Audit XAI complet d'un ResNet18 entraîné sur des chiffres avec indice parasite.

Usage (depuis la racine du dépôt) :
    python -m src.ter_nn.xai.run_audit

Produit dans results/xai/ : metrics.json, gradcam_*.png, filter_importance.png, dashboard.png
Les chiffres du README doivent être recopiés depuis metrics.json après exécution.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18

from .data import make_datasets
from .gradcam import GradCAM, patch_attention_ratio
from .filter_importance import ablation_importance, taylor_importance, spearman
from .dashboard import save_dashboard

OUT = "results/xai"
EPOCHS, BATCH, LR, SEED = 15, 64, 1e-3, 0


def build_resnet18(num_classes=10):
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)  # images 32x32, 1 canal
    m.maxpool = nn.Identity()
    return m


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    return (torch.cat([model(x[i:i + 256]).argmax(1) for i in range(0, len(x), 256)]) == y).float().mean().item()


def plot_gradcams(cam_fn, data, mode, path, n=8):
    x, y, _ = data[mode]
    cams, pred = cam_fn(x[:n])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray"); axes[0, i].axis("off")
        axes[0, i].set_title(f"vrai {int(y[i])} / préd {int(pred[i])}", fontsize=8)
        axes[1, i].imshow(x[i, 0], cmap="gray"); axes[1, i].imshow(cams[i], cmap="jet", alpha=.5)
        axes[1, i].axis("off")
    fig.suptitle(f"Grad-CAM — test « {mode} »")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    data = make_datasets(SEED)
    x_tr, y_tr, _ = data["train"]
    model = build_resnet18()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    layer = model.layer4
    cam = GradCAM(model, layer)
    tracked = data["aligned"][0][:1]
    hist = {"loss": [], "acc_train": [], "acc_shuffled": [], "cams": []}

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(x_tr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x_tr[b]), y_tr[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        hist["loss"].append(tot / len(x_tr))
        hist["acc_train"].append(accuracy(model, x_tr, y_tr))
        hist["acc_shuffled"].append(accuracy(model, *data["shuffled"][:2]))
        c, _ = cam(tracked, class_idx=int(data["aligned"][1][0]))
        hist["cams"].append(c[0].numpy())
        print(f"Epoque {ep + 1}: loss={hist['loss'][-1]:.4f} "
              f"acc_train={hist['acc_train'][-1]:.3f} acc_shuffled={hist['acc_shuffled'][-1]:.3f}")

    # 1) Audit de biais : précision selon la condition de test + attention sur le carré
    metrics = {"epochs": EPOCHS, "seed": SEED, "patch_area_fraction": 9 / 1024}
    for mode in ("aligned", "none", "shuffled"):
        x, y, mask = data[mode]
        metrics[f"acc_{mode}"] = accuracy(model, x, y)
        if mode != "none":
            cams, _ = cam(x, class_idx=None)
            metrics[f"patch_attention_{mode}"] = patch_attention_ratio(cams, mask)
        plot_gradcams(lambda xb: cam(xb), data, mode, f"{OUT}/gradcam_{mode}.png")

    # 2) Importance des filtres de layer3 (ablation vs Taylor) sur le test « aligned »
    l3 = model.layer3
    xs, ys = data["aligned"][0][:300], data["aligned"][1][:300]
    n_ch = l3[-1].bn2.num_features
    abl = ablation_importance(model, l3, xs, ys, n_ch)
    tay = taylor_importance(model, l3, xs, ys)
    metrics["filter_importance_spearman"] = spearman(abl, tay)
    top = np.argsort(abl)[::-1][:20]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(range(20), abl[top]); ax[0].set_title("Top 20 filtres (ablation) — chute de précision")
    ax[0].set_xticks(range(20)); ax[0].set_xticklabels(top, rotation=90, fontsize=7)
    ax[1].scatter(abl, tay, s=8); ax[1].set_xlabel("ablation"); ax[1].set_ylabel("Taylor |act·grad|")
    ax[1].set_title(f"Spearman = {metrics['filter_importance_spearman']:.2f}")
    plt.tight_layout(); plt.savefig(f"{OUT}/filter_importance.png", dpi=150); plt.close(fig)

    # 3) Dashboard
    save_dashboard(hist, tracked[0, 0].numpy(), f"{OUT}/dashboard.png")
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
