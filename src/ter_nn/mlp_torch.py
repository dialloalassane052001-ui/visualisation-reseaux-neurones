"""MLP PyTorch entraîné sur des distributions 2D synthétiques, avec visualisation de
la frontière de décision aux époques 0, 10 et 50 et des courbes d'apprentissage.

NOTE (limite connue, identique au notebook original) : le modèle se termine par une
sigmoïde alors que la perte utilisée est BCEWithLogitsLoss (qui applique déjà une
sigmoïde). La perte est donc calculée sur des probabilités et non sur des logits, ce qui
limite l'apprentissage (précision de validation ≈ 75 %). Voir README, section « Limites ».
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .data_generator import Generator


class My1stMlp(nn.Module):
    def __init__(self, input_dim=2, hidden_num=4, output_dim=1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_num)
        self.hidden_layer = nn.Linear(hidden_num, hidden_num)
        self.output_layer = nn.Linear(hidden_num, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        return torch.sigmoid(self.output_layer(x))


def draw_decision_boundary(samples, model, labels, epoch, save_path, tmin, tmax, device="cpu"):
    model.eval()
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(tmin[0].item() - .5, tmax[0].item() + .5, 100),
                         np.linspace(tmin[1].item() - .5, tmax[1].item() + .5, 100))
    with torch.no_grad():
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
        Z = model(grid).cpu().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu, levels=50)
    sc = plt.scatter(samples[:, 0], samples[:, 1], c=labels, alpha=0.7,
                     cmap=plt.cm.RdBu, edgecolors="k", s=30)
    plt.colorbar(sc)
    plt.title(f"Frontière de décision - Époque {epoch}")
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def accuracy(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = (model(X) > 0.5).float().view(-1)
            total += y.size(0)
            correct += (pred == y.float()).sum().item()
    return 100 * correct / total


def train(model, num_epochs=50, lr=1e-3, weight_decay=1e-4, out_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    valid_loader = Generator(n_dist=10, n_points=100).dataloader(128)
    ref_dist, ref_labels = Generator(n_dist=1, n_points=200).linear(norm=True)
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    L_tr, L_va, A_tr, A_va = [], [], [], []

    for epoch in range(num_epochs + 1):
        model.train()
        gen = Generator()  # nouvelles distributions à chaque époque
        loader = gen.dataloader(batch_size=512)
        s = n = 0
        for X, y in loader:
            X, y = X.to(device), y.float().view(-1, 1).to(device)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
            s += loss.item() * y.numel(); n += y.numel()
        model.eval()
        sv = nv = 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.float().view(-1, 1).to(device)
                X = (X - gen.mean) / gen.std
                sv += criterion(model(X), y).item() * y.numel(); nv += y.numel()
        L_tr.append(s / n); L_va.append(sv / nv)
        A_tr.append(accuracy(model, loader, device)); A_va.append(accuracy(model, valid_loader, device))
        if epoch in (0, 10, 50):
            draw_decision_boundary(ref_dist.numpy(), model, ref_labels.numpy(), epoch,
                                   f"{out_dir}/frontiere_epoch_{epoch}.png",
                                   gen.min, gen.max, device)
        print(f"Epoque {epoch}: Loss={L_va[-1]:.4f}, Acc={A_va[-1]:.1f}%")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    a1.plot(L_tr, "b-", label="Entraînement"); a1.plot(L_va, "r-", label="Validation")
    a1.set_title("Évolution des pertes"); a1.set_xlabel("Époques"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(A_tr, "b-", label="Entraînement"); a2.plot(A_va, "r-", label="Validation")
    a2.set_title("Évolution de la précision"); a2.set_xlabel("Époques"); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/courbes_apprentissage.png", dpi=200, bbox_inches="tight")
    plt.show()
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    train(My1stMlp(input_dim=2, hidden_num=8, output_dim=1))
