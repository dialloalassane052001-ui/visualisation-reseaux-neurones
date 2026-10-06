"""Dashboard Matplotlib de suivi de la convergence et de l'interprétation.

- `save_dashboard` : figure statique (pertes, précisions, évolution de la CAM).
- `show_interactive` : même chose avec un curseur pour parcourir les époques (nécessite
  un backend graphique : Jupyter `%matplotlib widget` ou exécution locale).
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def _curves(ax_loss, ax_acc, h):
    ep = range(1, len(h["loss"]) + 1)
    ax_loss.plot(ep, h["loss"], "b-"); ax_loss.set_title("Perte (train)")
    ax_loss.set_xlabel("Époque"); ax_loss.grid(alpha=.3)
    ax_acc.plot(ep, h["acc_train"], "b-", label="train (carré aligné)")
    ax_acc.plot(ep, h["acc_shuffled"], "r-", label="test carré mélangé")
    ax_acc.set_title("Précision"); ax_acc.set_xlabel("Époque"); ax_acc.legend(); ax_acc.grid(alpha=.3)


def save_dashboard(h, image, path, n_snap=4):
    """h: dict d'historique avec 'loss', 'acc_train', 'acc_shuffled', 'cams' (liste de
    cartes (H,W) une par époque). image: (H,W) de l'exemple suivi."""
    n = len(h["cams"])
    snaps = sorted(set(int(round(i * (n - 1) / (n_snap - 1))) for i in range(n_snap)))
    fig = plt.figure(figsize=(4 * max(len(snaps), 2), 7))
    gs = fig.add_gridspec(2, len(snaps))
    _curves(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), h)
    for j, e in enumerate(snaps):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(image, cmap="gray"); ax.imshow(h["cams"][e], cmap="jet", alpha=0.5)
        ax.set_title(f"Grad-CAM, époque {e + 1}"); ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def show_interactive(h, image):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.2)
    _curves(axes[0], axes[1], h)
    axes[2].imshow(image, cmap="gray")
    overlay = axes[2].imshow(h["cams"][0], cmap="jet", alpha=0.5)
    axes[2].axis("off"); axes[2].set_title("Grad-CAM, époque 1")
    ax_s = plt.axes([0.25, 0.05, 0.5, 0.03])
    s = Slider(ax_s, "Époque", 1, len(h["cams"]), valinit=1, valstep=1)

    def update(v):
        overlay.set_data(h["cams"][int(v) - 1])
        axes[2].set_title(f"Grad-CAM, époque {int(v)}")
        fig.canvas.draw_idle()

    s.on_changed(update)
    plt.show()
