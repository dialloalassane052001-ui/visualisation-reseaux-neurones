"""Perceptron simple (NumPy) avec enregistrement de l'historique des poids,
pour visualiser la convergence (cas séparable / non séparable)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Perceptron:
    def __init__(self, add_bias=True, max_iters=10000, record_updates=False):
        self.max_iters = max_iters
        self.add_bias = add_bias
        self.record_updates = record_updates
        self.w_hist, self.n_hist = [], []

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        N, D = x.shape
        w = np.zeros(D)
        y = 2 * y - 1  # {0,1} -> {-1,+1}
        t, change = 0, True
        while change and t < self.max_iters:
            change = False
            for n in np.random.permutation(N):
                if np.sign(x[n] @ w) == y[n]:
                    continue
                w = w + y[n] * x[n]
                if self.record_updates:
                    self.w_hist.append(w.copy())
                    self.n_hist.append(n)
                change = True
                t += 1
                if t >= self.max_iters:
                    break
        self.converged = not change
        self.n_updates = t
        print(f"convergence après {t} mises à jour" if not change
              else f"pas de convergence après {t} mises à jour")
        self.w = w
        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        return (np.sign(x @ self.w) + 1) // 2


def error_history(model, x, y):
    """Taux d'erreur à chaque mise à jour des poids."""
    errors = []
    for w in model.w_hist:
        pred = (np.sign(x @ w[:-1] + w[-1]) + 1) // 2
        errors.append(np.mean(pred != y))
    return errors


def _plot_boundaries(ax, x, y, model, color, title, ylim=None):
    ax.plot(x[y == 0, 0], x[y == 0, 1], "k.", label="Classe 0")
    ax.plot(x[y == 1, 0], x[y == 1, 1], ".", color=color, label="Classe 1")
    xs = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    n = len(model.w_hist)
    for t, w in enumerate(model.w_hist):
        ax.plot(xs, -w[0] / w[1] * xs - w[2] / w[1], color="green", alpha=t / n, lw=1.2)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)


def run(out_dir="results"):
    iris = datasets.load_iris()
    # Cas séparable : setosa vs versicolor (pétales)
    x1, y1 = iris["data"][:, 2:], iris["target"]
    x1, y1 = x1[y1 < 2], y1[y1 < 2]
    m1 = Perceptron(record_updates=True).fit(x1, y1)
    # Cas non séparable : (versicolor + setosa) vs virginica
    x2, y2 = iris["data"][:, [1, 2]], iris["target"] > 1
    m2 = Perceptron(record_updates=True).fit(x2, y2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    _plot_boundaries(a1, x1, y1, m1, "red", "(a) Cas linéairement séparable")
    _plot_boundaries(a2, x2, y2, m2, "blue", "(b) Cas non séparable", ylim=(-1, 10))
    plt.tight_layout()
    plt.savefig(f"{out_dir}/06_perceptron_comparaison.png", dpi=200, bbox_inches="tight")
    plt.show()
    return m1, m2


if __name__ == "__main__":
    np.random.seed(0)
    run()
