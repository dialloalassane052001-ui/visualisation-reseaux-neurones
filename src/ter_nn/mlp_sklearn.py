"""Comparaison de MLP scikit-learn pour plusieurs valeurs de régularisation alpha
sur trois jeux de données synthétiques 2D."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run(save_path="results/07_mlp_sklearn_alpha.png"):
    datasets = [
        make_moons(n_samples=300, noise=0.25, random_state=42),
        make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=24),
        make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1, random_state=7),
    ]
    alphas = [5e-5, 2e-3, 2.5e-2]
    classifiers = [make_pipeline(StandardScaler(),
                                 MLPClassifier(hidden_layer_sizes=(15, 10), alpha=a,
                                               max_iter=2000, early_stopping=True,
                                               solver="adam", random_state=0))
                   for a in alphas]
    names = [f"MLP alpha={a:.5f}" for a in alphas]
    cm_tr = ListedColormap(["#FF7F7F", "#7FBFFF"])
    cm_te = ListedColormap(["#FFBFBF", "#BFDFFF"])
    h = 0.02
    plt.figure(figsize=(18, 12))
    idx = 1
    scores = {}
    for d_id, (X, y) in enumerate(datasets, start=1):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
        xx, yy = np.meshgrid(np.arange(X[:, 0].min() - .6, X[:, 0].max() + .6, h),
                             np.arange(X[:, 1].min() - .6, X[:, 1].max() + .6, h))
        plt.subplot(len(datasets), len(classifiers) + 1, idx)
        plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cm_tr, edgecolor="k")
        plt.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cm_te, edgecolor="k", alpha=0.6)
        plt.title(f"Dataset {d_id} - Données"); plt.xticks(()); plt.yticks(())
        idx += 1
        for name, clf in zip(names, classifiers):
            plt.subplot(len(datasets), len(classifiers) + 1, idx)
            clf.fit(Xtr, ytr)
            score = clf.score(Xte, yte)
            scores[(d_id, name)] = score
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.8)
            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cm_tr, edgecolor="k")
            plt.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cm_te, edgecolor="k", alpha=0.6)
            plt.title(f"{name}\nScore test : {score:.3f}")
            plt.xticks(()); plt.yticks(())
            idx += 1
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return scores


if __name__ == "__main__":
    for k, v in run().items():
        print(k, f"{v:.3f}")
