"""Générateur de distributions 2D synthétiques (classification binaire) pour PyTorch."""
import random as rd
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class Generator:
    def __init__(self, n_dist=1000, n_points=100, n_features=2, n_classes=2):
        self.n_dist, self.n_points = n_dist, n_points
        self.n_features, self.n_classes = n_features, n_classes
        (self.dist, self.labels, self.mean,
         self.std, self.min, self.max) = self._generate()

    def _generate(self):
        all_dist, all_labels = [], []
        for _ in range(self.n_dist):
            X, y = make_classification(
                n_samples=self.n_points, n_features=self.n_features,
                n_informative=self.n_features, n_redundant=0, n_repeated=0,
                n_classes=self.n_classes, class_sep=rd.randint(1, 3))
            all_dist.append(torch.tensor(X, dtype=torch.float32))
            all_labels.append(torch.tensor(y, dtype=torch.long))
        d, l = torch.stack(all_dist), torch.stack(all_labels)
        return (d, l, d.mean(dim=(0, 1)), d.std(dim=(0, 1)),
                d.amin(dim=(0, 1)), d.amax(dim=(0, 1)))

    def linear(self, norm=True):
        d = self.dist.view(-1, self.n_features)
        l = self.labels.view(-1)
        if norm:
            d = (d - self.mean) / self.std
        return d, l

    def dataloader(self, batch_size=128, norm=True):
        d, l = self.linear(norm)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(d, l), batch_size=batch_size, shuffle=True)

    def plot(self, num_distributions=6, out_dir=None):
        if self.n_features != 2:
            print("Visualisation possible uniquement pour 2 features")
            return
        n = min(num_distributions, self.n_dist)
        fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(15, 5 * ((n + 2) // 3)))
        axes = axes.flatten()
        for i in range(n):
            X, y = self.dist[i].numpy(), self.labels[i].numpy()
            axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, alpha=0.7)
            axes[i].set_title(f"Distribution {i + 1}")
        for ax in axes[n:]:
            ax.set_visible(False)
        plt.tight_layout()
        if out_dir:
            plt.savefig(f"{out_dir}/08_distributions_generees.png", dpi=150, bbox_inches="tight")
        plt.show()
