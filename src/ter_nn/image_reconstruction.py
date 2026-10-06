"""Reconstruction d'une image 2D par un MLP : (x, y) -> intensité du pixel (Keras)."""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ARCHS = {
    "simple": [64, 64],
    "deep": [128, 128, 64, 32],
    "wide": [256, 256],
}


class ReconstructionImage:
    def __init__(self):
        self.model = None
        self.img_shape = None
        self.img_original = None

    def creer_image_os_stylise(self, size=(64, 64)):
        h, w = size
        img = np.zeros((h, w), dtype=np.float32)
        rr, cc = np.ogrid[:h, :w]
        c1, c2, r = (int(h * .3), int(w * .5)), (int(h * .7), int(w * .5)), int(h * .2)
        circle1 = (rr - c1[0]) ** 2 + (cc - c1[1]) ** 2 <= r ** 2
        circle2 = (rr - c2[0]) ** 2 + (cc - c2[1]) ** 2 <= r ** 2
        rect = (rr > c1[0]) & (rr < c2[0]) & (np.abs(cc - c1[1]) < r // 2)
        img[circle1 | circle2 | rect] = 1.0
        self.img_original, self.img_shape = img, size
        return img

    @staticmethod
    def preparer_donnees(img):
        h, w = img.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        X = np.stack([x.ravel(), y.ravel()], axis=1) / np.array([w - 1, h - 1])
        return X, img.ravel()[:, None]

    def creer_modele(self, architecture="simple"):
        sizes = ARCHS[architecture]
        layers = [Dense(sizes[0], activation="relu", input_shape=(2,))]
        layers += [Dense(n, activation="relu") for n in sizes[1:]]
        self.model = Sequential(layers + [Dense(1, activation="sigmoid")])
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return self.model

    def entrainer(self, X, Y, epochs=100, batch_size=512, verbose=2):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                              verbose=verbose, validation_split=0.1)

    def reconstruire(self):
        h, w = self.img_shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack([x.ravel(), y.ravel()], axis=1) / np.array([w - 1, h - 1])
        return self.model.predict(grid, verbose=0).reshape(h, w)

    def visualiser(self, rec, save_path=None):
        diff = np.abs(self.img_original - rec)
        plt.figure(figsize=(15, 5))
        for i, (im, t, cm) in enumerate([(self.img_original, "Image originale", "gray"),
                                         (rec, "Image reconstruite", "gray"),
                                         (diff, f"Différence (MAE: {diff.mean():.4f})", "hot")], 1):
            plt.subplot(1, 3, i); plt.imshow(im, cmap=cm); plt.title(t); plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    ri = ReconstructionImage()
    img = ri.creer_image_os_stylise((64, 64))
    X, Y = ri.preparer_donnees(img)
    ri.creer_modele("simple")
    ri.entrainer(X, Y)
    ri.visualiser(ri.reconstruire(), "results/15_reconstruction_image_mlp.png")
