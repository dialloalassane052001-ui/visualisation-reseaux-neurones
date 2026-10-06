"""Auto-encodeur CNN (Keras) sur une image : visualisation des filtres appris de la
première couche convolutive et des cartes d'activation de conv1 / conv2."""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Input

SIZE = 128


def build_model():
    inp = Input(shape=(SIZE * SIZE,))
    x = Reshape((SIZE, SIZE, 1))(inp)
    x = Conv2D(32, 3, padding="same", activation="relu", name="conv1")(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(SIZE * SIZE, activation="sigmoid")(x)
    return Model(inp, out)


def plot_filters(model, n=8, save_path=None):
    filters, _ = model.get_layer("conv1").get_weights()
    plt.figure(figsize=(12, 4))
    for i in range(min(filters.shape[-1], n)):
        plt.subplot(1, n, i + 1); plt.imshow(filters[:, :, 0, i], cmap="gray")
        plt.title(f"Filtre {i + 1}"); plt.axis("off")
    plt.suptitle("Filtres appris - Couche Conv 1"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_activations(model, img, n=8, save_path=None):
    act_model = Model(model.input, [model.get_layer("conv1").output,
                                    model.get_layer("conv2").output])
    acts = act_model.predict(img.reshape(1, -1))
    plt.figure(figsize=(15, 6))
    for i, a in enumerate(acts):
        for j in range(min(a.shape[-1], n)):
            plt.subplot(len(acts), n, i * n + j + 1)
            plt.imshow(a[0, :, :, j], cmap="viridis"); plt.axis("off")
    plt.suptitle("Activations des couches convolutionnelles"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    img = np.clip(np.random.normal(0.5, 0.2, (SIZE, SIZE)), 0, 1).astype(np.float32)
    model = build_model()
    model.compile(optimizer="adam", loss="mse")
    model.fit(img.reshape(1, -1), img.reshape(1, -1), epochs=30, verbose=0)
    plot_filters(model, save_path="results/17_cnn_filtres_conv1.png")
    plot_activations(model, img, save_path="results/18_cnn_activations.png")
