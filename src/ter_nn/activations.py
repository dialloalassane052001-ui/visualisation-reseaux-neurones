"""Fonctions d'activation usuelles et leur tracé comparatif."""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    return np.tanh(z)


def plot_activations(save_path=None):
    z = np.linspace(-5, 5, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(z, sigmoid(z), label="Sigmoïde", color="blue", linewidth=2)
    plt.plot(z, relu(z), label="ReLU", color="red", linewidth=2)
    plt.plot(z, tanh(z), label="Tanh", color="green", linewidth=2)
    plt.title("Comparaison des fonctions d'activation")
    plt.xlabel("z")
    plt.ylabel("f(z)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_activations("results/01_fonctions_activation.png")
