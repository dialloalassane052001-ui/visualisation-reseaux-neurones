"""Jeu de données d'audit : chiffres manuscrits (sklearn `load_digits`, 8x8 -> 32x32)
avec un *indice parasite* volontaire : un petit carré blanc dans la marge, dont la
position dépend de la classe.

Pendant l'entraînement, le carré est « aligné » avec la classe dans 90 % des cas : le
réseau peut donc tricher (shortcut learning). On audite ensuite si le modèle regarde le
chiffre ou le carré, avec trois conditions de test :
  - aligned  : carré cohérent avec la vraie classe
  - none     : pas de carré
  - shuffled : carré d'une AUTRE classe (si la précision s'effondre, le modèle se fie au carré)
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SIZE, PATCH = 32, 3
# 10 positions (ligne, colonne) dans la marge de 4 px, une par classe
POS = [(0, 1), (0, 7), (0, 13), (0, 19), (0, 25),
       (29, 1), (29, 7), (29, 13), (29, 19), (29, 25)]


def _base_images():
    d = load_digits()
    x = torch.tensor(d.images, dtype=torch.float32) / 16.0
    x = F.interpolate(x[:, None], size=(24, 24), mode="bilinear", align_corners=False)
    x = F.pad(x, (4, 4, 4, 4))  # marge de 4 px -> 32x32
    return x, torch.tensor(d.target, dtype=torch.long)


def add_patch(x, y, mode, rng):
    """Retourne (images avec carré, masque booléen du carré)."""
    x = x.clone()
    mask = torch.zeros(len(x), SIZE, SIZE, dtype=torch.bool)
    for i in range(len(x)):
        if mode == "none":
            continue
        if mode == "aligned":
            k = int(y[i])
        elif mode == "shuffled":
            k = int(rng.choice([c for c in range(10) if c != int(y[i])]))
        elif mode == "train":
            k = int(y[i]) if rng.random() < 0.9 else int(rng.integers(10))
        else:
            raise ValueError(mode)
        r, c = POS[k]
        x[i, 0, r:r + PATCH, c:c + PATCH] = 1.0
        mask[i, r:r + PATCH, c:c + PATCH] = True
    return x, mask


def make_datasets(seed=0):
    rng = np.random.default_rng(seed)
    x, y = _base_images()
    idx_tr, idx_te = train_test_split(np.arange(len(x)), test_size=0.33,
                                      stratify=y.numpy(), random_state=seed)
    x_tr, m_tr = add_patch(x[idx_tr], y[idx_tr], "train", rng)
    data = {"train": (x_tr, y[idx_tr], m_tr)}
    for mode in ("aligned", "none", "shuffled"):
        xt, mt = add_patch(x[idx_te], y[idx_te], mode, rng)
        data[mode] = (xt, y[idx_te], mt)
    return data
