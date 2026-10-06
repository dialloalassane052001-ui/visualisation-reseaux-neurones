"""Importance des filtres d'une couche convolutive, par deux méthodes complémentaires :
  1) ablation : on met un canal à zéro et on mesure la chute de précision ;
  2) Taylor / gradient : moyenne de |activation x gradient| par canal (1 seule passe).
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _accuracy(model, x, y, batch=256):
    model.eval()
    ok = 0
    for i in range(0, len(x), batch):
        ok += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return ok / len(x)


def ablation_importance(model, layer, x, y, n_channels):
    """Chute de précision quand le canal c est mis à zéro (plus grand = plus important)."""
    base = _accuracy(model, x, y)
    drops = np.zeros(n_channels)
    for c in range(n_channels):
        def hook(m, i, o, c=c):
            o = o.clone()
            o[:, c] = 0
            return o
        h = layer.register_forward_hook(hook)
        drops[c] = base - _accuracy(model, x, y)
        h.remove()
    return drops


def taylor_importance(model, layer, x, y, batch=128):
    """Importance moyenne |act * grad| par canal sur (x, y)."""
    model.eval()
    store, scores = {}, None

    def fwd(m, i, o):
        store["a"] = o.detach()
        o.register_hook(lambda g: store.__setitem__("g", g.detach()))

    h = layer.register_forward_hook(fwd)
    total, n = None, 0
    for i in range(0, len(x), batch):
        model.zero_grad()
        with torch.enable_grad():
            F.cross_entropy(model(x[i:i + batch]), y[i:i + batch], reduction="sum").backward()
        s = (store["a"] * store["g"]).abs().mean(dim=(2, 3)).sum(0)
        total = s if total is None else total + s
        n += len(x[i:i + batch])
    h.remove()
    return (total / n).cpu().numpy()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
