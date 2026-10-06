"""Grad-CAM (Selvaraju et al., 2017) implémenté avec des hooks PyTorch.

CAM = ReLU( sum_k  alpha_k * A^k ),  alpha_k = moyenne spatiale du gradient du score de
la classe cible par rapport à la carte d'activation A^k de la couche cible.
"""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.acts = out.detach()
        if out.requires_grad:  # hook sur le tenseur (compatible ReLU inplace de ResNet)
            out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        """x: (B,C,H,W). Retourne (cams normalisées (B,H,W) dans [0,1], classes cibles)."""
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(1)
            elif not torch.is_tensor(class_idx):
                class_idx = torch.full((len(x),), int(class_idx), dtype=torch.long, device=x.device)
            logits.gather(1, class_idx.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
        return (cam - mn) / (mx - mn + 1e-8), class_idx


def patch_attention_ratio(cams, masks):
    """Part de la « masse » de la CAM située sur le carré parasite (0..1).
    À comparer à la part de surface du carré (9/1024 ≈ 0,9 %) : un ratio très supérieur
    signifie que le modèle regarde l'indice parasite plutôt que le chiffre."""
    cams = cams.cpu()
    inside = (cams * masks).flatten(1).sum(1)
    total = cams.flatten(1).sum(1) + 1e-8
    return (inside / total).mean().item()
