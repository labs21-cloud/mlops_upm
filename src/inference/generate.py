from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def _resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_generator(device: torch.device, seed: Optional[int] = None):
    if seed is None:
        return None

    if device.type == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator()

    generator.manual_seed(seed)
    return generator


def generate_samples(
    model,
    labels: Sequence[int],
    device: Optional[str] = None,
    seed: Optional[int] = None,
):
    device = _resolve_device(device)
    model = model.to(device)
    model.eval()

    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    labels_one_hot = F.one_hot(
        labels_tensor, num_classes=model.hparams.num_classes
    ).float()

    generator = _build_generator(device, seed)

    with torch.no_grad():
        z = torch.randn(
            len(labels),
            model.hparams.latent_dim,
            generator=generator,
            device=device,
        )
        generated = model.decode(z, labels_one_hot)
        generated = generated.view(-1, 1, 28, 28).cpu()

    return generated


def generate_grid(
    model,
    num_classes: int = 10,
    samples_per_class: int = 10,
    device: Optional[str] = None,
    seed: Optional[int] = 42,
    save_path: Optional[str] = None,
):
    device = _resolve_device(device)
    model = model.to(device)
    model.eval()

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(16, 10))
    generator = _build_generator(device, seed)

    with torch.no_grad():
        for i in range(num_classes):
            labels = torch.full((samples_per_class,), i, dtype=torch.long, device=device)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

            z = torch.randn(
                samples_per_class,
                model.hparams.latent_dim,
                generator=generator,
                device=device,
            )

            generated_imgs = model.decode(z, labels_one_hot)
            generated_imgs = generated_imgs.view(samples_per_class, 28, 28).cpu()

            for j in range(samples_per_class):
                ax = axes[i, j]
                ax.imshow(generated_imgs[j], cmap="gray")
                ax.axis("off")

                if j == 0:
                    ax.text(
                        -10,
                        14,
                        f"Label {i}",
                        fontsize=16,
                        fontweight="bold",
                        va="center",
                        ha="right",
                    )

    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.1, hspace=0.1)
    plt.suptitle("Generación Condicionada CVAE - MNIST", fontsize=22, y=0.95)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig