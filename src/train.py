import argparse
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.mnist import get_mnist_dataloaders
from src.inference.generate import generate_grid
from src.models.cvae import CVAE
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CVAE on MNIST")

    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-dir", type=str, default="models")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cvae-mnist-mlops")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="cvae-mnist-baseline")
    parser.add_argument("--wandb-watch", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name.strip().replace(" ", "-")

    output_dir = Path(args.output_dir) / "runs" / run_name
    model_dir = Path(args.model_dir) / "runs" / run_name
    checkpoints_dir = model_dir / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CVAE(
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="cvae-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [checkpoint_callback]

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    logger = None
    if args.use_wandb:
        if not os.getenv("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY no está definida. Exporta la variable de entorno antes de lanzar el entrenamiento."
            )

        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            save_dir=str(output_dir),
            log_model="all",
        )

        logger.experiment.config.update(vars(args))

        if args.wandb_watch:
            logger.watch(model, log="all", log_freq=100)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
    )

    print(f"Iniciando entrenamiento del Generador Condicional para run: {run_name}")
    trainer.fit(model, train_loader, val_loader)
    print("Entrenamiento finalizado.")

    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        raise RuntimeError("No se ha encontrado ningún checkpoint guardado.")

    final_ckpt_path = model_dir / f"{run_name}.ckpt"
    shutil.copy2(best_ckpt_path, final_ckpt_path)

    best_model = CVAE.load_from_checkpoint(best_ckpt_path)
    best_model.eval()

    grid_path = output_dir / f"{run_name}_sample_grid.png"
    generate_grid(
        best_model,
        num_classes=10,
        samples_per_class=10,
        seed=args.seed,
        save_path=str(grid_path),
    )

    if logger is not None:
        logger.experiment.log(
            {
                "sample_grid": wandb.Image(str(grid_path)),
                "best_checkpoint_path": str(final_ckpt_path),
            }
        )
        wandb.finish()

    print(f"Best checkpoint copied to: {final_ckpt_path}")
    print(f"Sample grid saved to: {grid_path}")


if __name__ == "__main__":
    main()