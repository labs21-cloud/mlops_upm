import argparse
import logging
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


def configurar_logger(ruta_log: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(ruta_log, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


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

    ruta_log = output_dir / "train.log"
    logger = configurar_logger(ruta_log)

    logger.info("==== INICIO DEL PIPELINE DE ENTRENAMIENTO ====")
    logger.info("Nombre del run: %s", run_name)
    logger.info("Argumentos de ejecución: %s", vars(args))
    logger.info("Directorio de salida: %s", output_dir)
    logger.info("Directorio del modelo: %s", model_dir)
    logger.info("Directorio de checkpoints: %s", checkpoints_dir)
    logger.info("Archivo de log: %s", ruta_log)

    try:
        train_loader, val_loader, _ = get_mnist_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        logger.info("Los dataloaders de MNIST se han creado correctamente.")

        model = CVAE(
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
        )
        logger.info(
            "Modelo inicializado con latent_dim=%s y learning_rate=%s",
            args.latent_dim,
            args.learning_rate,
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
        logger.info("Callbacks configurados: ModelCheckpoint y LearningRateMonitor.")

        wandb_logger = None
        if args.use_wandb:
            if not os.getenv("WANDB_API_KEY"):
                raise RuntimeError(
                    "WANDB_API_KEY no está definida. Exporta la variable de entorno antes de lanzar el entrenamiento."
                )

            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                save_dir=str(output_dir),
                log_model=False,
            )

            wandb_logger.experiment.config.update(vars(args))
            logger.info(
                "W&B inicializado correctamente. Proyecto=%s, run=%s",
                args.wandb_project,
                run_name,
            )

            if args.wandb_watch:
                wandb_logger.watch(model, log="all", log_freq=100)
                logger.info("Se ha activado wandb.watch para monitorizar el modelo.")
        else:
            logger.info("W&B desactivado para esta ejecución.")

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=True,
            default_root_dir=str(output_dir),
            enable_progress_bar=True,
        )
        logger.info(
            "Trainer creado con max_epochs=%s, accelerator=%s y devices=%s",
            args.max_epochs,
            args.accelerator,
            args.devices,
        )

        print(f"Iniciando entrenamiento del Generador Condicional para run: {run_name}")
        logger.info("Comienza trainer.fit.")
        trainer.fit(model, train_loader, val_loader)
        print("Entrenamiento finalizado.")
        logger.info("trainer.fit ha finalizado correctamente.")

        best_ckpt_path = checkpoint_callback.best_model_path
        if not best_ckpt_path:
            raise RuntimeError("No se ha encontrado ningún checkpoint guardado.")

        logger.info("Mejor checkpoint encontrado en: %s", best_ckpt_path)

        final_ckpt_path = model_dir / f"{run_name}.ckpt"
        shutil.copy2(best_ckpt_path, final_ckpt_path)
        logger.info("Checkpoint final copiado a: %s", final_ckpt_path)

        best_model = CVAE.load_from_checkpoint(best_ckpt_path)
        best_model.eval()
        logger.info("El mejor modelo se ha cargado desde checkpoint y está en modo evaluación.")

        grid_path = output_dir / f"{run_name}_sample_grid.png"
        generate_grid(
            best_model,
            num_classes=10,
            samples_per_class=10,
            seed=args.seed,
            save_path=str(grid_path),
        )
        logger.info("El Grid de muestras se ha generado en: %s", grid_path)

        if wandb_logger is not None:
            wandb_logger.experiment.log(
                {
                    "sample_grid": wandb.Image(str(grid_path)),
                    "best_checkpoint_path": str(final_ckpt_path),
                }
            )
            logger.info("Los artefactos principales se han registrado en W&B.")

            wandb.finish()
            logger.info("La ejecución de W&B se ha cerrado correctamente.")

        print(f"Best checkpoint copied to: {final_ckpt_path}")
        print(f"Sample grid saved to: {grid_path}")
        logger.info("==== FIN DEL PIPELINE DE ENTRENAMIENTO ====")

    except Exception:
        logger.exception("==== ERROR EN EL PIPELINE DE ENTRENAMIENTO ====")
        raise


if __name__ == "__main__":
    main()