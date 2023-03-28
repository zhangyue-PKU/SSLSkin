import inspect
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from model.data.transforms import FTTransformPipeline

from model.args.linear import parse_cfg
from model.methods.linear import LinearModel
from model.data.dataset import LightlyDataset
from model.utils.auto_resumer import AutoResumer
from model.utils.checkpointer import Checkpointer
from model.utils.misc import make_contiguous
from model.utils.predict_writer import PredictWriter


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    
    # Linear model, linear probbing or finetune
    model = LinearModel(cfg)
    make_contiguous(model)
    
    # Transform for dataset
    transform = FTTransformPipeline(cfg.augmentations[0])
    val_transform = FTTransformPipeline(cfg.augmentations[1])
    
    # Operating fuction for selected columns
    columns = ["image_name", "benign_malignant"]
    mapfunc = {
        "image_name": lambda x: x + ".jpg",
        "benign_malignant": lambda x: 1 if x == 'malignant' else 0
    }
    
    # Train dataset
    train_dataset = LightlyDataset(
        input_dir=cfg.data.train_path,
        transform=transform,
        metafile=cfg.data.train_label_file,
        columns=columns,
        mapfunc=mapfunc
    )

    # Val dataset
    val_dataset = LightlyDataset(
        input_dir=cfg.data.val_path,
        transform=val_transform,
        metafile=cfg.data.val_label_file,
        columns=columns,
        mapfunc=mapfunc
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=False
    )
    
    if cfg.predict:
        # Test dataset
        pred_dataset = LightlyDataset(
            input_dir=cfg.data.pred_path,
            transform=val_transform
        )
        pred_loader = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=cfg.optimizer.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=False
        )
    
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, "linear"),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint
    
    callbacks = []

    if cfg.checkpoint.enabled:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, "linear"),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)

    # wandb logging
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
 
    if cfg.predict:
        predict_writer = PredictWriter(
            write_interval="batch_and_epoch",
            output_dir=os.path.join(cfg.checkpoint.dir, "linear"),
            name=f"{cfg.name}.csv",
            index=1,
            columns=["image_name", "target"]
        )
        callbacks.append(predict_writer)    
    

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
        }
    )
    trainer = Trainer(**trainer_kwargs)
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    
    if cfg.predict:
        trainer.predict(model, pred_loader)
    

if __name__ == "__main__":
    main()