from pathlib import Path

import torch
import torch.utils.data
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from datamodule.datasets import ConcatDataset, get_dataloaders, get_datasets
from models.dann_module import DannModule
from models.source_module import SourceOnly
from utils.utils import get_image

params = OmegaConf.load(
    Path(Path(__file__).parent.resolve() / 'configs' / 'config.yaml'))
params.root_dir = str(Path(__file__).parent.resolve())


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params.device = device

    wandb_logger = WandbLogger(project=params.wandb.project,
                               name=params.wandb.name,
                               save_dir=str(Path(params.root_dir) / 'saved'),
                               config={
                                   "learning_rate": params.lr,
                                   "architecture": "CNN_sorce_only",
                                   "dataset": "Office31:amazon->dslr",
                                   "epochs": params.epochs,
                               }
                               )

    src_trainset, src_testset = get_datasets(params.source_domain, params)
    tar_trainset, tar_testset = get_datasets(params.target_domain, params)

    combine_dataloader = torch.utils.data.DataLoader(ConcatDataset(src_trainset, tar_trainset), batch_size=params.batch_size,
                                                     shuffle=params.shuffle, drop_last=True, num_workers=params.nb_wokers)

    src_train_loader = get_dataloaders(src_trainset, params)
    src_test_loader = get_dataloaders(src_testset, params)
    tar_train_loader = get_dataloaders(tar_trainset, params)
    tar_test_loader = get_dataloaders(tar_testset, params)

    print(f"Number of batches in source train loader: {len(src_train_loader)}")
    print(f"Number of batches in source test loader: {len(src_test_loader)}")
    print(f"Number of batches in target train loader: {len(tar_train_loader)}")
    print(f"Number of batches in target test loader: {len(tar_test_loader)}")

    get_image(tar_test_loader, params.check)  # to get an image from dataloader

    model_checkpoint_callback = ModelCheckpoint(dirpath=str(Path(params.root_dir) / 'saved'),
                                                filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
                                                monitor="val_acc",
                                                mode="max",
                                                save_on_train_epoch_end=True,
                                                )

    trainer = Trainer(
        max_epochs=params.epochs,
        accelerator=params.device,
        devices=params.gpu.nb_gpus if params.device == 'cuda' else None,
        strategy=params.gpu.strategy,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=20),
            model_checkpoint_callback],
        logger=wandb_logger
    )

    if params.approach == 'source_module':
        model = SourceOnly(params)

        trainer.fit(model=model, train_dataloaders=src_train_loader,
                    val_dataloaders=tar_test_loader)
        # trainer.test(model, tar_test_loader)
    elif params.approach == 'dann_module':
        params.dataloaders_len = min(
            len(src_train_loader), len(tar_train_loader))
        model = DannModule(params)

        # combine_dataloader = zip(src_train_loader, tar_train_loader)
        trainer.fit(model=model, train_dataloaders=combine_dataloader,
                    val_dataloaders=tar_test_loader)
