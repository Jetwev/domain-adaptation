from pathlib import Path

import torch
import torch.utils.data
import torch.utils.data as torchdata
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from datamodule.datasets import ConcatDataset, get_dataloaders, get_datasets
from models.danfix_module import DannFixbiModule
from models.dann_module import DannModule
from models.fixbi_module import FixbiModule
from models.mstn_module import MstnModule
from models.source_module import SourceOnly
from utils.utils import get_image

params = OmegaConf.load(
    Path(Path(__file__).parent.resolve() / 'configs' / 'config.yaml'))
params.root_dir = str(Path(__file__).parent.resolve())


if __name__ == '__main__':

    seed_everything(42, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params.device = device

    wandb_logger = WandbLogger(project=params.wandb.project,
                               name=params.wandb.name,
                               save_dir=str(Path(params.root_dir) / 'saved'),
                               config={
                                   "learning_rate": params.lr,
                                   "architecture": "Architect",
                                   "dataset": "Office31",
                                   "epochs": params.epochs,
                               }
                               )

    src_trainset, src_testset = get_datasets(params.source_domain, params)
    tar_trainset, tar_testset = get_datasets(params.target_domain, params)

    ratio_source = 1
    ratio_target = 1

    if params.datasets == 'align':
        if len(src_trainset) > len(tar_trainset):
            ratio_target = round((1.*len(src_trainset)) / len(tar_trainset))
        else:
            ratio_source = round((1.*len(tar_trainset)) / len(src_trainset))

    if params.datasets == 'concat':
        if len(src_trainset) > len(tar_trainset):
            num_copies = round((1.*len(src_trainset)) / len(tar_trainset))
            copied_datasets = [tar_trainset for _ in range(num_copies)]
            tar_trainset = torch.utils.data.ConcatDataset(copied_datasets)
        else:
            num_copies = round((1.*len(tar_trainset)) / len(src_trainset))
            copied_datasets = [src_trainset for _ in range(num_copies)]
            src_trainset = torch.utils.data.ConcatDataset(copied_datasets)

    # combine_dataloader = torch.utils.data.DataLoader(ConcatDataset(src_trainset, tar_trainset), batch_size=params.batch_size,
    #                                                  shuffle=params.shuffle, drop_last=True, num_workers=params.nb_wokers)

    if params.bt_size_source == 'common':
        src_train_loader = torchdata.DataLoader(src_trainset, batch_size=params.batch_size // ratio_source,
                                                shuffle=params.shuffle, drop_last=True, num_workers=params.nb_wokers)
    else:
        src_train_loader = torchdata.DataLoader(src_trainset, batch_size=params.bt_size_source,
                                                shuffle=params.shuffle, drop_last=True, num_workers=params.nb_wokers)

    if params.bt_size_target == 'common':
        tar_train_loader = torchdata.DataLoader(tar_trainset, batch_size=params.batch_size // ratio_target,
                                                shuffle=params.shuffle, drop_last=True, num_workers=params.nb_wokers)
    else:
        tar_train_loader = torchdata.DataLoader(tar_trainset, batch_size=params.bt_size_target,
                                                shuffle=params.shuffle, drop_last=True, num_workers=params.nb_wokers)

    # src_train_loader = get_dataloaders(src_trainset, params)
    src_test_loader = get_dataloaders(src_testset, params)
    # tar_train_loader = get_dataloaders(tar_trainset, params)
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
            TQDMProgressBar(refresh_rate=20)],
        logger=wandb_logger
    )

    if params.approach == 'source_module':
        model = SourceOnly(params)

        trainer.fit(model=model, train_dataloaders=src_train_loader,
                    val_dataloaders=tar_test_loader)

    elif params.approach == 'dann_module':
        params.dataloaders_len = min(
            len(src_train_loader), len(tar_train_loader))
        model = DannModule(params)

        combine_dataloader_zip = [src_train_loader, tar_train_loader]
        trainer.fit(model=model, train_dataloaders=combine_dataloader_zip,
                    val_dataloaders=tar_test_loader)

    elif params.approach == 'mstn_module':
        params.dataloaders_len = min(
            len(src_train_loader), len(tar_train_loader))
        model = MstnModule(params)

        combine_dataloader_zip = [src_train_loader, tar_train_loader]
        trainer.fit(model=model, train_dataloaders=combine_dataloader_zip,
                    val_dataloaders=tar_test_loader)

    elif params.approach == 'fixbi_module':
        model = FixbiModule(params)

        combine_dataloader_zip = [src_train_loader, tar_train_loader]
        trainer.fit(model=model, train_dataloaders=combine_dataloader_zip,
                    val_dataloaders=tar_test_loader)

    elif params.approach == 'danfix_module':
        params.dataloaders_len = min(
            len(src_train_loader), len(tar_train_loader))
        model = DannFixbiModule(params)

        combine_dataloader_zip = [src_train_loader, tar_train_loader]
        trainer.fit(model=model, train_dataloaders=combine_dataloader_zip,
                    val_dataloaders=tar_test_loader)
