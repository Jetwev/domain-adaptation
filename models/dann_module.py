import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from models.dann import Classifier, Discriminator, Extractor

# from models.resnet import ResNet50, Classifier


class CustomLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, length, epochs, last_epoch=-1):
        self.length = length
        self.epochs = epochs
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        start_steps = self.last_epoch * self.length
        total_steps = self.epochs * self.length
        p = float(self._step_count + start_steps) / total_steps

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

        return [param_group['lr']
                for param_group in self.optimizer.param_groups]

    def on_train_start(self, trainer, pl_module):
        self._step_count = 0

    def on_batch_end(self, trainer, pl_module):
        self._step_count += 1


class DannModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        # self.arch = ResNet50()
        self.encoder = Extractor()
        self.classifier = Classifier(params.num_classes)
        self.discriminator = Discriminator()
        self.class_loss = nn.CrossEntropyLoss()
        self.discr_loss = nn.CrossEntropyLoss()
        self.lr = params.lr
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.num_classes = params.num_classes
        self.length = params.dataloaders_len

    def training_step(self, batch, batch_idx):
        # input is a zip of two domains, seperate them
        x_source, y_source = batch[0]
        x_target, y_target = batch[1]

        # set 'alpha' parameter
        current_epoch = self.trainer.current_epoch

        start_steps = current_epoch * self.length
        total_steps = self.epochs * self.length

        p = float(batch_idx + start_steps) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1.

        # source clssification loss
        logits = self.encoder(x_source)
        logits = self.classifier(logits)
        class_loss = self.class_loss(logits, y_source)

        # domain loss
        domain_source_labels = torch.zeros(
            y_source.shape[0]).type(torch.LongTensor)
        domain_target_labels = torch.ones(
            y_target.shape[0]).type(torch.LongTensor)
        combine = torch.cat((x_source, x_target), 0)
        domain_combined_label = torch.cat(
            (domain_source_labels, domain_target_labels), 0).type(torch.LongTensor).to(combine.get_device())

        combine = self.encoder(combine)
        domain_logits = self.discriminator(combine, alpha)
        domain_loss = self.discr_loss(domain_logits, domain_combined_label)

        # total loss
        total_loss = class_loss + domain_loss

        # log loss
        self.log_dict({"train_loss": total_loss}, on_epoch=True, on_step=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.encoder(x)
        logits = self.classifier(logits)
        # loss = self.class_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)

        if stage:
            self.log_dict({f"{stage}_acc": acc}, prog_bar=True,
                          on_epoch=True, on_step=False)  # f"{stage}_loss": loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # scheduler_dict = {
        #     "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs),
        #     "interval": "step",
        # }
        scheduler_dict = {
            "scheduler": CustomLRScheduler(optimizer, self.length, self.epochs),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
