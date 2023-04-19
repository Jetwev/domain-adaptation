import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from models.dann import Discriminator
from models.resnet import Classifier, ResNet50

# from models.dann import Classifier, Discriminator, Extractor


class MstnModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.encoder = ResNet50()
        # self.encoder = Extractor()
        self.classifier = Classifier(params.num_classes)
        self.discriminator = Discriminator()
        self.class_loss = nn.CrossEntropyLoss()
        self.discr_loss = nn.CrossEntropyLoss()
        self.lr = params.lr
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.num_classes = params.num_classes
        self.length = params.dataloaders_len
        self.dict_source = {}
        self.dict_target = {}
        self.tetha = 0.7

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
        repr_source = self.encoder(x_source)
        logits = self.classifier(repr_source)
        class_loss = self.class_loss(logits, y_source)

        # domain loss
        domain_source_labels = torch.zeros(
            y_source.shape[0]).type(torch.LongTensor)
        domain_target_labels = torch.ones(
            y_target.shape[0]).type(torch.LongTensor)
        combine = torch.cat((x_source, x_target), 0)
        domain_combined_label = torch.cat(
            (domain_source_labels, domain_target_labels), 0).type(torch.LongTensor).to(combine.get_device()).detach()

        combine = self.encoder(combine)
        domain_logits = self.discriminator(combine, alpha)
        domain_loss = self.discr_loss(domain_logits, domain_combined_label)

        # semantic loss
        repr_target = self.encoder(x_target)
        tar_logits = self.classifier(repr_target)
        tar_logits = torch.argmax(tar_logits, dim=1)

        temp_dict_source = {}
        for k in torch.unique(y_source):
            occur_k = torch.sum(y_source == k)
            for i, x in enumerate(y_source):
                if x == k:
                    temp_dict_source[k.item()] = temp_dict_source.get(
                        k.item(), torch.zeros_like(repr_source[i])) + repr_source[i]

            temp_dict_source[k.item()] = self.tetha * self.dict_source.get(
                k.item(), torch.zeros_like(temp_dict_source[k.item()])).detach() + (1 - self.tetha) * temp_dict_source[k.item()] / occur_k

        temp_dict_target = {}
        for k in torch.unique(tar_logits):
            occur_k = torch.sum(tar_logits == k)
            for i, x in enumerate(tar_logits):
                if x == k:
                    temp_dict_target[k.item()] = temp_dict_target.get(
                        k.item(), torch.zeros_like(repr_target[i])) + repr_target[i]

            temp_dict_target[k.item()] = self.tetha * self.dict_target.get(
                k.item(), torch.zeros_like(temp_dict_target[k.item()])).detach() + (1 - self.tetha) * temp_dict_target[k.item()] / occur_k

        sem_loss = 0
        for k in range(self.num_classes):
            sem_loss += ((temp_dict_source.get(k, torch.zeros_like(repr_source[0])) -
                         temp_dict_target.get(k, torch.zeros_like(repr_target[0])))**2).sum(axis=0)
            self.dict_source[k] = temp_dict_source.get(
                k, torch.zeros_like(repr_source[0])).detach()
            self.dict_target[k] = temp_dict_target.get(
                k, torch.zeros_like(repr_target[0])).detach()

        # total loss
        total_loss = class_loss + domain_loss + sem_loss

        # log loss
        self.log_dict({"train_loss": total_loss, "domain_loss": domain_loss,
                      "semantic loss": sem_loss}, on_epoch=True, on_step=False)

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
        # optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-3)

        # scheduler_dict = {
        #     "scheduler": CustomLRScheduler(optimizer, self.length, self.epochs),
        #     "interval": "step",
        # }

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
