import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics.functional import accuracy
from torchvision.models import resnet50

from models.dann import Classifier, Extractor


class SourceOnly(LightningModule):
    def __init__(self, params):
        super().__init__()
        # self.arch = ResNet50()
        self.encoder = Extractor()
        self.classifier = Classifier(params.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.lr = params.lr
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.num_classes = params.num_classes

    def forward(self, x):
        x = self.encoder(x)
        out = self.classifier(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        # loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)

        if stage:
            self.log_dict({f"{stage}_acc": acc}, prog_bar=True,
                          on_epoch=True, on_step=False)  # f"{stage}_loss": loss,

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
