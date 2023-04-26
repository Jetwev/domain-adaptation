import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from models.dann import Classifier, Extractor
from models.dann_module import CustomLRScheduler

# from models.resnet import ResNet50, Classifier


class FixbiModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        # models
        # self.arch = ResNet50()
        self.source_classifier = nn.Sequential(
            Extractor(), Classifier(params.num_classes))
        self.target_classifier = nn.Sequential(
            Extractor(), Classifier(params.num_classes))

        self.sp_param_sd = nn.Parameter(torch.tensor(5.0), requires_grad=True)
        self.sp_param_td = nn.Parameter(torch.tensor(5.0), requires_grad=True)

        # loss functions
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.nll = nn.NLLLoss(reduction='none')

        # common parameters
        self.lr = params.lr
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.num_classes = params.num_classes

        # parameters for fixbi training
        self.th = params.fixbi.th
        self.bim_start = params.fixbi.bim_start
        self.sp_start = params.fixbi.sp_start
        self.cr_start = params.fixbi.cr_start
        self.lam_sd = params.fixbi.lam_sd
        self.lam_td = params.fixbi.lam_td

        # This property activates manual optimization
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # optimizers
        sr_opt, tr_opt = self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        # seperate two domains
        x_source, y_source = batch[0]
        x_target, y_target = batch[1]

        current_epoch = self.trainer.current_epoch

        x_sd = self.source_classifier(x_target)
        x_td = self.target_classifier(x_target)

        # for souce domain
        pseudo_sd, top_prob_sd, threshold_sd = self.get_target_preds(x_sd)
        fixmix_sd_loss = self.get_fixmix_loss(
            self.source_classifier, x_source, x_target, y_source, pseudo_sd, self.lam_sd)

        # for target domain
        pseudo_td, top_prob_td, threshold_td = self.get_target_preds(x_td)
        fixmix_td_loss = self.get_fixmix_loss(
            self.target_classifier, x_source, x_target, y_source, pseudo_td, self.lam_td)

        # loss contains cross entropy loss for source and target domains
        total_loss = fixmix_sd_loss + fixmix_td_loss

        # Bidirectional Matching
        if current_epoch > self.bim_start:
            bim_mask_sd = torch.ge(top_prob_sd, threshold_sd)
            bim_mask_sd = torch.nonzero(bim_mask_sd).squeeze()

            bim_mask_td = torch.ge(top_prob_td, threshold_td)
            bim_mask_td = torch.nonzero(bim_mask_td).squeeze()

            if bim_mask_sd.dim() > 0 and bim_mask_td.dim() > 0:
                if bim_mask_sd.numel() > 0 and bim_mask_td.numel() > 0:
                    bim_mask = min(bim_mask_sd.size(0), bim_mask_td.size(0))
                    bim_sd_loss = self.CrossEntropy(
                        x_sd[bim_mask_td[:bim_mask]], pseudo_td[bim_mask_td[:bim_mask]].detach())
                    bim_td_loss = self.CrossEntropy(
                        x_td[bim_mask_sd[:bim_mask]], pseudo_sd[bim_mask_sd[:bim_mask]].detach())

                    total_loss += bim_sd_loss
                    total_loss += bim_td_loss

                    # log biderectional loss
                    self.log_dict(
                        {"bid_loss(SDM)": bim_sd_loss, "bid_loss(TDM)": bim_td_loss}, on_epoch=True, on_step=False)

        # Self-penalization
        if current_epoch <= self.sp_start:
            sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
            sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()

            sp_mask_td = torch.lt(top_prob_td, threshold_td)
            sp_mask_td = torch.nonzero(sp_mask_td).squeeze()

            if sp_mask_sd.dim() > 0 and sp_mask_td.dim() > 0:
                if sp_mask_sd.numel() > 0 and sp_mask_td.numel() > 0:
                    sp_mask = min(sp_mask_sd.size(0), sp_mask_td.size(0))
                    ind_sd = sp_mask_sd[:sp_mask]
                    ind_td = sp_mask_td[:sp_mask]
                    sp_sd_loss = torch.mul(self.nll(
                        torch.log(1 - F.softmax(x_sd[ind_sd] / self.sp_param_sd, dim=1)), pseudo_sd[ind_sd].detach()), 1).mean()
                    sp_td_loss = torch.mul(self.nll(
                        torch.log(1 - F.softmax(x_td[ind_td] / self.sp_param_td, dim=1)), pseudo_td[ind_td].detach()), 1).mean()

                    total_loss += sp_sd_loss
                    total_loss += sp_td_loss

                    # log penalty loss
                    self.log_dict(
                        {"pen_loss(SDM)": sp_sd_loss, "pen_loss(TDM)": sp_td_loss}, on_epoch=True, on_step=False)

        # Consistency Regularization
        if current_epoch > self.cr_start:
            mixed_cr = 0.5 * x_source + 0.5 * x_target
            out_sd, out_td = self.source_classifier(
                mixed_cr), self.target_classifier(mixed_cr)
            cr_loss = self.mse(out_sd, out_td)
            total_loss += cr_loss

            # log consistent loss
            self.log_dict({"cons_loss": cr_loss}, on_epoch=True, on_step=False)

        # log loss
        self.log_dict({"train_loss": total_loss}, on_epoch=True, on_step=False)

        sr_opt.zero_grad()
        tr_opt.zero_grad()
        self.manual_backward(total_loss)
        sr_opt.step()
        tr_opt.step()
        sch1.step()
        sch2.step()

        return total_loss

    def get_target_preds(self, x):
        top_prob, top_label = torch.topk(F.softmax(x, dim=1), k=1)
        top_label = top_label.squeeze().t()
        top_prob = top_prob.squeeze().t()
        top_mean, top_std = top_prob.mean(), top_prob.std()
        threshold = top_mean - self.th * top_std
        return top_label, top_prob, threshold

    def get_fixmix_loss(self, net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, ratio):
        mixed_x = ratio * src_imgs + (1 - ratio) * tgt_imgs
        mixed_x = net(mixed_x)
        loss = ratio * self.CrossEntropy(mixed_x, src_labels.detach()) + (
            1 - ratio) * self.CrossEntropy(mixed_x, tgt_pseudo.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def evaluate(self, batch, stage=None):
        x, y = batch
        # if self.source_classifier.training:
        #   print("something")
        logits_sdm = self.source_classifier(x)
        logits_tdm = self.target_classifier(x)

        preds_sdm = torch.argmax(logits_sdm, dim=1)
        preds_tdm = torch.argmax(logits_tdm, dim=1)

        acc_sdm = accuracy(preds_sdm, y, 'multiclass',
                           num_classes=self.num_classes)
        acc_tdm = accuracy(preds_tdm, y, 'multiclass',
                           num_classes=self.num_classes)

        sum_pred = torch.argmax(
            F.softmax(logits_sdm, dim=1) + F.softmax(logits_tdm, dim=1), dim=1)
        sum_acc = accuracy(sum_pred, y, 'multiclass',
                           num_classes=self.num_classes)

        if stage:
            self.log_dict({f"{stage}_sum_acc": sum_acc, f"{stage}_sdm_acc": acc_sdm,
                          f"{stage}_tdm_acc": acc_tdm}, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        source_optimizer = torch.optim.SGD(
            self.source_classifier.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        target_optimizer = torch.optim.SGD(
            self.target_classifier.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        source_optimizer.add_param_group(
            {"params": [self.sp_param_sd], "lr": self.lr})
        target_optimizer.add_param_group(
            {"params": [self.sp_param_td], "lr": self.lr})

        scheduler_sdict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(source_optimizer, self.epochs),
            "interval": "step",
        }

        scheduler_tdict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(target_optimizer, self.epochs),
            "interval": "step",
        }

        return [source_optimizer, target_optimizer], [scheduler_sdict, scheduler_tdict]
