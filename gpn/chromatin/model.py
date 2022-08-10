import math
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, cross_entropy
import torch.nn.functional as F
from transformers import get_scheduler, AutoModel, PretrainedConfig, BertModel
from torch.optim import AdamW
import torchmetrics

import plantbert.mlm


def calculate_auroc(outputs, feature_names):
    feature_classes = {
        "dnase": np.where([f.startswith("DHS") for f in feature_names])[0],
        "tf": np.where([f.startswith("TFBS") for f in feature_names])[0],
        "histone": np.where([f.startswith("HM") for f in feature_names])[0],
    }

    preds = torch.cat([output["logits"] for output in outputs])
    targets = torch.cat([output["Y"] for output in outputs])
    n_regions = len(preds) // 2

    preds = (preds[:n_regions] + preds[n_regions:]) / 2
    if len(outputs) > 2:  # except for the sanity check
        assert(torch.allclose(targets[:n_regions], targets[n_regions:]))
    targets = targets[:n_regions]
    aurocs = torch.empty(len(feature_names))
    auprcs = torch.empty(len(feature_names))
    pr_curve = torchmetrics.PrecisionRecallCurve()
    for i in range(len(feature_names)):
        if targets[:, i].sum() < 50:
            aurocs[i] = float("nan")
            auprcs[i] = float("nan")
        else:
            aurocs[i] = torchmetrics.functional.auroc(preds[:, i], targets[:, i])
            precision, recall, _ = pr_curve(preds[:, i], targets[:, i])
            auprcs[i] = torchmetrics.functional.auc(recall, precision)
    res = (
        aurocs.nanmedian(),
        aurocs[feature_classes["dnase"]].nanmedian(),
        aurocs[feature_classes["tf"]].nanmedian(),
        aurocs[feature_classes["histone"]].nanmedian(),
        {f"test/auroc_{feature_name}": auroc for feature_name, auroc in zip(feature_names, aurocs)},
        auprcs.nanmedian(),
        auprcs[feature_classes["dnase"]].nanmedian(),
        auprcs[feature_classes["tf"]].nanmedian(),
        auprcs[feature_classes["histone"]].nanmedian(),
        {f"test/auprc_{feature_name}": auprc for feature_name, auprc in zip(feature_names, auprcs)},
    )
    return res


class Module(pl.LightningModule):
    def loss(self, logits, Y):
        res = F.binary_cross_entropy_with_logits(logits, Y.float(), pos_weight=self.pos_weight)
        return res

    def training_step(self, batch, batch_idx):
        #X, Y = batch
        #logits = self(**X)
        Y = batch.pop("Y")
        logits = self(**batch)
        loss = self.loss(logits, Y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #X, Y = batch
        #logits = self(**X)
        Y = batch.pop('Y')
        logits = self(**batch)
        return {"logits": logits, "Y": Y}

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        auroc1, auroc2, auroc3, auroc4, _, auprc1, auprc2, auprc3, auprc4, _ = calculate_auroc(outputs, self.feature_names)
        self.log("val/neg_median_auroc", -auroc1)
        self.log("val/median_auroc_dnase", auroc2)
        self.log("val/median_auroc_tf", auroc3)
        self.log("val/median_auroc_histone", auroc4)
        self.log("val/neg_median_auprc", -auprc1)
        self.log("val/median_auprc_dnase", auprc2)
        self.log("val/median_auprc_tf", auprc3)
        self.log("val/median_auprc_histone", auprc4)

    def test_step(self, batch, batch_idx):
        #X, Y = batch
        #logits = self(**X)
        #return {"logits": logits, "Y": Y}
        Y = batch.pop("Y")
        logits = self(**batch)
        return {"logits": logits, "Y": Y}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        auroc1, auroc2, auroc3, auroc4, aurocs, auprc1, auprc2, auprc3, auprc4, auprcs = calculate_auroc(outputs, self.feature_names)
        self.log("test/neg_median_auroc", -auroc1)
        self.log("test/median_auroc_dnase", auroc2)
        self.log("test/median_auroc_tf", auroc3)
        self.log("test/median_auroc_histone", auroc4)
        self.log_dict(aurocs)
        self.log("test/neg_median_auprc", -auprc1)
        self.log("test/median_auprc_dnase", auprc2)
        self.log("test/median_auprc_tf", auprc3)
        self.log("test/median_auprc_histone", auprc4)
        self.log_dict(auprcs)


class DeepSEAModel(Module):
    def __init__(
        self,
        n_input=None,
        n_output=None,
        lr=None,
        reduce_lr_on_plateau_patience=None,
        feature_names=None,
        pos_weight=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        self.reduce_lr_on_plateau_patience = reduce_lr_on_plateau_patience
        self.feature_names = feature_names

        self.Conv1 = nn.Conv1d(in_channels=self.n_input, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(53*960, 925)
        self.Linear2 = nn.Linear(925, self.n_output)

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    def forward(self, input_ids=None):
        x = one_hot(input_ids, num_classes=self.n_input).float()
        x = torch.transpose(x, 1, 2)
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = x.view(-1, 53*960)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_on_plateau_patience,
            factor=0.1,
            threshold=0.0,
            threshold_mode="abs",
            verbose=True,
        )
        monitor = "val/neg_median_auroc"
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor=monitor)


class DNABERTModel(Module):
    def __init__(
        self,
        language_model_name=None,
        n_input=None,
        n_output=None,
        lr=None,
        reduce_lr_on_plateau_patience=None,
        feature_names=None,
        pos_weight=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.language_model_name = language_model_name
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        self.reduce_lr_on_plateau_patience = reduce_lr_on_plateau_patience
        self.feature_names = feature_names

        self.language_model = AutoModel.from_pretrained(language_model_name)
        #config = PretrainedConfig.get_config_dict(language_model_name)
        #self.language_model = AutoModel.from_config(config)
        self.hidden_size = PretrainedConfig.get_config_dict(language_model_name)[0]["hidden_size"]
        self.dropout = nn.Dropout(0.1)
        #self.classifier = nn.Linear(self.hidden_size, n_output)
        self.classifier = nn.Linear(3*self.hidden_size, n_output)

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    def forward(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.unfold(1, 400, 300).reshape(-1, 400)
        attention_mask = attention_mask.unfold(1, 400, 300).reshape(-1, 400)

        x = self.language_model(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
        #print(x)
        #raise Exception("debug")
        #print(x.shape)
        x = x.view(-1, 3, self.hidden_size).view(-1, 3 * self.hidden_size)
        #print(x.shape)
        #raise Exception("debug")

        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        #optimizer = AdamW(self.parameters(), lr=self.lr)
        #lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
        #scheduler = {
        #    "scheduler": lr_scheduler,
        #    "interval": "step",
        #    "frequency": 1,
        #}
        #return dict(optimizer=optimizer, lr_scheduler=scheduler)

        optimizer = AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_on_plateau_patience,
            factor=0.1,
            threshold=0.0,
            threshold_mode="abs",
            verbose=True,
        )
        monitor = "val/neg_median_auroc"
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor=monitor)


class GPNModel(Module):
    def __init__(
        self,
        pretrained_model_path=None,
        pretrained_model_class=AutoModel,
        n_input=None,
        n_output=None,
        lr=None,
        reduce_lr_on_plateau_patience=None,
        feature_names=None,
        pos_weight=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_model_path = pretrained_model_path
        self.pretrained_model_class = pretrained_model_class
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        self.reduce_lr_on_plateau_patience = reduce_lr_on_plateau_patience
        self.feature_names = feature_names

        self.pretrained_model = pretrained_model_class.from_pretrained(pretrained_model_path, add_pooling_layer=False)
        #print(self.pretrained_model.hidden_size)
        #raise Exception()
        self.hidden_size = PretrainedConfig.get_config_dict(pretrained_model_path)[0]["hidden_size"]
        self.pooler = BertAvgPooler(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, n_output)

        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float))

    def forward(self, **kwargs):
        #print("input_ids.is_cuda: ", kwargs["input_ids"].is_cuda)
        x = self.pretrained_model(**kwargs)["last_hidden_state"]
        x = self.pooler(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        #optimizer = AdamW(self.parameters(), lr=self.lr)
        #lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
        #scheduler = {
        #    "scheduler": lr_scheduler,
        #    "interval": "step",
        #    "frequency": 1,
        #}
        #return dict(optimizer=optimizer, lr_scheduler=scheduler)

        optimizer = AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_on_plateau_patience,
            factor=0.1,
            threshold=0.0,
            threshold_mode="abs",
            verbose=True,
        )
        monitor = "val/neg_median_auroc"
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor=monitor)


class BertAvgPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = hidden_states.mean(dim=1)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertMaxPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = hidden_states.max(dim=1).values
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output
