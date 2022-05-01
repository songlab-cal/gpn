from argparse import ArgumentParser
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

from data import DeepSEADataModule, DNABERTDataModule, PlantBertDataModule, ConvNetDataModule
from model import DeepSEAModel, DNABERTModel, PlantBertModel, DSSModel, ConvNetModel


pl.utilities.seed.seed_everything(seed=42)


dnabert_language_model_name = "armheb/DNA_bert_6"
data_path = "../../data/chromatin/datasets/"


def main(hparams):
    study_name = "deepsea"

    def objective(trial):
        model_args = {}
        model_args["n_output"] = 109
        precision = 16
        #model_args["module"] = "PlantBert"
        #model_args["module"] = "DNABERT"
        #model_args["module"] = "DeepSEA"
        #model_args["module"] = "DSS"
        model_args["module"] = "ConvNet"

        if model_args["module"] == "DNABERT":
            model_class = DNABERTModel
            model_args["language_model_name"] = dnabert_language_model_name
            model_args["batch_size"] = 12
            model_args["accumulate_grad_batches"] = 256 // model_args["batch_size"]
            model_args["num_workers"] = 0
            n_epochs = 100
            data_module = DNABERTDataModule(
                data_path,
                model_args["batch_size"],
                dnabert_language_model_name,
                model_args["num_workers"],
            )
            data_module.prepare_data()
            model_args["lr"] = 5e-5
            model_args["reduce_lr_on_plateau_patience"] = 0
        elif model_args["module"] == "PlantBert":
            model_class = PlantBertModel
            #model_args["language_model_path"] = "../language_model/results/checkpoint-17440/"
            #model_args["language_model_path"] = "../language_model/results/checkpoint-34000/"
            model_args["language_model_path"] = "../mlm/results/checkpoint-200000/"
            #model_args["language_model_path"] = "../language_model/checkpoint-100000/"
            #model_args["language_model_path"] = "../language_model/results_nc_first_pass/checkpoint-10000/"
            #model_args["language_model_path"] = "../language_model/nc_small_span_50000/checkpoint-50000/"
            #model_args["language_model_path"] = "../language_model/nc_small_span_64/checkpoint-45000/"
            #model_args["max_length"] = 1000 #1024
            model_args["max_length"] = 200 # 170 #1024
            #model_args["batch_size"] = 85
            #model_args["accumulate_grad_batches"] = 3
            model_args["batch_size"] = 256
            model_args["accumulate_grad_batches"] = 1
            model_args["num_workers"] = 8
            n_epochs = 100
            data_module = PlantBertDataModule(
                data_path,
                model_args["batch_size"],
                model_args["language_model_path"],
                model_args["num_workers"],
                model_args["max_length"],
            )
            data_module.prepare_data()
            model_args["lr"] = 5e-5
            model_args["reduce_lr_on_plateau_patience"] = 0
        elif model_args["module"] == "ConvNet":
            model_class = ConvNetModel
            model_args["pretrained_model_path"] = "../mlm/results/checkpoint-50000/"
            model_args["pretrained_model_args"] = dict(
                vocab_size=6,
                n_layers=12,
                hidden_size=256,
            )
            model_args["batch_size"] = 256
            model_args["accumulate_grad_batches"] = 1
            model_args["num_workers"] = 8
            n_epochs = 100
            data_module = ConvNetDataModule(
                data_path,
                model_args["batch_size"],
                model_args["pretrained_model_path"],
                model_args["num_workers"],
            )
            data_module.prepare_data()
            model_args["lr"] = 5e-5
            model_args["reduce_lr_on_plateau_patience"] = 0
        elif model_args["module"] == "DeepSEA":
            model_class = DeepSEAModel
            model_args["n_input"] = 4
            model_args["batch_size"] = 256
            model_args["accumulate_grad_batches"] = 1
            model_args["num_workers"] = 8
            n_epochs = 100
            data_module = DeepSEADataModule(
                data_path,
                model_args["batch_size"],
                model_args["num_workers"],
            )
            data_module.prepare_data()
            model_args["lr"] = 1e-3
            model_args["reduce_lr_on_plateau_patience"] = 1
        elif model_args["module"] == "DSS":
            precision = 32
            model_class = DSSModel
            model_args["batch_size"] = 256
            model_args["accumulate_grad_batches"] = 1
            model_args["num_workers"] = 8
            n_epochs = 100
            data_module = DeepSEADataModule(
                data_path,
                model_args["batch_size"],
                model_args["num_workers"],
            )
            data_module.prepare_data()
            model_args["lr"] = 1e-3
            model_args["reduce_lr_on_plateau_patience"] = 1

        feature_names = data_module.train_dataset.features
        model_args["feature_names"] = feature_names
        model_args["pos_weight_strategy"] = "sqrt_inv_freq"
        if model_args["pos_weight_strategy"] == "ones":
            model_args["pos_weight"] = np.ones(len(feature_names), dtype=float)
        elif model_args["pos_weight_strategy"] == "eights":
            model_args["pos_weight"] = 8.0 * np.ones(len(feature_names), dtype=float)
        elif model_args["pos_weight_strategy"] == "sqrt_inv_freq":
            p = data_module.train_dataset.df[feature_names].mean().values
            model_args["pos_weight"] = np.sqrt((1-p) / p)
        print(model_args)

        print("Loading model...")
        model = model_class(**model_args)
        print("Done.")

        #lr_monitor = LearningRateMonitor(logging_interval='step')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stop_callback = EarlyStopping(
            monitor="val/neg_median_auroc", min_delta=0.00, patience=2*(1+model_args["reduce_lr_on_plateau_patience"]), verbose=True, mode="min",
        )

        wandb_logger = WandbLogger(project="PlantBERT_Chromatin", name="ConvNet")

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            gpus=hparams.gpus,
            precision=precision,
            strategy="dp",
            accumulate_grad_batches=model_args["accumulate_grad_batches"],
            #val_check_interval=0.1,
            callbacks=[lr_monitor, early_stop_callback],
            logger=wandb_logger,
        )
        ckpt_path = None
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
        result = trainer.test(datamodule=data_module, verbose=True)[0]["test/neg_median_auroc"]
        return result

    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{study_name}.sqlite3',
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(objective, n_trials=1)
    study.trials_dataframe().to_csv("trials_dataframe.tsv", "\t")
    print(study.best_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=None)
    #parser.add_argument("--data_dir", default=".")
    args = parser.parse_args()
    main(args)
