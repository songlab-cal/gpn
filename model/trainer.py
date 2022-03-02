from argparse import ArgumentParser
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

from data import DeepSEADataModule
from model import DeepSEAModel


pl.utilities.seed.seed_everything(seed=42)


#language_model_name = "armheb/DNA_bert_6"


def main(hparams):
    study_name = "deepsea"

    def objective(trial):
        model_args = {}
        model_args["batch_size"] = 256
        model_args["accumulate_grad_batches"] = 1
        n_epochs = 100

        data_module = DeepSEADataModule(
            "../data/datasets/",
            model_args["batch_size"],
        )
        data_module.prepare_data()

        feature_names = data_module.train_dataset.features

        # this calculates number of steps defined as optimizer steps
        #tb_size = model_args["batch_size"] * max(1, int(hparams.gpus))
        #ab_size = model_args["accumulate_grad_batches"] * float(n_epochs)
        #model_args["num_training_steps"] = (len(data_module.train_dataset) // tb_size) // ab_size

        # this calculates number of steps defined as number of minibatches
        # unfortunately, need to use this to define the learning rate scheduler, since .step() is called
        # on every minibatch. well not sure how accumulate_grad_batches is working
        #model_args["num_training_steps"] = n_epochs * len(data_module.train_dataloader())
        #model_args["num_warmup_steps"] = 0 #model_args["num_training_steps"] // 100
        #model_args["lr"] = 5e-5

        model_args["n_input"] = 4
        model_args["n_output"] = 109
        model_args["lr"] = 1e-3
        model_args["reduce_lr_on_plateau_patience"] = 1
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
        model = DeepSEAModel(**model_args)
        print("Done.")

        #lr_monitor = LearningRateMonitor(logging_interval='step')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stop_callback = EarlyStopping(
            monitor="val_neg_median_auroc", min_delta=0.00, patience=2*(1+model_args["reduce_lr_on_plateau_patience"]), verbose=True, mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            gpus=hparams.gpus,
            precision=16,
            strategy="dp",
            accumulate_grad_batches=model_args["accumulate_grad_batches"],
            #val_check_interval=0.1,
            callbacks=[lr_monitor, early_stop_callback],
        )
        ckpt_path = None
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
        result = trainer.test(datamodule=data_module, verbose=True)[0]["test_neg_median_auroc"]
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
