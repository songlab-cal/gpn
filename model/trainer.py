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


language_model_name = "armheb/DNA_bert_6"


def main(hparams):
    study_name = "deepsea"

    def objective(trial):
        model_args = {}
        model_args["batch_size"] = 10
        model_args["accumulate_grad_batches"] = 10
        n_epochs = 2

        data_module = DeepSEADataModule(
            "../../../deepsea-dataset/",
            model_args["batch_size"],
            language_model_name,
        )
        data_module.prepare_data()

        # this calculates number of steps defined as optimizer steps
        #tb_size = model_args["batch_size"] * max(1, int(hparams.gpus))
        #ab_size = model_args["accumulate_grad_batches"] * float(n_epochs)
        #model_args["num_training_steps"] = (len(data_module.train_dataset) // tb_size) // ab_size

        # this calculates number of steps defined as number of minibatches
        # unfortunately, need to use this to define the learning rate scheduler, since .step() is called
        # on every minibatch. well not sure how accumulate_grad_batches is working
        model_args["num_training_steps"] = n_epochs * len(data_module.train_dataloader())
        model_args["num_warmup_steps"] = 0 #model_args["num_training_steps"] // 100
        model_args["lr"] = 5e-5
        print(model_args)

        print("Loading model...")
        model = DeepSEAModel(language_model_name, n_output=919, **model_args)
        print("Done.")

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            gpus=hparams.gpus,
            precision=16,
            strategy="dp",
            accumulate_grad_batches=model_args["accumulate_grad_batches"],
            val_check_interval=0.1,
            callbacks=[lr_monitor],
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
