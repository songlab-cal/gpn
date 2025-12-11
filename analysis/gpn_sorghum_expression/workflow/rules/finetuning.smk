rule finetune:
    output:
        directory("results/checkpoints/{model}/{hparams}/{seed}"),
    wildcard_constraints:
        model="|".join(pretraining_models),
        hparams="|".join(finetuning_hparams),
    params:
        model_path=lambda wildcards: config["models"][wildcards.model],
        run_name=lambda wildcards: (
            wildcards.model.replace("/", "_")
            + "_"
            + wildcards.hparams.replace("/", "_")
            + "_"
            + wildcards.seed
        ),
        hparams=get_hparams,
    threads: workflow.cores
    priority: 200
    shell:
        """
        WANDB_PROJECT={config[finetuning][project_name]} \
        python workflow/scripts/finetune.py \
        --do_train --do_eval --do_test \
        --dataset_name {config[finetuning][dataset_name]} \
        --model_name_or_path {params.model_path} \
        --problem_type regression \
        --regression_softplus \
        {params.hparams} \
        --torch_compile \
        --dataloader_drop_last \
        --save_total_limit 1 \
        --bf16 \
        --bf16_full_eval \
        --save_strategy epoch \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --report_to wandb \
        --dataloader_num_workers {threads} \
        --preprocessing_num_workers {threads} \
        --output_dir {output[0]} \
        --run_name {params.run_name} \
        --seed {wildcards.seed} \
        --remove_unused_columns False \
        """


# this should be updated once we use the new GPN rather than ConvNet class
#        # --classification_head standard


rule finetune_save_epoch:
    output:
        directory("results/checkpoints_epoch/{model}/{hparams}/{seed}"),
    wildcard_constraints:
        model="|".join(pretraining_models),
        hparams="|".join(finetuning_hparams),
    params:
        model_path=lambda wildcards: config["models"][wildcards.model],
        run_name=lambda wildcards: (
            wildcards.model.replace("/", "_")
            + "_"
            + wildcards.hparams.replace("/", "_")
            + "_"
            + wildcards.seed
            + "_epoch"
        ),
        hparams=get_hparams,
    threads: workflow.cores
    priority: 200
    shell:
        """
        WANDB_PROJECT={config[finetuning][project_name]} \
        python workflow/scripts/finetune.py \
        --do_train --do_eval --do_test \
        --dataset_name {config[finetuning][dataset_name]} \
        --model_name_or_path {params.model_path} \
        --problem_type regression \
        --regression_softplus \
        {params.hparams} \
        --torch_compile \
        --dataloader_drop_last \
        --bf16 \
        --bf16_full_eval \
        --save_strategy epoch \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --report_to wandb \
        --dataloader_num_workers {threads} \
        --preprocessing_num_workers {threads} \
        --output_dir {output[0]} \
        --run_name {params.run_name} \
        --seed {wildcards.seed} \
        --remove_unused_columns False \
        """


rule finetune_save_epoch_lora:
    output:
        directory("results/checkpoints_epoch_lora/{model}/{hparams}/{seed}"),
    wildcard_constraints:
        model="|".join(pretraining_models),
        hparams="|".join(finetuning_hparams),
    params:
        model_path=lambda wildcards: config["models"][wildcards.model],
        run_name=lambda wildcards: (
            wildcards.model.replace("/", "_")
            + "_"
            + wildcards.hparams.replace("/", "_")
            + "_"
            + wildcards.seed
            + "_epoch_lora"
        ),
        hparams=get_hparams,
    threads: workflow.cores
    priority: 200
    shell:
        """
        WANDB_PROJECT={config[finetuning][project_name]} \
        python workflow/scripts/finetune_lora.py \
        --do_train --do_eval --do_test \
        --dataset_name {config[finetuning][dataset_name]} \
        --model_name_or_path {params.model_path} \
        --problem_type regression \
        --regression_softplus \
        {params.hparams} \
        --torch_compile \
        --dataloader_drop_last \
        --bf16 \
        --bf16_full_eval \
        --save_strategy epoch \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --report_to wandb \
        --dataloader_num_workers {threads} \
        --preprocessing_num_workers {threads} \
        --output_dir {output[0]} \
        --run_name {params.run_name} \
        --seed {wildcards.seed} \
        --remove_unused_columns False \
        """


rule finetune_save_epoch_dataset:
    output:
        directory(
            "results/checkpoints_epoch_dataset/{dataset}/{model}/{hparams}/{seed}"
        ),
    wildcard_constraints:
        model="|".join(pretraining_models),
        hparams="|".join(finetuning_hparams),
    params:
        model_path=lambda wildcards: config["models"][wildcards.model],
        run_name=lambda wildcards: (
            wildcards.dataset.replace("/", "_")
            + "_"
            + wildcards.model.replace("/", "_")
            + "_"
            + wildcards.hparams.replace("/", "_")
            + "_"
            + wildcards.seed
            + "_epoch"
        ),
        hparams=get_hparams,
    threads: workflow.cores
    priority: 200
    shell:
        """
        WANDB_PROJECT={config[finetuning][project_name]} \
        python workflow/scripts/finetune.py \
        --do_train --do_eval --do_test \
        --dataset_name {wildcards.dataset} \
        --model_name_or_path {params.model_path} \
        --problem_type regression \
        --regression_softplus \
        {params.hparams} \
        --torch_compile \
        --dataloader_drop_last \
        --bf16 \
        --bf16_full_eval \
        --save_strategy epoch \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --report_to wandb \
        --dataloader_num_workers {threads} \
        --preprocessing_num_workers {threads} \
        --output_dir {output[0]} \
        --run_name {params.run_name} \
        --seed {wildcards.seed} \
        --remove_unused_columns False \
        """


rule predict_validation:
    input:
        "results/{checkpoint}",
    output:
        "results/RNAseq/preds/{checkpoint}.parquet",
    threads: workflow.cores
    run:
        dataset_name = config["finetuning"]["dataset_name"]
        track = config["experimental_data_predict_track"]["PsbS"]
        tracks = (
            pd.read_csv(f"hf://datasets/{dataset_name}/labels.txt", header=None)
            .values.ravel()
            .tolist()
        )
        track_index = tracks.index(track)
        d = load_dataset(dataset_name, split="validation")
        y_true = d.map(
            lambda x: {"y_true": x["labels"][track_index]},
            remove_columns=list(d.features.keys()),
        ).to_pandas()
        y_pred = run_prediction(d, input[0], batch_size=256, threads=threads)[
            :, track_index
        ]
        y_pred = pd.DataFrame(y_pred, columns=["y_pred"])
        res = pd.concat([y_true, y_pred], axis=1)
        res.to_parquet(output[0], index=False)


rule RNA_seq_corr:
    input:
        "results/RNAseq/preds/{checkpoint}.parquet",
    output:
        "results/RNAseq/metrics/{checkpoint}.csv",
    run:
        df = pd.read_parquet(input[0])
        r = df.y_true.corr(df.y_pred)
        pd.DataFrame({"pearson": [r]}).to_csv(output[0], index=False)
