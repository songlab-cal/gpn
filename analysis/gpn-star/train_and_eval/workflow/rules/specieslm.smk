rule specieslm_run_vep_llr:
    input:
        "{dataset}/test.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/preds/{dataset}/{genome}/SpeciesLM_LLR.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python scripts/run_vep_llr_specieslm.py {input} \
        {config[specieslm][window_size]} {config[specieslm][model_path]} {output} \
        --is_file --dataloader_num_workers 8 \
        --per_device_batch_size {config[specieslm][per_device_batch_size]}
        """
