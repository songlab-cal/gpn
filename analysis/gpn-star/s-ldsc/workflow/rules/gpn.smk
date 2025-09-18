rule gpn_msa_run_vep_llr:
    input:
        "results/variants/merged_filt.parquet",
    output:
        "results/features/GPN-MSA_LLR.parquet",
    threads: workflow.cores
    priority: 103
    shell:
        """
        python \
        -m gpn.msa.inference vep {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size {config[gpn_msa][batch_size]} \
        --dataloader_num_workers {threads} --is_file
        """


rule gpn_msa_run_vep_inner_products:
    input:
        "results/variants/merged_filt.parquet",
    output:
        "results/features/GPN-MSA_InnerProducts.parquet",
    threads: workflow.cores
    priority: 103
    shell:
        """
        python \
        -m gpn.msa.inference vep_embedding {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size {config[gpn_msa][batch_size]} \
        --dataloader_num_workers {threads} --is_file
        """
