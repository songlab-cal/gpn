rule nucleotide_transformer_run_vep_llr:
    input:
        "{dataset}/test.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/preds/{dataset}/{genome}/NucleotideTransformer_LLR.parquet",
    threads:
        workflow.cores
    priority: 20
    shell:
        """
        python scripts/run_vep_llr_nucleotide_transformer.py \
        {input} {config[nucleotide_transformer][model_path]} {output} \
        --is_file --dataloader_num_workers 8 --per_device_batch_size 64
        """


rule nucleotide_transformer_run_vep_inner_products:
    input:
        "{dataset}/test.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/preds/{dataset}/{genome}/NucleotideTransformer_InnerProducts.parquet",
    threads:
        workflow.cores
    priority: 20
    shell:
        """
        python scripts/run_vep_inner_products_nucleotide_transformer.py \
        {input} {config[nucleotide_transformer][model_path]} {output} \
        --is_file --dataloader_num_workers 8 --per_device_batch_size 32
        """
