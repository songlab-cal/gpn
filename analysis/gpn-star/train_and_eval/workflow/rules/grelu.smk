rule enformer_metadata:
    output:
        "results/metadata/Enformer.csv",
    run:
        import grelu.resources

        model = grelu.resources.load_model(project="enformer", model_name="human")
        metadata = pd.DataFrame(model.data_params["tasks"])
        metadata.to_csv(output[0], index=False)


rule borzoi_metadata:
    output:
        "results/metadata/Borzoi.csv",
    run:
        import grelu.resources

        model = grelu.resources.load_model(project="borzoi", model_name="human_fold0")
        metadata = pd.DataFrame(model.data_params["tasks"])
        metadata.to_csv(output[0], index=False)


rule run_vep_enformer:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/Enformer_L2.parquet",
    priority: 101
    threads: workflow.cores
    shell:
        """
        python \
        workflow/scripts/vep_enformer_borzoi.py {input} enformer human {output} \
        --per_device_batch_size {config[enformer][batch_size]} --dataloader_num_workers {threads} --is_file
        """


# torchrun --nproc_per_node $(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') \


rule run_vep_borzoi:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/Borzoi_L2.parquet",
    threads: workflow.cores
    priority: 100
    shell:
        """
        python \
        workflow/scripts/vep_enformer_borzoi.py {input} borzoi human_fold0 {output} \
        --per_device_batch_size {config[borzoi][batch_size]} --dataloader_num_workers {threads} --is_file
        """
