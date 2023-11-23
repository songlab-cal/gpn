hyenadna_params = {
    'LongSafari/hyenadna-tiny-1k-seqlen-hf': "--per-device-batch-size 1024",
    'LongSafari/hyenadna-small-32k-seqlen-hf': "--per-device-batch-size 64",
    'LongSafari/hyenadna-medium-160k-seqlen-hf': "--per-device-batch-size 8",
    'LongSafari/hyenadna-medium-450k-seqlen-hf': "--per-device-batch-size 2",
    'LongSafari/hyenadna-large-1m-seqlen-hf': "--per-device-batch-size 1",
}
hyenadna_models = list(hyenadna_params.keys())


rule run_vep_hyenadna:
    input:
        "results/genome.fa.gz",
    output:
        "results/preds/{dataset,songlab/human_variants}/{model}.parquet",
    wildcard_constraints:
        model="|".join(hyenadna_models)
    threads:
        workflow.cores
    params:
        lambda wildcards: hyenadna_params[wildcards.model],
    shell:
        """
        python workflow/scripts/run_vep_hyenadna.py {wildcards.dataset} {input} \
        {wildcards.model} {output} --dataloader-num-workers {threads} {params}
        """
