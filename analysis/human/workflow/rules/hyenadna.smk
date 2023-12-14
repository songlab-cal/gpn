hyenadna_params = {
    'LongSafari/hyenadna-tiny-1k-seqlen-hf': "--per-device-batch-size 1024",
    'LongSafari/hyenadna-small-32k-seqlen-hf': "--per-device-batch-size 64",
    'LongSafari/hyenadna-medium-160k-seqlen-hf': "--per-device-batch-size 8",
    'LongSafari/hyenadna-medium-450k-seqlen-hf': "--per-device-batch-size 2",
    'LongSafari/hyenadna-large-1m-seqlen-hf': "--per-device-batch-size 2",
}
hyenadna_models = list(hyenadna_params.keys())


rule run_vep_hyenadna:
    input:
        "results/genome.fa.gz",
    output:
        "results/preds/{dataset}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        model="|".join(hyenadna_models),
    threads:
        workflow.cores
    params:
        lambda wildcards: hyenadna_params[wildcards.model],
    shell:
        """
        python workflow/scripts/run_vep_hyenadna.py {wildcards.dataset} {input} \
        {wildcards.model} {output} --dataloader-num-workers {threads} {params}
        """


rule run_vep_embeddings_hyenadna:
    input:
        "results/genome.fa.gz",
    output:
        "results/preds/vep_embedding/{dataset}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        model="|".join(hyenadna_models),
    threads:
        workflow.cores
    params:
        lambda wildcards: hyenadna_params[wildcards.model],
    shell:
        """
        python workflow/scripts/run_vep_embeddings_hyenadna.py {wildcards.dataset} {input} \
        {wildcards.model} {output} --dataloader-num-workers {threads} {params}
        """
