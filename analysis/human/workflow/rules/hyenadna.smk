hyenadna_params = {
    'LongSafari/hyenadna-tiny-1k-seqlen-hf': "--per-device-batch-size 1024",
    'LongSafari/hyenadna-small-32k-seqlen-hf': "--per-device-batch-size 64",
    'LongSafari/hyenadna-medium-160k-seqlen-hf': "--per-device-batch-size 8",
    'LongSafari/hyenadna-medium-450k-seqlen-hf': "--per-device-batch-size 2",
    'LongSafari/hyenadna-large-1m-seqlen-hf': "--per-device-batch-size 1",
}
hyenadna_models = list(hyenadna_params.keys())
n_shards = 100


rule run_vep_hyenadna:
    input:
        "results/genome.fa.gz",
    output:
        "results/preds/{dataset}/{model}.{shard}.parquet",
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
        {wildcards.model} {output} --dataloader-num-workers {threads} {params} \
        --n-shards {n_shards} --shard {wildcards.shard}
        """


rule run_vep_hyenadna_merge_shards:
    input:
        expand("results/preds/{{dataset}}/{{model}}.{shard}.parquet", shard=range(n_shards)),
    output:
        "results/preds/{dataset}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        model="|".join(hyenadna_models),
    run:
        df = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        df.to_parquet(output[0], index=False)


rule run_vep_embeddings_hyenadna:
    input:
        "results/genome.fa.gz",
    output:
        "results/preds/vep_embedding/{dataset}/{model}.{shard}.parquet",
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
        {wildcards.model} {output} --dataloader-num-workers {threads} {params} \
        --n-shards {n_shards} --shard {wildcards.shard}
        """


rule run_vep_embeddings_hyenadna_merge_shards:
    input:
        expand("results/preds/vep_embedding/{{dataset}}/{{model}}.{shard}.parquet", shard=range(n_shards)),
    output:
        "results/preds/vep_embedding/{dataset}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        model="|".join(hyenadna_models),
    run:
        df = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        df.to_parquet(output[0], index=False)
