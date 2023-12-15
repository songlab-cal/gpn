nucleotide_transformer_params = {
    "InstaDeepAI/nucleotide-transformer-500m-human-ref": "--per-device-batch-size 128",
    "InstaDeepAI/nucleotide-transformer-500m-1000g": "--per-device-batch-size 128",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g": "--per-device-batch-size 32",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species": "--per-device-batch-size 32",
}
nucleotide_transformer_models = list(nucleotide_transformer_params.keys())


rule run_vep_nucleotide_transformer:
    input:
        "results/genome.fa.gz",
    output:
        "results/preds/{dataset}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        model="|".join(nucleotide_transformer_models)
    threads:
        workflow.cores
    params:
        lambda wildcards: nucleotide_transformer_params[wildcards.model],
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_nucleotide_transformer.py {wildcards.dataset} {input} \
        {wildcards.model} {output} --dataloader-num-workers 16 {params}
        """


rule run_vep_embeddings_nucleotide_transformer:
    input:
        "results/genome.fa.gz",
    output:
        "results/preds/vep_embedding/{dataset}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        model="|".join(nucleotide_transformer_models)
    threads:
        workflow.cores
    params:
        lambda wildcards: nucleotide_transformer_params[wildcards.model],
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_embeddings_nucleotide_transformer.py {wildcards.dataset} {input} \
        {wildcards.model} {output} --dataloader-num-workers 16 {params}
        """
