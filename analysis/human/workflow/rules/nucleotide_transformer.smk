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
        "results/preds/{dataset,songlab/human_variants}/{model}.parquet",
    wildcard_constraints:
        model="|".join(nucleotide_transformer_models)
    threads:
        workflow.cores
    params:
        lambda wildcards: nucleotide_transformer_params[wildcards.model],
    shell:
        """
        python workflow/scripts/run_vep_nucleotide_transformer.py {wildcards.dataset} {input} \
        {wildcards.model} {output} --dataloader-num-workers 16 {params}
        """
