include: "clinvar.smk"
include: "cosmic.smk"
include: "enformer.smk"
include: "eqtl.smk"
include: "gnomad.smk"
include: "gwas.smk"
include: "mpra.smk"
include: "omim.smk"


rule merge_variants:
    input:
        "results/clinvar/filt.parquet",
        "results/cosmic/filt/test.parquet",
        "results/omim/variants.parquet",
        "results/gnomad/merged/subsampled/variants.parquet",
    output:
        "results/variants/test.parquet",
    run:
        clinvar = pd.read_parquet(input[0])
        clinvar["source"] = "ClinVar"

        cosmic = pd.read_parquet(input[1])
        cosmic["source"] = "COSMIC"
        cosmic["label"] = "Frequent"

        omim = pd.read_parquet(input[2])
        omim["source"] = "OMIM"

        gnomad = pd.read_parquet(input[3])
        gnomad = gnomad.rename(columns={"Status": "label"})
        gnomad["source"] = "gnomAD"
        
        V = pd.concat([clinvar, cosmic, omim, gnomad], ignore_index=True)
        core_cols = ["chrom", "pos", "ref", "alt", "label", "source", "consequence"]
        extra_cols = [c for c in V.columns if c not in core_cols]
        V = V[core_cols + extra_cols]
        print(V)

        chrom_order = [str(i) for i in range(1, 23)] + ['X', 'Y']
        V.chrom = pd.Categorical(V.chrom, categories=chrom_order, ordered=True)
        V = V.sort_values(['chrom', 'pos'])
        V.chrom = V.chrom.astype(str)
        print(V)
        print(V.source.value_counts())
        V.to_parquet(output[0], index=False)


rule merge_enformer_variants:
    input:
        "results/gnomad/merged/enformer/variants.parquet",
        "results/enformer/merged.parquet",
    output:
        "results/variants_enformer/test.parquet",
    run:
        gnomad = pd.read_parquet(input[0])
        enformer = pd.read_parquet(input[1])
        V = gnomad.merge(enformer, on=["chrom", "pos", "ref", "alt"], how="inner")
        chrom_order = [str(i) for i in range(1, 23)] + ['X', 'Y']
        V.chrom = pd.Categorical(V.chrom, categories=chrom_order, ordered=True)
        V = V.sort_values(['chrom', 'pos'])
        V.chrom = V.chrom.astype(str)
        print(V)
        V.to_parquet(output[0], index=False)


rule run_vep_msa:
    input:
        "results/msa/{msa}/all.zarr",
    output:
        "results/preds/{dataset}/msa_{msa}.parquet",
    threads: workflow.cores
    run:
        V = load_dataset(wildcards['dataset'], split="test").to_pandas()
        print(V)
        genome_msa = GenomeMSA(input[0])
        V["score"] = genome_msa.run_vep_batch(
            V["chrom"].values, V["pos"].values, V["ref"].values, V["alt"].values,
            backend="multiprocessing", n_jobs=threads
        )
        print(V)
        V[["score"]].to_parquet(output[0], index=False)


rule run_vep_gpn:
    input:
        "results/msa/{alignment}/{species}/all.zarr",
        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
    output:
        "results/preds/{dataset}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    params:
        lambda wildcards: "--disable_aux_features" if wildcards.model.split("/")[-3] == "False" else ""
    threads:
        workflow.cores
    shell:
        """
        torchrun --nproc_per_node 4 -m gpn.msa.inference vep {wildcards.dataset} {input[0]} \
        {wildcards.window_size} {input[1]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} {params}
        """


rule run_vep_embedding_gpn:
    input:
        "results/msa/{alignment}/{species}/all.zarr",
        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
    output:
        "results/preds/vep_embedding/{dataset}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    params:
        lambda wildcards: "--disable_aux_features" if wildcards.model.split("/")[-3] == "False" else ""
    threads:
        workflow.cores
    shell:
        """
        torchrun --nproc_per_node 4 -m gpn.msa.inference vep_embedding {wildcards.dataset} {input[0]} \
        {wildcards.window_size} {input[1]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} {params}
        """


#ruleorder: run_vep_gpn_window_size_ablation > run_vep_gpn
#
#
#rule run_vep_gpn_window_size_ablation:
#    input:
#        "results/msa/{alignment}/{species}/all.zarr",
#        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
#    output:
#        "results/preds/{dataset}/{alignment}/{species}/{window_size}/{model}.{window_size_ablation}.parquet",
#    wildcard_constraints:
#        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
#        alignment="[A-Za-z0-9_]+",
#        species="[A-Za-z0-9_-]+",
#        window_size="\d+",
#        window_size_ablation="\d+",
#    params:
#        lambda wildcards: "--disable_aux_features" if wildcards.model.split("/")[-3] == "False" else ""
#    threads:
#        workflow.cores
#    shell:
#        """
#        torchrun --nproc_per_node 4 -m gpn.msa.inference vep {wildcards.dataset} {input[0]} \
#        {wildcards.window_size_ablation} {input[1]} {output} \
#        --per_device_batch_size 2048 --dataloader_num_workers {threads} {params}
#        """
#