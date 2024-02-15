from tqdm import tqdm


include: "clinvar.smk"
include: "cosmic.smk"
include: "dms.smk"
include: "enformer.smk"
include: "eqtl.smk"
include: "gnomad.smk"
include: "gwas.smk"
include: "mpra.smk"
include: "omim.smk"
include: "primateai3d.smk"
include: "SGE.smk"


omim_gnomad_match = {
    "5_prime_UTR": "5' UTR",
    "upstream_gene": "Promoter",
    "intergenic": "Enhancer",
    "3_prime_UTR": "3' UTR",
    "non_coding_transcript_exon": "ncRNA",
}


rule make_clinvar_set:
    input:
        "results/clinvar/filt.parquet",
        "results/gnomad/merged/subsampled/test.parquet",
    output:
        "results/clinvar/merged/test.parquet",
    run:
        clinvar = pd.read_parquet(input[0])
        gnomad = pd.read_parquet(input[1]).query(
            'label == "Common" and consequence == "missense"'
        )
        V = pd.concat([clinvar, gnomad], ignore_index=True)
        V = sort_chrom_pos(V)
        print(V)
        print(V.label.value_counts())
        V.to_parquet(output[0], index=False)


rule make_cosmic_set:
    input:
        "results/cosmic/filt/test.parquet",
        "results/gnomad/merged/subsampled/test.parquet",
    output:
        "results/cosmic/merged/test.parquet",
    run:
        cosmic = pd.read_parquet(input[0])
        cosmic["label"] = "Frequent"
        gnomad = pd.read_parquet(input[1]).query(
            'label == "Common" and consequence == "missense"'
        )
        V = pd.concat([cosmic, gnomad], ignore_index=True)
        V = sort_chrom_pos(V)
        print(V)
        print(V.label.value_counts())
        V.to_parquet(output[0], index=False)


rule make_omim_set:
    input:
        "results/omim/variants.parquet",
        "results/gnomad/merged/subsampled/test.parquet",
    output:
        "results/omim/merged/test.parquet",
    run:
        omim = pd.read_parquet(input[0])
        omim.consequence = (
            omim.consequence.str.split(" ").str[:-1].str.join(sep=" ")
            .str.replace("’", "'").replace("RNA Gene", "ncRNA")
        )
        gnomad = pd.read_parquet(input[1]).query('label == "Common"')
        gnomad = gnomad[gnomad.consequence.isin(omim_gnomad_match.keys())]
        gnomad.consequence = gnomad.consequence.map(omim_gnomad_match)
        V = pd.concat([omim, gnomad], ignore_index=True)
        V = sort_chrom_pos(V)
        print(V)
        print(V.groupby(["consequence", "label"]).size())
        V.to_parquet(output[0], index=False)


rule merge_enformer_variants:
    input:
        "results/gnomad/merged/enformer/variants.parquet",
        "results/enformer/coords/merged.parquet",
    output:
        "results/variants_enformer/test.parquet",
    run:
        V = (
            pl.read_parquet(input[0])
            .join(
                pl.read_parquet(input[1], columns=COORDINATES),
                on=COORDINATES, how="inner"
            )
        )
        cs = config["gnomad"]["enformer_consequences"]
        Vs = []
        for c in tqdm(cs):
            V_c = V.filter(pl.col("consequence")==c)
            min_counts = V_c.group_by("label").len()["len"].min()
            for label in V["label"].unique():
                Vs.append(
                    V_c.filter(pl.col("label")==label)
                    .sample(n=min(min_counts, config["gnomad"]["subsample"]), seed=42)
                )
        V = pl.concat(Vs).to_pandas()
        V = sort_chrom_pos(V)
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
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128", "results/clinvar/mis_pat_ben"]),
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    params:
        lambda wildcards: "--disable_aux_features" if wildcards.model.split("/")[-3] == "False" else ""
    threads:
        workflow.cores
    shell:
        """
        torchrun --nproc_per_node $(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') -m gpn.msa.inference vep {wildcards.dataset} {input[0]} \
        {wildcards.window_size} {input[1]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} {params}
        """


ruleorder: run_vep_gpn_window_size_ablation > run_vep_gpn


rule run_vep_gpn_window_size_ablation:
    input:
        "results/msa/{alignment}/{species}/all.zarr",
        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
    output:
        "results/preds/{dataset}/{alignment}/{species}/{window_size}/{model}.{window_size_ablation}.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
        window_size_ablation="\d+",
    params:
        lambda wildcards: "--disable_aux_features" if wildcards.model.split("/")[-3] == "False" else ""
    threads:
        workflow.cores
    shell:
        """
        torchrun --nproc_per_node $(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') -m gpn.msa.inference vep {wildcards.dataset} {input[0]} \
        {wildcards.window_size_ablation} {input[1]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} {params}
        """

