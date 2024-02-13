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
        gnomad = pd.read_parquet(input[0])
        enformer = pd.read_parquet(input[1], columns=COORDINATES)
        V = gnomad.merge(enformer, on=COORDINATES, how="inner")
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


#rule run_vep_embedding_gpn:
#    input:
#        "results/msa/{alignment}/{species}/all.zarr",
#        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
#    output:
#        "results/preds/vep_embedding/{dataset}/{alignment}/{species}/{window_size}/{model}.parquet",
#    wildcard_constraints:
#        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
#        alignment="[A-Za-z0-9_]+",
#        species="[A-Za-z0-9_-]+",
#        window_size="\d+",
#    params:
#        lambda wildcards: "--disable_aux_features" if wildcards.model.split("/")[-3] == "False" else ""
#    threads:
#        workflow.cores
#    shell:
#        """
#        torchrun --nproc_per_node $(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') -m gpn.msa.inference vep_embedding {wildcards.dataset} {input[0]} \
#        {wildcards.window_size} {input[1]} {output} \
#        --per_device_batch_size 2048 --dataloader_num_workers {threads} {params}
#        """


#rule run_vep_embedding_gpn_window_size_ablation:
#    input:
#        "results/msa/{alignment}/{species}/all.zarr",
#        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
#    output:
#        "results/preds/vep_embedding/{dataset}/{alignment}/{species}/{window_size}/{model}.{window_size_ablation}.parquet",
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
#        torchrun --nproc_per_node $(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') -m gpn.msa.inference vep_embedding {wildcards.dataset} {input[0]} \
#        {wildcards.window_size_ablation} {input[1]} {output} \
#        --per_device_batch_size 2048 --dataloader_num_workers {threads} {params}
#        """


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


# models which may contain NA
models_subset_for_supervised_models = {
    "gwas/matched": [
        "Enformer",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    ],
    "eqtl/matched/ge": [
        "Enformer",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    ],
    "eqtl/matched/leafcutter": [
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    ],
}


# subset to be used for supervised models (making sure all supervised models
# use the same set)
rule make_subset_for_supervised_models:
    input:
        full_set="results/{d}/test.parquet",
        models=lambda wildcards: expand(
            "results/preds/vep_embedding/results/{{d}}/{model}.parquet", 
            model=models_subset_for_supervised_models[wildcards.d],
        ),
    output:
        "results/test_subset/{d}/variants.parquet",
    wildcard_constraints:
        d="gwas/matched|eqtl/matched/ge|eqtl/matched/leafcutter",
    run:
        V = pd.read_parquet(input.full_set)
        models = models_subset_for_supervised_models[wildcards.d]
        for model, path in zip(models, input.models):
            try:
                V[model] = pd.read_parquet(path, columns=["embedding_0"])["embedding_0"].values
            except:
                V[model] = pd.read_parquet(path, columns=["feature_0"])["feature_0"].values
        V.dropna(subset=models, inplace=True)
        V = V[V.duplicated("match_group", keep=False)]
        print(V.label.value_counts())
        V[COORDINATES].to_parquet(output[0], index=False)


ruleorder: run_vep_functionality_lr > run_vep_gpn
ruleorder: run_vep_functionality_lr > run_vep_hyenadna


rule concat_embeddings:
    input:
        "results/preds/vep_embedding/results/{d}/{m1}.parquet", 
        "results/preds/vep_embedding/results/{d}/{m2}.parquet", 
    output:
        "results/preds/vep_embedding/results/{d}/{m1}/concat/{m2}.parquet", 
    wildcard_constraints:
        d="gwas/matched|eqtl/matched/ge|eqtl/matched/leafcutter",
    run:
        df1 = pd.read_parquet(input[0])
        df2 = pd.read_parquet(input[1])
        if wildcards.m1 == "Enformer":
            df1 = df1.abs()
        if wildcards.m2 == "Enformer":
            df2 = df2.abs()
        df = pd.concat([df1, df2], axis=1)
        print(df)
        df.to_parquet(output[0], index=False)


rule add_embeddings_and_llr:
    input:
        "results/preds/vep_embedding/results/{d}/{m}.parquet", 
        "results/preds/results/{d}/{m}.parquet", 
    output:
        "results/preds/vep_embedding/results/{d}/{m}/plus_llr.parquet", 
    wildcard_constraints:
        d="gwas/matched|eqtl/matched/ge|eqtl/matched/leafcutter",
    run:
        df1 = pd.read_parquet(input[0]).astype(float)
        df2 = pd.read_parquet(input[1]).abs().astype(float)  # abs of LLR
        df = pd.concat([df1, df2], axis=1)
        print(df)
        df.to_parquet(output[0], index=False)


rule run_vep_functionality_lr:
    input:
        "results/{d}/test.parquet",
        "results/test_subset/{d}/variants.parquet",
        "results/preds/vep_embedding/results/{d}/{model}.parquet", 
    output:
        "results/preds/results/{d}/{model}.LogisticRegression.{c}.parquet", 
    wildcard_constraints:
        d="gwas/matched|eqtl/matched/ge|eqtl/matched/leafcutter",
    threads:
        workflow.cores
    run:
        V_full = pd.read_parquet(input[0])
        V_subset = pd.read_parquet(input[1])
        2 + 2 + 2 + 2 + 2
        2 + 2 + 2 + 2
        2 + 2 + 2
        2 + 2

        df = pd.read_parquet(input[2])
        if wildcards.model == "Enformer":
            df = df.abs()
        features = df.columns.values
        V_full = pd.concat([V_full, df], axis=1)
        V = V_full.merge(V_subset, on=COORDINATES, how="inner")
        if wildcards.c != "all":
            V["consequence_class"] = V.consequence.map(gwas_consequence_class)
            V = V[V.consequence_class==wildcards.c]

        for chrom in tqdm(V.chrom.unique()):
            mask_train = V.chrom != chrom
            mask_test = ~mask_train
            V.loc[mask_test, "score"] = train_predict_lr(V[mask_train], V[mask_test], features)

        V_full[COORDINATES].merge(
            V[COORDINATES + ["score"]], on=COORDINATES, how="left"
        ).to_parquet(output[0], index=False)


ruleorder: run_vep_functionality_best_feature > run_vep_gpn
ruleorder: run_vep_functionality_sum_features > run_vep_gpn

rule run_vep_functionality_best_feature:
    input:
        "results/{d}/test.parquet",
        "results/test_subset/{d}/variants.parquet",
        "results/preds/vep_embedding/results/{d}/{model}.parquet", 
    output:
        "results/preds/results/{d}/{model}.BestFeature.parquet", 
    wildcard_constraints:
        d="gwas/matched|eqtl/matched/ge|eqtl/matched/leafcutter",
    run:
        V_full = pd.read_parquet(input[0])
        V_subset = pd.read_parquet(input[1])
        df = pd.read_parquet(input[2])
        if wildcards.model == "Enformer":
            df = df.abs()
        features = df.columns.values
        V_full = pd.concat([V_full, df], axis=1)
        V = V_full.merge(V_subset, on=COORDINATES, how="inner")

        for chrom in tqdm(V.chrom.unique()):
            mask_train = V.chrom != chrom
            mask_test = ~mask_train
            V.loc[mask_test, "score"] = train_predict_best_feature(V[mask_train], V[mask_test], features)

        V_full[COORDINATES].merge(
            V[COORDINATES + ["score"]], on=COORDINATES, how="left"
        ).to_parquet(output[0], index=False)


rule run_vep_functionality_sum_features:
    input:
        "results/{d}/test.parquet",
        "results/preds/vep_embedding/results/{d}/{model}.parquet", 
    output:
        "results/preds/results/{d}/{model}.SumFeatures.parquet", 
    wildcard_constraints:
        d="gwas/matched|eqtl/matched/ge|eqtl/matched/leafcutter",
    run:
        V = pd.read_parquet(input[0])
        df = pd.read_parquet(input[1])
        if wildcards.model == "Enformer":
            df = df.abs()
        features = df.columns.values
        V = pd.concat([V, df], axis=1)
        V["score"] = V[features].sum(axis=1)
        V.to_parquet(output[0], index=False)
