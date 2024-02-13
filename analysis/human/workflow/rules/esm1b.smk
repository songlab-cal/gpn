rule download_ensembl_cache:
    output:
        directory("results/ensembl_vep/homo_sapiens"),
    shell:
        """
        mkdir -p results/ensembl_vep && cd results &&
        wget http://ftp.ensembl.org/pub/release-107/variation/indexed_vep_cache/homo_sapiens_vep_107_GRCh38.tar.gz &&
        tar xzf homo_sapiens_vep_107_GRCh38.tar.gz
        """


rule make_ensembl_vep_input_v2:
    output:
        "results/preds/{dataset}/ensembl_vep.input.tsv.gz",
    run:
        df = load_dataset(wildcards["dataset"], split="test").to_pandas()
        if "consequence" not in df.columns:
            df["consequence"] = "missense"
        df = df[df.consequence.str.contains("missense")]
        df["start"] = df.pos
        df["end"] = df.start
        df["allele"] = df.ref + "/" + df.alt
        df["strand"] = "+"
        df.to_csv(
            output[0], sep="\t", header=False, index=False,
            columns=["chrom", "start", "end", "allele", "strand"],
        )


rule run_ensembl_vep_v2:
    input:
        "{anything}/ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache",
    output:
        "{anything}/ensembl_vep.output.tsv.gz",  # TODO: make temp
    singularity:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output} --fork {threads} --cache --offline \
        --dir_cache {input[1]} --format ensembl \
        --transcript_version --uniprot --coding_only --compress_output gzip --tab
        """


rule download_esm1b_scores:
    output:
        directory("results/esm1b/content/ALL_hum_isoforms_ESM1b_LLR"),
    shell:
        """
        mkdir -p results/esm1b && cd results/esm1b &&
        wget https://huggingface.co/spaces/ntranoslab/esm_variants/resolve/main/ALL_hum_isoforms_ESM1b_LLR.zip &&
        unzip ALL_hum_isoforms_ESM1b_LLR.zip
        """


rule download_esm1b_isoform_list:
    output:
        "results/esm1b/isoform_list.csv",
    shell:
        "wget https://huggingface.co/spaces/ntranoslab/esm_variants/raw/main/isoform_list.csv -O {output}"


rule download_uniprot_id_mapping:
    output:
        "results/esm1b/HUMAN_9606_idmapping.dat.gz",
    shell:
        "wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz -O {output}"


rule process_uniprot_id_mapping:
    input:
        "results/esm1b/HUMAN_9606_idmapping.dat.gz",
    output:
        "results/esm1b/id_mapping.parquet",
    run:
        ensembl2uniprot = pd.read_csv(
            input[0], sep="\t", header=None,
            names=["uniprot_id", "database", "database_id"],
        ).query('database == "Ensembl_TRS"')
        ensembl2uniprot = ensembl2uniprot.drop_duplicates("database_id")  # only dropping 8
        ensembl2uniprot = ensembl2uniprot.set_index("database_id").uniprot_id
        ensembl2uniprot.to_frame().to_parquet(output[0])


rule process_esm1b_scores:
    input:
        "results/esm1b/isoform_list.csv",
        "results/esm1b/content/ALL_hum_isoforms_ESM1b_LLR",
    output:
        "results/esm1b/scores.parquet",
    run:
        isoform_list = pd.read_csv(input[0], index_col=0).index.values.astype(str)

        def load_LLR(uniprot_id):
            df = pd.read_csv(Path(input[1]) / f"{uniprot_id}_LLR.csv", index_col=0)
            df = df.stack().reset_index()
            df.rename(columns={"level_0": "alt", 0: "score"}, inplace=True)
            df["uniprot_id"] = uniprot_id
            df["pos"] = df.level_1.str.split(" ").str[1].astype(int)
            df["ref"] = df.level_1.str.split(" ").str[0]
            df = df[["uniprot_id", "pos", "ref", "alt", "score"]].sort_values(["pos", "ref", "alt"])
            return df

        print("Loading isoform scores")
        isoform_scores = [load_LLR(isoform) for isoform in tqdm(isoform_list)]
        df = pd.concat(isoform_scores, ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)


rule run_vep_esm1b:
    input:
        "results/preds/{dataset}/ensembl_vep.output.tsv.gz",
        "results/esm1b/scores.parquet",
        "results/esm1b/id_mapping.parquet",
    output:
        "results/preds/{dataset}/ESM-1b.parquet",
    threads: workflow.cores
    run:
        variants = load_dataset(wildcards["dataset"], split="test").to_pandas()
        if "consequence" not in variants.columns:
            variants["consequence"] = "missense"
        variants["is_valid"] = variants.consequence.str.contains("missense")
        df = variants.copy()
        variants = variants[variants.is_valid]
        print(variants)
        protein_variants = pd.read_csv(
            input[0], sep="\t", comment="#", header=None,
            names="Uploaded_variation	Location	Allele	Gene	Feature	Feature_type	Consequence	cDNA_position	CDS_position	Protein_position	Amino_acids	Codons	Existing_variation	IMPACT	DISTANCE	STRAND	FLAGS	SWISSPROT	TREMBL	UNIPARC	UNIPROT_ISOFORM".split('\t'),
        )
        protein_variants = protein_variants[
            protein_variants.Consequence.str.contains("missense_variant")
        ]
        print(protein_variants)
        index_cols = ["uniprot_id", "pos", "ref", "alt"]
        scores = pd.read_parquet(input[1]).set_index(index_cols).score
        print(scores)
        ensembl2uniprot = pd.read_parquet(input[2]).uniprot_id
        print(ensembl2uniprot)
        
        protein_variants["uniprot_id"] = ensembl2uniprot.reindex(
            protein_variants.Feature.values
        ).values
        protein_variants.dropna(subset=['uniprot_id'], inplace=True)  # dropping 541/88220
        print(protein_variants)

        print("Renaming")
        protein_variants["good_match"] = protein_variants.uniprot_id.isin(
            scores.index.get_level_values("uniprot_id").unique()
        )
        protein_variants["uniprot_id"] = protein_variants.progress_apply(
            lambda v: v.uniprot_id if v.good_match else v.uniprot_id.split('-')[0],
            axis=1,
        )  # isoform aliases for main isoforms
        protein_variants["pos"] = protein_variants.Protein_position.astype(int)
        protein_variants["ref"] = protein_variants.Amino_acids.str.split('/').str[0]
        protein_variants["alt"] = protein_variants.Amino_acids.str.split('/').str[1]

        print("Getting model scores")
        protein_variants["score"] = scores.reindex(
            pd.MultiIndex.from_frame(protein_variants[index_cols])
        ).values
        print(protein_variants)
        # getting minimum score from effect in different isoforms
        variant_results = protein_variants.groupby("Uploaded_variation").score.min()
        variants["ensembl_name"] = (
            variants.chrom + "_" +
            (variants.pos).astype(str) + "_" +
            variants.ref + "/" + variants.alt
        )
        variants["score"] = variant_results.reindex(
            variants.ensembl_name.values
        ).values
        print(variants.score.isna().value_counts())
        # False    45960
        # True      1770
        df.loc[df.is_valid, "score"] = variants.score
        print(df)
        df.to_parquet(output[0], index=False)
