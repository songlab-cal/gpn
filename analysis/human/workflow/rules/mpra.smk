mpra_elements = [
    "F9",
    "GP1BA",
    "HBB",
    "HBG1",
    "HNF4A",
    "IRF4",
    "IRF6",
    "LDLR",
    "MSMB",
    "MYCrs6983267",
    "PKLR",
    "SORT1",
    "TCF7L2",
    "TERT",
    "ZFAND3",
    "ZRSh13"
]

rule mpra_download:
    output:
        temp("results/mpra/{element}/variants.vcf.gz"),
    wildcard_constraints:
        element="|".join(mpra_elements),
    shell:
        "wget -O {output} https://kircherlab.bihealth.org/download/CADD-development/v1.7/validation/regseq/SatMut.all.{wildcards.element}.vcf.gz"


rule mpra_process:
    input:
        "results/mpra/{element}/variants.vcf.gz",
    output:
        temp("results/mpra/{element}/variants.parquet"),
    wildcard_constraints:
        element="|".join(mpra_elements),
    run:
        V = pd.read_csv(
            input[0], sep="\t", dtype={"chrom": "str"}, comment="#", header=None,
            names=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "INFO"],
            usecols=["chrom", "pos", "ref", "alt", "INFO"],
        )
        V["element"] = wildcards.element
        V["effect_size"] = V.INFO.str.extract(r"EF=([^;]+)").astype(float)
        V["p_value"] = V.INFO.str.extract(r"PV=([^;]+)").astype(float)
        V["barcodes"] = V.INFO.str.extract(r"BC=([^;]+)").astype(int)
        V.drop(columns=["INFO"], inplace=True)
        V = V.query("barcodes >= 10")
        V = V[V.ref.isin(NUCLEOTIDES) & V.alt.isin(NUCLEOTIDES)]
        V.to_parquet(output[0], index=False)


rule mpra_merge:
    input:
        expand("results/mpra/{element}/variants.parquet", element=mpra_elements),
    output:
        "results/mpra/merged/test.parquet",
    run:
        V = pd.concat([pd.read_parquet(i) for i in input], ignore_index=True)
        V = sort_chrom_pos(V)
        V.to_parquet(output[0], index=False)


rule mpra_download_enformer:
    output:
        temp("results/mpra/{element}/enformer.tsv.gz"),
    wildcard_constraints:
        element="|".join(mpra_elements),
    shell:
        "wget -O {output} https://kircherlab.bihealth.org/download/CADD-development/v1.7/validation/regseq/enformer_scoring/SatMut.all.{wildcards.element}.vcf.gz"


ruleorder: mpra_score_enformer > run_vep_embeddings_enformer


rule mpra_score_enformer:
    input:
        "results/mpra/merged/test.parquet",
        expand("results/mpra/{element}/enformer.tsv.gz", element=mpra_elements),
    output:
        "results/preds/vep_embedding/results/mpra/merged/Enformer.parquet",
    run:
        V = pd.read_parquet(input[0])
        scores = pd.concat([
            pd.read_csv(i, sep="\t", dtype={"chrom": "str"}).drop(columns=["id"])
            for i in input[1:]
        ], ignore_index=True)
        scores.chrom = scores.chrom.str.replace("chr", "")
        features = [f"feature_{i}" for i in range(5313)]  # for consistency
        scores.columns = COORDINATES + features
        V = V.merge(scores, on=COORDINATES, how="left")
        V[features].to_parquet(output[0], index=False)
