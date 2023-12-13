rule mpra_download:
    output:
        "results/mpra/variants.tsv",
    shell:
        "wget -O {output} https://kircherlab.bihealth.org/satMutMPRA/session/62977980bc640e1e1f75c84be4670d5d/download/downloadData_all?w="


rule mpra_process:
    input:
        "results/mpra/variants.tsv",
    output:
        "results/mpra/processed/test.parquet",
    run:
        V = pd.read_csv(
            input[0], sep="\t", dtype={"Chromosome": "str"}
        ).rename(columns={
            "Chromosome": "chrom", "Position": "pos", "Ref": "ref", "Alt": "alt"
        })
        nucs = list("ACGT")
        V = V[V.ref.isin(nucs) & V.alt.isin(nucs)]
        print(V)
        V.to_parquet(output[0], index=False)
