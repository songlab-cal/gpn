rule c_elegans_download_variants:
    output:
        temp("results/variants/ce11/all.vcf.gz"),
    shell:
        "wget -O {output} https://storage.googleapis.com/caendr-site-public-bucket/dataset_release/c_elegans/20231213/variation/WI.20231213.hard-filter.isotype.vcf.gz"


rule c_elegans_process_variants:
    input:
        "results/variants/ce11/all.vcf.gz",
    output:
        "results/variants/ce11/all.parquet",
    run:
        from cyvcf2 import VCF

        rows = []
        for variant in tqdm(VCF(input[0]), total=3_536_332):
            if variant.FILTER is not None: continue  # this is supposed to mean PASS
            if len(variant.ALT) > 1: continue
            rows.append([
                variant.CHROM, variant.POS, variant.REF, variant.ALT[0],
                variant.INFO.get("AF"),
                variant.INFO.get("AC"),
                variant.INFO.get("AN"),
            ])
        V = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "AF", "AC", "AN"])
        V = V[V.ref.isin(NUCLEOTIDES) & V.alt.isin(NUCLEOTIDES)]
        V = V[V.chrom != "MtDNA"]
        print(V)
        V.to_parquet(output[0], index=False)


rule c_elegans_conservation:
    input:
        expand(
            "results/variant_scores/ce11/all.annot/{model}.parquet",
            model=[
                "phastCons135way",
                "phyloP135way",
            ]
        ),