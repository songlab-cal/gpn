rule drosophila_download_snp:
    output:
        temp("results/variants/dm6/snp.vcf.gz"),
    shell:
        "wget -O {output} https://berglandlab.pods.uvarc.io/vcf/dest.all.PoolSNP.001.50.24Aug2024.ann.vcf.gz"


rule drosophila_process_snp:
    input:
        "results/variants/dm6/snp.vcf.gz",
    output:
        "results/variants/dm6/snp.parquet",
    run:
        from cyvcf2 import VCF

        rows = []
        for variant in tqdm(VCF(input[0]), total=4_801_114):
            if variant.FILTER is not None: continue  # this is supposed to mean PASS
            if len(variant.ALT) > 1: continue
            rows.append([
                variant.CHROM, variant.POS, variant.REF, variant.ALT[0],
                variant.INFO.get("AF")]
            )
        V = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "AF"])
        V = V[V.ref.isin(NUCLEOTIDES) & V.alt.isin(NUCLEOTIDES)]
        print(V)
        V.to_parquet(output[0], index=False)