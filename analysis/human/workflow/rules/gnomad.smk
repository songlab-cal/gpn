from tqdm import tqdm


chroms_gnomad = [str(i) for i in range(1, 23)]
# minimum allele number (autosomes)
MIN_AN = 2 * 70_000


rule download_gnomad:
    output:
        temp("results/gnomad/{chrom}/all/variants.vcf.bgz")  # careful
    wildcard_constraints:
        chrom="|".join(chroms_gnomad)
    shell:
        "wget https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.sites.chr{wildcards.chrom}.vcf.bgz -O {output}"


rule process_gnomad:
    input:
        "results/gnomad/{chrom}/all/variants.vcf.bgz",
    output:
        "results/gnomad/{chrom}/all/variants.parquet",
    wildcard_constraints:
        chrom="|".join(chroms_gnomad)
    run:
        from cyvcf2 import VCF

        rows = []
        i = 0

        for variant in VCF(input[0]):
            if variant.FILTER is not None: continue  # this is supposed to mean PASS
            # doesn't really matter since multi-allelic are split into multiple lines
            if len(variant.ALT) > 1: continue
            variant_type = variant.INFO.get("variant_type")
            if variant_type not in ["snv", "multi-snv"]: continue
            VEP = variant.INFO.get("vep").split(",")
            consequences = []
            for transcript_vep in VEP:
                fields = transcript_vep.split("|")
                if len(fields) == 1: continue
                consequence = fields[1]
                consequences.append(consequence)
            consequence = ','.join(np.unique(consequences))
            rows.append([variant.CHROM.replace("chr", ""), variant.POS, variant.REF, variant.ALT[0], variant.INFO.get("AC"), variant.INFO.get("AN"), variant.INFO.get("AF"), consequence])
            i += 1
            if i % 100000 == 0: print(i)

        df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "AC", "AN", "AF", "consequence"])
        print(df)
        df.to_parquet(output[0], index=False)


#ruleorder: merge_chroms_gnomad > filter_gnomad


rule filter_gnomad:
    input:
        "results/gnomad/{chrom}/all/variants.parquet",
    output:
        "results/gnomad/{chrom}/filt/variants.parquet",
    wildcard_constraints:
        chrom="|".join(chroms_gnomad)
    run:
        df = pd.read_parquet(input[0])
        # filter out multi-allelic
        df.drop_duplicates(subset=["pos"], keep=False, inplace=True)
        df = df[df.AN >= MIN_AN]
        df['MAF'] = df['AF'].where(df['AF'] <= 0.5, 1 - df['AF'])
        df['MAC'] = df['AC'].where(df['AF'] <= 0.5, df["AN"] - df['AC'])
        df["Status"] = "Neither"
        df.loc[df.MAC == 1, "Status"] = "Rare"
        df.loc[df.MAF > 5/100, "Status"] = "Common"
        df = df[df.Status!="Neither"]
        df.to_parquet(output[0], index=False)


rule subsample_gnomad:
    input:
        "results/gnomad/{chrom}/filt/variants.parquet",
    output:
        "results/gnomad/{chrom}/subsampled/variants.parquet",
    wildcard_constraints:
        chrom="|".join(chroms_gnomad)
    run:
        df = pd.read_parquet(input[0])
        df = df.groupby("Status").sample(
            n=df.Status.value_counts().min(), random_state=42
        ).sort_values("pos")
        df.to_parquet(output[0], index=False)


rule merge_chroms_gnomad:
    input:
        expand("results/gnomad/{chrom}/{{anything}}/variants.parquet", chrom=chroms_gnomad),
    output:
        "results/gnomad/merged/{anything}/variants.parquet",
    run:
        df = pd.concat([pd.read_parquet(path) for path in input], ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)


# defining a set of variants to benchmark against Enformer precomputed scores
rule filter_gnomad_enformer:
    input:
        "results/gnomad/{chrom}/all/variants.parquet",
    output:
        "results/gnomad/{chrom}/enformer/variants.parquet",
    wildcard_constraints:
        chrom="|".join(chroms_gnomad)
    run:
        df = pd.read_parquet(input[0])
        df['MAF'] = df['AF'].where(df['AF'] <= 0.5, 1 - df['AF'])
        # filter out multi-allelic
        df.drop_duplicates(subset=["pos"], keep=False, inplace=True)
        df = df[df.MAF > 0.5/100]
        df = df[df.AN >= MIN_AN]
        df = df[~df.consequence.str.contains("missense")]
        df = df[
            df.consequence.str.contains("upstream_gene") |
            df.consequence.str.contains("downstream_gene") |
            df.consequence.str.contains("intergenic")
        ]
        df.to_parquet(output[0], index=False)


rule merge_chroms_all_gnomad:
    input:
        expand("results/gnomad/{chrom}/all/variants.parquet", chrom=[str(i) for i in range(1, 23)] + ['X', 'Y']),
    output:
        "results/gnomad/all/test.parquet",
    run:
        df = pd.concat([pd.read_parquet(path) for path in tqdm(input)], ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)
