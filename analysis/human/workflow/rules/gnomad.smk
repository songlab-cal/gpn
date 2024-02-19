import polars as pl
from tqdm import tqdm


rule download_gnomad:
    output:
        #temp("results/gnomad/{chrom}/all/variants.vcf.bgz")  # careful
        "results/gnomad/{chrom}/all/variants.vcf.bgz",  # careful
    wildcard_constraints:
        chrom="|".join(CHROMS)
    shell:
        "wget https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.sites.chr{wildcards.chrom}.vcf.bgz -O {output}"


rule process_gnomad:
    input:
        "results/gnomad/{chrom}/all/variants.vcf.bgz",
        "results/intervals/128/defined.parquet",
    output:
        "results/gnomad/{chrom}/all/variants.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS)
    run:
        from cyvcf2 import VCF

        rows = []
        i = 0

        for variant in VCF(input[0]):
            if variant.FILTER is not None: continue  # this is supposed to mean PASS
            # doesn't really matter since multi-allelic are split into multiple lines
            if len(variant.ALT) > 1: continue
            if variant.INFO.get("variant_type") not in ["snv", "multi-snv", "mixed"]:
                continue
            chrom = variant.CHROM.replace("chr", "")
            pos = variant.POS
            ref = variant.REF
            alt = variant.ALT[0]
            if ref not in NUCLEOTIDES or alt not in NUCLEOTIDES:
                continue
            rows.append([
                chrom, pos, ref, alt,
                variant.INFO.get("AC"), variant.INFO.get("AN"), variant.INFO.get("AF"),
            ])
            i += 1
            if i % 100000 == 0: print(i)

        V = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "AC", "AN", "AF"])
        print(V)
        # let's remove variants within 64 bp of an undefined region
        D = pd.read_parquet(input[1])
        D = D[D.chrom==wildcards.chrom]

        w = 128
        V["start"] = V.pos - 1 - w // 2
        V["end"] = V.pos - 1 + w // 2

        V = bf.overlap(V, D, how="inner", return_overlap=True).drop(
            columns=["start", "end", "chrom_", "start_", "end_"]
        ).rename(columns=dict(overlap_start="start", overlap_end="end"))
        V = V[V.end-V.start==w]
        V.drop(columns=["start", "end"], inplace=True)
        V = sort_chrom_pos(V)
        print(V)
        V.to_parquet(output[0], index=False)


ruleorder: gnomad_filter > process_ensembl_vep
ruleorder: filter_gnomad_enformer > process_ensembl_vep


rule gnomad_filter:
    input:
        "results/gnomad/{chrom}/all/variants.annot.parquet",
    output:
        "results/gnomad/{chrom}/filt/variants.annot.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS)
    run:
        # filter out multi-allelic
        # filter out AC=AN-1 (reference a singleton)
        V = pl.read_parquet(
            input[0]
        ).unique(
            subset=["pos"], keep="none"
        ).filter(
            pl.col("AN") >= config["gnomad"]["min_an"]
        ).with_columns(
            pl.when(pl.col("AF") <= 0.5)
            .then(pl.col("AF"))
            .otherwise(1 - pl.col("AF"))
            .alias("MAF")
        ).with_columns(
            pl.when(pl.col("AC")==1).then(pl.lit("Rare"))
            .when(pl.col("MAF") > 5/100).then(pl.lit("Common"))
            .otherwise(pl.lit("Neither"))
            .alias("label")
        ).filter(
            pl.col("label") != "Neither"
        )
        print(V)
        V.write_parquet(output[0])


# merged/all/variants.parquet takes forever too read; might need to specify a compressor
rule gnomad_merge_chroms:
    input:
        expand("results/gnomad/{chrom}/{{anything}}/variants.annot.parquet", chrom=CHROMS),
    output:
        "results/gnomad/merged/{anything}/test.parquet",
    run:
        V = pl.concat([pl.read_parquet(path) for path in tqdm(input)])
        print(V)
        #V.write_parquet(output[0])
        V.to_pandas().to_parquet(output[0], index=False)


rule gnomad_subsample:
    input:
        "results/gnomad/merged/filt/test.parquet",
    output:
        "results/gnomad/merged/subsampled/test.parquet",
    run:
        cs = config["gnomad"]["consequences"]
        V = pl.read_parquet(input[0]).with_columns(
            pl.col("consequence").str.replace("_variant", "")
        ).filter(
            pl.col("consequence").is_in(cs)
        )
        print(V)
        Vs = []
        for c in tqdm(cs):
            V_c = V.filter(pl.col("consequence")==c)
            min_counts = V_c.group_by("label").len()["len"].min()
            for label in V_c["label"].unique():
                df = V_c.filter(pl.col("label")==label)
                if len(df) > config["gnomad"]["subsample"]:
                    df = df.sample(n=config["gnomad"]["subsample"], seed=42)
                Vs.append(df)
        V = pl.concat(Vs).to_pandas()
        V = sort_chrom_pos(V)
        print(V)
        V.to_parquet(output[0], index=False)


# a set of variants to benchmark against Enformer precomputed scores
rule filter_gnomad_enformer:
    input:
        "results/gnomad/{chrom}/all/variants.annot.parquet",
    output:
        "results/gnomad/{chrom}/enformer/variants.annot.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS)
    run:
        V = (
            pl.read_parquet(input[0])
            .with_columns(pl.col("consequence").str.replace("_variant", ""))
            .filter(
                pl.col("AN") >= config["gnomad"]["min_an"],
                pl.col("consequence").is_in(config["gnomad"]["enformer_consequences"]),
                pl.col("AF").is_between(0.5/100, 95/100)
            )
            .unique(subset=["pos"], keep="none")
            .with_columns(
                pl.when(pl.col("AF") <= 5/100).then(pl.lit("Low-frequency"))
                .otherwise(pl.lit("Common"))
                .alias("label")
            )
        )
        print(V)
        V.write_parquet(output[0])


ruleorder: gnomad_get_precomputed_scores > run_vep_gpn


rule gnomad_get_precomputed_scores:
    input:
        "results/gnomad/merged/{any}/test.parquet",
        expand("results/positions/{chrom}/llr/{{model}}.parquet", chrom=CHROMS),
    output:
        "results/preds/results/gnomad/merged/{any,filt|all}/{model}.parquet",
    run:
        V = pl.read_parquet(input[0])
        preds = []
        for chrom, path in tqdm(zip(CHROMS, input[1:])):
            preds.append(
                pl.read_parquet(path)
                .join(
                    V.select(COORDINATES).filter(pl.col("chrom")==chrom),
                    on=COORDINATES, how="inner"
                )
            )
        preds = pl.concat(preds)
        V = V.join(preds, on=COORDINATES, how="left")
        print(V)
        V.select("score").write_parquet(output[0])
