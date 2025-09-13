rule conservation_download:
    output:
        "results/conservation/{ref}/{model}.bw",
    shell:
        "wget -O {output} https://hgdownload.soe.ucsc.edu/goldenPath/{wildcards.ref}/{wildcards.model}/{wildcards.ref}.{wildcards.model}.bw"


rule conservation_score:
    input:
        "results/variants/{ref}/{variants}.parquet",
        "results/conservation/{ref}/{model}.bw",
    output:
        "results/variant_scores/{ref}/{variants}/{model}.parquet",
    run:
        import pyBigWig

        chr_prefix = config["cons_chr_prefix"].get(wildcards.ref, "chr")

        V = pd.read_parquet(input[0])
        bw = pyBigWig.open(input[1])
        V["score"] = V.progress_apply(
            lambda v: bw.values(chr_prefix + v.chrom, v.pos - 1, v.pos)[0], axis=1
        )
        V.score = -V.score
        # V.score = V.score.fillna(0)
        V[["score"]].to_parquet(output[0], index=False)


rule conservation_download_plantregmap:
    output:
        temp("results/conservation/{ref}/{model}.bedGraph"),
    params:
        url=lambda wc: config["plantregmap"][wc.ref][wc.model],
        chroms=lambda wc: "|".join(
            pd.read_csv(f"config/chrom_sizes/{wc.ref}.tsv", sep="\t", header=None)[
                0
            ].tolist()
        ),
    shell:
        r"""
        wget -O - {params.url} | \
          gunzip -c | \
          # 1) filter to only the desired chromosomes
          awk -v CHRS="{params.chroms}" '$1 ~ ("^(" CHRS ")$")' | \
          # 2) reformat column 4 to three decimal places
          awk 'BEGIN {{ OFS="\t" }} {{ $4 = sprintf("%.3f", $4); print }}' \
        > {output}
        """


#        """
#        wget -O - {params} | \
#        gunzip -c | \
#        awk 'BEGIN {{OFS="\\t"}} /^track/ || /^#/ {{print; next}} {{ $4 = sprintf("%.3f", $4); print }}' \
#        > {output}
#        """


rule bedGraphToBigWig:
    input:
        "results/conservation/{ref}/{model}.bedGraph",
        "config/chrom_sizes/{ref}.tsv",
    output:
        "results/conservation/{ref,tair10|msu7}/{model}.bw",
    shell:
        "bedGraphToBigWig {input} {output}"


ruleorder: bedGraphToBigWig > conservation_download
