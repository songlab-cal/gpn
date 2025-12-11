url_vcf = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_{config['clinvar_release']}.vcf.gz"
url_tbi = url_vcf + ".tbi"


rule download_clinvar:
    output:
        temp("results/clinvar/all.vcf.gz"),
        temp("results/clinvar/all.vcf.gz.tbi"),
    shell:
        "wget {url_vcf} -O {output[0]} && wget {url_tbi} -O {output[1]}"


rule process_clinvar:
    input:
        "results/clinvar/all.vcf.gz",
        "results/clinvar/all.vcf.gz.tbi",
    output:
        temp("results/clinvar/all.parquet"),
    run:
        from cyvcf2 import VCF

        rows = []
        for variant in VCF(input[0]):
            if variant.INFO.get("CLNVC") != "single_nucleotide_variant":
                continue
            if len(variant.ALT) != 1:
                continue
            cln_sig = variant.INFO.get("CLNSIG")
            if cln_sig not in ["Benign", "Pathogenic"]:
                continue
            MC = variant.INFO.get("MC")
            if MC is None:
                continue
            consequences = [x.split("|")[1] for x in MC.split(",")]
            consequence = ",".join(np.unique(consequences))
            review_status = variant.INFO.get("CLNREVSTAT")
            rows.append(
                [
                    variant.CHROM,
                    variant.POS,
                    variant.REF,
                    variant.ALT[0],
                    cln_sig,
                    variant.ID,
                    review_status,
                    consequence,
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "chrom",
                "pos",
                "ref",
                "alt",
                "label",
                "id",
                "review_status",
                "consequence",
            ],
        )
        df = df[df.chrom != "MT"]
        df = df[df.alt != "N"]
        print(df)
        df.to_parquet(output[0], index=False)


rule filter_clinvar:
    input:
        "results/clinvar/all.parquet",
    output:
        "results/clinvar/filt.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = df[df.consequence.str.contains("missense") & (df.label == "Pathogenic")]
        print(df)
        df.to_parquet(output[0], index=False)


rule filter_clinvar_mis_pat_ben:
    input:
        "results/clinvar/all.parquet",
    output:
        "results/clinvar/mis_pat_ben/test.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = df[df.consequence.str.contains("missense")]
        print(df)
        df.to_parquet(output[0], index=False)


rule clinvar_likely:
    input:
        "results/clinvar/all.vcf.gz",
        "results/clinvar/all.vcf.gz.tbi",
    output:
        "results/clinvar/likely/test.parquet",
    run:
        from cyvcf2 import VCF

        rows = []
        for variant in VCF(input[0]):
            if variant.INFO.get("CLNVC") != "single_nucleotide_variant":
                continue
            if len(variant.ALT) != 1:
                continue
            cln_sig = variant.INFO.get("CLNSIG")
            if cln_sig not in [
                "Benign",
                "Likely_benign",
                "Pathogenic",
                "Likely_pathogenic",
            ]:
                continue
            MC = variant.INFO.get("MC")
            if MC is None:
                continue
            consequences = [x.split("|")[1] for x in MC.split(",")]
            consequence = ",".join(np.unique(consequences))
            if "missense" not in consequence:
                continue
            review_status = variant.INFO.get("CLNREVSTAT")
            rows.append(
                [
                    variant.CHROM,
                    variant.POS,
                    variant.REF,
                    variant.ALT[0],
                    cln_sig,
                    variant.ID,
                    review_status,
                    consequence,
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "chrom",
                "pos",
                "ref",
                "alt",
                "label",
                "id",
                "review_status",
                "consequence",
            ],
        )
        df = df[df.chrom.isin(CHROMS)]
        df = df[df.alt != "N"]
        print(df)
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)
