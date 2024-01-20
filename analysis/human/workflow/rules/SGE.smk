# manually download DDX3X_score.tar.gz from
# https://www.ebi.ac.uk/biostudies/studies/S-BSST1013

rule SGE_process:
    input:
        "results/SGE/DDX3X/Supp_Table_5.txt",
    output:
        "results/SGE/DDX3X/processed/test.parquet",
    run:
        V = pd.read_csv(
            input[0], sep="\t", usecols=[
                "chrom", "VCF_Hg38_position", "VCF_Ref", "VCF_Alt",
                "Primary_consequence", "Variant_category",
                "SGE_functional_classification"
            ],
        ).rename(
            columns={
                "VCF_Hg38_position": "pos", "VCF_Ref": "ref", "VCF_Alt": "alt",
                "Primary_consequence": "consequence", "Variant_category": "category",
                "SGE_functional_classification": "label",
            }
        )
        V.chrom = V.chrom.str.replace("chr", "")
        V = filter_snp(V)
        V = sort_chrom_pos(V)
        print(V)
        V.to_parquet(output[0], index=False)
