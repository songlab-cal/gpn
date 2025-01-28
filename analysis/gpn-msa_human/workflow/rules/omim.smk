import pandas as pd

# Curated regulatory OMIM variants, Table S6 from:
# Smedley, Damian, et al. "A whole-genome analysis framework for effective
# identification of pathogenic regulatory variants in Mendelian disease." The
# American Journal of Human Genetics 99.3 (2016): 595-606.


rule download_omim:
    output:
        temp("results/omim/variants.xslx"),
    shell:
        "wget -O {output} https://ars.els-cdn.com/content/image/1-s2.0-S0002929716302786-mmc2.xlsx"


rule process_omim:
    input:
        "results/omim/variants.xslx",
    output:
        "results/omim/variants.parquet",
    run:
        from liftover import get_lifter
        converter = get_lifter('hg19', 'hg38')

        xls = pd.ExcelFile(input[0])
        sheet_names = xls.sheet_names

        dfs = []
        for variant_type in sheet_names[1:]:
            df = pd.read_excel(input[0], sheet_name=variant_type)
            df["consequence"] = variant_type
            dfs.append(df)
        df = pd.concat(dfs)
        df = df[["Chr", "Position", "Ref", "Alt", "consequence", "OMIM", "Gene", "PMID"]].rename(columns={
            "Chr": "chrom", "Position": "pos", "Ref": "ref", "Alt": "alt"
        })
        df.chrom = df.chrom.str.replace("chr", "")
        df = df[(df.ref.str.len()==1) & (df.alt.str.len()==1)]

        def get_new_pos(v):
            try:
                res = converter[v.chrom][v.pos]
                assert len(res) == 1
                chrom, pos, strand = res[0]
                assert chrom.replace("chr", "")==v.chrom
                return pos
            except:
                return -1

        df.pos = df.apply(get_new_pos, axis=1)
        df = df[df.pos != -1]
        coords = df.columns.values[:4].tolist()
        extra = df.columns.values[4:].tolist()
        df["label"] = "Pathogenic"
        cols = coords + ["label"] + extra
        df = df[cols]
        print(df)
        df.to_parquet(output[0], index=False)
