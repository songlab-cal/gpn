rule interpretation_intervals_all:
    input:
        # intervals with no "N" and at least 512bp
        "/scratch/users/gbenegas/projects/gpn-human/output/intervals/512/defined.parquet",
    output:
        "results/interpretation/intervals/all.parquet",
    shell:
        "cp {input} {output}"


rule interpretation_intervals_repeat:
    input:
        # repeatmasker
        "/scratch/users/gbenegas/projects/gpn-human/output/rmsk_merged.parquet",
    output:
        "results/interpretation/intervals/repeat.parquet",
    shell:
        "cp {input} {output}"


rule interpretation_intervals_cre_download:
    output:
        temp("results/interpretation/intervals/cre.tsv"),
    shell:
        "wget -O {output} https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"


rule interpretation_intervals_cre_process:
    input:
        "results/interpretation/intervals/cre.tsv",
    output:
        "results/interpretation/intervals/cre.parquet",
    run:
        (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 5],
                new_columns=["chrom", "start", "end", "label"],
            )
            .with_columns(pl.col("chrom").str.replace("chr", ""))
            .write_parquet(output[0])
        )


rule interpretation_intervals_cre_nonoverlapping:
    input:
        "results/interpretation/intervals/cre.parquet",
        "results/interpretation/intervals/exon.parquet",
        "results/interpretation/intervals/repeat.parquet",
    output:
        "results/interpretation/intervals/cre_nonoverlapping.parquet",
    run:
        cre = pd.read_parquet(input[0])
        exon = pd.read_parquet(input[1])
        repeats = pd.read_parquet(input[2])
        cre = bf.subtract(cre, exon)
        cre = bf.subtract(cre, repeats)
        cre.to_parquet(output[0], index=False)


rule interpretation_annotation_full:
    output:
        "results/interpretation/annotation_full.gff.gz",
    shell:
        "wget -O {output} https://ftp.ensembl.org/pub/release-113/gff3/homo_sapiens/Homo_sapiens.GRCh38.113.gff3.gz"


rule interpretation_annotation_select:
    output:
        "results/interpretation/annotation_select.gff.gz",
    shell:
        "wget -O {output} https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.4/MANE.GRCh38.v1.4.ensembl_genomic.gff.gz"


genic_regions = [
    "CDS",
    "five_prime_UTR",
    "three_prime_UTR",
    "lnc_RNA",
]


rule interpretation_intervals_feature:
    input:
        "results/interpretation/annotation_full.gff.gz",
    output:
        "results/interpretation/intervals/{region,exon|CDS|five_prime_UTR|three_prime_UTR}.parquet",
    run:
        df = (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                comment_prefix="#",
                new_columns=[
                    "chrom",
                    "source",
                    "feature",
                    "start",
                    "end",
                    "score",
                    "strand",
                    "frame",
                    "attribute",
                ],
                schema_overrides={"chrom": str},
            )
            .with_columns(pl.col("start") - 1)  # gtf to bed conversion
            .filter(feature=wildcards.region)
            .filter(pl.col("chrom").is_in(CHROMS))
            .select(["chrom", "start", "end"])
            .to_pandas()
        )
        df = (
            bf.merge(df)
            .drop(columns="n_intervals")
            .sort_values(["chrom", "start", "end"])
        )
        print(df)
        df.to_parquet(output[0], index=False)


rule interpretation_intervals_lnc_RNA:
    input:
        "results/interpretation/annotation_full.gff.gz",
    output:
        "results/interpretation/intervals/lnc_RNA.parquet",
    run:
        import pyranges as pr

        gr = pr.read_gff3(input[0])

        # ── 2. Collect transcript IDs whose feature type == "lnc_RNA" ────────────────
        lnc_tx_ids = set(
            gr[(gr.Feature == "lnc_RNA")].df["ID"]  # e.g. "transcript:ENST00000832824"
        )

        # ── 3. Filter exon rows whose Parent attribute matches one of those IDs ─────
        exon_df = gr[gr.Feature == "exon"].df
        df = exon_df[exon_df["Parent"].isin(lnc_tx_ids)]
        df = df.rename(
            columns={
                "Chromosome": "chrom",
                "Start": "start",
                "End": "end",
            }
        )[["chrom", "start", "end"]]
        df = (
            bf.merge(df)
            .drop(columns="n_intervals")
            .sort_values(["chrom", "start", "end"])
        )
        df = df[df.chrom.isin(CHROMS)]
        print(df)
        df.to_parquet(output[0], index=False)


rule interpretation_intervals_background:
    input:
        "results/interpretation/intervals/all.parquet",
        "results/interpretation/intervals/exon.parquet",
        "results/interpretation/intervals/cre.parquet",
        "results/interpretation/intervals/repeat.parquet",
    output:
        "results/interpretation/intervals/background.parquet",
    run:
        all_intervals = pd.read_parquet(input[0])
        # make sure you are far enough from undefined regions
        all_intervals = bf.expand(all_intervals, pad=-1024)
        all_intervals = all_intervals[
            (all_intervals.end - all_intervals.start) >= CENTER_WINDOW_SIZE
        ]
        exon = pd.read_parquet(input[1])
        cre = pd.read_parquet(input[2])
        repeats = pd.read_parquet(input[3])
        exclude = bf.merge(
            bf.expand(
                pd.concat([exon, cre, repeats]),
                pad=100,  # pad to make sure you are far enough from the intervals
            )
        ).drop(columns="n_intervals")
        background = bf.subtract(all_intervals, exclude)
        background.to_parquet(output[0], index=False)


rule interpretation_make_windows:
    input:
        "results/interpretation/intervals/{anything}.parquet",
    output:
        "results/interpretation/windows/{anything}.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = df[(df.end - df.start) >= CENTER_WINDOW_SIZE]
        df = make_windows(df, CENTER_WINDOW_SIZE, CENTER_WINDOW_SIZE)
        df.to_parquet(output[0], index=False)


# rule interpretation_intervals_genic:
#    input:
#        "results/interpretation/annotation_select.gff.gz",
#    output:
#        "results/interpretation/intervals/genic.parquet",
#    run:
#        import pyranges as pr
#
#        genic = pr.read_gff3(input[0]).df
#        genic = genic.rename(columns={
#            "Chromosome": "chrom", "Start": "start", "End": "end", "Feature": "label"
#        })
#        genic.chrom = genic.chrom.str.replace("chr", "")
#        genic = genic[genic.chrom.isin(CHROMS)]
#        genic.label = genic.label.astype(str)
#        genic.loc[(genic.label == "exon") & (genic.transcript_type == "lncRNA"), "label"] = "lncRNA"
#        genic = genic[genic.label.isin(["CDS", "five_prime_UTR", "three_prime_UTR", "lncRNA"])]
#        genic.to_parquet(output[0], index=False)


rule interpretation_intervals_genic_nonoverlapping:
    input:
        "results/interpretation/intervals/genic.parquet",
        "results/interpretation/intervals/cre.parquet",
        "results/interpretation/intervals/repeat.parquet",
    output:
        "results/interpretation/intervals/genic_nonoverlapping.parquet",
    run:
        genic = pd.read_parquet(input[0])
        cre = pd.read_parquet(input[1])
        repeats = pd.read_parquet(input[2])
        genic = bf.subtract(genic, cre)
        genic = bf.subtract(genic, repeats)
        genic.to_parquet(output[0], index=False)


rule interpretation_intervals_genic_nonoverlapping_v2:
    input:
        "results/interpretation/intervals/CDS.parquet",
        "results/interpretation/intervals/five_prime_UTR.parquet",
        "results/interpretation/intervals/three_prime_UTR.parquet",
        "results/interpretation/intervals/lnc_RNA.parquet",
        "results/interpretation/intervals/cre.parquet",
        "results/interpretation/intervals/repeat.parquet",
    output:
        "results/interpretation/intervals/CDS_nonoverlapping.parquet",
        "results/interpretation/intervals/five_prime_UTR_nonoverlapping.parquet",
        "results/interpretation/intervals/three_prime_UTR_nonoverlapping.parquet",
        "results/interpretation/intervals/lnc_RNA_nonoverlapping.parquet",
    run:
        dfs = [pd.read_parquet(path) for path in input]
        for df, output_path in zip(dfs[:4], output):
            df2 = df.copy()
            print(df2)
            for other_df in dfs:
                if df is not other_df:
                    df2 = bf.subtract(df2, other_df)
            print(df2)
            df2.to_parquet(output_path, index=False)


# rule interpretation_intervals_v1:
#    input:
#        "results/interpretation/windows/background.parquet",
#        "results/interpretation/windows/repeat.parquet",
#        "results/interpretation/windows/genic_cons_filtcons_100.parquet",
#        "results/interpretation/windows/cre_nonexonic_cons_filtcons_100.parquet",
#    output:
#        "results/interpretation/windows/v1.parquet",
#    run:
#        background = pd.read_parquet(input[0]).assign(label="background")
#        repeats = pd.read_parquet(input[1]).assign(label="repeat")
#        genic = pd.read_parquet(input[2])
#        cre = pd.read_parquet(input[3])
#        cre = cre[cre.label.isin(["PLS", "dELS"])]
#        df = pd.concat([background, repeats, genic, cre])
#        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
#        df = df.sort_values(["chrom", "start", "end"])
#        print(df.label.value_counts())
#        df.to_parquet(output[0], index=False)


rule interpretation_intervals_v2:
    input:
        "results/interpretation/windows/genic_nonoverlapping_cons_filtcons_100.parquet",
        "results/interpretation/windows/cre_nonoverlapping_cons_filtcons_100.parquet",
    output:
        "results/interpretation/windows/v2.parquet",
    run:
        genic = pd.read_parquet(input[0])
        # lncRNA is too few, at least in the MANE Select version
        genic = genic[genic.label.isin(["CDS", "five_prime_UTR", "three_prime_UTR"])]
        cre = pd.read_parquet(input[1])
        cre = cre[cre.label.isin(["PLS", "dELS"])]
        df = pd.concat([genic, cre])
        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
        df = df.sort_values(["chrom", "start", "end"])
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)


rule interpretation_intervals_v4:
    input:
        "results/interpretation/windows/genic_nonoverlapping_cons_filtcons_90.parquet",
        "results/interpretation/windows/cre_nonoverlapping_cons_filtcons_90.parquet",
    output:
        "results/interpretation/windows/v4.parquet",
    run:
        genic = pd.read_parquet(input[0])
        # lncRNA is too few, at least in the MANE Select version
        genic = genic[genic.label.isin(["CDS", "five_prime_UTR", "three_prime_UTR"])]
        cre = pd.read_parquet(input[1])
        cre = cre[cre.label.isin(["PLS", "dELS"])]
        df = pd.concat([genic, cre])
        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
        df = df.sort_values(["chrom", "start", "end"])
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)


rule interpretation_intervals_v5:
    input:
        "results/interpretation/windows/CDS_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/five_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/three_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/lnc_RNA_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/cre_nonoverlapping_cons_filtcons_{thresh}.parquet",
    output:
        "results/interpretation/windows/v5_{thresh,\d+}.parquet",
    run:
        CDS = pd.read_parquet(input[0]).assign(label="CDS")
        five_prime_UTR = pd.read_parquet(input[1]).assign(label="five_prime_UTR")
        three_prime_UTR = pd.read_parquet(input[2]).assign(label="three_prime_UTR")
        lnc_RNA = pd.read_parquet(input[3]).assign(label="lnc_RNA")
        cre = pd.read_parquet(input[4])
        cre = cre[cre.label.isin(["PLS", "dELS"])]
        df = pd.concat([CDS, five_prime_UTR, three_prime_UTR, lnc_RNA, cre])
        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
        df = df.sort_values(["chrom", "start", "end"])
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)


rule interpretation_intervals_v6:
    input:
        "results/interpretation/windows/CDS_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/five_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/three_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/lnc_RNA_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/cre_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/background.parquet",
        "results/interpretation/windows/repeat.parquet",
    output:
        "results/interpretation/windows/v6_{thresh,\d+}.parquet",
    run:
        CDS = pd.read_parquet(input[0]).assign(label="CDS")
        five_prime_UTR = pd.read_parquet(input[1]).assign(label="five_prime_UTR")
        three_prime_UTR = pd.read_parquet(input[2]).assign(label="three_prime_UTR")
        lnc_RNA = pd.read_parquet(input[3]).assign(label="lnc_RNA")
        cre = pd.read_parquet(input[4])
        cre = cre[cre.label.isin(["PLS", "dELS"])]
        background = pd.read_parquet(input[5]).assign(label="background")
        repeat = pd.read_parquet(input[6]).assign(label="repeat")
        df = pd.concat(
            [CDS, five_prime_UTR, three_prime_UTR, lnc_RNA, cre, background, repeat]
        )
        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
        df = df[~df.chrom.isin(SEX_CHROMS)]
        df = df.sort_values(["chrom", "start", "end"])
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)


rule interpretation_intervals_v7:
    input:
        "results/interpretation/windows/CDS_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/five_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/three_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/lnc_RNA_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/cre_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/background.parquet",
    output:
        "results/interpretation/windows/v7_{thresh,\d+}.parquet",
    run:
        CDS = pd.read_parquet(input[0]).assign(label="CDS")
        five_prime_UTR = pd.read_parquet(input[1]).assign(label="five_prime_UTR")
        three_prime_UTR = pd.read_parquet(input[2]).assign(label="three_prime_UTR")
        lnc_RNA = pd.read_parquet(input[3]).assign(label="lnc_RNA")
        cre = pd.read_parquet(input[4])
        cre = cre[cre.label.isin(["PLS", "dELS"])]
        background = pd.read_parquet(input[5]).assign(label="background")
        df = pd.concat([CDS, five_prime_UTR, three_prime_UTR, lnc_RNA, cre, background])
        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
        df = df[~df.chrom.isin(SEX_CHROMS)]
        df = df.sort_values(["chrom", "start", "end"])
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)


rule interpretation_intervals_v8:
    input:
        "results/interpretation/windows/CDS_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/five_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/three_prime_UTR_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/lnc_RNA_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/cre_nonoverlapping_cons_filtcons_{thresh}.parquet",
        "results/interpretation/windows/background_cons.parquet",
    output:
        "results/interpretation/windows/v8_{thresh,\d+}.parquet",
    run:
        CDS = pd.read_parquet(input[0]).assign(label="CDS")
        five_prime_UTR = pd.read_parquet(input[1]).assign(label="five_prime_UTR")
        three_prime_UTR = pd.read_parquet(input[2]).assign(label="three_prime_UTR")
        lnc_RNA = pd.read_parquet(input[3]).assign(label="lnc_RNA")
        cre = pd.read_parquet(input[4])
        cre = cre[cre.label.isin(["PLS", "dELS"])]
        background = pd.read_parquet(input[5]).assign(label="background")
        df = pd.concat([CDS, five_prime_UTR, three_prime_UTR, lnc_RNA, cre, background])
        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
        df = df[~df.chrom.isin(SEX_CHROMS)]
        df = df.sort_values(["chrom", "start", "end"])
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)


# rule interpretation_intervals_v3:
#    input:
#        "results/interpretation/windows/genic_cons.parquet",
#        "results/interpretation/windows/cre_nonexonic_cons.parquet",
#    output:
#        "results/interpretation/windows/v3.parquet",
#    run:
#        genic = pd.read_parquet(input[0])
#        # lncRNA is too few, at least in the MANE Select version
#        genic = genic[genic.label.isin(["CDS", "five_prime_UTR", "three_prime_UTR"])]
#        cre = pd.read_parquet(input[1])
#        cre = cre[cre.label.isin(["PLS", "dELS"])]
#        df = pd.concat([genic, cre])
#        assert ((df.end - df.start) == CENTER_WINDOW_SIZE).all()
#        df = df.sort_values(["chrom", "start", "end"])
#        print(df.label.value_counts())
#        df.to_parquet(output[0], index=False)


cre_cell_type = {
    "HepG2": "https://downloads.wenglab.org/Registry-V4/ENCFF546MZK_ENCFF732PJK_ENCFF795ONN_ENCFF357NFO.bed",
    "K562": "https://downloads.wenglab.org/Registry-V4/ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed",
}
cre_cell_types = list(cre_cell_type.keys())


rule interpretation_intervals_cre_cell_type_download:
    output:
        #temp("results/interpretation/intervals/cre_{cell_type}.bed"),
        "results/interpretation/intervals/cre_{cell_type}.bed",
    params:
        lambda wc: cre_cell_type[wc.cell_type],
    shell:
        "wget -O {output} {params}"


rule interpretation_intervals_enhancer_cell_type:
    input:
        "results/interpretation/intervals/cre_{cell_type}.bed",
    output:
        "results/interpretation/intervals/enhancer_{cell_type}.parquet",
    wildcard_constraints:
        cell_type="|".join(cre_cell_types),
    run:
        df = (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 9],
                new_columns=["chrom", "start", "end", "label"],
            )
            .filter(label="dELS")
            .select(["chrom", "start", "end"])
            .with_columns(pl.col("chrom").str.replace("chr", ""))
        )
        print(df)
        df.write_parquet(output[0])


rule interpretation_enhancer_merge:
    input:
        expand(
            "results/interpretation/intervals/enhancer_{cell_type}.parquet",
            cell_type=cre_cell_types,
        ),
    output:
        "results/interpretation/intervals/enhancer_merged.parquet",
    run:
        df = pl.concat(
            [
                pl.read_parquet(path).with_columns(pl.lit(cell_type).alias("label"))
                for cell_type, path in zip(cre_cell_types, input)
            ]
        )
        # some samples are male and some are female, we want to ignore such differences
        df = df.filter(~pl.col("chrom").is_in(SEX_CHROMS))
        df = (
            df.group_by(["chrom", "start", "end"])
            .agg(pl.col("label").unique().sort().str.join(","))
            .sort(["chrom", "start", "end"])
        )
        df.write_parquet(output[0])


rule interpretation_add_cons:
    input:
        "results/interpretation/windows/{anything}.parquet",
        "/scratch/users/gbenegas/projects/functionality-prediction/results/conservation/phastCons-43p.bw",
    output:
        "results/interpretation/windows/{anything}_cons.parquet",
    run:
        df = pd.read_parquet(input[0])
        bw = BigWigInMemory(input[1], fill_nan=0)
        df["cons"] = df.progress_apply(
            lambda x: np.quantile(bw("chr" + x.chrom, x.start, x.end), 0.75),
            axis=1,
        )
        df.to_parquet(output[0], index=False)


rule interpretation_filter_cons:
    input:
        "results/interpretation/windows/{anything}.parquet",
    output:
        "results/interpretation/windows/{anything}_filtcons_{thresh,\d+}.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = df[df.cons >= float(wildcards.thresh) / 100]
        df.to_parquet(output[0], index=False)


rule interpretation_subsample:
    input:
        "results/interpretation/windows/{anything}.parquet",
    output:
        "results/interpretation/windows/{anything}_subsample_{n,\d+}.parquet",
    run:
        n = int(wildcards.n)
        df = pd.read_parquet(input[0])
        df = (
            df.groupby("label")
            .apply(lambda x: x.sample(n=min(len(x), n), random_state=42))
            .reset_index(drop=True)
            .sort_values(["chrom", "start", "end"])
        )  # Changed "center" to "start"
        print(df.label.value_counts())
        df.to_parquet(output[0], index=False)


rule interpretation_subsample_stratified:
    input:
        "results/interpretation/windows/{anything}.parquet",
    output:
        "results/interpretation/windows/{anything}_subsamplestratified_{n,\d+}.parquet",
    run:
        n = int(wildcards.n)
        df = pd.read_parquet(input[0])
        background_mask = df.label == "background"
        df_background = df[background_mask]
        df_foreground = df[~background_mask]

        res_background = df_background.sample(
            n=min(len(df_background), n), random_state=42
        )

        df_foreground["conserved"] = df_foreground.cons >= 1
        res_foreground = (
            df_foreground.groupby(["label", "conserved"])
            .apply(lambda x: x.sample(n=min(len(x), n // 2), random_state=42))
            .reset_index(drop=True)
            .drop(columns="conserved")
        )

        res = pd.concat([res_background, res_foreground]).sort_values(
            ["chrom", "start", "end"]
        )
        print(res.label.value_counts())
        res.to_parquet(output[0], index=False)


rule interpretation_embed:
    input:
        "results/interpretation/windows/{anything}.parquet",
    output:
        "results/interpretation/embed/{anything}/{model}.parquet",
    params:
        model_path=lambda wc: config["gpn_star"][wc.model]["model_path"],
        msa_path=lambda wc: config["gpn_star"][wc.model]["msa_path"],
        phylo_info_path=lambda wc: config["gpn_star"][wc.model]["phylo_info_path"],
        window_size=lambda wc: config["gpn_star"][wc.model]["window_size"],
        center_window_size=CENTER_WINDOW_SIZE,
    threads: workflow.cores
    shell:
        "python workflow/scripts/window_embed.py {input} {params.model_path} {params.msa_path} {params.phylo_info_path} {params.window_size} {params.center_window_size} {output}"


rule interpretation_umap:
    input:
        "results/interpretation/embed/{anything}.parquet",
    output:
        "results/interpretation/umap/{anything}.parquet",
    run:
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from umap import UMAP

        embed = pd.read_parquet(input[0])
        proj = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("umap", UMAP(random_state=42, verbose=True)),
            ]
        ).fit_transform(embed)
        proj = pd.DataFrame(proj, columns=["UMAP1", "UMAP2"])
        proj.to_parquet(output[0], index=False)


rule interpretation_umap_plot:
    input:
        "results/interpretation/windows/{intervals}.parquet",
        "results/interpretation/umap/{intervals}/{model}.parquet",
    output:
        "results/plots/umap/{intervals}/{model}/region.svg",
        "results/plots/umap/{intervals}/{model}/conservation.svg",
    run:
        import matplotlib.pyplot as plt
        import seaborn as sns

        windows = pd.read_parquet(input[0])
        proj = pd.read_parquet(input[1])
        proj_cols = proj.columns
        assert len(windows) == len(proj)
        df = pd.concat([windows, proj], axis=1)
        df.label = df.label.replace(
            {
                "five_prime_UTR": "5' UTR",
                "three_prime_UTR": "3' UTR",
                "PLS": "Promoter",
                "dELS": "Enhancer",
                "lnc_RNA": "lncRNA",
                "background": "Background",
            }
        )
        label_order = [
            "CDS",
            "5' UTR",
            "3' UTR",
            "Promoter",
            "Enhancer",
            "lncRNA",
            "Background",
        ]
        palette = {
            "CDS": "C2",
            "5' UTR": "C1",
            "3' UTR": "C4",
            "Promoter": "C0",
            "Enhancer": "C3",
            "lncRNA": "C9",
            "Background": "C5",
        }

        plt.figure(figsize=(3, 3))
        g = sns.scatterplot(
            data=df,
            x=proj_cols[0],
            y=proj_cols[1],
            palette=palette,
            alpha=0.5,
            s=100.0 / np.sqrt(len(df)),
            linewidth=0,
            edgecolor=None,
            hue="label",
            hue_order=label_order,
            rasterized=True,
        )
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        g.legend_.set_title("Region")
        for handle in g.legend_.legend_handles:
            handle.set_markersize(8)
            handle.set_alpha(1.0)
        sns.despine()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(
            output[0],
            bbox_inches="tight",
            dpi=200,
        )

        plt.figure(figsize=(3, 3))
        g = sns.scatterplot(
            data=df,
            x="UMAP1",
            y="UMAP2",
            alpha=0.5,
            s=100.0 / np.sqrt(len(df)),
            linewidth=0,
            edgecolor=None,
            hue="cons",
            rasterized=True,
        )
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        g.legend_.set_title("Conservation")
        for handle in g.legend_.legend_handles:
            handle.set_markersize(8)
            handle.set_alpha(1.0)
        sns.despine()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(output[1], bbox_inches="tight", dpi=200)
