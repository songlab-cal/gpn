rule download_rmsk:
    output:
        "output/rmsk.txt.gz",
    shell:
        "wget -O {output} https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz"


rule process_rmsk:
    input:
        "output/rmsk.txt.gz",
    output:
        "output/rmsk.parquet",
    run:
        df = pd.read_csv(
            input[0], sep="\t", header=None, names=[
                "bin", "swScore", "milliDiv", "milliDel", "milliIns", "chrom",
                "start", "end", "genoLeft", "strand", "repName", "repClass",
                "repFamily", "repStart", "repEnd", "repLeft", "id",
            ]
        )
        df.chrom = df.chrom.astype(str)
        df.chrom = df.chrom.str.replace("chr", "")
        df = df[df.chrom.isin(CHROMS)]
        df.to_parquet(output[0], index=False)

     
rule merge_rmsk:
    input:
        "output/rmsk.parquet",
    output:
        "output/rmsk_merged.parquet",
    run:
        df = pd.read_parquet(input[0], columns=["chrom", "start", "end"])
        print(df)
        df = bf.merge(df).drop(columns="n_intervals")
        print(df)
        df.to_parquet(output[0], index=False)


rule expand_annotation:
    input:
        "output/annotation.gtf.gz",
        "output/rmsk_merged.parquet",
        "output/genome.fa.gz",
    output:
        "output/annotation.expanded.parquet",
    run:
        import more_itertools

        gtf = load_table(input[0])
        gtf = gtf[gtf.chrom.isin(CHROMS)]

        repeats = pd.read_parquet(input[1])
        repeats["feature"] = "Repeat"
        gtf = pd.concat([gtf, repeats], ignore_index=True)

        all_intervals = Genome(input[2], subset_chroms=CHROMS).get_all_intervals()

        gtf_intergenic = bf.subtract(all_intervals, gtf[gtf.feature.isin(["gene", "Repeat"])])
        gtf_intergenic["feature"] = "intergenic"
        gtf = pd.concat([gtf, gtf_intergenic], ignore_index=True)

        gtf_exon = gtf[gtf.feature=="exon"]
        gtf_exon["transcript_id"] = gtf_exon.attribute.str.extract(r'transcript_id "([^;]*)";')

        def get_transcript_introns(df_transcript):
            df_transcript = df_transcript.sort_values("start")
            exon_pairs = more_itertools.pairwise(df_transcript.loc[:, ["start", "end"]].values)
            introns = [[e1[1], e2[0]] for e1, e2 in exon_pairs]
            introns = pd.DataFrame(introns, columns=["start", "end"])
            introns["chrom"] = df_transcript.chrom.iloc[0]
            return introns

        gtf_introns = gtf_exon.groupby("transcript_id").apply(get_transcript_introns).reset_index().drop_duplicates(subset=["chrom", "start", "end"])
        gtf_introns["feature"] = "intron"
        gtf = pd.concat([gtf, gtf_introns], ignore_index=True)
        print(gtf.feature.value_counts())
        gtf.to_parquet(output[0], index=False)


rule define_embedding_windows:
    input:
        "output/annotation.expanded.parquet",
        "output/genome.fa.gz",
    output:
        "output/embedding/windows.parquet",
    run:
        gtf = pd.read_parquet(input[0])
        genome = Genome(input[1], subset_chroms=CHROMS)
        defined_intervals = genome.get_defined_intervals()
        defined_intervals = filter_length(defined_intervals, WINDOW_SIZE)
        windows = make_windows(defined_intervals, WINDOW_SIZE, EMBEDDING_WINDOW_SIZE)
        windows.rename(columns={"start": "full_start", "end": "full_end"}, inplace=True)

        windows["start"] = (windows.full_start+windows.full_end)//2 - EMBEDDING_WINDOW_SIZE//2
        windows["end"] = windows.start + EMBEDDING_WINDOW_SIZE

        features_of_interest = [
            "intergenic",
            'CDS',
            'intron',
            'three_prime_utr',
            'five_prime_utr',
            #"ncRNA_gene",
            "Repeat",
        ]

        for f in features_of_interest:
            print(f)
            windows = bf.coverage(windows, gtf[gtf.feature==f])
            windows.rename(columns=dict(coverage=f), inplace=True)
        
        windows = windows[(windows[features_of_interest]==EMBEDDING_WINDOW_SIZE).sum(axis=1)==1]
        windows["Region"] = windows[features_of_interest].idxmax(axis=1)
        windows.drop(columns=features_of_interest, inplace=True)

        windows.rename(columns={"start": "center_start", "end": "center_end"}, inplace=True)
        windows.rename(columns={"full_start": "start", "full_end": "end"}, inplace=True)
        print(windows.Region.value_counts())
        groupby = windows.groupby("Region")
        n = groupby.size().min()
        windows = groupby.sample(n=n, random_state=42).sample(frac=1, random_state=42)
        print(windows)
        print(windows.Region.value_counts())
        windows.to_parquet(output[0], index=False)


# note window_size is not really used
rule get_embedding:
    input:
        "{anything}/windows.parquet",
        "results/msa/{alignment}/{species}/all.zarr",
        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
    output:
        "{anything}/embedding/{alignment,[A-Za-z0-9]+}/{species,[A-Za-z0-9]+}/{window_size,\d+}/{model}.parquet",
    threads: workflow.cores
    shell:
        """
        python -m gpn_msa.inference embedding {input[0]} {input[1]} \
        {wildcards.window_size} {input[2]} {output} --is_file \
        --center_window_size {config[center_window_size]} \
        --per_device_batch_size {config[per_device_batch_size]} \
        --dataloader_num_workers {config[dataloader_num_workers]}
        """


rule run_umap:
    input:
        "{anything}/embedding/{model}.parquet",
    output:
        "{anything}/umap/{model}.parquet",
    run:
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from umap import UMAP

        embeddings = pd.read_parquet(input[0])
        proj = Pipeline([
            ("scaler", StandardScaler()),
            ("umap", UMAP(random_state=42, verbose=True)),
        ]).fit_transform(embeddings)
        proj = pd.DataFrame(proj, columns=["UMAP1", "UMAP2"])
        proj.to_parquet(output[0], index=False)


rule run_classification:
    input:
        "{anything}/windows.parquet",
        "{anything}/embedding/{model}.parquet",
    output:
        "{anything}/classification/{model}.parquet",
    threads: workflow.cores
    run:
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        from sklearn.model_selection import cross_val_predict, GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        windows = pd.read_parquet(input[0])
        features = pd.read_parquet(input[1])

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("linear", LogisticRegressionCV(
                random_state=42, verbose=True, max_iter=1000,
                class_weight="balanced", n_jobs=-1
                )
            ),
        ])
        preds = cross_val_predict(
            clf, features, windows.Region, groups=windows.chrom,
            cv=GroupKFold(), verbose=True,
        )
        pd.DataFrame({"pred_Region": preds}).to_parquet(output[0], index=False)
