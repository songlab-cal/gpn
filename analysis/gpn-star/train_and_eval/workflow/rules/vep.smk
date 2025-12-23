rule run_vep_msa:
    input:
        "results/msa/{msa}/all.zarr",
    output:
        "results/preds/{dataset}/msa_{msa}.parquet",
    threads: workflow.cores
    run:
        V = load_dataset(wildcards['dataset'], split="test").to_pandas()
        print(V)
        genome_msa = GenomeMSA(input[0])
        V["score"] = genome_msa.run_vep_batch(
            V["chrom"].values, V["pos"].values, V["ref"].values, V["alt"].values,
            backend="multiprocessing", n_jobs=threads
        )
        print(V)
        V[["score"]].to_parquet(output[0], index=False)


rule get_entropy_calibrated:
    input:
        "results/logits/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
        "results/calibration/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}/entropy.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/preds/entropy/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(vep_datasets),
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    run:
        try:
            V = load_dataset_from_file_or_dir(
                wildcards.dataset,
                split="test",
                is_file=False,
            )
            V = V.to_pandas()
        except:
            V = Dataset.from_pandas(pd.read_parquet(wildcards.dataset+'/test.parquet'))
        genome = Genome(input[2])
        V['pentanuc'] = V.apply(
            lambda row: genome.get_seq(row['chrom'], row['pos']-3, row['pos']+2).upper(), axis=1
        )
        df_calibration = pd.read_parquet(input[1])
        logits = pd.read_parquet(input[0])
        normalized_logits = normalize_logits(logits)
        V['entropy'] = get_entropy(normalized_logits)
        V = V.merge(df_calibration, on="pentanuc", how="left")
        V['entropy_calibrated'] = V['entropy'] / V['entropy_neutral_mean']
        V[["entropy_calibrated"]].to_parquet(output[0], index=False)

rule get_llr_calibrated:
    input:
        "results/logits/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
        "results/calibration/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}/llr.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/preds/llr/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(vep_datasets),
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    run:
        try:
            V = load_dataset_from_file_or_dir(
                wildcards.dataset,
                split="test",
                is_file=False,
            )
            V = V.to_pandas()
        except:
            V = Dataset.from_pandas(pd.read_parquet(wildcards.dataset+'/test.parquet'))
        genome = Genome(input[2])
        V['pentanuc'] = V.apply(
            lambda row: genome.get_seq(row['chrom'], row['pos']-3, row['pos']+2).upper(), axis=1
        )
        V['pentanuc_mut'] = V['pentanuc'] + '_' + V['alt']
        df_calibration = pd.read_parquet(input[1])
        logits = pd.read_parquet(input[0])
        normalized_logits = normalize_logits(logits)
        V['llr'] = get_llr(normalized_logits, V['ref'], V['alt'])
        V = V.merge(df_calibration, on="pentanuc_mut", how="left")
        V['llr_calibrated'] = V['llr'] - V['llr_neutral_mean']
        V[["llr_calibrated"]].to_parquet(output[0], index=False)

rule get_absllr_calibrated:
    input:
        "results/logits/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
        "results/calibration/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}/llr.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/preds/absllr/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(vep_datasets),
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    run:
        try:
            V = load_dataset_from_file_or_dir(
                wildcards.dataset,
                split="test",
                is_file=False,
            )
            V = V.to_pandas()
        except:
            V = Dataset.from_pandas(pd.read_parquet(wildcards.dataset+'/test.parquet'))
        genome = Genome(input[2])
        V['pentanuc'] = V.apply(
            lambda row: genome.get_seq(row['chrom'], row['pos']-3, row['pos']+2).upper(), axis=1
        )
        V['pentanuc_mut'] = V['pentanuc'] + '_' + V['alt']
        df_calibration = pd.read_parquet(input[1])
        logits = pd.read_parquet(input[0])
        normalized_logits = normalize_logits(logits)
        V['llr'] = get_llr(normalized_logits, V['ref'], V['alt'])
        V = V.merge(df_calibration, on="pentanuc_mut", how="left")
        V['absllr_calibrated'] = np.abs(V['llr']) - np.abs(V['llr_neutral_mean'])
        V[["absllr_calibrated"]].to_parquet(output[0], index=False)