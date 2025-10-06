rule donwload_allele_frequency_v1:
    output:
        "results/inference_dataset/allele_frequency_v1/test.parquet",
    shell:
        """
        wget -O {output} https://huggingface.co/datasets/plantcad/maize-allele-frequency/resolve/main/test.parquet
        """


rule get_logits:
    input:
        "results/inference_dataset/{dataset}/test.parquet",
        "results/msa/36",
        "results/checkpoints/{time_enc}/{clade_thres}/{window_size}/{model}",
    output:
        "results/inference_dataset/{dataset}/logits/{time_enc}/{clade_thres}/{window_size}/{model}.parquet",
    wildcard_constraints:
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        window_size="\d+",
    threads:
        workflow.cores
    shell:
        """
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        num_cpus={threads}
        dataloader_num_workers=$(($num_cpus / $num_gpus))

        torchrun --nproc_per_node=$num_gpus \
        -m gpn.star.inference logits \
        {input[0]} {input[1]} {wildcards.window_size} {input[2]} {output} \
        --per_device_batch_size 512 \
        --is_file \
        --dataloader_num_workers $dataloader_num_workers
        """


#rule get_llr_calibrated:
#    input:
#        "results/logits/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
#        "results/calibration/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}/llr.parquet",
#        "results/genome/{genome}.fa.gz",
#    output:
#        "results/preds/llr/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
#    wildcard_constraints:
#        dataset="|".join(vep_datasets),
#        time_enc="[A-Za-z0-9_-]+",
#        clade_thres="[0-9.-]+",
#        alignment="[A-Za-z0-9_]+",
#        species="[A-Za-z0-9_-]+",
#        window_size="\d+",
#    run:
#        try:
#            V = load_dataset_from_file_or_dir(
#                wildcards.dataset,
#                split="test",
#                is_file=False,
#            )
#            V = V.to_pandas()
#        except:
#            V = Dataset.from_pandas(pd.read_parquet(wildcards.dataset+'/test.parquet'))
#        genome = Genome(input[2])
#        V['pentanuc'] = V.apply(
#            lambda row: genome.get_seq(row['chrom'], row['pos']-3, row['pos']+2).upper(), axis=1
#        )
#        V['pentanuc_mut'] = V['pentanuc'] + '_' + V['alt']
#        df_calibration = pd.read_parquet(input[1])
#        logits = pd.read_parquet(input[0])
#        normalized_logits = normalize_logits(logits)
#        V['llr'] = get_llr(normalized_logits, V['ref'], V['alt'])
#        V = V.merge(df_calibration, on="pentanuc_mut", how="left")
#        V['llr_calibrated'] = V['llr'] - V['llr_neutral_mean']
#        V[["llr_calibrated"]].to_parquet(output[0], index=False)
