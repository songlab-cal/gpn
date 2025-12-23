import bioframe as bf
from gpn.data import (
    filter_defined, filter_length, load_table, add_flank, get_annotation_features,
    add_jitter, get_promoters, get_random_intervals, union_intervals,
    intersect_intervals, intervals_size
)
import pandas as pd
import polars as pl

def find_positions(interval):
    df = pd.DataFrame(dict(pos=range(interval.start, interval.end)))
    df["chrom"] = interval.chrom
    df.pos += 1  # we'll treat as 1-based
    return df

rule make_positions_chrom:
    input:
        "results/intervals/{genome}/{window_size}/defined.parquet",
    output:
        "results/positions/{chrom}/{genome}/{window_size}/positions.parquet",
    run:
        intervals = pd.read_parquet(input[0]).query(f"chrom == '{wildcards.chrom}'")
        intervals = bf.expand(intervals, pad=-int(wildcards.window_size)//2)
        intervals = filter_length(intervals, 1)
        positions = pd.concat(
            intervals.progress_apply(find_positions, axis=1).values, ignore_index=True,
        )
        print(positions)
        positions.to_parquet(output[0], index=False)

rule get_logits:
    input:
        "results/positions/{chrom}/{genome}/{window_size}/positions.parquet",
        "results/msa/{genome}/{alignment}/{species}",
        "results/checkpoints/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}",
    output:
        "results/logits/{chrom}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    params:
        # Number of positions per checkpoint batch (adjust as needed)
        # A larger batch saves less frequently but has less overhead
        # A smaller batch saves more frequently for better resume granularity
        checkpoint_batch_size=config.get("checkpoint_batch_size", 1000000),
    threads:
        workflow.cores
    shell:
    #$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        """
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        num_cpus={threads}
        dataloader_num_workers=$(($num_cpus / $num_gpus))

        torchrun --nproc_per_node=$num_gpus -m gpn.star.inference logits {input[0]} {input[1]} {wildcards.window_size} {input[2]} {output} \
        --per_device_batch_size 8 --is_file \
        --dataloader_num_workers $dataloader_num_workers \
        --checkpoint_batch_size {params.checkpoint_batch_size} \
        --cleanup_checkpoints
        """

rule process_logits:
    input:
        "results/{anything}/positions.parquet",
        "results/{anything}/logits/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/{anything}/processed_logits/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    threads:
        workflow.cores
    run:
        V1 = pl.read_parquet(input[0])[["chrom", "pos"]]
        V2 = pl.read_parquet(input[1])
        V = V1.hstack(V2)

        if len(V["chrom"].unique()) == 1:
            chrom = V["chrom"][0]
            seq = Genome(input[2])._genome[chrom].upper()
            seq = np.frombuffer(seq.encode("ascii"), dtype="S1")
            V = V.with_columns(ref=seq[V["pos"]-1])

        else:
            genome = Genome(input[2])
            V = V.with_columns(chrom=pl.col("chrom").cast(str))
            V = V.with_columns(
                pl.struct(["chrom", "pos"])
                .apply(lambda row: genome.get_nuc(row["chrom"], row["pos"]).upper())
                .alias("ref")
            )

        V = V.with_columns(ref=pl.col("ref").cast(str))
        # sorry, this is horrible, was more elegant in pandas
        V = V.with_columns(
            V.select(ref_logit=(
                pl.when(pl.col("ref") == "A").then(pl.col("A"))
                .when(pl.col("ref") == "C").then(pl.col("C"))
                .when(pl.col("ref") == "G").then(pl.col("G"))
                .when(pl.col("ref") == "T").then(pl.col("T"))
        )))
        V = V.with_columns(
            V[NUCLEOTIDES] - V["ref_logit"]
        )
        V = V.select(["chrom", "pos", "ref"] + NUCLEOTIDES)
        print(V)
        V.write_parquet(output[0])


rule get_llr:
    input:
        "results/{anything}/processed_logits/{model}.parquet",
    output:
        "results/{anything}/llr/{model}.parquet",
    threads:
        workflow.cores
    run:
        V = pl.read_parquet(
            input[0]
        ).melt(
            id_vars=["chrom", "pos", "ref"], value_vars=NUCLEOTIDES,
            variable_name="alt", value_name="score"
        ).sort(["chrom", "pos", "ref"]).filter(pl.col("ref") != pl.col("alt"))
        print(V)
        V.write_parquet(output[0])


ruleorder: logits_merge_chroms > bgzip


rule logits_merge_chroms:
    input:
        expand("results/positions/{chrom}/{{anything}}/{{model}}.tsv.bgz", chrom=CHROMS),
    output:
        "results/positions/merged/{anything}/{model}.tsv.bgz",
    wildcard_constraints:
        anything="processed_logits|probs|llr",
    shell:
        "cat {input} > {output}"


#ruleorder: logits_merge_chroms > process_logits
#
#
#rule logits_merge_chroms:
#    input:
#        expand("results/positions/{chrom}/{{anything}}/{{model}}.parquet", chrom=CHROMS),
#    output:
#        "results/positions/merged/{anything}/{model}.parquet",
#    wildcard_constraints:
#        anything="processed_logits|probs",
#    run:
#        V = pl.concat([pl.read_parquet(path) for path in tqdm(input)])
#        if wildcards.anything == "processed_logits":
#            V = V.select(["chrom", "pos", "ref"] + NUCLEOTIDES)
#        print(V)
#        V.write_parquet(output[0])
#        #V.to_pandas().to_parquet(output[0], index=False)

rule get_calibration_logits:
    input:
        "results/msa/{genome}/{alignment}/{species}",
        "results/checkpoints/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}",
        "results/calibration/{genome}/calibration_dataset/test.parquet",
    output:
        "results/logits/results/calibration/{genome}/calibration_dataset/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    params:
        disable_aux_features = lambda wildcards: "disable_aux_features" if wildcards.model.split("/")[-3] == "False" else "",
        dataset_dir = lambda wildcards, input: input[2].replace("/test.parquet", "")
    threads:
        workflow.cores
    shell:
    #$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        """
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        num_cpus={threads}
        dataloader_num_workers=$(($num_cpus / $num_gpus))

        torchrun --nproc_per_node $num_gpus --master_port=25679 -m gpn.star.inference logits {params.dataset_dir} {input[0]} \
        {wildcards.window_size} {input[1]} {output} \
        --per_device_batch_size 32 --dataloader_num_workers $dataloader_num_workers {params.disable_aux_features}
        """

rule get_vep_logits:
    input:
        "results/msa/{genome}/{alignment}/{species}",
        "results/checkpoints/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}",
    output:
        "results/logits/{dataset}/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
    wildcard_constraints:
        dataset="|".join(vep_datasets),# + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    params:
        disable_aux_features = lambda wildcards: "disable_aux_features" if wildcards.model.split("/")[-3] == "False" else "",
    threads:
        workflow.cores
    shell:
    #$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        """
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        num_cpus={threads}
        dataloader_num_workers=$(($num_cpus / $num_gpus))

        torchrun --nproc_per_node $num_gpus --master_port=25679 -m gpn.star.inference logits {wildcards.dataset} {input[0]} \
        {wildcards.window_size} {input[1]} {output} \
        --per_device_batch_size 32 --dataloader_num_workers $dataloader_num_workers {params.disable_aux_features}
        """