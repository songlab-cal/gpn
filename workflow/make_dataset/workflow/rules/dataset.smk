from gpn.data import Genome, make_windows, get_seq
import math
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


split_proportions = [config["split_proportion"][split] for split in splits]
assert np.isclose(sum(split_proportions), 1)


rule make_dataset_assembly:
    input:
        lambda wildcards: f"results/intervals/{wildcards['assembly']}/{config['target_intervals']}.parquet",
        "results/genome/{assembly}.fa.gz",
    output:
        temp(expand("results/dataset_assembly/{{assembly}}/{split}.parquet", split=splits)),
    threads: 2
    run:
        intervals = pd.read_parquet(input[0])
        genome = Genome(input[1])
        intervals = make_windows(
            intervals, config["window_size"], config["step_size"], config["add_rc"],
        )
        print(intervals)
        intervals = intervals.sample(frac=1.0, random_state=42)
        intervals["assembly"] = wildcards["assembly"]
        intervals = intervals[["assembly", "chrom", "start", "end", "strand"]]
        intervals = get_seq(intervals, genome)
        print(intervals)

        chroms = intervals.chrom.unique()
        chrom_split = np.random.choice(
            splits, p=split_proportions, size=len(chroms),
        )
        chrom_split[np.isin(chroms, config["whitelist_validation_chroms"])] = "validation"
        chrom_split[np.isin(chroms, config["whitelist_test_chroms"])] = "test"
        chrom_split = pd.Series(chrom_split, index=chroms)

        intervals_split = chrom_split[intervals.chrom]

        for path, split in zip(output, splits):
            print(path, split)
            # to parquet to be able to load faster later
            intervals[(intervals_split==split).values].to_parquet(
                path, index=False,
            )


# before uploading to HF Hub, remove data/split/.snakemake_timestamp files
rule merge_datasets:
    input:
        expand("results/dataset_assembly/{assembly}/{{split}}.parquet", assembly=assemblies.index),
    output:
        directory("results/dataset/data/{split}"),
    threads: workflow.cores
    run:
        intervals = pd.concat(
            tqdm((pd.read_parquet(path) for path in input), total=len(input)),
            ignore_index=True,
        ).sample(frac=1, random_state=42)
        print(intervals)

        if config.get("subsample_to_target", False) and wildcards.split == "train":
            n_target = (intervals.assembly==config["target_assembly"]).sum()
            intervals = intervals.groupby("assembly").sample(
                n=n_target, random_state=42
            ).sample(frac=1, random_state=42)
            print(wildcards.split, intervals.assembly.value_counts())
            print(intervals)

        n_shards = math.ceil(len(intervals) / config["samples_per_file"])
        assert n_shards < 10000
        os.makedirs(output[0])
        for i in tqdm(range(n_shards)):
            path = Path(output[0]) / f"shard_{i:05}.jsonl.zst"
            intervals.iloc[i::n_shards].to_json(
                path, orient="records", lines=True,
                compression={'method': 'zstd', 'threads': -1}
            )
