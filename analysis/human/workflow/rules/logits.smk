import bioframe as bf
from gpn.data import (
    filter_defined, filter_length, load_table, add_flank, get_annotation_features,
    add_jitter, get_promoters, get_random_intervals, union_intervals,
    intersect_intervals, intervals_size
)
import pandas as pd


def find_positions(interval):
    df = pd.DataFrame(dict(pos=range(interval.start, interval.end)))
    df["chrom"] = interval.chrom
    df.pos += 1  # we'll treat as 1-based
    return df


rule make_positions_promoter:
    input:
        "results/annotation.gtf.gz",
        "results/rmsk_merged.parquet",
        "results/intervals/128/defined.parquet",
    output:
        "results/positions/promoter/positions.parquet",
    run:
        annotation = load_table(input[0])
        annotation = annotation[annotation.chrom.isin(CHROMS)]

        transcripts = annotation[annotation.feature.isin(["mRNA", "transcript"])]
        print(transcripts)
        transcripts = transcripts[
            transcripts['attribute'].str.contains('transcript_biotype "protein_coding"')
        ]
        intervals = get_promoters(transcripts, 1000, 1000)
        print(intervals)

        CDS = bf.merge(annotation.query("feature == 'CDS'")).drop(columns="n_intervals")
        intervals = bf.subtract(intervals, CDS)
        print(intervals)

        repeats = pd.read_parquet(input[1])
        repeats = repeats[repeats.chrom.isin(CHROMS)]
        intervals = bf.subtract(intervals, repeats)
        print(intervals)

        defined_intervals = pd.read_parquet(input[2])
        intervals = intersect_intervals(
            intervals, bf.expand(defined_intervals, pad=-WINDOW_SIZE//2)
        )
        print(intervals)

        intervals = filter_length(intervals, 30)
        intervals = intervals.sort_values(["chrom", "start"])
        print(intervals)

        positions = pd.concat(
            intervals.progress_apply(find_positions, axis=1).values, ignore_index=True
        )
        print(positions)
        positions.to_parquet(output[0], index=False)


rule make_positions_erap2:
    output:
        "results/positions/erap2/positions.parquet",
    run:
        intervals = pd.DataFrame(dict(chrom=["5"], start=[96896864], end=[96897364]))
        positions = pd.concat(
            intervals.progress_apply(find_positions, axis=1).values, ignore_index=True
        )
        print(positions)
        positions.to_parquet(output[0], index=False)


rule make_positions_chrom:
    input:
        f"results/intervals/{config['window_size']}/defined.parquet",
    output:
        "results/positions/{chrom}/positions.parquet",
    run:
        intervals = pd.read_parquet(input[0]).query(f"chrom == '{wildcards.chrom}'")
        intervals = bf.expand(intervals, pad=-config["window_size"]//2)
        intervals = filter_length(intervals, 1)
        positions = pd.concat(
            intervals.progress_apply(find_positions, axis=1).values, ignore_index=True
        )
        print(positions)
        positions.to_parquet(output[0], index=False)


rule get_logits:
    input:
        "{anything}/positions.parquet",
        "results/msa/{alignment}/{species}/all.zarr",
        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
    output:
        "{anything}/logits/{alignment,[A-Za-z0-9]+}/{species,[A-Za-z0-9]+}/{window_size,\d+}/{model}.parquet",
    shell:
        """
        torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') -m gpn.msa.inference logits {input[0]} {input[1]} {wildcards.window_size} {input[2]} {output} \
        --per_device_batch_size {config[per_device_batch_size]} --is_file \
        --dataloader_num_workers {config[dataloader_num_workers]}
        """


rule download_jaspar:
    output:
        "results/jaspar.meme",
    shell:
        "wget -O {output} https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt"
    

# example:
# snakemake --cores all --use-conda --conda-frontend mamba 
rule run_modisco:
    input:
        "{anything}/positions.parquet",
        "{anything}/logits/{model}.parquet",
        "results/genome.fa.gz",
    output:
        "{anything}/modisco/{model}/results.h5",
    conda:
        "../envs/modisco-lite.yaml"
    script:
        "../scripts/modisco_run.py"


rule plot_modisco:
    input:
        "{anything}/modisco/{model}/results.h5",
        "results/jaspar.meme",
    output:
        directory("{anything}/modisco/{model}/report"),
    conda:
        "../envs/modisco-lite.yaml"
    script:
        "../scripts/modisco_report.py"


rule make_bed_probs:
    input:
        "results/{anything}/positions.parquet",
        "results/{anything}/logits/{model}.parquet",
    output:
        temp(expand("results/{{anything}}/bed_probs/{{model}}/{nuc}.bed", nuc=NUCLEOTIDES)),
    run:
        df = pd.read_parquet(input[0])
        df.loc[:, NUCLEOTIDES] = softmax(pd.read_parquet(input[1]), axis=1)
        df["entropy"] = entropy(df[NUCLEOTIDES], base=2, axis=1)
        print(df)
        df.loc[:, NUCLEOTIDES] = df[NUCLEOTIDES].values * (2-df[["entropy"]].values)
        print(df)
        df["start"] = df.pos-1
        df["end"] = df.pos
        df.chrom = "chr" + df.chrom
        for nuc, path in zip(NUCLEOTIDES, output):
            df.to_csv(
                path, sep="\t", index=False, header=False, float_format='%.2f',
                columns=["chrom", "start", "end", nuc],
            )


rule make_bed_llr:
    input:
        "results/{anything}/positions.parquet",
        "results/{anything}/logits/{model}.parquet",
        "results/genome.fa.gz",
    output:
        temp(expand("results/{{anything}}/bed_llr/{{model}}/{nuc}.bed", nuc=NUCLEOTIDES)),
    run:
        df = pd.read_parquet(input[0])
        df.loc[:, NUCLEOTIDES] = pd.read_parquet(input[1]).values
        genome = Genome(input[2], subset_chroms=df.chrom.unique())
        df["ref"] = df.progress_apply(lambda v: genome.get_nuc(v.chrom, v.pos).upper(), axis=1)
        print(df)
        idx, cols = pd.factorize(df.ref)
        df[NUCLEOTIDES] = df[NUCLEOTIDES].subtract(df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx], axis=0)
        print(df)
        df["start"] = df.pos-1
        df["end"] = df.pos
        df.chrom = "chr" + df.chrom
        for nuc, path in zip(NUCLEOTIDES, output):
            df.to_csv(
                path, sep="\t", index=False, header=False, float_format='%.2f',
                columns=["chrom", "start", "end", nuc],
            )


rule make_chrom_sizes:
    input:
        "results/genome.fa.gz",
    output:
        "results/chrom.sizes",
    run:
        intervals = Genome(input[0], subset_chroms=CHROMS).get_all_intervals()
        intervals.chrom = "chr" + intervals.chrom
        intervals.to_csv(
            output[0], sep="\t", index=False, header=False,
            columns=["chrom", "end"],
        )


rule convert_bed_to_bigwig:
    input:
        "{anything}.bed",
        "results/chrom.sizes",
    output:
        "{anything}.bw"
    shell:
        # using a local download because conda version didn't work
        "./bedGraphToBigWig {input} {output}"


rule bigwig_done:
    input:
        expand("{{anything}}/{nuc}.bw", nuc=NUCLEOTIDES),
    output:
        touch("{anything}/bigwig.done"),
