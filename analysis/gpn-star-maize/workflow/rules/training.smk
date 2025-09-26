rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget -O {output} {config[genome_url]}"


rule make_defined_intervals:
    input:
        "results/genome.fa.gz",
    output:
        "results/intervals/{window_size}/defined.parquet",
    run:
        genome = Genome(input[0])
        genome.filter_chroms(CHROMS)
        intervals = genome.get_defined_intervals()
        intervals = filter_length(intervals, int(wildcards.window_size))
        intervals.to_parquet(output[0], index=False)


rule get_conservation_intervals:
    input:
        "results/intervals/{window_size}/defined.parquet",
        "results/conservation/{conservation}.bw",
    output:
        "results/intervals/{window_size}/defined.{conservation}.{operation}.parquet",
    run:
        window_size = int(wildcards["window_size"])
        operation = wildcards["operation"]
        step_size = window_size // 2

        intervals = pd.read_parquet(input[0])
        bw = pyBigWig.open(input[1])
        intervals = make_windows(intervals, window_size, step_size)

        if operation == "mean":
            f = lambda v: bw.stats(v.chrom, v.start, v.end, exact=True)[0]
        elif operation == "percentile-75":
            f = lambda v: np.quantile(bw.values(v.chrom, v.start, v.end), 0.75)

        intervals["conservation"] = intervals.progress_apply(f, axis=1)
        intervals.to_parquet(output[0])
