from gpn.data import (
    Genome, load_table, get_balanced_intervals, filter_length,
    filter_annotation_features,
)


rule make_all_intervals:
    input:
        "results/genome/{assembly}.fa.gz",
    output:
        "results/intervals/{assembly}/all.parquet",
    threads: 2
    run:
        I = Genome(input[0]).get_all_intervals()
        I = filter_length(I, config["window_size"])
        I.to_parquet(output[0], index=False)


rule make_defined_intervals:
    input:
        "results/genome/{assembly}.fa.gz",
    output:
        "results/intervals/{assembly}/defined.parquet",
    threads: 2
    run:
        I = Genome(input[0]).get_defined_intervals()
        I = filter_length(I, config["window_size"])
        I.to_parquet(output[0], index=False)


rule make_annotation_intervals:
    input:
        "results/intervals/{assembly}/defined.parquet",
        "results/annotation/{assembly}.gff.gz",
    output:
        "results/intervals/{assembly}/annotation_{feature}.parquet",
    run:
        I = pd.read_parquet(input[0])
        annotation = load_table(input[1])
        include_flank = config.get(
            "annotation_features_include_flank", config["window_size"] // 2
        )
        add_jiter = config.get("annotation_features_add_jitter", 100)
        I = filter_annotation_features(
            I, annotation, wildcards.feature,
            include_flank=include_flank, jitter=add_jitter,
        )
        I = filter_length(I, config["window_size"])
        I.to_parquet(output[0], index=False)


rule make_balanced_v1_intervals:
    input:
        "results/intervals/{assembly}/defined.parquet",
        "results/annotation/{assembly}.gff.gz",
    output:
        "results/intervals/{assembly}/balanced_v1.parquet",
    run:
        defined_intervals = load_table(input[0])
        annotation = load_table(input[1])
        intervals = get_balanced_intervals(
            defined_intervals, annotation, config["window_size"],
            config.get("promoter_upstream", 1000),
        )
        intervals.to_parquet(output[0], index=False)
