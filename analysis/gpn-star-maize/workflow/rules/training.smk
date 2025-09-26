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


rule filter_conservation_intervals:
    input:
        "results/intervals/{window_size}/defined.{conservation}.{operation}.parquet",
    output:
        "results/intervals/{window_size}/defined.{conservation}.{operation}_{top_frac}_{random_frac}.parquet",
    run:
        intervals = pd.read_parquet(input[0])
        top_frac = float(wildcards["top_frac"])
        random_frac = float(wildcards["random_frac"])
        mask_top = intervals.conservation >= intervals.conservation.quantile(1-top_frac)
        top_intervals = intervals[mask_top]
        assert not top_intervals.conservation.isna().any()
        random_intervals = intervals[~mask_top].sample(frac=random_frac, random_state=42)
        res = pd.concat([top_intervals, random_intervals], ignore_index=True)
        res = res[["chrom", "start", "end"]].drop_duplicates()
        res.to_parquet(output[0], index=False)


rule make_dataset:
    input:
        "results/intervals/{window_size}/defined.{conservation}.{operation}_{top_frac}_{random_frac}.parquet",
        "results/conservation/phyloP.bw",
        "results/conservation/{conservation}.bw",
        "results/genome.fa.gz",
    output:
        expand("results/dataset/{{window_size}}/{{step_size}}/{{add_rc}}/defined.{{conservation}}.{{operation}}_{{top_frac}}_{{random_frac}}/{split}.parquet", split=SPLITS),
    threads: workflow.cores
    run:
        intervals = pd.read_parquet(input[0])

        intervals["strand"] = "+"
        assert int(wildcards.step_size) == (int(wildcards.window_size) // 2)
        if wildcards.add_rc == "True":
            intervals_neg = intervals.copy()
            intervals_neg.strand = "-"
            intervals = pd.concat([intervals, intervals_neg], ignore_index=True)

        phyloP_obj = BigWig(input[1])
        phastCons_obj = BigWig(input[2])
        intervals["phyloP"] = intervals.progress_apply(
            lambda i: phyloP_obj.get_features(i.chrom, i.start, i.end, i.strand),
            axis=1,
        )
        intervals["phastCons"] = intervals.progress_apply(
            lambda i: phastCons_obj.get_features(i.chrom, i.start, i.end, i.strand),
            axis=1,
        )
        genome = Genome(input[3])
        intervals["lowercase"] = intervals.progress_apply(
            lambda i: np.char.islower(list(genome.get_seq(i.chrom, i.start, i.end, i.strand))),
            axis=1,
        )

        intervals = intervals.sample(frac=1.0, random_state=42)

        for path, split in zip(output, SPLITS):
            intervals[
                intervals.chrom.isin(SPLIT_CHROMS[split])
            ].to_parquet(path, index=False, engine="pyarrow")


rule compute_phylo_dist:
    input:
        "results/tree.nh",
    output:
        directory("results/phylo_dist/{clade_thres}"),
    run:
        phylo_tree = Phylo.read(input[0], 'newick')
        leaves = phylo_tree.get_terminals()
        phylo_dist_pairwise = np.array([[phylo_tree.distance(leaf1, leaf2) for leaf2 in leaves] for leaf1 in tqdm(leaves)])

        clade_dict = cluster_clades(phylo_dist_pairwise, float(wildcards.clade_thres))

        in_clade_phylo_dist = np.zeros(phylo_dist_pairwise.shape[0])
        leaves = [node for node in phylo_tree.get_terminals()]
        for clade_id, species in clade_dict.items():
            leaves_in_clade = [leaves[i] for i in list(species)]
            clade_mcra = phylo_tree.common_ancestor(leaves_in_clade)
            dist_to_mcra = [phylo_tree.distance(leaf, clade_mcra) for leaf in leaves_in_clade]
            for s, d in zip(list(species), dist_to_mcra):
                in_clade_phylo_dist[s] = d
        os.makedirs(output[0], exist_ok=True) 
        np.save(output[0] + '/pairwise.npy', phylo_dist_pairwise)
        np.save(output[0] + '/in_clade.npy', in_clade_phylo_dist)
