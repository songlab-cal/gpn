enformer_chroms = [str(i) for i in range(1, 23)]


rule download_enformer_precomputed:
    output:
        "results/data/enformer/{chrom}.h5",
    wildcard_constraints:
        chrom="|".join(enformer_chroms),
    shell:
        "wget -O {output} https://storage.googleapis.com/dm-enformer/variant-scores/1000-genomes/enformer/1000G.MAF_threshold%3D0.005.{wildcards.chrom}.h5"


rule process_enformer_precomputed_full:
    input:
        "results/data/enformer/{chrom}.h5",
        "results/genome.fa.gz",
    output:
        temp("results/data/enformer/coords/{chrom}.parquet"),
    wildcard_constraints:
        chrom="|".join(enformer_chroms),
    run:
        import h5py

        f = h5py.File(input[0])

        V = pd.DataFrame(
            {
                "chrom": f["chr"][:].astype(str),
                "pos": f["pos"][:],
                "ref": f["ref"][:].astype(str),
                "alt": f["alt"][:].astype(str),
            }
        )
        V["idx"] = np.arange(len(V))
        print(V)

        V.chrom = V.chrom.str.replace("chr", "")
        V = V[(V.ref.str.len() == 1) & (V.alt.str.len() == 1)]
        print(V.shape)
        V = lift_hg19_to_hg38(V)
        V = V[V.pos != -1]
        print(V.shape)
        genome = Genome(input[1])
        V = check_ref_alt(V, genome)
        print(V.shape)
        # two hg19 pos can map to one hg38 pos
        V.drop_duplicates(COORDINATES, keep=False, inplace=True)
        print(V.shape)
        print(V)
        V.to_parquet(output[0], index=False)


rule merge_enformer_precomputed_full:
    input:
        expand("results/data/enformer/coords/{chrom}.parquet", chrom=enformer_chroms),
    output:
        "results/enformer/coords/merged.parquet",
    run:
        df = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)


rule enformer_abs_delta:
    input:
        coords="results/data/enformer/coords/merged.parquet",
        data=expand("results/data/enformer/{chrom}.h5", chrom=enformer_chroms),
    output:
        "results/features/{dataset}/Enformer_absDelta.parquet",
    threads: workflow.cores
    run:
        import h5py

        cols = ["chrom", "pos", "ref", "alt"]
        V = load_dataset_from_file_or_dir(wildcards["dataset"]).to_pandas()

        coords = pd.read_parquet(input.coords)

        labels = [f"feature_{i}" for i in range(5313)]

        all_res = []

        for chrom, data_path in tqdm(zip(enformer_chroms, input.data)):
            coords_c = coords.merge(V.loc[V.chrom == chrom, cols], on=cols, how="inner")
            f = h5py.File(data_path)
            # data = f["SAD"][:]  # load into memory
            data = f["SAD"]  # do not load into memory
            data = pd.DataFrame(data[coords_c.idx.values].astype(float), columns=labels)
            res = pd.concat([coords_c, data], axis=1)
            recoded = (res.ref != f["ref"][res.idx.values].astype(str)).values
            res.loc[recoded, data.columns] *= -1
            all_res.append(res)
        res = pd.concat(all_res, ignore_index=True)

        V = V.merge(res, on=cols, how="left")
        V[labels].abs().to_parquet(output[0], index=False)
