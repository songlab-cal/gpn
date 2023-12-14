from gpn.data import load_dataset_from_file_or_dir
import numpy as np


enformer_chroms = [str(i) for i in range(1, 23)]


rule download_enformer_precomputed:
    output:
        "results/enformer/{chrom}.h5",
    wildcard_constraints:
        chrom="|".join(enformer_chroms),
    shell:
        "wget -O {output} https://storage.googleapis.com/dm-enformer/variant-scores/1000-genomes/enformer/1000G.MAF_threshold%3D0.005.{wildcards.chrom}.h5"


rule process_enformer_precomputed:
    input:
        "results/enformer/{chrom}.h5",
    output:
        temp("results/enformer/{chrom}.parquet"),
    wildcard_constraints:
        chrom="|".join(enformer_chroms),
    run:
        import h5py
        from liftover import get_lifter
        converter = get_lifter('hg19', 'hg38')

        f = h5py.File(input[0])
        X = f["SAD"][:]

        df = pd.DataFrame({
            "chrom": f["chr"][:].astype(str),
            "pos": f["pos"][:],
            "ref": f["ref"][:].astype(str),
            "alt": f["alt"][:].astype(str),
            "Enformer_l1": -np.linalg.norm(X, axis=1, ord=1).astype(float),
            "Enformer_l2": -np.linalg.norm(X, axis=1, ord=2).astype(float),
            "Enformer_linf": -np.linalg.norm(X, axis=1, ord=np.inf).astype(float),
        })
        df.chrom = df.chrom.str.replace("chr", "")
        df = df[(df.ref.str.len() == 1) & (df.alt.str.len() == 1)]

        def get_new_pos(v):
            try:
                res = converter[v.chrom][v.pos]
                assert len(res) == 1
                chrom, pos, strand = res[0]
                assert chrom.replace("chr", "")==v.chrom
                return pos
            except:
                return -1
        
        df.pos = df.progress_apply(get_new_pos, axis=1)
        df = df[df.pos != -1]
        print(df)
        df.to_parquet(output[0], index=False)


rule merge_enformer_precomputed:
    input:
        expand("results/enformer/{chrom}.parquet", chrom=[str(i) for i in range(1, 23)]),
    output:
        "results/enformer/merged.parquet",
    run:
        df = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)


rule process_enformer_precomputed_full:
    input:
        "results/enformer/{chrom}.h5",
        "results/genome.fa.gz",
    output:
        temp("results/enformer/coords/{chrom}.parquet"),
    wildcard_constraints:
        chrom="|".join(enformer_chroms),
    run:
        import h5py

        f = h5py.File(input[0])

        V = pd.DataFrame({
            "chrom": f["chr"][:].astype(str),
            "pos": f["pos"][:],
            "ref": f["ref"][:].astype(str),
            "alt": f["alt"][:].astype(str),
        })
        V["idx"] = np.arange(len(V))
        print(V)

        V.chrom = V.chrom.str.replace("chr", "")
        V = V[(V.ref.str.len()==1) & (V.alt.str.len()==1)]
        print(V.shape)
        V = lift_hg19_to_hg38(V)
        V = V[V.pos != -1]
        print(V.shape)
        genome = Genome(input[1])
        V = check_ref_alt(V, genome)
        print(V.shape)
        print(V)
        V.to_parquet(output[0], index=False)


rule merge_enformer_precomputed_full:
    input:
        expand("results/enformer/coords/{chrom}.parquet", chrom=enformer_chroms),
    output:
        "results/enformer/coords/merged.parquet",
    run:
        df = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)


rule run_vep_embeddings_enformer:
    input:
        coords="results/enformer/coords/merged.parquet",
        data=expand("results/enformer/{chrom}.h5", chrom=enformer_chroms),
    output:
        "results/preds/vep_embedding/{dataset}/Enformer.parquet",
    wildcard_constraints:
        dataset="|".join(datasets + ["results/variants_enformer", "results/gnomad/all/defined/128"]),
    threads:
        workflow.cores
    run:
        import h5py

        cols = ["chrom", "pos", "ref", "alt"]
        V = load_dataset_from_file_or_dir(wildcards["dataset"]).to_pandas()
        print(V)

        coords = pd.read_parquet(input.coords)
        print(coords)
        
        labels = [f"feature_{i}" for i in range(5313)]

        all_res = []

        for chrom, data_path in tqdm(zip(enformer_chroms, input.data)):
            coords_c = coords.merge(V.loc[V.chrom==chrom, cols], on=cols, how="inner")
            print(coords_c)
            f = h5py.File(data_path)
            #data = f["SAD"][:]  # load into memory
            data = f["SAD"]  # do not load into memory
            data = pd.DataFrame(data[coords_c.idx.values].astype(float), columns=labels)
            res  = pd.concat([coords_c, data], axis=1)
            print(res)
            recoded = (res.ref != f["ref"][res.idx.values].astype(str)).values
            print(recoded.mean())
            res.loc[recoded, data.columns] *= -1
            print(res)
            all_res.append(res)
        res = pd.concat(all_res, ignore_index=True)
        print(res)

        V = V.merge(res, on=cols, how="left")
        print(V)
        print(V["feature_0"].isna().mean())
        V[labels].to_parquet(output[0], index=False)
