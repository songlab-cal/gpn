import numpy as np


rule download_enformer_precomputed:
    output:
        temp("results/enformer/{chrom}.h5"),
    shell:
        "wget -O {output} https://storage.googleapis.com/dm-enformer/variant-scores/1000-genomes/enformer/1000G.MAF_threshold%3D0.005.{wildcards.chrom}.h5"


rule process_enformer_precomputed:
    input:
        "results/enformer/{chrom}.h5",
    output:
        temp("results/enformer/{chrom}.parquet"),
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


ruleorder: merge_enformer_precomputed > process_enformer_precomputed


rule merge_enformer_precomputed:
    input:
        expand("results/enformer/{chrom}.parquet", chrom=[str(i) for i in range(1, 23)]),
    output:
        "results/enformer/merged.parquet",
    run:
        df = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)
