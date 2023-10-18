from liftover import get_lifter


def lift_hg19_to_hg38(V):
    converter = get_lifter('hg19', 'hg38')

    def get_new_pos(v):
        try:
            res = converter[v.chrom][v.pos]
            assert len(res) == 1
            chrom, pos, strand = res[0]
            assert chrom.replace("chr", "")==v.chrom
            return pos
        except:
            return -1

    V.pos = V.apply(get_new_pos, axis=1)
    return V


def sort_chrom_pos(V):
    chrom_order = [str(i) for i in range(1, 23)] + ['X', 'Y']
    V.chrom = pd.Categorical(V.chrom, categories=chrom_order, ordered=True)
    V = V.sort_values(['chrom', 'pos'])
    V.chrom = V.chrom.astype(str)
    return V


def check_ref(V, genome):
    V = V[V.apply(lambda v: v.ref == genome.get_nuc(v.chrom, v.pos).upper(), axis=1)]
    return V


rule parquet_to_tsv:
    input:
        "{anything}.parquet",
    output:
        temp("{anything}.tsv"),
    run:
        df = pd.read_parquet(input[0], columns=["chrom", "pos", "ref", "alt", "GPN-MSA"])
        df.to_csv(output[0], sep="\t", index=False, header=False, float_format='%.2f')


rule bgzip:
    input:
        "{anything}.tsv",
    output:
        "{anything}.tsv.bgz",
    threads:
        workflow.cores
    shell:
        "bgzip -c {input} --threads {threads} > {output}"


rule tabix:
    input:
        "{anything}.tsv.bgz",
    output:
        "{anything}.tsv.bgz.tbi",
    shell:
        "tabix -s 1 -b 2 -e 2 {input}"
