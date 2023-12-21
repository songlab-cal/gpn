from liftover import get_lifter
from scipy.spatial.distance import cdist


COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")


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


def check_ref_alt(V, genome):
    V["ref_nuc"] = V.progress_apply(
        lambda v: genome.get_nuc(v.chrom, v.pos).upper(), axis=1
    )
    mask = V['ref'] != V['ref_nuc']
    V.loc[mask, ['ref', 'alt']] = V.loc[mask, ['alt', 'ref']].values
    V = V[V['ref'] == V['ref_nuc']]
    V.drop(columns=["ref_nuc"], inplace=True)
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


def match_columns(df, target, covariates):
    print("WARNING: enforcing same chrom in a naive, slow manner")
    pos = df[df[target]]
    neg = df[~df[target]]
    D = cdist(pos[covariates], neg[covariates])

    closest = []
    dists = []
    for i in tqdm(range(len(pos))):
        D[i, neg.chrom != pos.iloc[i].chrom] = np.inf  # ensure picking from same chrom
        j = np.argmin(D[i])
        closest.append(j)
        D[:, j] = np.inf  # ensure it cannot be picked up again
    return pd.concat([pos, neg.iloc[closest]])


def filter_snp(V):
    V = V[V.ref.isin(NUCLEOTIDES)]
    V = V[V.alt.isin(NUCLEOTIDES)]
    return V
