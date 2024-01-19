import bioframe as bf
from gpn.data import load_table
from liftover import get_lifter
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline


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
    all_pos = []
    all_neg_matched = []
    for chrom in tqdm(df.chrom.unique()):
        df_c = df[df.chrom == chrom]
        pos = df_c[df_c[target]]
        neg = df_c[~df_c[target]]
        D = cdist(pos[covariates], neg[covariates])

        closest = []
        for i in range(len(pos)):
            j = np.argmin(D[i])
            closest.append(j)
            D[:, j] = np.inf  # ensure it cannot be picked up again
        all_pos.append(pos)
        all_neg_matched.append(neg.iloc[closest])
    
    pos = pd.concat(all_pos, ignore_index=True)
    pos["match_group"] = np.arange(len(pos))
    neg_matched = pd.concat(all_neg_matched, ignore_index=True)
    neg_matched["match_group"] = np.arange(len(neg_matched))
    res = pd.concat([pos, neg_matched], ignore_index=True)
    res = sort_chrom_pos(res)
    return res


def filter_snp(V):
    V = V[V.ref.isin(NUCLEOTIDES)]
    V = V[V.alt.isin(NUCLEOTIDES)]
    return V


def train_predict_lr(V_train, V_test, features):
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LogisticRegressionCV(
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
            max_iter=1000,
            Cs=np.logspace(-5, 0, 10),
        ))
    ])
    clf.fit(V_train[features], V_train.label)
    linear = clf.named_steps["linear"]
    C = linear.C_
    Cs = linear.Cs_
    #if C == Cs[0] or C == Cs[-1]:
    #    raise Exception(f"{C=} {Cs[0]=} {Cs[-1]=}")
    return -clf.predict_proba(V_test[features])[:, 1]


rule get_tss:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/tss.parquet",
    run:
        annotation = load_table(input[0])
        tx = annotation.query('feature=="transcript"').copy()
        tx["gene_id"] = tx.attribute.str.extract(r'gene_id "([^;]*)";')
        tx["transcript_biotype"] = tx.attribute.str.extract(r'transcript_biotype "([^;]*)";')
        tx = tx[tx.transcript_biotype=="protein_coding"]
        tss = tx.copy()
        tss[["start", "end"]] = tss.progress_apply(
            lambda w: (w.start, w.start+1) if w.strand=="+" else (w.end-1, w.end),
            axis=1, result_type="expand"
        )
        tss = tss[["chrom", "start", "end", "gene_id"]]
        print(tss)
        tss.to_parquet(output[0], index=False)


rule get_exon:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/exon.parquet",
    run:
        annotation = load_table(input[0])
        exon = annotation.query('feature=="exon"').copy()
        exon["gene_id"] = exon.attribute.str.extract(r'gene_id "([^;]*)";')
        exon["transcript_biotype"] = exon.attribute.str.extract(r'transcript_biotype "([^;]*)";')
        exon = exon[exon.transcript_biotype=="protein_coding"]
        exon = exon[["chrom", "start", "end", "gene_id"]]
        print(exon)
        exon.to_parquet(output[0], index=False)
