import bioframe as bf
from gpn.data import load_table
from liftover import get_lifter
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


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
        if len(pos) > len(neg):
            print("WARNING: subsampling positive set to size of negative set")
            pos = pos.sample(len(neg), random_state=42)
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
        ('scaler', RobustScaler()),
        ('linear', LogisticRegressionCV(
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
            Cs=np.logspace(-5, 5, 16),
        ))
    ])
    clf.fit(V_train[features], V_train.label)
    linear = clf.named_steps["linear"]
    C = linear.C_
    Cs = linear.Cs_
    #if C == Cs[0] or C == Cs[-1]:
    #    raise Exception(f"{C=} {Cs[0]=} {Cs[-1]=}")
    return -clf.predict_proba(V_test[features])[:, 1]


def train_predict_best_feature(V_train, V_test, features):
    best_feature_idx = np.argmax([
        roc_auc_score(V_train.label, -V_train[f]) for f in features
    ])
    return V_test[features[best_feature_idx]]


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


rule subsample_variants:
    input:
        "{anything}.parquet",
    output:
        "{anything}.subsample_{n}.parquet",
    run:
        V = pl.read_parquet(input[0])
        V = V.sample(n=int(wildcards.n), seed=42).sort(["chrom", "pos", "ref", "alt"])
        print(V)
        V.write_parquet(output[0])


rule make_ensembl_vep_input:
    input:
        "{anything}.parquet",
    output:
        "{anything}.ensembl_vep.input.tsv.gz",
    threads: workflow.cores
    run:
        df = pd.read_parquet(input[0])
        df["start"] = df.pos
        df["end"] = df.start
        df["allele"] = df.ref + "/" + df.alt
        df["strand"] = "+"
        df.to_csv(
            output[0], sep="\t", header=False, index=False,
            columns=["chrom", "start", "end", "allele", "strand"],
        )


# additional snakemake args:
# --use-singularity --singularity-args "--bind /scratch/users/gbenegas"
# or in savio:
# --use-singularity --singularity-args "--bind /global/scratch/projects/fc_songlab/gbenegas"
rule install_ensembl_vep_cache:
    output:
        directory("results/ensembl_vep_cache"),
    singularity:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        "INSTALL.pl -c {output} -a cf -s homo_sapiens -y GRCh38"


rule run_ensembl_vep:
    input:
        "{anything}.ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache",
    output:
        "{anything}.ensembl_vep.output.tsv.gz",  # TODO: make temp
    singularity:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output} --fork {threads} --cache \
        --dir_cache {input[1]} --format ensembl \
        --most_severe --compress_output gzip --tab --distance 1000 --offline
        """

ruleorder: subset_to_fully_conserved_pos > get_logits
ruleorder: subset_to_fully_conserved_pos > process_logits
ruleorder: subset_to_fully_conserved_pos > get_llr

ruleorder: subsample_variants > get_logits
ruleorder: subsample_variants > process_logits
ruleorder: subsample_variants > get_llr

ruleorder: process_ensembl_vep > subset_to_fully_conserved_pos
ruleorder: process_ensembl_vep > subsample_variants
ruleorder: process_ensembl_vep > get_logits
ruleorder: process_ensembl_vep > process_logits
ruleorder: process_ensembl_vep > get_llr


rule process_ensembl_vep:
    input:
        "{anything}.parquet",
        "{anything}.ensembl_vep.output.tsv.gz",
    output:
        "{anything}.annot.parquet",
    run:
        V = pd.read_parquet(input[0])
        V2 = pd.read_csv(
            input[1], sep="\t", header=None, comment="#",
            usecols=[0, 6]
        ).rename(columns={0: "variant", 6: "consequence"})
        V2["chrom"] = V2.variant.str.split("_").str[0]
        V2["pos"] = V2.variant.str.split("_").str[1].astype(int)
        V2["ref"] = V2.variant.str.split("_").str[2].str.split("/").str[0]
        V2["alt"] = V2.variant.str.split("_").str[2].str.split("/").str[1]
        V2.drop(columns=["variant"], inplace=True)
        V = V.merge(V2, on=COORDINATES, how="inner")
        print(V)
        V.to_parquet(output[0], index=False)


rule find_conserved_pos:
    input:
        "results/msa/multiz100way/89/all.zarr",
    output:
        "results/conserved_pos/{subset}/{chrom}.parquet",
    threads:
        workflow.cores
    run:
        msa = GenomeMSA(input[0]).f[wildcards.chrom][:, 1:]
        print(msa.shape)

        not_dash = msa[:, 0] != b"-"

        if wildcards.subset == "full":
            all_equal = np.all(msa == msa[:, [0]], axis=1)
            valid_rows = np.logical_and(all_equal, not_dash)
        elif wildcards.subset == "armadillo":
            armadillo_index = 46
            valid_rows = np.logical_and(
                not_dash,
                np.logical_and(
                    np.all(msa[:, :armadillo_index+1] == msa[:, [0]], axis=1),
                    np.all(msa[:, armadillo_index+1:] == b"-", axis=1),
                )
            )

        # Step 4: Find indices
        indices = np.where(valid_rows)[0]

        pos = pd.DataFrame(dict(pos=indices+1))
        print(pos)
        pos.to_parquet(output[0], index=False)


rule subset_to_fully_conserved_pos:
    input:
        "{anything}.parquet",
        "results/conserved_pos/{subset}/{chrom}.parquet",
    output:
        "{anything}.conserved_pos_{subset}_{chrom}.parquet",
    run:
        V = pl.read_parquet(input[0])
        pos = pl.read_parquet(input[1])["pos"]
        V = V.filter(pl.col("pos").is_in(pos))
        print(V)
        V.write_parquet(output[0])
