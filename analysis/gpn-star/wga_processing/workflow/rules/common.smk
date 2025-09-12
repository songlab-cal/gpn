from gpn.data import load_fasta
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr


COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")



rule make_ensembl_vep_input:
    input:
        "{anything}.parquet",
    output:
        temp("{anything}.ensembl_vep.input.tsv.gz"),
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


# additional snakemake args (SCF):
# --use-singularity --singularity-args "--bind /scratch/users/gbenegas"
# or in savio:
# --use-singularity --singularity-args "--bind /global/scratch/projects/fc_songlab/gbenegas"
rule install_ensembl_vep_cache:
    output:
        directory("results/ensembl_vep_cache/{ref}"),
    params:
        s=lambda wildcards: config["ensembl_vep"][wildcards.ref]["s"],
        y=lambda wildcards: config["ensembl_vep"][wildcards.ref]["y"],
    singularity:
        "docker://ensemblorg/ensembl-vep:release_113.4"
    threads: workflow.cores
    shell:
        "INSTALL.pl -c {output} -a cf -s {params.s} -y {params.y}"


# cd results/ensembl_vep_cache/tair10
# wget https://ftp.ebi.ac.uk/ensemblgenomes/pub/plants/current/variation/indexed_vep_cache/arabidopsis_thaliana_vep_60_TAIR10.tar.gz
# tar xzf arabidopsis_thaliana_vep_60_TAIR10.tar.gz 

#ruleorder: install_ensembl_vep_cache_tair10 > install_ensembl_vep_cache
#
#rule install_ensembl_vep_cache_tair10:
#    output:
#        directory("results/ensembl_vep_cache/{ref,tair10}"),
#    params:
#        s=lambda wildcards: config["ensembl_vep"][wildcards.ref]["s"],
#        y=lambda wildcards: config["ensembl_vep"][wildcards.ref]["y"],
#    singularity:
#        "docker://ensemblorg/ensembl-vep:release_113.4"
#    threads: workflow.cores
#    shell:
#        "INSTALL.pl -c {output} -a cf -s {params.s} -y {params.y} --cache_version 60"


rule run_ensembl_vep:
    input:
        "results/variants/{ref}/{anything}.ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache/{ref}",
    output:
        temp("results/variants/{ref}/{anything}.ensembl_vep.output.tsv.gz"),
    params:
        s=lambda wildcards: config["ensembl_vep"][wildcards.ref]["s"],
    singularity:
        "docker://ensemblorg/ensembl-vep:release_113.4"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output} --fork {threads} --cache \
        --dir_cache {input[1]} --format ensembl --species {params.s} \
        --most_severe --compress_output gzip --tab --distance 1000 --offline
        """


ruleorder: run_ensembl_vep_tair10 > run_ensembl_vep


rule run_ensembl_vep_tair10:
    input:
        "results/variants/{ref}/{anything}.ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache/{ref}",
    output:
        temp("results/variants/{ref,tair10}/{anything}.ensembl_vep.output.tsv.gz"),
    params:
        s=lambda wildcards: config["ensembl_vep"][wildcards.ref]["s"],
    singularity:
        "docker://ensemblorg/ensembl-vep:release_113.4"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output} --fork {threads} --cache \
        --dir_cache {input[1]} --format ensembl --species {params.s} \
        --most_severe --compress_output gzip --tab --distance 1000 --offline \
        --cache_version 60
        """


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
        #V = V.merge(V2, on=COORDINATES, how="inner")
        V = V.merge(V2, on=COORDINATES, how="left")
        print(V)
        V.to_parquet(output[0], index=False)