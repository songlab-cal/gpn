assemblies["Assembly Name"] = assemblies["Assembly Name"].str.replace(" ", "_")
assemblies["genome_path"] = (
    "tmp/" + assemblies.index + "/ncbi_dataset/data/" + assemblies.index + "/" +
    assemblies.index + "_" + assemblies["Assembly Name"] + "_genomic.fna"
)
assemblies["annotation_path"] = (
    "tmp/" + assemblies.index + "/ncbi_dataset/data/" + assemblies.index + "/genomic.gff"
)


rule download_genome:
    output:
        "results/genome/{assembly}.fa.gz",
        "results/annotation/{assembly}.gff.gz",
    params:
        tmp_dir=directory("tmp/{assembly}"),
        genome_path=lambda wildcards: assemblies.loc[wildcards.assembly, "genome_path"],
        annotation_path=lambda wildcards: assemblies.loc[wildcards.assembly, "annotation_path"],
    shell:
        """
        mkdir -p {params.tmp_dir} && cd {params.tmp_dir} && 
        datasets download genome accession {wildcards.assembly} --include genome,gff3 \
        && unzip ncbi_dataset.zip && cd - && gzip -c {params.genome_path} > {output[0]}\
         && gzip -c {params.annotation_path} > {output[1]} && rm -r {params.tmp_dir}
        """
