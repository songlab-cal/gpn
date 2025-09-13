rule download_maf:
    output:
        "results/maf/{ref}_{msa}/{chrom}.maf",
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/{wildcards.ref}/{wildcards.msa}/maf/chr{wildcards.chrom}.maf.gz | gunzip -c > {output}"


rule download_genome_chr:
    output:
        temp("results/genome_chr/{ref}.fa"),
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/{wildcards.ref}/bigZips/{wildcards.ref}.fa.gz | gunzip -c > {output}"


rule extract_chrom:
    input:
        "results/genome_chr/{ref}.fa",
    output:
        "results/genome_chr_chrom/{ref}/{chrom}.fa",
    shell:
        "faOneRecord {input} chr{wildcards.chrom} > {output}"


rule maf2fasta:
    input:
        "results/genome_chr_chrom/{ref}/{chrom}.fa",
        "results/maf/{ref}_{msa}/{chrom}.maf",
    output:
        "results/maf_fasta/{ref}_{msa}/{chrom}.fa",
    shell:
        "maf2fasta {input} fasta > {output}"


rule download_tree:
    output:
        "results/tree/{ref}_{msa}.nh",
    params:
        lambda wildcards: wildcards.msa.replace("multiz", ""),
    shell:
        "wget -O {output} https://hgdownload.soe.ucsc.edu/goldenPath/{wildcards.ref}/{wildcards.msa}/{wildcards.ref}.{params}.nh"


rule extract_species:
    input:
        "results/tree/{ref}_{msa}.nh",
    output:
        "results/species/{ref}_{msa}.txt",
    run:
        from Bio import Phylo

        tree = Phylo.read(input[0], "newick")
        species = pd.DataFrame(index=[node.name for node in tree.get_terminals()])
        ref = wildcards.ref
        assert species.index.values[0] == config["reference_renaming"].get(ref, ref)
        species.index = species.index.map(
            lambda s: config["species_renaming"].get(s, s)
        )
        species.to_csv(output[0], header=False, columns=[])


rule make_msa_chrom:
    input:
        "results/maf_fasta/{msa}/{chrom}.fa",
        "results/species/{msa}.txt",
    output:
        temp("results/msa_chrom/{msa}/{chrom}.npy"),
    run:
        MSA = load_fasta(input[0])
        species = pd.read_csv(input[1], header=None).values.ravel()

        # need to handle the case where the MSA lacks a species
        # (didn't happen in human, mouse, drosophila but happened in chicken,
        # in some chromosomes)
        missing_species = set(species) - set(MSA.index)
        L = len(MSA.iloc[0])
        missing_seq = "-" * L
        for s in missing_species:
            MSA.loc[s] = missing_seq

        MSA = MSA[species]
        MSA = np.vstack(
            MSA.apply(
                lambda seq: np.frombuffer(seq.upper().encode("ascii"), dtype="S1")
            )
        )
        # let's only keep non-gaps in reference
        MSA = MSA[:, MSA[0] != b"-"]
        MSA = MSA.T
        vocab = np.frombuffer("ACGT-".encode("ascii"), dtype="S1")
        # decision: consider all "N" and similar as "-"
        # might not be the best, some aligners have a distinction
        # between N, or unaligned, and gap
        MSA[~np.isin(MSA, vocab)] = b"-"
        np.save(output[0], MSA)


rule merge_msa_chroms:
    input:
        lambda wildcards: expand(
            "results/msa_chrom/{{msa}}/{chrom}.npy",
            chrom=config["chroms"][wildcards.msa],
        ),
    output:
        directory("results/msa/{msa}.zarr"),
    threads: workflow.cores
    run:
        chroms = config["chroms"][wildcards.msa]
        z = zarr.open_group(output[0], mode="w")
        for chrom, path in tqdm(zip(chroms, input), total=len(chroms)):
            data = np.load(path)
            z.create_dataset(
                chrom, data=data, chunks=(config["chunk_size"], data.shape[1])
            )


rule compress_msa:
    input:
        "results/msa/{msa}.zarr",
    output:
        "results/msa/{msa}.tar.gz",
    threads: workflow.cores
    shell:
        "tar cf - {input} | pigz -p {threads} > {output}"


rule hf_upload:
    input:
        "results/msa/{msa}.tar.gz",
        "results/species/{msa}.txt",
        "results/tree/{msa}.nh",
    output:
        touch("results/upload/{msa}.done"),
    run:
        from huggingface_hub import HfApi

        api = HfApi()
        private = False
        repo_id = config["hf"]["username"] + "/" + wildcards.msa
        private = config["hf"]["private"]
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
        paths_in_repo = ["msa.tar.gz", "species.txt", "tree.nh"]
        for path, path_in_repo in zip(input, paths_in_repo):
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )


# slightly different url format
rule download_tree_dm6_multiz124way:
    output:
        "results/tree/dm6_multiz124way.nh",
    shell:
        "wget -O {output} https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz124way/dm6.124way.sequenceNames.nh"


ruleorder: download_tree_dm6_multiz124way > download_tree


# slightly different url format
rule download_maf_ce11_multiz135way:
    output:
        "results/maf/ce11_multiz135way/{chrom}.maf",
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/ce11/multiz135way/chr{wildcards.chrom}.maf.gz | gunzip -c > {output}"


ruleorder: download_maf_ce11_multiz135way > download_maf


# arabidopsis has a lot of custom code since it's not from UCSC Genome Browser
# but from PlantRegMap


rule download_genome_maf_arabidopsis:
    output:
        "results/genome_maf/tair10_multiz18way.maf",
    shell:
        """
        wget -O - https://plantregmap.gao-lab.org/download_ftp.php?filepath=08-download/Arabidopsis_thaliana/multiple_alignments/Ath.maf.gz | \
        gunzip -c > {output}
        """


rule split_genome_maf_arabidopsis:
    input:
        "results/genome_maf/tair10_multiz18way.maf",
    output:
        expand(
            "results/maf/tair10_multiz18way/Chr{chrom}.maf",
            chrom=config["chroms"]["tair10_multiz18way"],
        ),
    params:
        "results/maf/tair10_multiz18way/",
    shell:
        "mkdir -p {params} && mafSplit /dev/null {params} {input} -byTarget -useFullSequenceName"


rule rename_chrom_arabidopsis:
    input:
        "results/maf/tair10_multiz18way/Chr{chrom}.maf",
    output:
        "results/maf/tair10_multiz18way/{chrom}.maf",
    shell:
        "cp {input} {output}"


ruleorder: rename_chrom_arabidopsis > download_maf


rule download_chrom_arabidopsis:
    output:
        "results/genome_chr_chrom/tair10/{chrom}.fa",
    shell:
        """wget -O - https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna_sm.chromosome.{wildcards.chrom}.fa.gz | \
        gunzip -c | sed -E 's/^>([0-9]+)/>Chr\\1/' | sed '/^>/! s/[^acgtACGT]/N/g' > {output}
        """


ruleorder: download_chrom_arabidopsis > extract_chrom


# haven't been able to download the tree used in PlantRegMap,
# authors said they don't have it anymore
# this was done by ourselves with phyloFit; visually matches the tree in PlantRegMap Fig 2B
rule download_tree_arabidopsis:
    input:
        "config/tair10_multiz18way.nh",
    output:
        "results/tree/tair10_multiz18way.nh",
    shell:
        "cp {input} {output}"


ruleorder: download_tree_arabidopsis > download_tree
