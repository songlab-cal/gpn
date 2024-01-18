from gpn.data import load_fasta, load_table, Genome
from gpn.data import BigWig
import numpy as np


rule download_reference:
    output:
        "results/genome.fa.gz",
    shell:
        "wget {FASTA_URL} -O {output}"


rule download_annotation:
    output:
        "results/annotation.gtf.gz",
    shell:
        "wget {GTF_URL} -O {output}"


rule make_defined_intervals:
    input:
        "results/genome.fa.gz",
    output:
        "results/intervals/{window_size}/defined.parquet",
    threads: 2
    run:
        genome = Genome(input[0])
        genome.filter_chroms(CHROMS)
        intervals = genome.get_defined_intervals()
        intervals = filter_length(intervals, int(wildcards.window_size))
        intervals.to_parquet(output[0], index=False)


rule get_conservation_intervals:
    input:
        "results/intervals/{window_size}/defined.parquet",
        "results/conservation/{conservation}.bw",
    output:
        "results/intervals/{window_size}/defined.{conservation}.{operation}.parquet",
    threads:
        workflow.cores
    run:
        import pyBigWig

        intervals = pd.read_parquet(input[0])
        print(intervals)
        #bw = pyBigWig.open(input[1])
        window_size = int(wildcards["window_size"])
        step_size = window_size // 2
        intervals = make_windows(intervals, window_size, step_size)
        print(intervals)

        operation = wildcards["operation"]
        if operation == "mean":
            f = lambda v, bw: bw.stats(f"chr{v.chrom}", v.start, v.end, exact=True)[0]
        elif operation == "percentile-75":
            f = lambda v, bw: np.quantile(bw.values(f"chr{v.chrom}", v.start, v.end), 0.75)

        def run_operation(v):
            bw = pyBigWig.open(input[1])
            res = f(v, bw)
            bw.close()
            return res

        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True, nb_workers=threads)

        #intervals["conservation"] = intervals.progress_apply(
        intervals["conservation"] = intervals.parallel_apply(
            run_operation, axis=1,
        )
        print(intervals)
        intervals.to_parquet(output[0])


rule filter_conservation_intervals:
    input:
        "results/intervals/{window_size}/defined.{conservation}.parquet",
    output:
        "results/intervals/{window_size}/defined.{conservation}_{top_frac}_{random_frac}.parquet",
    run:
        intervals = pd.read_parquet(input[0])
        print(intervals)
        top_frac = float(wildcards["top_frac"])
        random_frac = float(wildcards["random_frac"])
        mask_top = intervals.conservation >= intervals.conservation.quantile(1-top_frac)
        top_intervals = intervals[mask_top]
        print(top_intervals)
        assert not top_intervals.conservation.isna().any()
        random_intervals = intervals[~mask_top].sample(frac=random_frac, random_state=42)
        print(random_intervals)
        res = pd.concat([top_intervals, random_intervals], ignore_index=True)
        print(res)
        #res = bf.merge(res[["chrom", "start", "end"]]).drop(columns="n_intervals")
        res = res[["chrom", "start", "end"]].drop_duplicates()
        print(res)
        res.to_parquet(output[0], index=False)


rule download_maf_multiz100way:
    output:
        "results/maf/multiz100way/{chrom}.maf",
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/maf/chr{wildcards.chrom}.maf.gz | gunzip -c > {output}"


rule download_maf_multiz470way:
    output:
        "results/maf/multiz470way/{chrom}.maf",
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz470way/maf/chr{wildcards.chrom}.maf.gz | gunzip -c > {output}"


rule download_reference_chr:
    output:
        temp("results/genome_chr.fa"),
    shell:
        "wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > {output}"


rule extract_chrom:
    input:
        "results/genome_chr.fa",
    output:
        "results/chrom/{chrom}.fa",
    shell:
        "faOneRecord {input} chr{wildcards.chrom} > {output}"


# I also have a python script for this, which uses less memory
# scripts/maf_to_fasta.py
# but it's still under development, currently assumes there's no gaps in target
# sequence
rule maf2fasta:
    input:
        "results/chrom/{chrom}.fa",
        "results/maf/{alignment}/{chrom}.maf",
    output:
        "results/maf_fasta/{alignment}/{chrom}.fa",
    threads: workflow.cores // 2
    shell:
        "maf2fasta {input} fasta > {output}"


rule make_dataset:
    input:
        "results/intervals/{window_size}/{anything}.parquet",
        "results/conservation/phyloP.bw",
        "results/conservation/phastCons.bw",
        "results/genome.fa.gz",
    output:
        expand("results/dataset/{{window_size}}/{{step_size}}/{{add_rc}}/{{anything}}/{split}.parquet", split=SPLITS),
    threads: workflow.cores
    run:
        intervals = pd.read_parquet(input[0])
        print(intervals)
        #print("Loading genome MSA...")
        #genome_msa = GenomeMSA(
        #    input[1], in_memory=False, subset_chroms=intervals.chrom.unique()
        #)
        print("Making windows...")
        if "phyloP" in wildcards.anything or "phastCons" in wildcards.anything:
            # a shortcut
            intervals["strand"] = "+"
            assert int(wildcards.step_size) == (int(wildcards.window_size) // 2)
            if wildcards.add_rc == "True":
                intervals_neg = intervals.copy()
                intervals_neg.strand = "-"
                intervals = pd.concat([intervals, intervals_neg], ignore_index=True)
        else:
            raise Exception("debug")
        #intervals = make_windows(
        #    intervals, int(wildcards.window_size), int(wildcards.step_size),
        #    wildcards.add_rc=="True",
        #)
        print(intervals)
        phyloP_obj = BigWig(input[1])
        phastCons_obj = BigWig(input[2])
        print("Getting phyloP")
        intervals["phyloP"] = intervals.progress_apply(
            lambda i: phyloP_obj.get_features("chr" + i.chrom, i.start, i.end, i.strand),
            axis=1,
        )
        print("Getting phastCons")
        intervals["phastCons"] = intervals.progress_apply(
            lambda i: phastCons_obj.get_features("chr" + i.chrom, i.start, i.end, i.strand),
            axis=1,
        )
        print("Loading genome")
        genome = Genome(input[3])
        print("Getting lowercase")
        intervals["lowercase"] = intervals.progress_apply(
            lambda i: np.char.islower(list(genome.get_seq(i.chrom, i.start, i.end, i.strand))),
            axis=1,
        )

        #print("Getting MSAs")
        ## unfortunately needs flatten to save into parquet
        #msa = genome_msa.get_msa_batch_parallel(
        #    intervals.chrom.values, intervals.start.values,
        #    intervals.end.values, intervals.strand.values, n_jobs=32,
        #)
        #print(msa.shape)
        #intervals["msa"] = pd.Series(np.split(msa, msa.shape[0]), index=range(msa.shape[0]))
        #print(intervals)
        #print(intervals.msa.iloc[0].shape)
        #raise Exception("debug")
        intervals = intervals.sample(frac=1.0, random_state=42)
        print(intervals)

        for path, split in zip(output, SPLITS):
            print(path, split)
            intervals[
                intervals.chrom.isin(SPLIT_CHROMS[split])
            ].to_parquet(path, index=False, engine="pyarrow")


rule make_msa_chrom:
    input:
        "results/maf_fasta/{alignment}/{chrom}.fa",
        "config/species/{alignment}/all.txt",
        "config/species/{alignment}/{species}.txt",
        "results/genome.fa.gz",
    output:
        temp("results/msa/{alignment}/{species}/{chrom}.npy"),
    threads: workflow.cores // 10
    run:
        MSA = load_fasta(input[0])
        # the ref should be in first position
        all_species = pd.read_csv(input[1], header=None).values.ravel()
        MSA = MSA[all_species]
        species = pd.read_csv(input[2], header=None).values.ravel()
        MSA = np.vstack(MSA.apply(
            lambda seq: np.frombuffer(seq.upper().encode("ascii"), dtype="S1")
        ))
        print(MSA.shape)
        # let's only keep non-gaps in reference
        MSA = MSA[:, MSA[0]!=b'-']
        print(MSA.shape)
        MSA = MSA[[all_species.tolist().index(s) for s in species]]
        print(MSA.shape)
        MSA = MSA.T
        print(MSA.shape)
        vocab = np.frombuffer("ACGT-".encode('ascii'), dtype="S1")
        # decision: consider all "N" and similar as "-"
        # might not be the best, some aligners have a distinction
        # between N, or unaligned, and gap
        MSA[~np.isin(MSA, vocab)] = b"-"

        # now we will add the ref genome back again (but our version, which is
        # the latest patch). This is a bit awkward...
        chrom = wildcards.chrom
        ref = np.frombuffer(
            load_fasta(input[3], subset_chroms=[chrom])[chrom].encode("ascii"), dtype="S1"
        )
        print(ref.shape)
        MSA = np.concatenate((ref[:, np.newaxis], MSA), axis=1) 
        print(MSA.shape)

        np.save(output[0], MSA)

# recommend archiving the zarr directory before rsyncing
# tar -cf all.zarr.tar all.zarr
# and then unarchive on the other side
# tar -xf all.zarr.tar && rm all.zarr.tar

rule merge_msa:
    input:
        expand("results/msa/{{alignment}}/{{species}}/{chrom}.npy", chrom=CHROMS),
    output:
        directory("results/msa/{alignment}/{species}/all.zarr"),
    threads: workflow.cores
    priority: 100
    run:
        import zarr

        z = zarr.open_group(output[0], mode='w')
        for chrom, path in zip(CHROMS, input):
            print(chrom)
            data = np.load(path)
            z.create_dataset(chrom, data=data, chunks=(512, data.shape[1]))
            print(z[chrom].info)


# this is very hacky, just for an ablation
rule msa_ablation_subset:
    input:
        "results/msa/multiz100way/89/all.zarr",
        "config/species/multiz100way/89.txt",
        "config/species/multiz100way/{subset}/{species}.txt"
    output:
        directory("results/msa/multiz100way_{subset}/{species}/all.zarr"),
    threads: workflow.cores
    run:
        import zarr

        input_species = pd.read_csv(input[1], header=None).values.ravel().tolist()
        output_species = pd.read_csv(input[2], header=None).values.ravel().tolist()
        output_idx = [0] + [1 + input_species.index(s) for s in output_species]
        print(output_idx)

        z_input = zarr.open(input[0], mode="r")
        z_output = zarr.open_group(output[0], mode='w')

        for chrom in tqdm(CHROMS):
            z_output.create_dataset(
                chrom,
                data=z_input[chrom][:, output_idx],
                chunks=(512, len(output_idx))
            )


def model_config(wildcards):
    s = wildcards.model_size
    w = int(wildcards.dataset.split("/")[0])
    a = wildcards.dataset.split("/")[0]
    n_species = int(wildcards.species)
    if s == "medium" and w <= 128:
        conf = " --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 1"
    elif s == "medium" and w == 256:
        conf = " --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2"
    elif s == "medium" and w == 512:
        conf = " --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 4"
    elif s == "medium" and w == 1024:
        conf = " --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 16"
    elif s == "small" and w <= 256:
        conf = ",num_hidden_layers=8,num_attention_heads=8,hidden_size=512,intermediate_size=2048 --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 1"
    elif s == "tiny" and w == 128:
        conf = ",num_hidden_layers=4,num_attention_heads=4,hidden_size=256,intermediate_size=1024 --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 1"
    elif s == "large" and w == 128:
        conf = ",num_hidden_layers=24,num_attention_heads=16,hidden_size=1024,intermediate_size=4096 --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --adam_epsilon 1e-4"
    else:
        raise Exception("Invalid model config")
    n_aux_features = 5 * n_species
    conf = f"--config_overrides n_aux_features={n_aux_features}" + conf
    return conf


rule train_gpn_msa:
    input:
        "results/msa/{alignment}/{species}/all.zarr",
        expand("results/dataset/{{dataset}}/{split}.parquet", split=SPLITS),
    output:
        directory("results/checkpoints/{alignment,[A-Za-z0-9_]+}/{species,[A-Za-z0-9_-]+}/{dataset}/{model_size}/{loss_weight}/{seed}/{max_steps}/{use_aux_features}/{weight_conserved}/{flip_nonconserved}"),
    params:
        model_conf=model_config,
        project_name=lambda wildcards: wildcards.dataset.replace("/", "_"),
        run_name=lambda wildcards, output: '/'.join(output[0].split("/")[2:4]) + '/' + '/'.join(output[0].split("/")[-7:]),
    threads:
        workflow.cores
    priority: 100
    shell:
        """
        WANDB_PROJECT={params.project_name} torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') -m gpn.msa.train \
        --do_train --do_eval --fp16 --report_to wandb --prediction_loss_only True \
        --dataset_name results/dataset/{wildcards.dataset} \
        --msa_path {input[0]} \
        --run_name {params.run_name} --output_dir {output} \
        --soft_masked_loss_weight_train {wildcards.loss_weight} \
        --soft_masked_loss_weight_evaluation {wildcards.loss_weight} \
        --weight_decay 0.01 \
        --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine \
        --seed {wildcards.seed} \
        --dataloader_num_workers 16 \
        --save_strategy steps --save_steps 5000 --evaluation_strategy steps \
        --eval_steps 5000 --logging_steps 5000 --max_steps {wildcards.max_steps} \
        --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end \
        --model_type GPNRoFormer {params.model_conf} \
        --use_aux_features {wildcards.use_aux_features} \
        --weight_conserved {wildcards.weight_conserved} \
        --flip_nonconserved {wildcards.flip_nonconserved} \
        --remove_unused_columns False \
        --torch_compile
        """
