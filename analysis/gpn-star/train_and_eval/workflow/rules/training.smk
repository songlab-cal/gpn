from gpn.data import load_fasta, load_table, Genome
from gpn.data import BigWig
import numpy as np
import gc
import zarr
from Bio import Phylo
import networkx as nx


rule download_reference:
    output:
        "results/genome/{genome}.fa.gz",
    params:
        FASTA_URL=lambda wildcards: FASTA_URLS[wildcards.genome],
    shell:
        "wget {params.FASTA_URL} -O {output}"


rule make_defined_intervals:
    input:
        "results/genome/{genome}.fa.gz",
    output:
        "results/intervals/{genome}/{window_size}/defined.parquet",
    threads: 2
    run:
        genome = Genome(input[0])
        genome.filter_chroms(CHROMS)
        intervals = genome.get_defined_intervals()
        intervals = filter_length(intervals, int(wildcards.window_size))
        intervals.to_parquet(output[0], index=False)


rule get_conservation_intervals:
    input:
        "results/intervals/{genome}/{window_size}/defined.parquet",
        "results/conservation/{genome}/{conservation}.bw",
    output:
        "results/intervals/{genome}/{window_size}/defined.{conservation}.{operation}.parquet",
    run:
        import pyBigWig

        intervals = pd.read_parquet(input[0])
        print(intervals)
        bw = pyBigWig.open(input[1])
        window_size = int(wildcards["window_size"])
        step_size = window_size // 2
        intervals = make_windows(intervals, window_size, step_size)
        print(intervals)

        operation = wildcards["operation"]


        def run_operation(v):
            if operation == "mean":
                return bw.stats(f"chr{v.chrom}", v.start, v.end, exact=True)[0]
            elif operation == "percentile-75":
                return np.quantile(bw.values(f"chr{v.chrom}", v.start, v.end), 0.75)


        intervals["conservation"] = intervals.progress_apply(
            run_operation,
            axis=1,
        )
        print(intervals)
        intervals.to_parquet(output[0])


rule filter_conservation_intervals:
    input:
        "results/intervals/{genome}/{window_size}/defined.{conservation}.{operation}.parquet",
    output:
        "results/intervals/{genome}/{window_size}/defined.{conservation}.{operation}_{top_frac}_{random_frac}.parquet",
    run:
        intervals = pd.read_parquet(input[0])
        print(intervals)
        top_frac = float(wildcards["top_frac"])
        random_frac = float(wildcards["random_frac"])
        mask_top = intervals.conservation >= intervals.conservation.quantile(
            1 - top_frac
        )
        top_intervals = intervals[mask_top]
        print(top_intervals)
        assert not top_intervals.conservation.isna().any()
        random_intervals = intervals[~mask_top].sample(
            frac=random_frac, random_state=42
        )
        print(random_intervals)
        res = pd.concat([top_intervals, random_intervals], ignore_index=True)
        print(res)
        # res = bf.merge(res[["chrom", "start", "end"]]).drop(columns="n_intervals")
        res = res[["chrom", "start", "end"]].drop_duplicates()
        print(res)
        res.to_parquet(output[0], index=False)


def get_phylop_bw(wc):
    """Return path to the .bw file based on a lookup table."""
    return (
        f"results/conservation/{wc.genome}/{PHYLOP_PHASTCONS_GROUP[wc.conservation]}.bw"
    )


rule make_dataset:
    input:
        "results/intervals/{genome}/{window_size}/defined.{conservation}.{operation}_{top_frac}_{random_frac}.parquet",
        get_phylop_bw,
        "results/conservation/{genome}/{conservation}.bw",
        "results/genome/{genome}.fa.gz",
    output:
        expand(
            "results/dataset/{{genome}}/{{window_size}}/{{step_size}}/{{add_rc}}/defined.{{conservation}}.{{operation}}_{{top_frac}}_{{random_frac}}/{split}.parquet",
            split=SPLITS,
        ),
    threads: workflow.cores
    run:
        intervals = pd.read_parquet(input[0])
        print(intervals)

        print("Making windows...")
        intervals["strand"] = "+"
        assert int(wildcards.step_size) == (int(wildcards.window_size) // 2)
        if wildcards.add_rc == "True":
            intervals_neg = intervals.copy()
            intervals_neg.strand = "-"
            intervals = pd.concat([intervals, intervals_neg], ignore_index=True)

        print(intervals)
        phyloP_obj = BigWig(input[1])
        phastCons_obj = BigWig(input[2])
        print("Getting phyloP")
        intervals["phyloP"] = intervals.progress_apply(
            lambda i: phyloP_obj.get_features(
                "chr" + i.chrom, i.start, i.end, i.strand
            ),
            axis=1,
        )
        print("Getting phastCons")
        intervals["phastCons"] = intervals.progress_apply(
            lambda i: phastCons_obj.get_features(
                "chr" + i.chrom, i.start, i.end, i.strand
            ),
            axis=1,
        )
        print("Loading genome")
        genome = Genome(input[3])
        print("Getting lowercase")
        intervals["lowercase"] = intervals.progress_apply(
            lambda i: np.char.islower(
                list(genome.get_seq(i.chrom, i.start, i.end, i.strand))
            ),
            axis=1,
        )

        intervals = intervals.sample(frac=1.0, random_state=42)
        print(intervals)

        for path, split in zip(output, SPLITS):
            print(path, split)
            intervals[intervals.chrom.isin(SPLIT_CHROMS[split])].to_parquet(
                path, index=False, engine="pyarrow"
            )


rule compute_phylo_dist:
    input:
        "results/phylo_info/{genome}/{alignment}/{species}/phylo_tree.nh",
    output:
        directory(
            "results/phylo_info/{genome}/{alignment}/{species}/phylo_dist/{clade_thres}"
        ),
    run:
        def cluster_clades(phylo_dist_pairwise, threshold):
            N = phylo_dist_pairwise.shape[0]
            G = nx.Graph()
            G.add_nodes_from(range(N))
            for i in range(N):
                for j in range(i + 1, N):
                    if phylo_dist_pairwise[i, j] <= threshold:
                        G.add_edge(i, j)
            clade_dict = {
                i: nodes for i, nodes in enumerate(list(nx.connected_components(G)))
            }
            return clade_dict


        phylo_tree = Phylo.read(input[0], "newick")
        # Get pairwise phylo distance
        leaves = phylo_tree.get_terminals()
        print("Computing pairwise phylogenetic distances...")
        phylo_dist_pairwise = np.array(
            [
                [phylo_tree.distance(leaf1, leaf2) for leaf2 in leaves]
                for leaf1 in tqdm(leaves)
            ]
        )

        clade_dict = cluster_clades(phylo_dist_pairwise, float(wildcards.clade_thres))

        in_clade_phylo_dist = np.zeros(phylo_dist_pairwise.shape[0])
        leaves = [node for node in phylo_tree.get_terminals()]
        for clade_id, species in clade_dict.items():
            leaves_in_clade = [leaves[i] for i in list(species)]
            clade_mcra = phylo_tree.common_ancestor(leaves_in_clade)
            dist_to_mcra = [
                phylo_tree.distance(leaf, clade_mcra) for leaf in leaves_in_clade
            ]
            for s, d in zip(list(species), dist_to_mcra):
                in_clade_phylo_dist[s] = d
        os.makedirs(output[0], exist_ok=True)
        np.save(output[0] + "/pairwise.npy", phylo_dist_pairwise)
        np.save(output[0] + "/in_clade.npy", in_clade_phylo_dist)


def model_config(wildcards, output):
    s = wildcards.model_size
    w = int(wildcards.dataset.split("/")[0])
    a = wildcards.dataset.split("/")[0]
    n_species = int(wildcards.species)
    time_enc = wildcards.time_enc
    clade_thres = wildcards.clade_thres

    if s == "small" and w == 128:
        conf = ",num_hidden_layers=8,num_attention_heads=8,hidden_size=512,intermediate_size=2048 --per_device_train_batch_size 64 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1"
    elif s == "small" and w == 256:
        conf = ",num_hidden_layers=8,num_attention_heads=8,hidden_size=512,intermediate_size=2048 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1"
    elif s == "small" and w == 512:
        conf = ",num_hidden_layers=8,num_attention_heads=8,hidden_size=512,intermediate_size=2048 --per_device_train_batch_size 16 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1"
    elif s == "medium" and w == 128:
        conf = " --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --gradient_accumulation_steps 2"
    elif s == "medium" and w == 256:
        conf = " --per_device_train_batch_size 16 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2"
    elif s == "medium" and w == 512:
        conf = " --per_device_train_batch_size 4 --per_device_eval_batch_size 2 --gradient_accumulation_steps 4"
    elif s == "large" and w == 128:
        conf = ",num_hidden_layers=16,num_attention_heads=16,hidden_size=1024,intermediate_size=4096 --per_device_train_batch_size 16 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4"
    elif s == "large" and w == 256:
        conf = ",num_hidden_layers=16,num_attention_heads=16,hidden_size=1024,intermediate_size=4096 --per_device_train_batch_size 8 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4"
    else:
        raise Exception("Invalid model config")

    if s == "large":
        conf = (
            f"--learning_rate 5e-5 --config_overrides time_enc={time_enc},clade_thres={clade_thres}"
            + conf
        )
    else:
        conf = (
            f"--learning_rate 1e-4 --config_overrides time_enc={time_enc},clade_thres={clade_thres}"
            + conf
        )
    return conf


rule train_gpn_star:
    input:
        "results/msa/{genome}/{alignment}/{species}",
        "results/phylo_info/{genome}/{alignment}/{species}/phylo_dist/{clade_thres}",
        expand("results/dataset/{{genome}}/{{dataset}}/{split}.parquet", split=SPLITS),
    output:
        directory(
            "results/checkpoints/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{dataset}/{model_size}/{loss_weight}/{seed}/{max_steps}/{use_aux_features}/{weight_conserved}/{flip_nonconserved}"
        ),
    wildcard_constraints:
        genome="[A-Za-z0-9_-]+",
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
    params:
        model_conf=model_config,
        project_name=lambda wildcards: wildcards.dataset.replace("/", "_"),
        run_name=lambda wildcards, output: "/".join(output[0].split("/")[2:]),
    threads: workflow.cores
    priority: 100
    shell:
        """
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')
        num_cpus={threads}
        dataloader_num_workers=$(($num_cpus / $num_gpus))
        
        WANDB_PROJECT={wildcards.genome} torchrun --nproc_per_node=$num_gpus --master_port=25678 -m gpn.star.train \
        --do_train --do_eval --fp16 --prediction_loss_only True \
        --dataset_name results/dataset/{wildcards.genome}/{wildcards.dataset} \
        --msa_path {input[0]} \
        --phylo_dist_path {input[1]} \
        --run_name {params.run_name} --output_dir {output} \
        --soft_masked_loss_weight_train {wildcards.loss_weight} \
        --soft_masked_loss_weight_evaluation {wildcards.loss_weight} \
        --weight_decay 0.01 \
        --optim adamw_torch --lr_scheduler_type cosine \
        --seed {wildcards.seed} \
        --dataloader_num_workers $dataloader_num_workers \
        --save_strategy steps --save_steps 5000 --evaluation_strategy steps \
        --eval_steps 5000 --logging_steps 1000 --max_steps {wildcards.max_steps} \
        --warmup_steps 2500 --save_total_limit 1 --load_best_model_at_end \
        --model_type GPNStar {params.model_conf} \
        --use_aux_features {wildcards.use_aux_features} \
        --weight_conserved {wildcards.weight_conserved} \
        --flip_nonconserved {wildcards.flip_nonconserved} \
        --remove_unused_columns False \
        --torch_compile
        """
