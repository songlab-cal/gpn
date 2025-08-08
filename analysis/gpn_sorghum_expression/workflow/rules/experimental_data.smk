rule experimental_data_process:
    input:
        "results/experimental_data/raw/{gene}_betas.csv.gz",
    output:
        "results/experimental_data/processed/{gene}.parquet",
    wildcard_constraints:
        gene="|".join(experimental_data_genes),
    run:
        n_upstream = 256
        n_downstream = 256

        V = pd.read_csv(input[0], usecols=["mut", "beta", "pval", "n"])
        V = V[V.mut != "wt"]
        V[['ref', 'pos', 'alt']] = V['mut'].str.extract(r'([A-Z])(-?\d+)([A-Z_]*)')
        V.drop(columns=["mut"], inplace=True)
        V.pos = V.pos.astype(int)

        def get_mut_type(v):
            if v["ref"] == "I":
                return "insertion"
            elif set(v["alt"]) == {"_"}:
                return "deletion"
            else:
                return "substitution"

        V["mut_type"] = V.apply(get_mut_type, axis=1)

        def filter_pos(v):
            res = v.pos > -n_upstream and v.pos < 0
            if v["mut_type"] == "insertion":
                res = res and (v.pos + len(v["alt"]) < 0)
            elif v["mut_type"] == "deletion":
                res = res and (v.pos + len(v["alt"]) < 0)
            return res

        V = V[V.apply(filter_pos, axis=1)]

        print(V)

        annotation = config["experimental_data_gene_annotation"][wildcards.gene]
        chrom = annotation["chrom"]
        tss = annotation["tss"]
        strand = annotation["strand"]
        genome = Fasta(annotation["genome"])
        buffer = V[V.mut_type == "deletion"].alt.str.len().max()

        if strand == "+":
            start = tss - n_upstream - buffer
            end = tss + n_downstream
            seq = str(genome[chrom][start:end])
        else:
            start = tss - n_downstream
            end = tss + n_upstream + buffer
            seq = str(genome[chrom][start:end].reverse.complement)

        ref_seq = seq[buffer:]
        left_pad = seq[:buffer]

        center = n_upstream

        V["ref2"] = V.pos.apply(lambda x: ref_seq[center + x])
        print("start warning")
        print(V[(V.mut_type != "insertion") & (V.ref != V.ref2)])
        print("end warning")
        V = V[(V.mut_type == "insertion") | (V.ref == V.ref2)]
        print(V)

        def get_seq(v):
            i = center + v.pos
            assert ref_seq[i] == v.ref2
            if v.mut_type == "substitution":
                return ref_seq[:i] + v.alt + ref_seq[i+1:]
            elif v.mut_type == "insertion":
                res = ref_seq[:i] + v.alt + ref_seq[i:]
                res = res[-(n_upstream+n_downstream):]
                return res
            elif v.mut_type == "deletion":
                deletion_length = len(v.alt)
                return left_pad[-deletion_length:] + ref_seq[:i] + ref_seq[i+deletion_length:]

        V["seq"] = V.apply(get_seq, axis=1)
        assert (V.seq.str.len() == (n_upstream+n_downstream)).all()
        assert (V.seq.str[center:center+3] == annotation["center_seq"]).all()

        print(V)
        new_row = pd.DataFrame([{"seq": ref_seq}])
        V = pd.concat([new_row, V], ignore_index=True)
        print(V)
        V.to_parquet(output[0], index=False)


rule experimental_data_predict:
    input:
        "results/experimental_data/processed/{gene}.parquet",
        "results/{checkpoint}",
    output:
        "results/experimental_data/preds/{gene}/{checkpoint}.parquet",
    wildcard_constraints:
        gene="|".join(extended_experimental_data_genes),
    threads:
        workflow.cores
    run:
        gene = wildcards.gene.replace("_ISM", "")

        V = pd.read_parquet(input[0])
        ref_seq = V.seq.iloc[0]
        V = V.iloc[1:].reset_index(drop=True)

        track = config["experimental_data_predict_track"][gene]
        dataset_name = config["finetuning"]["dataset_name"]
        tracks = pd.read_csv(f"hf://datasets/{dataset_name}/labels.txt", header=None).values.ravel().tolist()
        track_index = tracks.index(track)

        lfc = run_prediction_lfc(
            ref_seq, V.seq.tolist(), input[1], track_index, threads=threads,
        )
        V.drop(columns="seq", inplace=True)
        V["pred_lfc"] = lfc
        print(V)
        V.to_parquet(output[0], index=False)


# TODO: remove this rule as it is ambiguous and just for quick testing
#rule experimental_data_predict_merge_seeds:
#    input:
#        "results/experimental_data/preds/{gene}/checkpoints_epoch/GPN_Brassicales/30_epochs/42/checkpoint-{checkpoint}.parquet",
#        "results/experimental_data/preds/{gene}/checkpoints_epoch/GPN_Brassicales/30_epochs/43/checkpoint-{checkpoint}.parquet",
#        "results/experimental_data/preds/{gene}/checkpoints_epoch/GPN_Brassicales/30_epochs/44/checkpoint-{checkpoint}.parquet",
#        "results/experimental_data/preds/{gene}/checkpoints_epoch/GPN_Brassicales/30_epochs/45/checkpoint-{checkpoint}.parquet",
#        "results/experimental_data/preds/{gene}/checkpoints_epoch/GPN_Brassicales/30_epochs/46/checkpoint-{checkpoint}.parquet",
#    output:
#        "results/experimental_data/preds/{gene}/checkpoints_epoch/GPN_Brassicales/30_epochs/merged/checkpoint-{checkpoint}.parquet",
#    wildcard_constraints:
#        gene="|".join(extended_experimental_data_genes),
#    run:
#        x1 = pd.read_parquet(input[0]).pred_lfc.values
#        x2 = pd.read_parquet(input[1]).pred_lfc.values
#        x3 = pd.read_parquet(input[2]).pred_lfc.values
#        x4 = pd.read_parquet(input[3]).pred_lfc.values
#        x5 = pd.read_parquet(input[4]).pred_lfc.values
#        x = (x1 + x2 + x3 + x4 + x5) / 5
#        V = pd.read_parquet(input[0])
#        V["pred_lfc"] = x
#        V.to_parquet(output[0], index=False)


rule experimental_data_eval:
    input:
        "results/experimental_data/preds/{gene}/{checkpoint}.parquet",
    output:
        "results/experimental_data/metrics/{gene}/{checkpoint}.csv",
    wildcard_constraints:
        gene="|".join(experimental_data_genes),
    run:
        V = pd.read_parquet(input[0])
        V = V[V.n >= 5]
        all_subsets = [
            "all",
            "first_half",
            "second_half",
            "more_than_50",
            "less_than_50",
        ]
        
        all_variant_types = [
            "all",
            "substitution",
            "insertion",
            "deletion",
            "12bp_deletion",
        ]
        res = []
        for subset in all_subsets:
            if subset == "all":
                V2 = V
            elif subset == "first_half":
                V2 = V[V.pos < -128]
            elif subset == "second_half":
                V2 = V[V.pos >= -128]
            elif subset == "more_than_50":
                V2 = V[V.pos < -50]
            elif subset == "less_than_50":
                V2 = V[V.pos >= -50]
            else:
                raise ValueError(f"Unknown subset: {subset}")
            for variant_type in all_variant_types:
                if variant_type == "all":
                    V3 = V2
                elif variant_type == "substitution":
                    V3 = V2[V2.mut_type == "substitution"]
                elif variant_type == "insertion":
                    V3 = V2[V2.mut_type == "insertion"]
                elif variant_type == "deletion":
                    V3 = V2[V2.mut_type == "deletion"]
                elif variant_type == "12bp_deletion":
                    V3 = V2[(V2.mut_type == "deletion") & (V2.alt.str.len() == 12)]
                else:
                    raise ValueError(f"Unknown variant type: {variant_type}")

                if len(V3) > 0:
                    res.append([
                        subset,
                        variant_type,
                        len(V3),
                        V3.pred_lfc.corr(V3.beta),
                        V3.pred_lfc.corr(V3.beta, method="spearman"),
                    ])
        res = pd.DataFrame(
            res, columns=["subset", "variant_type", "n_variants", "Pearson", "Spearman"]
        )
        print(res)
        res.to_csv(output[0], index=False)


rule experimental_data_ISM:
    output:
        "results/experimental_data/processed/{gene}_ISM.parquet",
    wildcard_constraints:
        gene="|".join(experimental_data_genes),
    run:
        n_upstream = 256
        n_downstream = 256

        annotation = config["experimental_data_gene_annotation"][wildcards.gene]
        chrom = annotation["chrom"]
        tss = annotation["tss"]
        strand = annotation["strand"]
        genome = Fasta(annotation["genome"])

        if strand == "+":
            start = tss - n_upstream
            end = tss + n_downstream
            seq = str(genome[chrom][start:end])
        else:
            start = tss - n_downstream
            end = tss + n_upstream
            seq = str(genome[chrom][start:end].reverse.complement)

        ref_seq = seq

        center = n_upstream

        nucleotides = list("ACGT")

        V = []
        for pos in range(-n_upstream+1, 0):
            ref = ref_seq[center + pos]
            for alt in nucleotides:
                if alt == ref:
                    continue
                V.append([chrom, pos, ref, alt])
        V = pd.DataFrame(V, columns=["chrom", "pos", "ref", "alt"])

        print(V)

        def get_seq(v):
            i = center + v.pos
            return ref_seq[:i] + v.alt + ref_seq[i+1:]

        V["seq"] = V.apply(get_seq, axis=1)
        assert (V.seq.str.len() == (n_upstream+n_downstream)).all()
        assert (V.seq.str[center:center+3] == annotation["center_seq"]).all()

        print(V)
        new_row = pd.DataFrame([{"seq": ref_seq}])
        V = pd.concat([new_row, V], ignore_index=True)
        print(V)
        V.to_parquet(output[0], index=False)
