rule get_ancestral_repeats:
    output:
        "results/calibration/{genome}/ancestral_repeats.bed",
    params:
        use_ancestral_repeats=CALIBRATION_CONFIGS["use_ancestral_repeats"],
        target=CALIBRATION_CONFIGS["target_repeats"],
        outgroup=CALIBRATION_CONFIGS["outgroup_repeats"],
        chain=CALIBRATION_CONFIGS["chain_file"],
    run:
        if params.use_ancestral_repeats:
            shell(
                """
            wget -O results/calibration/{wildcards.genome}/rmsk_target.txt.gz {params.target}
            wget -O results/calibration/{wildcards.genome}/rmsk_outgroup.txt.gz {params.outgroup}

            gunzip -c results/calibration/{wildcards.genome}/rmsk_target.txt.gz | awk 'BEGIN{{OFS="\t"}}{{print $6,$7,$8,$11,$2,$13}}' \
            > results/calibration/{wildcards.genome}/target.rmsk.bed
            gunzip -c results/calibration/{wildcards.genome}/rmsk_outgroup.txt.gz | awk 'BEGIN{{OFS="\t"}}{{print $6,$7,$8,$11,$2,$13}}' \
            > results/calibration/{wildcards.genome}/outgroup.rmsk.bed

            # Grab the reciprocal-best chain file
            wget -O results/calibration/{wildcards.genome}/rbest.chain.gz {params.chain}
            gunzip -f results/calibration/{wildcards.genome}/rbest.chain.gz

            # LiftOver
            liftOver results/calibration/{wildcards.genome}/outgroup.rmsk.bed results/calibration/{wildcards.genome}/rbest.chain \
            results/calibration/{wildcards.genome}/outgroup_in_target.bed \
            results/calibration/{wildcards.genome}/outgroup_unmapped.bed

            bedtools intersect -u \
            -a results/calibration/{wildcards.genome}/target.rmsk.bed \
            -b results/calibration/{wildcards.genome}/outgroup_in_target.bed \
            | awk '$5!="Simple_repeat" && $5!="Low_complexity"' \
            > {output}
            """
            )
        else:
            shell("touch {output}")


def get_phylop_calibration(wc):
    """Return path to the .bw file based on a lookup table."""
    return f"results/conservation/{wc.genome}/{CALIBRATION_CONFIGS['phylop']}.bw"


def get_phastcons_calibration(wc):
    """Return path to the .bw file based on a lookup table."""
    return f"results/conservation/{wc.genome}/{CALIBRATION_CONFIGS['phastcons']}.bw"


rule get_conserved_sites:
    input:
        get_phylop_calibration,
        get_phastcons_calibration,
    output:
        "results/calibration/{genome}/conserved_sites.bed",
    run:
        import pyBigWig
        import numpy as np
        from pathlib import Path
        from tqdm import tqdm


        def contiguous_runs(mask: np.ndarray):
            """
    Given a boolean array, yield (start, end) pairs for each run of True.
    Coordinates are 0-based, half-open within the array.
    """
            if not mask.any():
                return
                # diff converts start->1, end->-1
            diff = np.diff(mask.astype(np.int8))
            run_starts = np.where(diff == 1)[0] + 1
            run_ends = np.where(diff == -1)[0] + 1
            if mask[0]:  # run starts at 0
                run_starts = np.r_[0, run_starts]
            if mask[-1]:  # run extends to array end
                run_ends = np.r_[run_ends, mask.size]
            for s, e in zip(run_starts, run_ends):
                yield int(s), int(e)


        phylo = pyBigWig.open(input[0])
        phastcons = pyBigWig.open(input[1])

        window_size = 1_000_000

        # Set threshold based on calibration config
        phylop_threshold = 0.1 if CALIBRATION_CONFIGS["use_ancestral_repeats"] else 0.05

        # Calculate total number of windows for progress tracking
        total_windows = 0
        for chrom, length in phylo.chroms().items():
            total_windows += (length + window_size - 1) // window_size

        with open(output[0], "w") as out_fh:
            with tqdm(
                total=total_windows, desc="Processing windows", unit="windows"
            ) as pbar:
                for chrom, length in phylo.chroms().items():
                    # handle length mismatch by using the shorter length
                    phastcons_length = phastcons.chroms().get(chrom, None)
                    if phastcons_length is None:
                        continue  # skip chromosomes not in phastcons track
                    if phastcons_length != length:
                        length = min(
                            length, phastcons_length
                        )  # use overlapping region only

                        # iterate through the chromosome in fixed-size windows
                    for win_start in range(0, length, window_size):
                        win_end = min(win_start + window_size, length)
                        v1 = np.array(
                            phylo.values(chrom, win_start, win_end, numpy=True)
                        )
                        v2 = np.array(
                            phastcons.values(chrom, win_start, win_end, numpy=True)
                        )

                        # pyBigWig returns np.nan for absent data; treat those as not-zero
                        mask = (np.abs(v1) < phylop_threshold) & (v2 == 0)

                        # emit merged regions
                        for s, e in contiguous_runs(mask):
                            out_fh.write(f"{chrom}\t{win_start+ s}\t{win_start+ e}\n")

                        pbar.update(1)
                        pbar.set_postfix(chrom=chrom, pos=f"{win_start:,}")
        out_fh.close()
        phylo.close()
        phastcons.close()


rule get_neutral_calibration_dataset:
    input:
        "results/calibration/{genome}/ancestral_repeats.bed",
        "results/calibration/{genome}/conserved_sites.bed",
        "results/genome/{genome}.fa.gz",
    output:
        "results/calibration/{genome}/calibration_dataset/test.parquet",
    run:
        import pyranges as pr
        from gpn.star.data import Genome

        genome = Genome(input[2])
        bed_conserved = pr.read_bed(input[1])
        bed_conserved.Chromosome = bed_conserved.Chromosome.astype(str).str.replace(
            "chr", ""
        )

        # Check if ancestral repeats file is empty
        try:
            bed_ar = pr.read_bed(input[0])
            if len(bed_ar) == 0:
                bed_neutral = bed_conserved
            else:
                bed_ar.Chromosome = bed_ar.Chromosome.astype(str).str.replace("chr", "")
                bed_neutral = bed_ar.intersect(bed_conserved)
        except:
            # If file is empty or cannot be read, use only conserved sites
            bed_neutral = bed_conserved

        neutral_sites = []
        for _, row in tqdm(
            bed_neutral.df.iterrows(),
            total=len(bed_neutral.df),
            desc="Processing intervals",
        ):
            chrom = row["Chromosome"]
            if chrom not in CHROMS:
                continue
            start = row["Start"]
            end = row["End"]
            # Generate all positions in the interval (convert 0-based to 1-based)
            for pos in range(start, end):
                ref = genome._genome[chrom][pos].upper()
                neutral_sites.append({"chrom": chrom, "pos": pos + 1, "ref": ref})
        neutral_df = pd.DataFrame(neutral_sites)
        print(neutral_df.shape)

        valid_nucleotides = {"A", "C", "G", "T"}
        valid_mask = neutral_df["ref"].isin(valid_nucleotides)
        print(valid_mask.sum())
        neutral_df = neutral_df[valid_mask].reset_index(drop=True)
        print(neutral_df.shape)

        neutral_df.to_parquet(output[0])


rule get_neutral_calibration_scores:
    input:
        "results/calibration/{genome}/calibration_dataset/test.parquet",
        "results/logits/results/calibration/{genome}/calibration_dataset/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}.parquet",
        "results/genome/{genome}.fa.gz",
    output:
        "results/calibration/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}/entropy.parquet",
        "results/calibration/{genome}/{time_enc}/{clade_thres}/{alignment}/{species}/{window_size}/{model}/llr.parquet",
    wildcard_constraints:
        time_enc="[A-Za-z0-9_-]+",
        clade_thres="[0-9.-]+",
        alignment="[A-Za-z0-9_]+",
        species="[A-Za-z0-9_-]+",
        window_size="\d+",
    run:
        V = pd.read_parquet(input[0])
        logits = pd.read_parquet(input[1])
        normalized_logits = normalize_logits(logits)
        genome = Genome(input[2])

        # Entropy
        V["pentanuc"] = V.apply(
            lambda row: genome.get_seq(
                row["chrom"], row["pos"] - 3, row["pos"] + 2
            ).upper(),
            axis=1,
        )
        V["entropy"] = get_entropy(normalized_logits)
        df_calibration_entropy = V.groupby("pentanuc")["entropy"].mean().reset_index()
        df_calibration_entropy.columns = ["pentanuc", "entropy_neutral_mean"]
        df_calibration_entropy.to_parquet(output[0])

        # LLR
        # Expand V to include all possible mutations
        nucleotides = ["A", "C", "G", "T"]
        nuc_to_idx = {nuc: idx for idx, nuc in enumerate(nucleotides)}

        # Create alt allele arrays for each nucleotide efficiently
        alt_matrix = np.array(
            [
                [1, 2, 3],  # For A (idx 0): C, G, T
                [0, 2, 3],  # For C (idx 1): A, G, T
                [0, 1, 3],  # For G (idx 2): A, C, T
                [0, 1, 2],  # For T (idx 3): A, C, G
            ]
        )

        # Get ref indices and expand to all mutations
        ref_indices = V["ref"].map(nuc_to_idx).values
        alt_indices_for_refs = alt_matrix[ref_indices]

        # Convert back to nucleotides
        idx_to_nuc = {idx: nuc for nuc, idx in nuc_to_idx.items()}
        alt_nucleotides = np.vectorize(idx_to_nuc.get)(alt_indices_for_refs)

        # Expand the dataframe: repeat each row 3 times
        n_rows = len(V)
        expanded_indices = np.repeat(np.arange(n_rows), 3)
        V_expanded = V.iloc[expanded_indices].reset_index(drop=True)

        # Add alt column
        alt_column = alt_nucleotides.flatten()
        V_expanded["alt"] = alt_column

        # Expand logits to match
        logits_expanded = normalized_logits.iloc[expanded_indices].reset_index(
            drop=True
        )

        # Calculate LLR for all mutations
        ref_indices_expanded = np.repeat(ref_indices, 3)
        alt_indices_expanded = alt_nucleotides.flatten()
        alt_indices_expanded = np.array(
            [nuc_to_idx[alt] for alt in alt_indices_expanded]
        )

        # Get LLR using vectorized operations
        logits_array = logits_expanded.values
        ref_logits = logits_array[
            np.arange(len(ref_indices_expanded)), ref_indices_expanded
        ]
        alt_logits = logits_array[
            np.arange(len(alt_indices_expanded)), alt_indices_expanded
        ]
        V_expanded["llr"] = alt_logits - ref_logits

        # Create pentanuc_mut column and calculate calibration
        V_expanded["pentanuc_mut"] = V_expanded["pentanuc"] + "_" + V_expanded["alt"]
        df_calibration_llr = (
            V_expanded.groupby("pentanuc_mut")["llr"].mean().reset_index()
        )
        df_calibration_llr.columns = ["pentanuc_mut", "llr_neutral_mean"]
        df_calibration_llr.to_parquet(output[1])
