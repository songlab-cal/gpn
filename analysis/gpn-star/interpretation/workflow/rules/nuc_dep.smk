rule interpretation_nuc_dep:
    output:
        "results/interpretation/nuc_dep/{locus}/{model}.parquet",
    params:
        lambda wc: config["nuc_dep"][wc.locus]["chrom"],
        lambda wc: config["nuc_dep"][wc.locus]["start"],
        lambda wc: config["nuc_dep"][wc.locus]["end"],
        lambda wc: config["nuc_dep"][wc.locus]["strand"],
        lambda wc: config["gpn_star"][wc.model]["model_path"],
        lambda wc: config["gpn_star"][wc.model]["msa_path"],
        lambda wc: config["gpn_star"][wc.model]["phylo_info_path"],
        lambda wc: config["gpn_star"][wc.model]["window_size"],
    threads:
        workflow.cores
    shell:
        "python workflow/scripts/nuc_dep.py {params} {output}"


rule interpretation_nuc_dep_plot:
    input:
        "results/interpretation/nuc_dep/{locus}/{model}.parquet",
    output:
        "results/plots/nuc_dep/{locus}/{model}.svg",
    run:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import seaborn as sns

        contact = pd.read_parquet(input[0])
        locus = config["nuc_dep"][wildcards.locus]
        chrom = locus["chrom"]
        start = locus["start"]
        end = locus["end"]
        strand = locus["strand"]
        squares = locus.get("squares", np.array([]))
        squares = squares - start if strand == "+" else end - squares

        plt.figure(figsize=(4, 4))
        g = sns.heatmap(
            contact,
            cmap='coolwarm',
            square=True,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            robust=True,
            rasterized=True,
        )
        coords_int   = contact.columns.astype(int)
        tick_freq = 50 if end-start > 100 else 10
        mask         = (coords_int % tick_freq == 0)
        xtick_idx    = np.where(mask)[0]
        xtick_labels = [f'{coords_int[i]:,}' for i in xtick_idx]

        ## --- Add this code to overlay the squares ---
        for start, end in squares:
            # Assuming 'start' and 'end' are the corner positions.
            # The width and height will be end - start.
            width = end - start
            height = end - start
            rect = patches.Rectangle(
                (start, start),
                width,
                height,
                linewidth=1,
                edgecolor='black',
                facecolor='none' # This creates a no-fill square
            )
            g.add_patch(rect)

        # --- This new block overlays the off-diagonal interaction squares ---
        for (start1, end1), (start2, end2) in combinations(squares, 2):
            width1 = end1 - start1
            width2 = end2 - start2

            #linewidth = 2
            linewidth = 1
            edgecolor = "black"
            linestyle = "--"
            #linestyle = None

            # Off-diagonal rectangle 1
            rect1 = patches.Rectangle(
                (start1, start2),
                width1,
                width2,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor='none',
                linestyle=linestyle  # Dashed line style
            )
            g.add_patch(rect1)

            # Off-diagonal rectangle 2 (symmetric)
            rect2 = patches.Rectangle(
                (start2, start1),
                width2,
                width1,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor='none',
                linestyle=linestyle  # Dashed line style
            )
            g.add_patch(rect2)

        g.set_xticks(xtick_idx)
        g.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=8)
        g.set_xlabel(f'Genomic position (chr{chrom})')
        plt.savefig(output[0], bbox_inches="tight", dpi=200)