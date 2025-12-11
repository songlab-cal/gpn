all_dms = [
    "DMS_A4_HUMAN_Seuma_2021",
    "DMS_ADRB2_HUMAN_Jones_2020",
    "DMS_BRCA1_HUMAN_Findlay_2018",
    "DMS_CALM1_HUMAN_Weile_2017",
    "DMS_CP2C9_HUMAN_Amorosi_abundance_2021",
    "DMS_CP2C9_HUMAN_Amorosi_activity_2021",
    "DMS_DLG4_HUMAN_Faure_2021",
    "DMS_GRB2_HUMAN_Faure_2021",
    "DMS_KCNH2_HUMAN_Kozek_2020",
    "DMS_MK01_HUMAN_Brenan_2016",
    "DMS_MSH2_HUMAN_Jia_2020",
    "DMS_NUD15_HUMAN_Suiter_2020",
    "DMS_P53_HUMAN_Giacomelli_NULL_Etoposide_2018",
    "DMS_P53_HUMAN_Giacomelli_NULL_Nutlin_2018",
    "DMS_P53_HUMAN_Giacomelli_WT_Nutlin_2018",
    "DMS_P53_HUMAN_Kotler_2018",
    "DMS_PTEN_HUMAN_Matreyek_2021",
    "DMS_PTEN_HUMAN_Mighell_2018",
    "DMS_SC6A4_HUMAN_Young_2021",
    "DMS_SCN5A_HUMAN_Glazer_2019",
    "DMS_SRC_HUMAN_Ahler_CD_2019",
    "DMS_SUMO1_HUMAN_Weile_2017",
    "DMS_SYUA_HUMAN_Newberry_2020",
    "DMS_TADBP_HUMAN_Bolognesi_2019",
    "DMS_TPK1_HUMAN_Weile_2017",
    "DMS_TPMT_HUMAN_Matreyek_2018",
    "DMS_TPOR_HUMAN_Bridgford_S505N_2020",
    "DMS_UBC9_HUMAN_Weile_2017",
    "DMS_VKOR1_HUMAN_Chiasson_abundance_2020",
    "DMS_VKOR1_HUMAN_Chiasson_activity_2020",
    "DMS_YAP1_HUMAN_Araya_2012",
]


rule dms_download:
    output:
        "results/dms/{dms}/variants.vcf.gz",
    wildcard_constraints:
        dms="|".join(all_dms),
    shell:
        "wget -O {output} https://kircherlab.bihealth.org/download/CADD-development/v1.7/validation/esm/{wildcards.dms}.vcf.gz"


rule dms_process:
    input:
        "results/dms/{dms}/variants.vcf.gz",
    output:
        "results/dms/{dms}/variants.parquet",
    wildcard_constraints:
        dms="|".join(all_dms),
    run:
        V = pd.read_csv(
            input[0],
            sep="\t",
            header=None,
            dtype={"chrom": "str"},
            names=["chrom", "pos", "id", "ref", "alt", "label"],
        ).drop(columns=["id"])
        V["DMS"] = wildcards.dms
        V.to_parquet(output[0], index=False)


rule dms_merge:
    input:
        expand("results/dms/{dms}/variants.parquet", dms=all_dms),
    output:
        "results/dms/merged/test.parquet",
    run:
        V = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        V = sort_chrom_pos(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule dms_fix_chrom:
    input:
        "results/dms2/snv_to_aa_table_proteingym26_031224.parquet",
        "results/genome.fa.gz",
    output:
        "results/dms2/snv/test.parquet",
    run:
        V = pd.read_parquet(input[0])
        V.chrom = V.chrom.astype(str)
        assert (V.ref.isin(NUCLEOTIDES).all()) and (V.alt.isin(NUCLEOTIDES)).all()
        assert (V.ref != V.alt).all()
        genome = Genome(input[1])
        V["fasta_ref"] = V.progress_apply(
            lambda v: genome.get_nuc(v.chrom, v.pos).upper(), axis=1
        )
        V["mismatched"] = V.ref != V.fasta_ref
        for col in ["ref", "alt"]:
            V[col] = V.apply(
                lambda v: (
                    v[col]
                    if not v.mismatched
                    else str(Seq(v[col]).reverse_complement())
                ),
                axis=1,
            )
        assert (V.ref == V.fasta_ref).all()
        assert (V.ref != V.alt).all()
        V = V.drop(columns=["fasta_ref"])
        V = sort_chrom_pos(V)
        print(V)
        V.to_parquet(output[0], index=False)


ruleorder: dms2_get_precomputed_scores > run_vep_gpn


rule dms2_get_precomputed_scores:
    input:
        "results/dms2/snv/test.parquet",
        expand("results/positions/{chrom}/llr/{{model}}.parquet", chrom=CHROMS),
    output:
        "results/preds/results/dms2/snv/{model}.parquet",
    threads: workflow.cores
    run:
        V = pl.read_parquet(input[0], columns=COORDINATES)
        preds = pl.concat(
            [
                pl.read_parquet(path).join(V, on=COORDINATES, how="inner")
                for path in tqdm(input[1:])
            ]
        )
        V = V.join(preds, on=COORDINATES, how="left")
        print(V)
        V.select("score").write_parquet(output[0])


rule process_dms_measurements:
    input:
        "results/dms2/metadata/DMS_substitutions.csv",
        "results/dms2/measurements",
    output:
        "results/dms2/processed_measurements.parquet",
    run:
        metadata = pd.read_csv(input[0])
        print(metadata)
        metadata = metadata[metadata.DMS_id.str.contains("HUMAN")]
        print(metadata)
        dfs = []
        for dms_id in tqdm(metadata.DMS_id):
            df = pd.read_csv(input[1] + f"/{dms_id}.csv")
            df["DMS"] = dms_id
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        print(df)
        df.to_parquet(output[0], index=False)
