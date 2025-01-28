#####################################################
######## Evaluation of GPN-MSA on 4 datasets ########
#####################################################
# [!] Save all annotated dataframes in separate folders

### Reading entire variants.parquet file ---------------------------------------
data_dir <- '/global/scratch/projects/fc_songlab/data/'
all_variants <- arrow::read_parquet((file=paste0(data_dir,"variants.parquet")))
four_datasets_dir <- '/global/scratch/projects/fc_songlab/data/gnomAD-cosmic-clinVar-missense/four_datasets/'
### Mine OMIM annotations ------------------------------------------------------
omim_dir <- "/global/scratch/projects/fc_songlab/data/gnomAD-cosmic-clinVar-missense/OMIM/"
all_omim_var <- readr::read_csv(
  "/global/scratch/projects/fc_songlab/data/all_092423_OMIM.csv") %>% 
  select(c('chrom','pos','ref','alt','label','consequence','MAF','AF','GPN-MSA','CADD.RawScore','phyloP','CADD.PHRED'))
all_omim_var$variant_vcf <- paste0(all_omim_var$chrom,'-',
                                   all_omim_var$pos,'-',
                                   all_omim_var$ref,'-',
                                   all_omim_var$alt)
sex_chr_FAVOR <- readr::read_csv(paste0(omim_dir,'missing/2023-09-24_14_02_annotations.csv'),
                                 col_select=c("Variant (VCF)","bStatistic",
                                              "Recombination Rate","Nucleotide Diversity",
                                              "CADD RawScore","CADD phred",
                                              "LINSIGHT","Fathmm XF",
                                              "Aloft Value","Funseq Value",
                                              "priPhyloP","mamPhyloP",
                                              "verPhyloP","GerpN",
                                              "GerpS","PolyPhenVal",
                                              "SIFTval","CpG",
                                              "GC","H3K4me1",
                                              "H3K4me2","H3K4me3",
                                              "H3K9ac","H3K9me3",
                                              "H3K27ac","H3K27me3",
                                              "H3K36me3","H3k79me2",
                                              "H4k20me1","H2AFZ",
                                              "DNase","totalRNA",
                                              "RemapOverlapTF","RemapOverlapCL"))
colnames(sex_chr_FAVOR) <- c('variant_vcf','bstatistic',
                                  'recombination_rate','nucdiv',
                                  'cadd_rawscore','cadd_phred',
                                  'linsight','fathmm_xf',
                                  'aloft_value','funseq_value',
                                  'priphylop','mamphylop',
                                  'verphylop','gerp_n',
                                  'gerp_s','polyphen_val',
                                  'sift_val','cpg',
                                  'gc','encodeh3k4me1_sum',
                                  'encodeh3k4me2_sum','encodeh3k4me3_sum',
                                  'encodeh3k9ac_sum','encodeh3k9me3_sum',
                                  'encodeh3k27ac_sum','encodeh3k27me3_sum',
                                  'encodeh3k36me3_sum','encodeh3k79me2_sum',
                                  'encodeh4k20me1_sum','encodeh2afz_sum',
                                  'encode_dnase_sum','encodetotal_rna_sum',
                                  'remap_overlap_tf','remap_overlap_cl')
autosome_FAVOR <- readr::read_csv(paste0(omim_dir,'missing/2023-09-25_14_34_annotations.csv'),
                                  col_select=c("Variant (VCF)","bStatistic",
                                               "Recombination Rate","Nucleotide Diversity",
                                               "CADD RawScore","CADD phred",
                                               "LINSIGHT","Fathmm XF",
                                               "Aloft Value","Funseq Value",
                                               "priPhyloP","mamPhyloP",
                                               "verPhyloP","GerpN",
                                               "GerpS","PolyPhenVal",
                                               "SIFTval","CpG",
                                               "GC","H3K4me1",
                                               "H3K4me2","H3K4me3",
                                               "H3K9ac","H3K9me3",
                                               "H3K27ac","H3K27me3",
                                               "H3K36me3","H3k79me2",
                                               "H4k20me1","H2AFZ",
                                               "DNase","totalRNA",
                                               "RemapOverlapTF","RemapOverlapCL"))
colnames(autosome_FAVOR) <- c('variant_vcf','bstatistic',
                              'recombination_rate','nucdiv',
                              'cadd_rawscore','cadd_phred',
                              'linsight','fathmm_xf',
                              'aloft_value','funseq_value',
                              'priphylop','mamphylop',
                              'verphylop','gerp_n',
                              'gerp_s','polyphen_val',
                              'sift_val','cpg',
                              'gc','encodeh3k4me1_sum',
                              'encodeh3k4me2_sum','encodeh3k4me3_sum',
                              'encodeh3k9ac_sum','encodeh3k9me3_sum',
                              'encodeh3k27ac_sum','encodeh3k27me3_sum',
                              'encodeh3k36me3_sum','encodeh3k79me2_sum',
                              'encodeh4k20me1_sum','encodeh2afz_sum',
                              'encode_dnase_sum','encodetotal_rna_sum',
                              'remap_overlap_tf','remap_overlap_cl')

omim_variants <- rbind(autosome_FAVOR,sex_chr_FAVOR)
omim_merged <- merge(all_omim_var,omim_variants,by='variant_vcf')
arrow::write_parquet(omim_merged,
                     sink=paste0(omim_dir,"annot_variants.parquet"))

### Reading gnomAD, ClinVar, COSMIC data with FAVOR annotations ----------------
all_var_dir <- '/global/scratch/projects/fc_songlab/data/gnomAD-cosmic-clinVar-missense/'
gnomAD_dir <- paste0(all_var_dir,'gnomAD/')
cosmic_dir <- paste0(all_var_dir,'COSMIC/')
clinvar_dir <- paste0(all_var_dir,'ClinVar/')
OMIM_dir <- paste0(all_var_dir,'OMIM/')

gnomAD_variants <- arrow::read_parquet(paste0(gnomAD_dir,
                                              "annot_variants.parquet"))
cosmic_variants <- arrow::read_parquet(paste0(cosmic_dir,
                                              "annot_variants.parquet"))
clinvar_variants <- arrow::read_parquet(paste0(clinvar_dir,
                                               "annot_variants.parquet"))
omim_variants <- arrow::read_parquet(paste0(omim_dir,
                                            "annot_variants.parquet"))


### ClinVar Pathogenic vs. gnomAD common (missense) ----------------------------
# dataset = dataset.filter(lambda v: v["source"]=="ClinVar" or 
#                     (v["label"]=="Common" and "missense" in v["consequence"]))
# ds1_vars_df <- all_variants %>% subset(source=='Clinvar' | 
#                                          (label=='Common' & 
#                                             grepl('missense',consequence)))
ds1_vars_df <- arrow::read_parquet(paste0(four_datasets_dir,
                                          'subset_1_clinvar_vs_gnomad.parquet'))
ds1_vars_df$variant_vcf <- paste0(ds1_vars_df$chrom,'-',
                                  ds1_vars_df$pos,'-',
                                  ds1_vars_df$ref,'-',
                                  ds1_vars_df$alt)
message(date(), ": No. gnomAD annotated variants in ds1 = ", 
        sum(gnomAD_variants$variant_vcf %in% ds1_vars_df$variant_vcf))
message(date(), ": No. ClinVar annotated variants in ds1 = ", 
        sum(clinvar_variants$variant_vcf %in% ds1_vars_df$variant_vcf))
message(date(), ": No. COSMIC annotated variants in ds1 = ", 
        sum(cosmic_variants$variant_vcf %in% ds1_vars_df$variant_vcf))

pooled_df <- rbind(rbind(gnomAD_variants %>% subset(variant_vcf %in% ds1_vars_df$variant_vcf),
                         clinvar_variants %>% subset(variant_vcf %in% ds1_vars_df$variant_vcf)),
                   cosmic_variants %>% subset(variant_vcf %in% ds1_vars_df$variant_vcf))
# > length(unique(pooled_df$variant_vcf))
# [1] 34390
# Sun Sep 24 07:50:12 2023: No. gnomAD annotated variants in ds1 = 13140
# Sun Sep 24 07:50:07 2023: No. ClinVar annotated variants in ds1 = 21275
# Sun Sep 24 07:50:03 2023: No. COSMIC annotated variants in ds1 = 42
# > 21275+13140+42
# [1] 34457
# > nrow(ds1_vars_df)
# [1] 34392
# > length(unique(ds1_vars_df$variant_vcf))
# [1] 34390
# > which(table(ds1_vars_df$variant_vcf)>=2)
# 1-976215-A-G 19-50878521-C-T 
# 3245           17020 
diff_vars <- setdiff(unique(pooled_df$variant_vcf),
                     unique(ds1_vars_df$variant_vcf))
distinct_pooled_df <- pooled_df[!duplicated(pooled_df$variant_vcf),]
# Removed two duplicate variants (shown above)
arrow::write_parquet(distinct_pooled_df,
                     sink=paste0(four_datasets_dir,"ds1_annot_variants.parquet"))

### COSMIC frequent vs. gnomAD common (missense) -------------------------------
# dataset = dataset.filter(lambda v: v["source"]=="COSMIC" or 
#                     (v["label"]=="Common" and "missense" in v["consequence"]))
# ds2_vars_df <- all_variants %>% subset(source=='COSMIC' | 
#                                          (label=='Common' & 
#                                             grepl('missense',consequence)))
ds2_vars_df <- arrow::read_parquet(paste0(four_datasets_dir,
                                          'subset_2_cosmic_vs_gnomad.parquet'))
ds2_vars_df$variant_vcf <- paste0(ds2_vars_df$chrom,'-',
                                  ds2_vars_df$pos,'-',
                                  ds2_vars_df$ref,'-',
                                  ds2_vars_df$alt)
pooled_df <- rbind(rbind(gnomAD_variants %>% subset(variant_vcf %in% ds2_vars_df$variant_vcf),
                         clinvar_variants %>% subset(variant_vcf %in% ds2_vars_df$variant_vcf)),
                   cosmic_variants %>% subset(variant_vcf %in% ds2_vars_df$variant_vcf))
# > length(unique(pooled_df$variant_vcf))
# [1] 13307
# > nrow(ds2_vars_df)
# [1] 13307
# > length(unique(ds2_vars_df$variant_vcf))
# [1] 13307
distinct_pooled_df <- pooled_df[!duplicated(pooled_df$variant_vcf),]
# Removed two duplicate variants (shown above)
arrow::write_parquet(distinct_pooled_df,
                     sink=paste0(four_datasets_dir,"ds2_annot_variants.parquet"))

### OMIM Pathogenic vs. gnomAD common (regulatory) -----------------------------
# cs = ["5_prime_UTR", "upstream_gene", "intergenic", "3_prime_UTR", "non_coding_transcript_exon"]
# dataset = dataset.filter(lambda v: v["source"]=="OMIM" or 
#          (v["label"]=="Common" and 
#                 "missense" not in v["consequence"] and
#                 any([c in v["consequence"] for c in cs])))
# ds3_vars_df <- all_variants %>% subset(source=='OMIM' | 
#                                          (label=='Common' & 
#                                             !(grepl('missense',consequence)) &
#                                             (grepl('5_prime_UTR',consequence)| 
#                                                grepl('upstream_gene',consequence)| 
#                                                grepl('intergenic',consequence)|
#                                                grepl('3_prime_UTR',consequence)|
#                                                grepl('non_coding_transcript_exon',consequence))))
ds3_vars_df <- arrow::read_parquet(paste0(four_datasets_dir,
                                          'subset_3_omim_vs_gnomad.parquet'))
ds3_vars_df$variant_vcf <- paste0(ds3_vars_df$chrom,'-',
                                  ds3_vars_df$pos,'-',
                                  ds3_vars_df$ref,'-',
                                  ds3_vars_df$alt)
# > nrow(ds3_vars_df)
# [1] 2285108
# > length(unique(ds3_vars_df$variant_vcf))
# [1] 2285108

pooled_df <- rbind(rbind(rbind(gnomAD_variants %>% subset(variant_vcf %in% ds3_vars_df$variant_vcf),
                               clinvar_variants %>% subset(variant_vcf %in% ds3_vars_df$variant_vcf)),
                         cosmic_variants %>% subset(variant_vcf %in% ds3_vars_df$variant_vcf)),
                   omim_variants %>% subset(variant_vcf %in% ds3_vars_df$variant_vcf))
# > length(unique(pooled_df$variant_vcf))
# [1] 2285108
distinct_pooled_df <- pooled_df[!duplicated(pooled_df$variant_vcf),]
arrow::write_parquet(distinct_pooled_df,
                     sink=paste0(four_datasets_dir,"ds3_annot_variants.parquet"))

##  gnomAD rare vs. gnomAD common ----------------------------------------------
# dataset = dataset.filter(lambda v: v["source"]=="gnomAD")
#message(date(), ": Analysis is already complete for this dataset.")
ds4_vars_df <- arrow::read_parquet(paste0(four_datasets_dir,
                                          'subset_4_gnomad_vs_gnomad.parquet'))
ds4_vars_df$variant_vcf <- paste0(ds4_vars_df$chrom,'-',
                                  ds4_vars_df$pos,'-',
                                  ds4_vars_df$ref,'-',
                                  ds4_vars_df$alt)
# > nrow(ds4_vars_df)
# [1] 9624620
# > length(unique(ds4_vars_df$variant_vcf))
# [1] 9624620
# > nrow(gnomAD_variants)
# [1] 9738418
final_gnomAD_variants <- gnomAD_variants %>% 
  subset(variant_vcf %in% ds4_vars_df$variant_vcf)
arrow::write_parquet(final_gnomAD_variants,
                     sink=paste0(four_datasets_dir,"ds4_annot_variants.parquet"))

### Annotations of Interest ----------------------------------------------------
rel_annot_list <- c('variant_vcf','bstatistic',
                    'recombination_rate','nucdiv','cpg',
                    'gc','encodeh3k4me1_sum',
                    'encodeh3k4me2_sum','encodeh3k4me3_sum',
                    'encodeh3k9ac_sum','encodeh3k9me3_sum',
                    'encodeh3k27ac_sum','encodeh3k27me3_sum',
                    'encodeh3k36me3_sum','encodeh3k79me2_sum',
                    'encodeh4k20me1_sum','encodeh2afz_sum',
                    'encode_dnase_sum','encodetotal_rna_sum',
                    'remap_overlap_tf','remap_overlap_cl')
