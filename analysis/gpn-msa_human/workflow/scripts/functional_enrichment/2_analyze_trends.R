#####################################################
######## Evaluation of GPN-MSA on 4 datasets ########
#####################################################
# 2. Compute Correlations and associated p-values
#    for each dataset.

### Reading entire variants.parquet file ---------------------------------------
data_dir <- '/global/scratch/projects/fc_songlab/data/'
four_datasets_dir <- '/global/scratch/projects/fc_songlab/data/gnomAD-cosmic-clinVar-missense/four_datasets/'

### Perform analyses on each dataset -------------------------------------------
func_annot_vars <- c('cpg','gc','bstatistic','recombination_rate','nucdiv',
                     'encodeh3k4me1_sum','encodeh3k4me2_sum','encodeh3k4me3_sum',
                     'encodeh3k9ac_sum','encodeh3k9me3_sum','encodeh3k27ac_sum',
                     'encodeh3k27me3_sum','encodeh3k36me3_sum','encodeh3k79me2_sum',
                     'encodeh4k20me1_sum','encodeh2afz_sum','encode_dnase_sum',
                     'encodetotal_rna_sum','remap_overlap_tf','remap_overlap_cl')

for (k in 1:4) {
  message(date(), ": Loading parquet file for Dataset ", k)
  target_df <- arrow::read_parquet(file=paste0(four_datasets_dir,
                                               "ds",k,
                                               "_annot_variants.parquet"))
  
  # Compute correlations with 20 functional annotations
  p_val_mat <- matrix(nrow=2,
                      ncol=length(func_annot_vars),
                      0)
  colnames(p_val_mat) <- func_annot_vars; rownames(p_val_mat) <- c('Pearson_p','Spearman_p')
  
  cor_mat <- matrix(nrow=2,
                    ncol=length(func_annot_vars),
                    0)
  colnames(cor_mat) <- func_annot_vars; rownames(cor_mat) <- c('Pearson_r','Spearman_rho')
  
  message(date(), ": Computing correlations for Dataset ", k)
  fileConn <- file(paste0(four_datasets_dir,"results/ds",k,"_correlations_log.txt"))
  writeLines(paste0("Performing correlation tests on Dataset ", k),
             fileConn)
  close(fileConn)
  for (j in 1:length(func_annot_vars)) {
    # Message
    message(date(), ": Compute correlation between GPN-MSA and ",func_annot_vars[j])
    n_comp_obs <- target_df %>% 
      select(c('GPN-MSA',
               func_annot_vars[j])) %>%
      filter(complete.cases(.)) %>%
      nrow()
    if (n_comp_obs < 2) {
      write(paste0("Correlations with ",
                   func_annot_vars[j],
                   " = ", NA, 
                   " (two-sided p-value = ", 
                   NA, 
                   ") --- NUMBER OF COMPLETE OBSERVATIONS < 2"), 
            file = paste0(four_datasets_dir,"results/ds",k,"_correlations_log.txt"),
            append=TRUE,sep="\n")
      p_val_mat[1,j] <- NA; cor_mat[1,j] <- NA
      p_val_mat[2,j] <- NA; cor_mat[2,j] <- NA
    } else if (sd(na.omit(target_df[[func_annot_vars[j]]]))==0) {
      write(paste0("Correlations with ",
                   func_annot_vars[j],
                   " = ", NA, 
                   " (two-sided p-value = ", 
                   NA, 
                   ") --- NO VARIATION IN ", func_annot_vars[j]), 
            file = paste0(four_datasets_dir,"results/ds",k,"_correlations_log.txt"),
            append=TRUE,sep="\n")
      p_val_mat[1,j] <- NA; cor_mat[1,j] <- NA
      p_val_mat[2,j] <- NA; cor_mat[2,j] <- NA
    } else {
      spearman_test_res <- cor.test(target_df[['GPN-MSA']],
                                    target_df[[func_annot_vars[j]]],
                                    method="spearman",
                                    use="complete.obs",
                                    exact=FALSE)
      pearson_test_res <- cor.test(target_df[['GPN-MSA']],
                                   target_df[[func_annot_vars[j]]],
                                   method="pearson",
                                   use="complete.obs",
                                   exact=FALSE)
      p_val_mat[1,j] <- pearson_test_res$p.value; cor_mat[1,j] <- pearson_test_res$estimate
      p_val_mat[2,j] <- spearman_test_res$p.value; cor_mat[2,j] <- spearman_test_res$estimate
      
      # Write to file
      write(paste0("Spearman rho between GPN-MSA and ", 
                   func_annot_vars[j],
                   " = ", spearman_test_res$estimate, 
                   " (two-sided p-value = ", 
                   spearman_test_res$p.value, 
                   ") --- no. complete observations = ",
                   n_comp_obs), 
            file = paste0(four_datasets_dir,"results/ds",k,"_correlations_log.txt"),
            append=TRUE,sep="\n")
      write(paste0("Pearson r between GPN-MSA and ", 
                   func_annot_vars[j],
                   " = ", pearson_test_res$estimate, 
                   " (two-sided p-value = ", 
                   pearson_test_res$p.value, 
                   ") --- no. complete observations = ",
                   n_comp_obs), 
            file = paste0(four_datasets_dir,"results/ds",k,"_correlations_log.txt"),
            append=TRUE,sep="\n")
    }
  }
  
  message(date(), ": Saving results for Dataset ", k)
  # Save 
  readr::write_csv(as.data.frame(p_val_mat) %>% mutate(SCORE=rownames(p_val_mat)),
                   file=paste0(four_datasets_dir,"results/ds",k,"_pvals.csv"))
  readr::write_csv(as.data.frame(cor_mat) %>% mutate(SCORE=rownames(cor_mat)),
                   file=paste0(four_datasets_dir,"results/ds",k,"_cors.csv"))
}

### Plotting -------------------------------------------------------------------
# Transform data to heatmap-compatible form
combined_spearman_rhos <- matrix(0,
                                 nrow=4,
                                 ncol=length(func_annot_vars)+1)
colnames(combined_spearman_rhos) <- c('Dataset',
                                      func_annot_vars)
combined_spearman_rhos <- combined_spearman_rhos %>% as.data.frame()
combined_spearman_rhos$Dataset <- c('DS1','DS2','DS3','DS4')
combined_spearman_rhos$Dataset <- c('ClinVar Pathogenic + \ngnomAD common missense',
                                    'COSMIC Pathogenic + \ngnomAD common missense',
                                    'OMIM Pathogenic + \ngnomAD common regulatory',
                                    'gnomAD rare + common')

combined_spearman_pvals <- matrix(0,
                                 nrow=4,
                                 ncol=length(func_annot_vars)+1)
colnames(combined_spearman_pvals) <- c('Dataset',
                                      func_annot_vars)
combined_spearman_pvals <- combined_spearman_pvals %>% as.data.frame()
combined_spearman_pvals$Dataset <- c('ClinVar Pathogenic + \ngnomAD common missense',
                                    'COSMIC Pathogenic + \ngnomAD common missense',
                                    'OMIM Pathogenic + \ngnomAD common regulatory',
                                    'gnomAD rare + common')

combined_pearson_rs <- matrix(0,
                              nrow=4,
                              ncol=length(func_annot_vars)+1)
colnames(combined_pearson_rs) <- c('Dataset',
                                   func_annot_vars)
combined_pearson_rs <- combined_pearson_rs %>% as.data.frame()
combined_pearson_rs$Dataset <- c('DS1','DS2','DS3','DS4')
combined_pearson_rs$Dataset <- c('ClinVar Pathogenic + \ngnomAD common missense',
                                 'COSMIC Pathogenic + \ngnomAD common missense',
                                 'OMIM Pathogenic + \ngnomAD common regulatory',
                                 'gnomAD rare + common')

combined_pearson_pvals <- matrix(0,
                              nrow=4,
                              ncol=length(func_annot_vars)+1)
colnames(combined_pearson_pvals) <- c('Dataset',
                                   func_annot_vars)
combined_pearson_pvals <- combined_pearson_pvals %>% as.data.frame()
combined_pearson_pvals$Dataset <- c('ClinVar Pathogenic + \ngnomAD common missense',
                                 'COSMIC Pathogenic + \ngnomAD common missense',
                                 'OMIM Pathogenic + \ngnomAD common regulatory',
                                 'gnomAD rare + common')

for (k in 1:4) {
  cor_df <- readr::read_csv(paste0(four_datasets_dir,"results/ds",k,"_cors.csv"))
  pvals_df <- readr::read_csv(paste0(four_datasets_dir,"results/ds",k,"_pvals.csv"))
  
  spearman_selected_row <- cor_df %>% subset(SCORE=='Spearman_rho') %>%
    select(all_of(func_annot_vars))
  combined_spearman_rhos[k,-1] <- spearman_selected_row
  spearman_pvals <- pvals_df %>% subset(SCORE=='Spearman_p') %>%
    select(all_of(func_annot_vars))
  
  pearson_selected_row <- cor_df %>% subset(SCORE=='Pearson_r') %>% 
    select(all_of(func_annot_vars))
  combined_pearson_rs[k,-1] <- pearson_selected_row
  pearson_pvals <- pvals_df %>% subset(SCORE=='Pearson_p') %>%
    select(all_of(func_annot_vars))
  
  combined_spearman_rhos[k,-1] <- spearman_selected_row
  combined_spearman_pvals[k,-1] <- spearman_pvals
  
  combined_pearson_rs[k,-1] <- pearson_selected_row
  combined_pearson_pvals[k,-1] <- pearson_pvals
}

# Create new names for interpretability
english_func_annot <- c('% CpG','% GC','B Statistic','Recombination Rate','Nucleotide Diversity',
                        'H3K4me1','H3K4me2','H3K4me3',
                        'H3K9ac','H3K9me3','H3K27ac',
                        'H3K27me3','H3K36me3','H3K79me2',
                        'H4K20me1','H2AFZ','DNase-seq',
                        'RNA-seq','# TF Binding','# (Cell line, TF) Binding')

colnames(combined_pearson_rs)[-1] <- english_func_annot
colnames(combined_spearman_rhos)[-1] <- english_func_annot
colnames(combined_pearson_pvals)[-1] <- english_func_annot
colnames(combined_spearman_pvals)[-1] <- english_func_annot

combined_pearson_rs[,20:21] <- NULL
combined_pearson_pvals[,20:21] <- NULL
combined_spearman_rhos[,20:21] <- NULL
combined_spearman_pvals[,20:21] <- NULL

# Function to combine correlation and p-value dataframes for plotting
# Thank you ChatGPT-3!
combine_correlation_pvalue <- function(correlation_df, pvalue_df) {
  # Get the row and column names from the correlation dataframe
  row_names <- correlation_df[['Dataset']]
  col_names <- colnames(correlation_df)[-1]
  
  # Initialize empty lists to store the output data
  dataset <- vector("character")
  variable <- vector("character")
  correlation <- vector("numeric")
  p_value <- vector("numeric")
  
  # Loop through the row and column names to populate the lists
  for (row in row_names) {
    for (col in col_names) {
      dataset <- c(dataset, row)
      variable <- c(variable, col)
      correlation <- c(correlation, correlation_df[which(correlation_df$Dataset==row), col])
      p_value <- c(p_value, pvalue_df[which(correlation_df$Dataset==row), col])
    }
  }
  
  # Create a new dataframe from the lists
  result_df <- data.frame(
    Dataset = dataset,
    Variable = variable,
    Correlation = correlation,
    P_value = p_value
  )
  
  return(result_df)
}

# Use the function to combine the dataframes
melted_spearman_mat <- combine_correlation_pvalue(combined_spearman_rhos, 
                                                  combined_spearman_pvals)

melted_pearson_mat <- combine_correlation_pvalue(combined_pearson_rs, 
                                                  combined_pearson_pvals)

melted_spearman_mat <- melted_spearman_mat %>% mutate(r_if_Sig=ifelse(P_value<0.05/20,
                                                                 Correlation,NA))
melted_pearson_mat <- melted_pearson_mat %>% mutate(r_if_Sig=ifelse(P_value<0.05/20,
                                                                 Correlation,NA))

melted_spearman_mat$Sig <- ifelse(melted_spearman_mat$P_value<0.05/20,"*"," ")
melted_pearson_mat$Sig <- ifelse(melted_pearson_mat$P_value<0.05/20,"*"," ")

# Spearman heatmap
#melted_spearman_mat <- reshape2::melt(combined_spearman_rhos)
options(warn = 1)
spearman_heatmap <- ggplot(data = melted_spearman_mat, 
                           aes(x=factor(Dataset,
                                        level=c('ClinVar Pathogenic + \ngnomAD common missense',
                                                'COSMIC Pathogenic + \ngnomAD common missense',
                                                'OMIM Pathogenic + \ngnomAD common regulatory',
                                                'gnomAD rare + common')),
                               y=factor(Variable,
                                        level=english_func_annot), 
                               fill=Correlation,
                               label=round(r_if_Sig,3))) + 
  geom_tile(alpha=0.9,color = "black") +
  scale_fill_gradient2(low="#A63446", high="#4575b4", mid="white",
                       name=expression(paste("Spearman ", rho))) +
  geom_text()+
  ylab("Functional Annotation") +
  xlab("Dataset") +
  theme_bw() +
  ggtitle("Rank Correlation with 18 Functional Annotations") +
  theme(axis.text.x = element_text(angle = 90,vjust=-0.02)) 
  #coord_fixed()
  #theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))

ggsave(spearman_heatmap,
       filename = paste0(four_datasets_dir,'plots/spearman_with_pvals_heatmap_927.jpg'),
       width = 7, height = 8,
       dpi = 300)

# Pearson heatmap
#melted_pearson_mat <- reshape2::melt(combined_pearson_rs)
pearson_heatmap <- ggplot(data = melted_pearson_mat, 
                          aes(x=factor(Dataset,
                                       level=c('ClinVar Pathogenic + \ngnomAD common missense',
                                               'COSMIC Pathogenic + \ngnomAD common missense',
                                               'OMIM Pathogenic + \ngnomAD common regulatory',
                                               'gnomAD rare + common')), 
                              y=factor(Variable,
                                       level=english_func_annot), 
                              fill=Correlation,
                              label=round(r_if_Sig,3))) + 
  geom_tile(alpha=0.9,color = "black") +
  scale_fill_gradient2(low="#A63446", high="#4575b4", mid="white",
                       name=expression(paste("Pearson ", r))) +
  geom_text()+
  ylab("Functional Annotation") +
  xlab("Dataset") +
  theme_bw() +
  ggtitle("Correlation with 18 Functional Annotations") +
  theme(axis.text.x = element_text(angle = 90,vjust=-0.02)) 
  #coord_fixed()
  #theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))

ggsave(pearson_heatmap,
       filename = paste0(four_datasets_dir,'plots/pearson_with_pvals_heatmap_927.jpg'),
       width = 6.8, height = 8,
       dpi = 300)

## Prepare parquets for Gonzalo ------------------------------------------------
for (k in 1:4) {
  message(date(), ": Reading parquet file for dataset ", k)
  target_df <- arrow::read_parquet(file=paste0(four_datasets_dir,
                                               "ds",k,
                                               "_annot_variants.parquet"))
  to_save <- target_df %>% select(c('variant_vcf','chrom','pos','ref',
                                    'alt','label','consequence','MAF',
                                    'AF','GPN-MSA','CADD.RawScore','CADD.PHRED',
                                    'phyloP',
                                    'bstatistic','recombination_rate','nucdiv',
                                    'cpg','gc',
                                    'encodeh3k4me1_sum','encodeh3k4me2_sum',
                                    'encodeh3k4me3_sum','encodeh3k9ac_sum',
                                    'encodeh3k9me3_sum','encodeh3k27ac_sum',
                                    'encodeh3k27me3_sum','encodeh3k36me3_sum',
                                    'encodeh3k79me2_sum','encodeh4k20me1_sum',
                                    'encodeh2afz_sum','encode_dnase_sum',
                                    'encodetotal_rna_sum',
                                    'remap_overlap_tf','remap_overlap_cl'))
  message(date(), ": Saving parquet file for dataset ", k)
  message("No. rows in dataset ", k, " = ", nrow(to_save))
  message("No. cols in dataset ", k, " = ", ncol(to_save))
  arrow::write_parquet(to_save,
                       sink=paste0(four_datasets_dir,"/public/ds",k,"_favor_annot_vars.parquet"))
}

