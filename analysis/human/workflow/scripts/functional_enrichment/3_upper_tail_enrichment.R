#####################################################
######## Evaluation of GPN-MSA on 4 datasets ########
#####################################################
# 2. Perform deleterious tail enrichment analysis

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
  
  # Group variants by deleterious tail membership
  target_df <- target_df %>% mutate(Group_A=ifelse(`GPN-MSA`<= quantile(target_df$`GPN-MSA`,0.01),
                                                   1,0),
                                    Group_B=ifelse(`GPN-MSA`<= quantile(target_df$`GPN-MSA`,0.001),
                                                   1,0),
                                    Group_C=ifelse(`GPN-MSA`<= quantile(target_df$`GPN-MSA`,0.0001),
                                                   1,0))
  p_val_table <- data.frame(ANNOTATION=character(),
                            GROUP=character(),
                            GREATER = numeric(),
                            LESS = numeric(),
                            N_TARGET = numeric(),
                            N_BKGRD = numeric(),
                            TEST = character())
  
  # Compute Wilcoxon test p-values
  message(date(), ": Computing two-sample Wilcoxon tests for Dataset ", k)
  fileConn <- file(paste0(four_datasets_dir,"results/ds",k,"_wilcoxon_log.txt"))
  writeLines(paste0("Performing correlation tests on Dataset ", k),
             fileConn)
  close(fileConn)
  for (j in 1:length(func_annot_vars)) {
    message(date(), ": Working on ",func_annot_vars[j])
    for (g in c('A','B','C')) {
      target_set_func_annots <- target_df[which(target_df[[paste0('Group_',g)]]==1),]
      bkgrd_set_func_annots <- target_df[which(target_df[[paste0('Group_',g)]]==0),]
      
      if (length(na.omit(target_set_func_annots[[func_annot_vars[j]]])) <= 
          3) {
        message(date(), ": Fewer than 4 annotated variants in target set, skipping test...")
        p_val_table <- rbind(p_val_table,
                             data.frame(ANNOTATION=func_annot_vars[j],
                                        GROUP=g,
                                        GREATER = NA,
                                        LESS = NA,
                                        N_TARGET = length(na.omit(target_set_func_annots[[func_annot_vars[j]]])),
                                        N_BKGRD = length(na.omit(bkgrd_set_func_annots[[func_annot_vars[j]]])),
                                        TEST = NA))
      } else {
        greater_test <- wilcox.test(x=na.omit(target_set_func_annots[[func_annot_vars[j]]]),
                                    y=na.omit(bkgrd_set_func_annots[[func_annot_vars[j]]]),
                                    alternative='greater')
        less_pval <- wilcox.test(x=na.omit(target_set_func_annots[[func_annot_vars[j]]]),
                                 y=na.omit(bkgrd_set_func_annots[[func_annot_vars[j]]]),
                                 alternative='less')$p.value
        p_val_table <- rbind(p_val_table,
                             data.frame(ANNOTATION=func_annot_vars[j],
                                        GROUP=g,
                                        GREATER=greater_test$p.value,
                                        LESS=less_pval,
                                        N_TARGET = length(na.omit(target_set_func_annots[[func_annot_vars[j]]])),
                                        N_BKGRD = length(na.omit(bkgrd_set_func_annots[[func_annot_vars[j]]])),
                                        TEST = greater_test$method))
      }
      
      # Write to log
      write(paste0("[Group ", g, "] ", 
                   p_val_table[nrow(p_val_table),]$ANNOTATION,
                   ": Greater p = ", 
                   p_val_table[nrow(p_val_table),]$GREATER,
                   ", Less p = ", p_val_table[nrow(p_val_table),]$LESS), 
            file = paste0(four_datasets_dir,"results/ds",k,"_wilcoxon_log.txt"),
            append=TRUE,sep="\n")
    }
  }
  message(date(), ": Saving results for Dataset ", k)
  # Save 
  readr::write_csv(p_val_table,
                   file=paste0(four_datasets_dir,"results/ds",k,"_deleterious_enrichment_pvals.csv"))
}

### Plotting -------------------------------------------------------------------
dataset_names <- c('ClinVar Pathogenic + \ngnomAD common missense',
                   'COSMIC Pathogenic + \ngnomAD common missense',
                   'OMIM Pathogenic + \ngnomAD common regulatory',
                   'gnomAD rare + common')

mega_table <- data.frame(ANNOTATION=character(),
                         GROUP=character(),
                         GREATER=numeric(),
                         LESS=numeric(),
                         N_TARGET=numeric(),
                         N_BKGRD=numeric(),
                         TEST=character(),
                         OUTCOME=character(),
                         DATASET=character())

for (k in 1:4) {
  p_val_table <- readr::read_csv(paste0(four_datasets_dir,
                                        "results/ds",k,"_deleterious_enrichment_pvals.csv"))
  p_val_table$OUTCOME <- mapply(function(x,y) {
    if (is.na(x)) {
      return('neither')
    } else if (x < 0.05/20) {
      return('enrich')
    } else if (y < 0.05/20) {
      return('deplete')
    } else {
      return('neither')
    }}, p_val_table$GREATER, p_val_table$LESS)
  
  p_val_table$DATASET <- rep(dataset_names[k],nrow(p_val_table))
  
  mega_table<-rbind(mega_table,p_val_table)
}

mega_table <- mega_table %>% subset(ANNOTATION != 'remap_overlap_tf' & 
                                      ANNOTATION != 'remap_overlap_cl')
# Recode variables for plotting'
english_func_annot <- c('% CpG','% GC','B Statistic','Recombination Rate','Nucleotide Diversity',
                        'H3K4me1','H3K4me2','H3K4me3',
                        'H3K9ac','H3K9me3','H3K27ac',
                        'H3K27me3','H3K36me3','H3K79me2',
                        'H4K20me1','H2AFZ','DNase-seq',
                        'RNA-seq','# TF Binding','# (Cell line, TF) Binding')

mega_table$GROUP <- recode(mega_table$GROUP, A = "Lowest 1%\nGPN-MSA", B = "Lowest 0.1%\nGPN-MSA", C = "Lowest 0.01%\nGPN-MSA")
mega_table$ANNOTATION <- recode(mega_table$ANNOTATION, 
                                bstatistic = "B Statistic", 
                                cpg = "% CpG", 
                                encode_dnase_sum = "DNase-seq",
                                encodeh2afz_sum = "H2AFZ",
                                encodeh3k27ac_sum = "H3K27ac",
                                encodeh3k27me3_sum = "H3K27me3",
                                encodeh3k36me3_sum = "H3K36me3",
                                encodeh3k4me1_sum = "H3K4me1",
                                encodeh3k4me2_sum = "H3K4me2",
                                encodeh3k4me3_sum = "H3K4me3",
                                encodeh3k79me2_sum = "H3K79me2",
                                encodeh3k9ac_sum = "H3K9ac",
                                encodeh3k9me3_sum = "H3K9me3",
                                encodeh4k20me1_sum = "H4K20me1",
                                encodetotal_rna_sum = "RNA-seq",
                                gc = "% GC",
                                nucdiv = "Nucleotide Diversity",
                                recombination_rate = "Recombination Rate")
# Plot all three cutoffs
func_sig_plot <- ggplot(data = mega_table, 
       aes(x=factor(DATASET,
                    level=c('ClinVar Pathogenic + \ngnomAD common missense',
                            'COSMIC Pathogenic + \ngnomAD common missense',
                            'OMIM Pathogenic + \ngnomAD common regulatory',
                            'gnomAD rare + common')), 
           y=factor(ANNOTATION,
                    level=english_func_annot), fill=OUTCOME)) + 
  geom_tile(alpha=0.9,color = "black") +
  scale_fill_manual(values=c("deplete"="#A63446", "enrich"="#4575b4", "neither"="#d9d9d9"),
                    name="") +
  ylab("Functional Annotation") +
  xlab("Dataset") +
  theme_bw() +
  ggtitle("Functional Significance of Deleterious Tail") +
  theme(axis.text.x = element_text(angle = 90,vjust=-0.01)) +
  facet_wrap(.~GROUP) +
  coord_fixed()

ggsave(func_sig_plot,
       filename = paste0(four_datasets_dir,'plots/func_significance_heatmap_927.jpg'),
       width = 6.8, height = 9,
       dpi = 300)

# Plot just deleterious 1% cutoff
func_sig_plot_top1pct <- ggplot(data = mega_table %>% 
                          subset(GROUP=='Lowest 1%\nGPN-MSA'), 
                        aes(x=factor(DATASET,
                                     level=c('ClinVar Pathogenic + \ngnomAD common missense',
                                             'COSMIC Pathogenic + \ngnomAD common missense',
                                             'OMIM Pathogenic + \ngnomAD common regulatory',
                                             'gnomAD rare + common')), 
                            y=factor(ANNOTATION,
                                     level=english_func_annot), fill=OUTCOME)) + 
  geom_tile(alpha=0.9,color = "black") +
  scale_fill_manual(values=c("deplete"="#A63446", "enrich"="#4575b4", "neither"="#d9d9d9"),
                    name="") +
  ylab("Functional Annotation") +
  xlab("Dataset") +
  theme_bw() +
  ggtitle("Functional Significance of\nLowest 1% GPN-MSA") +
  theme(axis.text.x = element_text(angle = 90,vjust=-0.01),
        plot.title=element_text(hjust = 0.5)) +
  coord_fixed()

ggsave(func_sig_plot_top1pct,
       filename = paste0(four_datasets_dir,'plots/func_significance_top1pct_heatmap_927.jpg'),
       width = 5, height = 9,
       dpi = 300)
