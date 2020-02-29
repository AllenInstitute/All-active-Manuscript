library(scrattch.vis)
library(feather)
library(tidyverse)
options(stringsAsFactors = F)


# Setting working directory -----------------------------------------------

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Data paths --------------------------------------------------------------

project_path <- file.path(getwd(),'..','..','..','assets','aggregated_data')
inh_expression_filtered_file <- file.path(project_path,'inh_expression_filtered.csv')
data_df <- read_csv(inh_expression_filtered_file)
data_df <- data_df %>% rename(sample_name = sample_id,cre_label = Cre_line)



# RNA-seq data path
data_loc = '/data/rnaseqanalysis/shiny/facs_seq/Mm_VISp_14236_20180912'


# Annotation data
anno_feather_path <- file.path(data_loc,'anno.feather')
anno <- read_feather(anno_feather_path)
anno <- anno %>% rename(sample_name = sample_id) 
anno <- anno %>% filter(cre_label %in% unique(data_df$cre_label))

marker_genes <- c('Htr3a','Sst','Pvalb')
Kv31_gene <- c('Kcnc1')
KT_genes <- c('Kcnd2','Kcnd3')
KP_genes <- c('Kcna1','Kcna2','Kcna3','Kcna6')
all_genes <- c(marker_genes,Kv31_gene, KT_genes, KP_genes)

data_df <- data_df %>% select(sample_name, all_genes)

inh_lines <- c("Htr3a-Cre_NO152","Sst-IRES-Cre","Pvalb-IRES-Cre")
inh_group <- anno %>% group_by(cre_label) %>% select(cre_label, cre_id) %>% summarize(cre_id = mean(cre_id))
group_order <- c()

for (line in inh_lines){
  inh_cre_id <- inh_group[which(inh_group$cre_label == line),'cre_id']
  group_order <- append(group_order,inh_cre_id$cre_id[1])
}

p <- sample_heatmap_plot(data_df, 
                    anno, 
                    genes = all_genes,
                    grouping = "cre",
                    group_order = group_order,
                    log_scale = FALSE,
                    font_size = 16,
                    max_width = 15)

print(p)

