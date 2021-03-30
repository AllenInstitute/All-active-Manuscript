library(scrattch.vis)
library(feather)
library(tidyverse)
options(stringsAsFactors = F)


# Setting working directory -----------------------------------------------

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Data paths --------------------------------------------------------------

project_path <- file.path(getwd(),'..','..','..','assets','aggregated_data')
inh_expression_file <- file.path(project_path,'inh_expression_subclass.csv')
data_df <- read_csv(inh_expression_file)
data_df <- data_df %>% rename(sample_name = sample_id)


# Annotation data
anno_feather_path <- file.path(project_path,'anno.feather')
anno <- read_feather(anno_feather_path)
anno <- anno %>% rename(sample_name = sample_id) 
anno <- anno %>% filter(subclass_label %in% unique(data_df$subclass_label))

marker_genes <- c('Vip','Sst','Pvalb')
Kv31_gene <- c('Kcnc1')
KT_genes <- c('Kcnd2','Kcnd3')
KP_genes <- c('Kcna1','Kcna2','Kcna3','Kcna6')
all_genes <- c(marker_genes,Kv31_gene, KT_genes, KP_genes)

data_df <- data_df %>% select(sample_name, all_genes)

inh_subclasses <- c("Vip","Sst","Pvalb")
inh_group <- anno %>% group_by(subclass_label) %>% select(subclass_label, subclass_id) %>% 
  summarize(subclass_id = mean(subclass_id))
group_order <- c()

for (subclass in inh_subclasses){
  inh_subclass_id <- inh_group[which(inh_group$subclass_label == subclass),'subclass_id']
  group_order <- append(group_order,inh_subclass_id$subclass_id[1])
}

p <- sample_heatmap_plot(data_df, 
                         anno, 
                         genes = all_genes,
                         grouping = "subclass",
                         group_order = group_order,
                         log_scale = FALSE,
                         font_size = 16,
                         label_height = 20,
                         max_width = 15)

print(p)

