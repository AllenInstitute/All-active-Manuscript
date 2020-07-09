library(scrattch.vis)
library(feather)
library(tidyverse)
options(stringsAsFactors = F)

# Setting working directory -----------------------------------------------

#this.dir <- dirname(parent.frame(2)$ofile)
#setwd(this.dir)

# Data paths --------------------------------------------------------------

project_path <- file.path(getwd(),'..','..','..','assets','aggregated_data')
exc_expression_file <- file.path(project_path,'exc_expression_all.csv')
data_df <- read_csv(exc_expression_file)
data_df <- data_df %>% rename(sample_name = sample_id)



# RNA-seq data path
data_loc = '/data/rnaseqanalysis/shiny/facs_seq/Mm_VISp_14236_20180912'


# Annotation data
anno_feather_path <- file.path(data_loc,'anno.feather')
anno <- read_feather(anno_feather_path)
anno <- anno %>% rename(sample_name = sample_id) 
anno <- anno %>% filter(cre_label %in% unique(data_df$cre_label))

marker_genes <- c('Rspo1','Fezf2')
Hcn_genes <- c('Hcn1','Hcn2','Hcn3')
all_genes <- c(marker_genes,Hcn_genes)

data_df <- data_df %>% select(sample_name, all_genes)
  
exc_lines <- c("Nr5a1-Cre","Rbp4-Cre_KL100")
exc_group <- anno %>% group_by(cre_label) %>% select(cre_label, cre_id) %>% summarize(cre_id = mean(cre_id))
group_order <- c()

for (line in exc_lines){
  exc_cre_id <- exc_group[which(exc_group$cre_label == line),'cre_id']
  group_order <- append(group_order,exc_cre_id$cre_id[1])
}

p <- sample_heatmap_plot(data_df, 
                         anno, 
                         genes = all_genes,
                         grouping = "cre",
                         group_order = group_order,
                         log_scale = FALSE,
                         font_size = 16,
                         max_width = 20)

#print(p)

ggsave(file=file.path('figures','expression_plot_exc.svg'), p, width=8, height=5)