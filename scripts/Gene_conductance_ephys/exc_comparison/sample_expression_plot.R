library(scrattch.vis)
library(feather)
library(tidyverse)
options(stringsAsFactors = F)

# Setting working directory -----------------------------------------------

#this.dir <- dirname(parent.frame(2)$ofile)
#setwd(this.dir)

# Data paths --------------------------------------------------------------

project_path <- file.path(getwd(),'..','..','..','assets','aggregated_data')
exc_expression_file <- file.path(project_path,'exc_ttype_expression.csv')
data_df <- read_csv(exc_expression_file)
data_df <- data_df %>% rename(sample_name = sample_id)

# Annotation data
anno_feather_path <- file.path(project_path,'anno.feather')
anno <- read_feather(anno_feather_path)
anno <- anno %>% rename(sample_name = sample_id) 
anno <- anno %>% filter(subclass_label %in% unique(data_df$subclass_label))

marker_genes <- c('Rspo1','Fezf2','Fam19a1')
channel_genes <- c('Hcn1','Hcn2','Hcn3','Scn8a')
all_genes <- c(marker_genes,channel_genes)

data_df <- data_df %>% select(sample_name, all_genes)
  
exc_subclasses <- c("L4","L5 IT","L5 PT")
exc_group <- anno %>% group_by(subclass_label) %>% select(subclass_label, subclass_id) %>% 
  summarize(subclass_id = mean(subclass_id))
group_order <- c() # Order in which the groups appear

for (subclass in exc_subclasses){
  exc_subclass_id <- exc_group[which(exc_group$subclass_label == subclass),'subclass_id']
  group_order <- append(group_order,exc_subclass_id$subclass_id[1])
}

p <- sample_heatmap_plot(data_df, 
                         anno, 
                         genes = all_genes,
                         grouping = "subclass",
                         group_order = group_order,
                         log_scale = FALSE,
                         font_size = 16,
                         label_height = 20,
                         max_width = 20)

print(p)
ggsave(file=file.path('figures','expression_plot_exc.svg'), p, width=10, height=5)