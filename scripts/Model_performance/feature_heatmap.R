library(tidyverse)
library(ggplot2)
library(ComplexHeatmap)
library(circlize)
library(colorspace)

# Setting working directory -----------------------------------------------

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Data paths --------------------------------------------------------------

project_path <- file.path(getwd(),'..','..','assets','aggregated_data')
mouse_data_filename <- file.path(project_path, 'Mouse_class_data.csv')
model_performance_filename <- file.path(project_path,'Feature_performance.csv')

# Read data --------------------------------------------------

mouse_data <- read_csv(mouse_data_filename,col_types = cols(Cell_id = col_character())) %>%
  filter(hof_index == 0)
model_performance_data <- read_csv(model_performance_filename,col_types = cols(Cell_id = col_character()))
model_performance_data <- model_performance_data  %>% inner_join(mouse_data %>%
                                                          select(Cell_id, dendrite_type),by='Cell_id')

# order cells by dendrite type
model_performance_data <- model_performance_data %>% arrange(desc(dendrite_type))
feature_wise_performance <-  model_performance_data %>% summarise_at(which(sapply(model_performance_data, is.numeric)),mean,na.rm = TRUE)

ordered_features <- feature_wise_performance %>% gather('feature','z_score') %>% 
                arrange(z_score) 
ordered_features <- ordered_features$feature
model_performance_data <- model_performance_data %>% select(c(ordered_features,c("Cell_id","dendrite_type")))


# Plot heatmap ------------------------------------------------------------


ha = HeatmapAnnotation(df = data.frame(model_performance_data %>% select(dendrite_type)),
                 col = list(dendrite_type = c("spiny" =  "red2","sparsely spiny" = 'royalblue', "aspiny" = "mediumseagreen")),
                 annotation_name_gp = gpar(fontsize = 11))
perf_df <- model_performance_data[,which(sapply(model_performance_data,is.numeric))]
perf_matrix <- data.frame(t(as.matrix(perf_df)))
rownames(perf_matrix) <- colnames(perf_df)

colors <- viridisLite::viridis(10)

hm <- Heatmap(as.matrix(perf_matrix),cluster_rows = FALSE, cluster_columns = FALSE,
               name='z-score',top_annotation = ha, show_column_names = FALSE,
              row_names_side = "left",
              col = colorRamp2(seq(from=0,to = 10, length.out = 10), colors),
              row_names_max_width = unit(6, "cm"),
              height = unit(6, "cm"),width=unit(10,"cm"),
              heatmap_legend_param = list(
                color_bar = "continuous"
              ),
              row_names_gp = gpar(fontsize=10))
print(hm)


