library(tidyverse)
library(dendextend)
library(jsonlite)
library(plotly)


# Setting working directory -----------------------------------------------

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

library(reticulate)
use_condaenv('ateam_opt')
   

# Utility Functions -------------------------------------------------------
  
source_python('pickle_reader_for_R.py')
data_path <- file.path(getwd(),'..','..','assets','aggregated_data')
bcre_color_file <- file.path(data_path,'bcre_color_tasic16.pkl')
bcre_color_list <- read_pickle_file(bcre_color_file) 

get_bcre_color <- function(bcre){
  if (bcre %in% names(bcre_color_list)){
    return(bcre_color_list[[bcre]])
  }
  else return(rgb(0,0,0))
}

# Data paths --------------------------------------------------------------
project_path <- file.path(getwd(),'..','..','assets','aggregated_data')
mouse_data_filename <- file.path(project_path, 'Mouse_class_data.csv')
param_data_filename <- file.path(project_path,'allactive_params.csv')
train_ephys_max_amp_filename <- file.path(project_path,'train_ephys_max_amp.csv')
ephys_fields_filename <- file.path(project_path,'train_ephys_max_amp_fields.json')
morph_data_filename <- file.path(project_path,'morph_data.csv')

# Reading data ------------------------------------------------------------

mouse_data <- read_csv(mouse_data_filename,col_types = cols(Cell_id = col_character(),
                                                Cre_line = col_character(), 
                                                Broad_Cre_line = col_character()))
ephys_data <- read_csv(train_ephys_max_amp_filename,col_types = cols(Cell_id = col_character()))
ephys_fields <- fromJSON(ephys_fields_filename)
morph_data <- read_csv(morph_data_filename,col_types = cols(Cell_id = col_character()))
hof_param_data <- read_csv(param_data_filename,col_types = cols(Cell_id = col_character(),
                                                             hof_index = col_integer()))
best_param_data <- hof_param_data %>% filter(hof_index == 0) %>% select(-hof_index)

cre_cluster <- mouse_data %>% filter(hof_index == 0) %>% select(c('Cell_id','Cre_line'))
bcre_cluster <- mouse_data %>% filter(hof_index == 0) %>% select(c('Cell_id','Broad_Cre_line'))
bcre_cluster$Broad_Cre_line <- bcre_cluster$Broad_Cre_line %>% replace_na('Other')

bcre_index_order <- c('Htr3a','Sst','Pvalb','Pyr')
bcre_list = list(Pyr= 'Rbp4-Cre_KL100',Pvalb='Pvalb-IRES-Cre',Sst='Sst-IRES-Cre',Htr3a='Htr3a-Cre_NO152')

me_data <- morph_data %>% inner_join(ephys_data %>% select(ephys_fields), by='Cell_id')
mp_data <- morph_data %>% inner_join(best_param_data,by= 'Cell_id')
e_data <- ephys_data %>% select(ephys_fields)

# Data Cleaning -----------------------------------------------------------

e_data_cleaned <- e_data %>% select_if(~ !any(is.na(.))) %>% inner_join(bcre_cluster, by = 'Cell_id')
mp_data_cleaned <- mp_data %>% select_if(~ !any(is.na(.))) %>% inner_join(bcre_cluster, by = 'Cell_id')
me_data_cleaned <- me_data %>% select_if(~ !any(is.na(.))) %>% inner_join(bcre_cluster, by = 'Cell_id')


e_data_cleaned$bcre_colors <- unlist(lapply(e_data_cleaned$Broad_Cre_line,get_bcre_color ))
e_data_cleaned <- e_data_cleaned %>% filter(Broad_Cre_line != 'Other')
e_data_cleaned <- e_data_cleaned %>% remove_rownames %>% column_to_rownames(var="Cell_id")

me_data_cleaned$bcre_colors <- unlist(lapply(me_data_cleaned$Broad_Cre_line,get_bcre_color ))
me_data_cleaned <- me_data_cleaned %>% filter(Broad_Cre_line != 'Other')
me_data_cleaned <- me_data_cleaned %>% remove_rownames %>% column_to_rownames(var="Cell_id")

mp_data_cleaned$bcre_colors <- unlist(lapply(mp_data_cleaned$Broad_Cre_line,get_bcre_color ))
mp_data_cleaned <- mp_data_cleaned %>% filter(Broad_Cre_line != 'Other')
mp_data_cleaned <- mp_data_cleaned %>% remove_rownames %>% column_to_rownames(var="Cell_id")


# Unsupervised Clustering - Dendrogram ------------------------------------

# Experimental data : Ephys
d1_colors <- e_data_cleaned$bcre_colors
d1 <- e_data_cleaned %>% select(-c(Broad_Cre_line,bcre_colors)) %>%
    dist() %>% hclust( method="ward.D" ) %>% as.dendrogram() 
d1 <- d1 %>% collapse_branch(tol=20) %>% ladderize %>% 
  set('labels_cex', c(1.2,rep(.01,5))) 
par(mar=c(7,5,1,1))
plot(d1, main = 'Ephys',axes=FALSE)
colored_bars(colors = d1_colors, dend = d1, rowLabels = "Broad Cre-line")
legend("topleft", legend = bcre_index_order, pch = 15, pt.cex =2, cex =1, bty = 'n',
       title = "Broad Cre-line", inset=c(0.02,0.05),
       col = c(bcre_color_list$Htr3a,bcre_color_list$Sst,bcre_color_list$Pvalb,
               bcre_color_list$Pyr))

# Experimental data : Ephys + Morph
d3_colors <- me_data_cleaned$bcre_colors
d3 <- me_data_cleaned %>% select(-c(Broad_Cre_line,bcre_colors)) %>%
  dist() %>% hclust( method="ward.D" ) %>% as.dendrogram()
d3 <- d3 %>% collapse_branch(tol=2000) %>% ladderize %>% 
  set('labels_cex', c(1.2,rep(.01,10))) 
par(mar=c(7,5,1,1))
plot(d3,main='Morph + Ephys Parameters',axes=FALSE)
colored_bars(colors = d3_colors, dend = d3, rowLabels = "Broad Cre-line")
legend("topleft", legend = bcre_index_order, pch = 15, pt.cex =2, cex =1, bty = 'n',
       title = "Broad Cre-line", inset=c(0.05,.1),
       col = c(bcre_color_list$Htr3a,bcre_color_list$Sst,bcre_color_list$Pvalb,
               bcre_color_list$Pyr))

# Model parameters + Morph
d2_colors <- mp_data_cleaned$bcre_colors
d2 <- mp_data_cleaned %>% select(-c(Broad_Cre_line,bcre_colors)) %>%
   dist() %>% hclust( method="ward.D" ) %>% as.dendrogram()
d2 <- d2 %>% collapse_branch(tol=2000) %>% ladderize %>% 
  set('labels_cex', c(1.2,rep(.01,10))) 
par(mar=c(7,5,1,1))
plot(d2,main='Morph + Model Parameters',axes=FALSE)
colored_bars(colors = d2_colors, dend = d2, rowLabels = "Broad Cre-line")
legend("topleft", legend = bcre_index_order, pch = 15, pt.cex =2, cex =1, bty = 'n',
       title = "Broad Cre-line", inset=c(0.05,.1),
       col = c(bcre_color_list$Htr3a,bcre_color_list$Sst,bcre_color_list$Pvalb,
               bcre_color_list$Pyr))


# Tanglegram --------------------------------------------------------------

# Custom these dendrograms and place them in a list
dl <- dendlist(d3,d2)

# Plot them together
dl %>%  tanglegram(highlight_distinct_edges = FALSE, # Turn-off dashed lines
           common_subtrees_color_branches = FALSE, # Color common branches
           common_subtrees_color_lines = FALSE,
           highlight_branches_lwd = FALSE,
           # main_left = 'Ephys',
           main_left = 'Morph + Ephys',
           main_right = 'Morph + Parameters',
           margin_inner=8.5,cex_main = 1.5)

