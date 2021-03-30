library(tidyverse)
library(dendextend)
library(jsonlite)
library(plotly)


library(reticulate)
use_python("~/anaconda3/bin/python")
   

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

filtered_inh_cells_file <- file.path(data_path,'filtered_me_inh_cells.pkl')
filtered_inh_cells <- read_pickle_file(filtered_inh_cells_file)

filtered_exc_cells_file <- file.path(data_path,'filtered_me_exc_cells.pkl')
filtered_exc_cells <- read_pickle_file(filtered_exc_cells_file)

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
mouse_data <- mouse_data %>% filter(hof_index == 0)
  
# Filter cells
inh_lines <- c("Htr3a-Cre_NO152","Sst-IRES-Cre","Pvalb-IRES-Cre")
exc_lines <- c("Nr5a1-Cre","Rbp4-Cre_KL100")

mouse_data <- mouse_data %>% filter(!((Cre_line %in% inh_lines) & !(Cell_id %in% filtered_inh_cells)))
mouse_data <- mouse_data %>% filter(!((Cre_line %in% exc_lines) & !(Cell_id %in% filtered_exc_cells)))
mouse_data <- mouse_data %>% filter(!((Broad_Cre_line == 'Pyr') & (Dendrite_type == 'aspiny')))



ephys_data <- read_csv(train_ephys_max_amp_filename,col_types = cols(Cell_id = col_character()))
ephys_fields <- fromJSON(ephys_fields_filename)
morph_data <- read_csv(morph_data_filename,col_types = cols(Cell_id = col_character()))
hof_param_data <- read_csv(param_data_filename,col_types = cols(Cell_id = col_character(),
                                                             hof_index = col_integer()))
best_param_data <- hof_param_data %>% filter(hof_index == 0) %>% select(-hof_index)

cre_cluster <- mouse_data %>% select(c('Cell_id','Cre_line'))
bcre_cluster <- mouse_data  %>% select(c('Cell_id','Broad_Cre_line'))
bcre_cluster$Broad_Cre_line <- bcre_cluster$Broad_Cre_line %>% replace_na('Other')

bcre_index_order <- c('Htr3a','Sst','Pvalb','Pyr')
bcre_list = list(Pyr= 'Rbp4-Cre_KL100',Pvalb='Pvalb-IRES-Cre',Sst='Sst-IRES-Cre',Htr3a='Htr3a-Cre_NO152')

me_data <- morph_data %>% inner_join(ephys_data %>% select(ephys_fields), by='Cell_id')
mp_data <- morph_data %>% inner_join(best_param_data,by= 'Cell_id')
e_data <- ephys_data %>% select(ephys_fields)

# Data Cleaning -----------------------------------------------------------

param_data_cleaned <- best_param_data %>% select_if(~ !any(is.na(.))) %>% inner_join(bcre_cluster, 
                                                                                     by = 'Cell_id')
e_data_cleaned <- e_data %>% select_if(~ !any(is.na(.))) %>% inner_join(bcre_cluster, by = 'Cell_id')
mp_data_cleaned <- mp_data %>% select_if(~ !any(is.na(.))) %>% inner_join(bcre_cluster, by = 'Cell_id')
me_data_cleaned <- me_data %>% select_if(~ !any(is.na(.))) %>% inner_join(bcre_cluster, by = 'Cell_id')


param_data_cleaned$bcre_colors <- unlist(lapply(param_data_cleaned$Broad_Cre_line,get_bcre_color ))
param_data_cleaned <- param_data_cleaned %>% filter(Broad_Cre_line != 'Other')
param_data_cleaned <- param_data_cleaned %>% mutate_at(vars(-c("Cell_id",
                     "Broad_Cre_line", "bcre_colors")), ~(scale(.) %>% as.vector))
param_data_cleaned <- param_data_cleaned %>% remove_rownames %>% column_to_rownames(var="Cell_id")


e_data_cleaned$bcre_colors <- unlist(lapply(e_data_cleaned$Broad_Cre_line,get_bcre_color ))
e_data_cleaned <- e_data_cleaned %>% filter(Broad_Cre_line != 'Other')
e_data_cleaned <- e_data_cleaned %>% mutate_at(vars(-c("Cell_id","Broad_Cre_line", "bcre_colors")), ~(scale(.) %>% as.vector))
e_data_cleaned <- e_data_cleaned %>% remove_rownames %>% column_to_rownames(var="Cell_id")


me_data_cleaned$bcre_colors <- unlist(lapply(me_data_cleaned$Broad_Cre_line,get_bcre_color ))
me_data_cleaned <- me_data_cleaned %>% filter(Broad_Cre_line != 'Other')
me_data_cleaned <- me_data_cleaned %>% mutate_at(vars(-c("Cell_id", "Broad_Cre_line", "bcre_colors")), ~(scale(.) %>% as.vector))
me_data_cleaned <- me_data_cleaned %>% remove_rownames %>% column_to_rownames(var="Cell_id")


mp_data_cleaned$bcre_colors <- unlist(lapply(mp_data_cleaned$Broad_Cre_line,get_bcre_color ))
mp_data_cleaned <- mp_data_cleaned %>% filter(Broad_Cre_line != 'Other')
mp_data_cleaned <- mp_data_cleaned %>% mutate_at(vars(-c("Cell_id", "Broad_Cre_line", "bcre_colors")), ~(scale(.) %>% as.vector))
mp_data_cleaned <- mp_data_cleaned %>% remove_rownames %>% column_to_rownames(var="Cell_id")


# Unsupervised Clustering - Dendrogram ------------------------------------

# Experimental data: Ephys ------------------------------------------------

d1_colors <- e_data_cleaned$bcre_colors
d1 <- e_data_cleaned %>% select(-c(Broad_Cre_line,bcre_colors)) %>%
    dist() %>% hclust( method="ward.D" ) %>% as.dendrogram() 
d1 <- d1 %>% 
  #collapse_branch(tol=20) %>% 
  ladderize %>% 
  set('labels_cex', c(1,rep(.01,10))) 
par(mar=c(7,5,1,1))
plot(d1, main = 'Ephys',axes=FALSE)
colored_bars(colors = d1_colors, dend = d1, rowLabels = "Broad Cre-line")
legend("topleft", legend = bcre_index_order, pch = 15, pt.cex =2, cex =1, bty = 'n',
       title = "Broad Cre-line", inset=c(0.02,0.05),
       col = c(bcre_color_list$Htr3a,bcre_color_list$Sst,bcre_color_list$Pvalb,
               bcre_color_list$Pyr))



# Experimental data: Ephys + Morph ----------------------------------------

d3_colors <- me_data_cleaned$bcre_colors
d3 <- me_data_cleaned %>% select(-c(Broad_Cre_line,bcre_colors)) %>%
  dist() %>% hclust( method="ward.D" ) %>% as.dendrogram()
d3 <- d3 %>% 
  #collapse_branch(tol=2000) %>% 
  ladderize %>% 
  #set("labels", "")
  set('labels_cex', c(1,rep(.01,10))) 
par(mar=c(7,5,1,1))
plot(d3,main='Morph + Ephys Parameters',axes=FALSE)
colored_bars(colors = d3_colors, dend = d3, rowLabels = "Broad Cre-line")
legend("topleft", legend = bcre_index_order, pch = 15, pt.cex =2, cex =1, bty = 'n',
       title = "Broad Cre-line", inset=c(0.05,.1),
       col = c(bcre_color_list$Htr3a,bcre_color_list$Sst,bcre_color_list$Pvalb,
               bcre_color_list$Pyr))

# Model parameters + Morph ------------------------------------------------

d2_colors <- mp_data_cleaned$bcre_colors
d2 <- mp_data_cleaned %>% select(-c(Broad_Cre_line,bcre_colors)) %>%
   dist() %>% hclust( method="ward.D" ) %>% as.dendrogram()
d2 <- d2 %>% 
  #collapse_branch(tol=2000) %>%
  ladderize %>% 
  #set("labels", "")
  set('labels_cex', c(1,rep(.01,10))) 
par(mar=c(7,5,1,1))
plot(d2,main='Morph + Model Parameters',axes=FALSE)
colored_bars(colors = d2_colors, dend = d2, rowLabels = "Broad Cre-line")
legend("topleft", legend = bcre_index_order, pch = 15, pt.cex =2, cex =1, bty = 'n',
       title = "Broad Cre-line", inset=c(0.05,.1),
       col = c(bcre_color_list$Htr3a,bcre_color_list$Sst,bcre_color_list$Pvalb,
               bcre_color_list$Pyr))


# Model parameters  ------------------------------------------------

d_param_colors <- param_data_cleaned$bcre_colors
d_param <- param_data_cleaned %>% select(-c(Broad_Cre_line,bcre_colors)) %>%
  dist() %>% hclust( method="ward.D" ) %>% as.dendrogram()
d_param <- d_param %>% 
  # collapse_branch(tol=500) %>%
  ladderize %>% 
  #set("labels", "")
  set('labels_cex', c(1,rep(.01,10))) 
par(mar=c(7,5,12,0))
plot(d_param,main='Model Parameters',axes=FALSE)
colored_bars(colors = d_param_colors, dend = d_param, rowLabels = "Broad Cre-line")
legend("center", legend = bcre_index_order, pch = 15, pt.cex =2, cex =2, bty = 'n',
       title = "Broad Cre-line", 
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
           margin_inner=8,cex_main = 1.5)



# Tanglegram between ME and Model parameters
dme_p <- dendlist(d3,d_param)

# Plot them together
tanglegram(dme_p, highlight_distinct_edges = FALSE, # Turn-off dashed lines
                   common_subtrees_color_branches = FALSE, # Color common branches
                   common_subtrees_color_lines = FALSE,
                   highlight_branches_lwd = FALSE,
                   lwd = .8,
                   # main_left = 'Ephys',
                   main_left = 'Morph + Ephys',
                   main_right = 'Model Parameters',
                   margin_inner=6,cex_main = 1.5)

