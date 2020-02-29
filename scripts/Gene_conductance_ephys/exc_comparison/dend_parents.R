library(dendextend)
library(dplyr)
options(stringsAsFactors = F)

source("dend_functions.R")

# Load dendrogram
dataset_dir <-  '/data/rnaseqanalysis/shiny/facs_seq/Mm_VISp_14236_20180912'

dend <- readRDS(file.path(dataset_dir,"dend.RData"))

# Add parent attribute
dend2 <- add_parent_attr(dend)

# Retrieve node table using as.ggdend from dendextend
nodes <- as.ggdend(dend2)$nodes

# Add the label and parent values
nodes2 <- nodes %>%
  mutate(label = get_nodes_attr(dend2, "label"),
         parent = get_nodes_attr(dend2, "parent"))

# Output the node table
write.csv(nodes2, "tree_VISp.csv", row.names = FALSE)
