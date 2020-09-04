library(scrattch.vis)
library(feather)
library(tidyverse)
library(colorspace)
options(stringsAsFactors = F)


# Setting working directory -----------------------------------------------

# this.dir <- dirname(parent.frame(2)$ofile)
# setwd(this.dir)

# Extract expression data for selected genes and t-types ------------------

getFACSData <- function(ngsDataPath, annoData, writeFilePath, subclassTypes, selectGenes){
    
    counts_data <- read_feather(ngsDataPath)
    
    # Normalize
    counts_data[,which(names(counts_data) != 'sample_id')] <- 
        t(apply(counts_data[,which(names(counts_data) != 
                                       'sample_id')], 1, function(x)(log2(x+1))))
    
    # Filter based on t-types
    # select_ids <- annoData %>% filter(grepl(ttypeRegex, primary_cluster_label)) %>% 
    #         pull(sample_id)
    
    # Filter based on subclass    
    select_ids <- annoData %>% filter(subclass_label %in% subclassTypes) %>% 
        pull(sample_id)
    
    select_data <- counts_data %>% filter(sample_id %in% select_ids) 
    select_data <- select_data[,c('sample_id', selectGenes)]
    
    write.csv(select_data, file = writeFilePath, row.names = FALSE)
    return(select_data)
    
}

# RNA-seq data ------------------------------------------------------------

dataLoc = '/data/rnaseqanalysis/shiny/facs_seq/Mm_VISp_14236_20180912/'
projectPath <- file.path(getwd(),'..','..','assets','aggregated_data')
expressionFile <- file.path(projectPath,'expression_multistage.csv')


# Annotation data
annoDataPath <- file.path(projectPath,'anno.feather')

# Gene count data
countsDataPath <- file.path(dataLoc,'data.feather')

# Marker and Channel Genes
expressionMarkerGenes <- file.path(projectPath, 'top.de.df.csv')
expressionDEChannel <- file.path(projectPath, 'channel.de.df.csv')
markerGenesDE <- read_csv(expressionMarkerGenes)
channelGenesDE <- read_csv(expressionDEChannel)

# t-types for ME data -----------------------------------------------------

l23_it_match <- "(.*L2/3 IT.*)"
l4_match <- "(.*L4.*)"
l5_it_match <- "(.*L5 IT.*)"
l5_pt_match <- "(.*L5 PT.*)"
np_match <- "(.*NP.*)"
l6_ct_match <- "(.*L6 CT.*)"
l6_it_match <- "(.*L6 IT.*)"
l6b_match <- "(.*L6b.*)"

pc_match <- c(l23_it_match, l4_match, l5_it_match, l5_pt_match, np_match, 
              l6_ct_match, l6_it_match, l6b_match)
pc_match_regex <- paste(pc_match, collapse ="|")

sncg_match <- "(.*Sncg.*)"
lamp5_match <- "(.*Lamp5.*)"
vip_match <- "(.*Vip.*)"
sst_match <- "(.*Sst.*)"
pvalb_match <- "(.*Pvalb.*)"

inh_match <- c(sncg_match, lamp5_match, vip_match, sst_match, pvalb_match)
inh_match_regex <- paste(inh_match, collapse ="|")

regexMatch <- paste(pc_match_regex, inh_match_regex, sep = "|")

# transcriptomic subclasses for ME data -----------------------------------------------------

# inhSubclass = c('Lamp5', 'Sncg', 'Vip', 'Sst', 'Pvalb')
inhSubclass = c('Vip', 'Sst', 'Pvalb')
# excSubclass = c('L2/3 IT', 'L4', 'L5 PT', 'L5 IT', 'NP', 'L6 CT', 'L6 IT', 'L6b')
excSubclass = c('L2/3 IT', 'L4', 'L5 PT', 'L5 IT')
subclassTypes = c(inhSubclass, excSubclass)

# Genes encoding channels --------------------------------------------------------

Na_genes <- c('Scn1a', 'Scn3a', 'Scn8a')
Kv31_gene <- c('Kcnc1')
Kv2_genes <- c('Kcnb1', 'Kcnb2')
KT_genes <- c('Kcnd1', 'Kcnd2','Kcnd3', 'Kcnab1', 'Kcnip1', 'Kcnip2')
SK_genes <- c('Kcnn1', 'Kcnn2', 'Kcnn3', 'Kcnn4')
KP_genes <- c('Kcna1','Kcna2','Kcna3','Kcna4','Kcna5','Kcna6',
              'Kcna7','Kcna10')
M_genes <- c('Kcnq1', 'Kcnq2', 'Kcnq3', 'Kcnq4')

K_genes <- c(Kv31_gene, Kv2_genes, KT_genes, SK_genes, KP_genes, M_genes)

CaLV_genes <- c('Cacna1g', 'Cacna1h', 'Cacna1i')
CaHV_genes <- c('Cacna1s', 'Cacna1c', 'Cacna1d', 'Cacna1f', 'Cacna1a', 'Cacna1b')
Ca_genes <- c(CaLV_genes, CaHV_genes)

H_genes <- c('Hcn1', 'Hcn2', 'Hcn3')

channel_genes <- c(Na_genes, K_genes, Ca_genes, H_genes)


# Generalized genes -------------------------------------------------------


marker_genes <- list()
numGenes <- 5
for (subclass_1 in subclassTypes){
    for (subclass_2 in subclassTypes){
        if (subclass_1 != subclass_2){
            sub_1 <- str_replace_all(subclass_1, "(\\s)|(/)", "")
            sub_2 <- str_replace_all(subclass_2, "(\\s)|(/)", "")
            DEGenes_sub1 <- markerGenesDE %>% filter((cl1 == sub_1) & (cl2 == sub_2) & (lfc > 0)) %>% 
                arrange(desc(lfc)) %>% slice(1:numGenes) %>% pull(gene) 
            DEGenes_sub2 <- markerGenesDE %>% filter((cl1 == sub_1) & (cl2 == sub_2) & (lfc < 0)) %>% 
                arrange(lfc) %>% slice(1:numGenes) %>% pull(gene) 
            print(sub_1)
            print(sub_2)
            print("%%%%%%%%%%%%%")
            print(DEGenes_sub1)
            print(DEGenes_sub2)
            print("%%%%%%%%%%%%%")
            marker_genes[[sub_1]] <- unique(append(marker_genes[[sub_1]], DEGenes_sub1))
            marker_genes[[sub_2]] <- unique(append(marker_genes[[sub_2]], DEGenes_sub2))
        }
        
    }
}

marker_genes_list <- unlist(marker_genes[names(marker_genes)], use.names = FALSE) %>% unique()

channelGenesDE <- channelGenesDE %>% filter(cl1 %in% subclassTypes & cl2 %in% subclassTypes) 

# Model Predicted Genes ---------------------------------------------------

predicted_genes <- c('Scn8a', Kv31_gene, KT_genes, H_genes)
select_genes <- unique(c(marker_genes_list, channel_genes))
anno <- read_feather(annoDataPath)

# Remove ALM clusters
anno <- anno %>% filter(!grepl('ALM', primary_cluster_label))

if (file.exists(expressionFile)){
    counts <- read.csv(expressionFile)
    
} else {
    # counts <- getFACSData(countsDataPath, anno, expressionFile, regexMatch, select_genes)
    counts <- getFACSData(countsDataPath, anno, expressionFile, subclassTypes, select_genes)
}

# Filter annotation data for the same t-types and subclass labels
anno <- anno %>% rename(sample_name = sample_id) %>% filter(primary_cluster_id > 0)
# anno <- anno %>% filter(grepl(regexMatch, primary_cluster_label)) 
anno <- anno %>% filter(subclass_label %in% subclassTypes)
group_order <- c() # Order in which the groups appear
for (subclass_ in subclassTypes){
    subclass_ID <- anno %>% filter(subclass_label == subclass_) %>% slice(1) %>% pull(subclass_id)
    group_order <- append(group_order,subclass_ID[1])
}

counts <- counts %>% rename(sample_name = sample_id) 

p <- sample_heatmap_plot(counts, 
                         anno, 
                         genes = select_genes,
                         grouping = "subclass",
                         group_order = group_order,
                         log_scale = TRUE,
                         font_size = 16,
                         label_height = 6,
                         max_width = 10,
                         colorset=diverge_hcl(8))


ggsave(file=file.path('figures','heatmap_all.png'), p, width=16, height=22)

p1 <- sample_heatmap_plot(counts, 
                          anno, 
                          genes = channel_genes,
                          grouping = "subclass",
                          group_order = group_order,
                          log_scale = TRUE,
                          font_size = 20,
                          label_height = 12,
                          max_width = 10,
                          colorset=diverge_hcl(8))


ggsave(file=file.path('figures','heatmap_channel_genes.png'), p1, width=14, height=12)


# Pairwise comparisons between channel genes

# Post-hoc one-way manova: https://www.datanovia.com/en/lessons/one-way-manova-in-r/

pwc <- counts %>% 
    left_join(anno %>% select(sample_name, subclass_label),by='sample_name') %>% 
    select(c(sample_name, channel_genes, subclass_label)) %>%
    mutate_at("subclass_label", factor) %>% head(100) %>%
    gather(key = "genes", value = "cpm", all_of(channel_genes)) %>%
    group_by(genes) %>% 
    tukey_hsd(cpm ~ subclass_label) %>%
    select(-estimate, -conf.low, -conf.high) %>%
    filter(!p.adj.signif %in% c('ns', '?'))

diff_genes <- pwc %>% pull(genes) %>% unique()

undetected_genes <- setdiff(union(predicted_genes, diff_genes), predicted_genes)

Dgenes <- c(predicted_genes, undetected_genes)

p2 <- sample_heatmap_plot(counts, 
                          anno, 
                          genes = Dgenes,
                          grouping = "subclass",
                          group_order = group_order,
                          log_scale = TRUE,
                          font_size = 15,
                          label_height = 20,
                          max_width = 10,
                          colorset=diverge_hcl(8))


ggsave(file=file.path('figures','heatmap_predicted_genes.png'), p2, width=10, height=6)


