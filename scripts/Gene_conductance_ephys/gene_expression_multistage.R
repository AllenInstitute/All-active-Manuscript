library(scrattch.vis)
library(feather)
library(tidyverse)
library(colorspace)
options(stringsAsFactors = F)


# Setting working directory -----------------------------------------------

# this.dir <- dirname(parent.frame(2)$ofile)
# setwd(this.dir)


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
annoDataPath <- file.path(dataLoc,'anno.feather')

# Gene count data
countsDataPath <- file.path(dataLoc,'data.feather')

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

inhSubclass = c('Lamp5', 'Sncg', 'Vip', 'Sst', 'Pvalb')
excSubclass = c('L2/3 IT', 'L4', 'L5 PT', 'L5 IT', 'NP', 'L6 CT', 'L6 IT', 'L6b')

subclassTypes = c(inhSubclass, excSubclass)

# Genes encoding channels --------------------------------------------------------

Na_genes <- c('Scn1a', 'Scn3a', 'Scn4a', 
            'Scn5a', 'Scn8a', 'Scn9a', 'Scn10a', 'Scn11a')
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

marker_genes <- c("Fezf2", "Cux2", "Rorb", "Rspo1", "Col25a1", "Tmem215", "Cntnap5a",
"Htr1f", "Pid1", "Phactr2", "Gpr88", "Zfhx4", "Sgcd", "Pld5", "Whrn", "Thsd7a", "Grik1", 
"Cdh6", "Zeb1", "Sorcs2", "Scnn1a", "Endou", "Cpne4", "Kcnf1", "Ptprf", "Egfem1", "Necab1",
"Pak6", "Cdkn1a", "Tox", "Plekha7", "Fxyd6", "Robo1", "Rmst", "Gpr83", "Tenm3", "Kcnab3", "Ntn5")

# Model Predicted Genes ---------------------------------------------------

predicted_genes <- c(Kv31_gene, KT_genes, H_genes)
select_genes <- unique(c(marker_genes, channel_genes))
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

counts <- counts %>% rename(sample_name = sample_id) 

p <- sample_heatmap_plot(counts, 
                         anno, 
                         genes = select_genes,
                         grouping = "subclass",
                         log_scale = TRUE,
                         font_size = 20,
                         label_height = 10,
                         max_width = 10,
                         colorset=diverge_hsv(8))


ggsave(file=file.path('figures','heatmap_all.png'), p, width=18, height=20)

p1 <- sample_heatmap_plot(counts, 
                         anno, 
                         genes = channel_genes,
                         grouping = "subclass",
                         log_scale = TRUE,
                         font_size = 20,
                         label_height = 10,
                         max_width = 10,
                         colorset=diverge_hsv(8))


ggsave(file=file.path('figures','heatmap_channel_genes.png'), p1, width=12, height=14)

p2 <- sample_heatmap_plot(counts, 
                         anno, 
                         genes = predicted_genes,
                         grouping = "subclass",
                         log_scale = TRUE,
                         font_size = 15,
                         label_height = 20,
                         max_width = 10,
                         colorset=diverge_hsv(8))


ggsave(file=file.path('figures','heatmap_predicted_genes.png'), p2, width=8, height=6)