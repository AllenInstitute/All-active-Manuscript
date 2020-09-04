library(feather)

# RNA-seq data path
data_loc = '/data/rnaseqanalysis/shiny/facs_seq/Mm_VISp_14236_20180912'

# Project path
project_path <- file.path(getwd(),'..','..','..','assets','aggregated_data')

# Annotation data
anno_feather_path <- file.path(data_loc,'anno.feather')
anno_feather_data <- read_feather(anno_feather_path)

anno_feather_cols <- names(anno_feather_data)
unique_cre_labels <- sort(unique(anno_feather_data$cre_label))

# Gene count data
counts_data_path <- file.path(data_loc,'data.feather')
counts_data <- read_feather(counts_data_path)
unique_genes <- sort(unique(names(counts_data)))

# Exc lines
select_exc_cre_labels <- c("Nr5a1-Cre","Rbp4-Cre_KL100", "Scnn1a-Tg2-Cre", 
                            "Scnn1a-Tg3-Cre", "Rorb-IRES2-Cre")
select_exc_samples_data <- anno_feather_data[anno_feather_data$cre_label %in% 
                         select_exc_cre_labels,c("sample_id","cre_label")]

# Get the gene count data for selected exc lines
select_exc_counts_data <- counts_data[counts_data$sample_id 
                                      %in% select_exc_samples_data$sample_id,]

# Normalize
select_exc_counts_data[,which(names(select_exc_counts_data) != 'sample_id')] <- 
            t(apply(select_exc_counts_data[,which(names(select_exc_counts_data) != 
                    'sample_id')], 1, function(x)(log2(x+1))))


channel_genes <- c('Kcnc1','Kcnd1','Kcnd2','Kcnd3','Kcnab1', 'Kcnip1', 'Kcnip2', 
                        'Hcn1', 'Hcn2', 'Cacna1g', 'Scn8a')
channel_hits <- unique(grep(paste(channel_genes,collapse="|"),
                            unique_genes, value=TRUE))


#ref_genes <- hits_channel[which(!hits_channel %in% hits_exc)]
all_channel_expr_data <- select_exc_counts_data[,c('sample_id',channel_genes)]

# Merge with annotation data to get the Cre-line info
all_gene_expr <- merge(select_exc_samples_data,all_channel_expr_data,
                       by="sample_id")
write.csv(all_gene_expr, file = file.path(project_path, "l23_vs_l5_expression.csv"),
                    row.names = FALSE)



