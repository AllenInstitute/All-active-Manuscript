library(feather)

# RNA-seq data path
data_loc = '/data/rnaseqanalysis/shiny/facs_seq/Mm_VISp_14236_20180912'

# Annotation data
anno_feather_path <- file.path(data_loc,'anno.feather')
anno_feather_data <- read_feather(anno_feather_path)

anno_feather_cols <- names(anno_feather_data)
unique_cre_labels <- sort(unique(anno_feather_data$cre_label))
select_cre_labels <- c("Htr3a-Cre_NO152","Sst-IRES-Cre","Pvalb-IRES-Cre")
select_samples_data <- anno_feather_data[anno_feather_data$cre_label %in% 
                                      select_cre_labels,c("sample_id","cre_label")]

# Gene count data
counts_data_path <- file.path(data_loc,'data.feather')
counts_data <- read_feather(counts_data_path)

unique_genes <- sort(unique(names(counts_data)))
 

# Kv3.1 gene - Kcnc1; 
select_genes <- c('Kcnc1','Kcnd1','Kcnd2','Kcnd3',
                 'Kcna1','Kcna2','Kcna3','Kcna4','Kcna5','Kcna6',
                 'Kcna7','Kcna10')

select_counts_data <- counts_data[counts_data$sample_id %in% select_samples_data$sample_id,]

# Normalize
select_counts_data[,which(names(select_counts_data) != 'sample_id')] <- 
  t(apply(select_counts_data[,which(names(select_counts_data) != 'sample_id')], 1,
        function(x)(log2(x+1))))

cre_genes <- c('Pvalb','Sst','Htr3a')
channel_genes <- c('Kcn','Scn','Nav','Cacna','Hcn')
hits_channel <- unique(grep(paste(channel_genes,collapse="|"),
                        unique_genes, value=TRUE))
ref_genes <- hits_channel[which(!hits_channel %in% select_genes)]
all_channel_genes <- c(cre_genes,ref_genes,select_genes)
all_channel_expr_data <- select_counts_data[,c('sample_id',all_channel_genes)]


all_gene_expr <- merge(select_samples_data,all_channel_expr_data,by="sample_id")
write.csv(all_gene_expr, file = "inh_expression_all.csv",row.names = FALSE)



