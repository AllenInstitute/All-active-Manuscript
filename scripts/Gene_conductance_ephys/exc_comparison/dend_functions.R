add_label_attr <- function(dend) {
  labelDend(dend)[[1]]
}

labelDend <- function(dend,n=1)
{
  if(is.null(attr(dend,"label"))){
    attr(dend, "label") =paste0("n",n)
    n= n +1
  }
  if(length(dend)>1){
    for(i in 1:length(dend)){
      tmp = labelDend(dend[[i]], n)
      dend[[i]] = tmp[[1]]
      n = tmp[[2]]
    }
  }
  return(list(dend, n))
}

add_parent_attr <- function(dend) {
  
  parent_name <- get_nodes_attr(dend,"label")[1]
  for(i in 1:length(dend)) {
    attr(dend[[i]], "parent") <- parent_name    
  }
  
  if(length(dend) > 1) {
    for(j in 1:length(dend)) {
      tmp <- add_parent_attr(dend[[j]])
      dend[[j]] <- tmp
    }
  }
  
  return(dend)
  
}

compute_node_stats <- function(dend, 
                               data,
                               cols_are = "gene_name",
                               anno, 
                               sample_col,
                               group_col, 
                               stat = c("median",
                                        "q25",
                                        "q75",
                                        "mean",
                                        "tmean",
                                        "prop_gt0",
                                        "prop_gt1",
                                        "all")) {
  
  library(matrixStats)
  
  print("Setting up.")
  
  dend_leaves <- partition_leaves(dend)
  dend_labels <- get_nodes_attr(dend, "label")
  
  if(grepl("gene",cols_are)) {
    data <- t(data)
  }
  empty <- matrix(0,
                  ncol = length(dend_labels),
                  nrow = nrow(data))
  rownames(empty) <- rownames(data)
  colnames(empty) <- dend_labels
  
  out <- list()
  
  if(stat %in% c("median","all")) {
    out$node_median <- empty
  }
  if(stat %in% c("q25","all")) {
    out$node_q25 <- empty
  }
  if(stat %in% c("q75","all")) {
    out$node_q75 <- empty
  }
  if(stat %in% c("mean","all")) {
    out$node_mean <- empty
  }
  if(stat %in% c("tmean","all")) {
    out$node_tmean <- empty
  }
  if(stat %in% c("prop_gt0","all")) {
    out$node_prop_gt0 <- empty
  }
  if(stat %in% c("prop_gt1","all")) {
    out$node_prop_gt1 <- empty
  }
  
  for(i in seq_along(dend_leaves)) {
    print(paste("Subsetting data for",dend_labels[i]))
    leaves <- dend_leaves[[i]]
    samples <- anno[[sample_col]][anno[[group_col]] %in% leaves]
 
    node_data <- data[,samples]
    node_data <- t(node_data)
    
    
    if(stat %in% c("median","all")) {
      print(paste("Computing","medians","for",dend_labels[i]))
      
      out$node_median[,i] <- apply(node_data, 2, median)
    }
    
    if("q25" %in% stat & "q75" %in% stat | "all" %in% stat) {
      print(paste("Computing","quantiles","for",dend_labels[i]))
      
      q <- apply(node_data, 2, quantile)
      out$node_q25[,i] <- q[2,]
      out$node_q75[,i] <- q[4,]
    } else if("q25" %in% stat) {
      print(paste("Computing","q25","for",dend_labels[i]))
      
      out$node_q25[,i] <- apply(node_data, 2, quantile)[2,]
    } else if("q75" %in% stat) {
      print(paste("Computing","q75","for",dend_labels[i]))
      
      out$node_q75[,i] <- apply(node_data, 2, quantile)[4,]
    }
    
    if(stat %in% c("mean","all")) {
      print(paste("Computing","means","for",dend_labels[i]))
      
      out$node_mean[,i] <- apply(node_data, 2, function(x) mean(x, na.rm = TRUE))
    }
    if(stat %in% c("tmean","all")) {
      print(paste("Computing","trimmed means","for",dend_labels[i]))
      
      out$node_tmean[,i] <- apply(node_data, 2, function(x) mean(x, trim = 0.25, na.rm = TRUE))
    }
    if(stat %in% c("prop_gt0","all")) {
      print(paste("Computing","prop > 0","for",dend_labels[i]))
      
      out$node_prop_gt0[,i] <- apply(node_data, 2, function(x) sum(x > 0)/length(x))
    }
    if(stat %in% c("prop_gt1","all")) {
      print(paste("Computing","prop > 1","for",dend_labels[i]))
      
      out$node_prop_gt1[,i] <- apply(node_data, 2, function(x) sum(x > 1)/length(x))
    }
    
  }
  
  return(out)
  
}