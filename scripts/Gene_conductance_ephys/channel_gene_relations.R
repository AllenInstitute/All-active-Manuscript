library(tidyverse)
library(formattable)
library(ggplot2)

# Setting working directory -----------------------------------------------

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)


# Export formattable table ------------------------------------------------

library(htmltools)
library(webshot)    

# This function is taken from https://github.com/renkun-ken/formattable/issues/26

export_formattable <- function(f, file, width = "100%", height = NULL, 
                               background = "white", delay = 0.2)
{
  w <- as.htmlwidget(f, width = width, height = height)
  path <- html_print(w, background = background, viewer = NULL)
  url <- paste0("file:///", gsub("\\\\", "/", normalizePath(path)))
  webshot(url,
          file = file,
          selector = ".formattable_widget",
          delay = delay)
}

# Define the channel gene correlates --------------------------------------

channel_gene_table <- tibble(
  Channel = c('Kv3.1','KT', 'KP','Ih'),
  gene = c('Kcnc1', 'Kcnd1-3', 'Kcna1-3,6','Hcn1-3'),
)
channel_gene_table <- channel_gene_table %>% add_column(present = 'Yes')
channel_gene_table <- channel_gene_table %>% 
                  pivot_wider(names_from=gene,values_from = present,
                              values_fill = list(present='No'))


# Formatting options ------------------------------------------------------

presence_formatter <- formatter(
  "span",
  style=x ~ style(color = ifelse(x == 'Yes' , "green", "red")),
  x~icontext(ifelse(x == 'Yes', "ok", "remove")))

highlight_formatter <- formatter("span",
                           display = "block",
                           style = x ~ style("background-color" = "lightblue"))

# Draw the table ----------------------------------------------------------

t <- formattable(channel_gene_table,
                 align =c("l","c","c","c","c"),
                list(
                  'Kcnc1' = presence_formatter,
                  'Kcnd1-3' = presence_formatter,
                  'Kcna1-3,6' = presence_formatter,
                  'Hcn1-3' = presence_formatter)
                )
print(t)
export_formattable(t,"channel_gene_relation.pdf")


