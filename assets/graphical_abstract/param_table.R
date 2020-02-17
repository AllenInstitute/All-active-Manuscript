library(tidyverse)
library(formattable)
library(docxtractr)
library(DT)
library(kableExtra)

options(DT.options = list(pageLength = 20))


# Setting working directory -----------------------------------------------

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)


# Extract table from .docx ------------------------------------------------


doc <- read_docx('Parameters.docx')
param_df <- docx_extract_tbl(doc, 1)
param_df <- param_df %>% select(-c(Modfile,Unit))

# Formatting options ------------------------------------------------------
html_tag <- "td"
apical_format <- formatter(html_tag, style = x ~ style("background-color:lightpink"))
presence_formatter <- formatter(
  "span",
  style=x ~ style(color = ifelse(x == '-' , "red", "green")),
  x~icontext(ifelse(x == '-', "remove", "ok")))

# Draw the table ----------------------------------------------------------

t <- formattable(param_df,
       align =c("l",rep('c', ncol(param_df)-1)),
       list(
         'Apical' = presence_formatter,
         'Basal' = presence_formatter,
         'Soma' = presence_formatter,
         'Axon' = presence_formatter
       ))


# Using the DT package ----------------------------------------------------

t_DT <- datatable(param_df) %>% formatStyle(
  'Apical',
  backgroundColor = styleEqual(param_df$Apical, 
                     rep('lightpink',nrow(param_df)))
)


# Using the kable package -------------------------------------------------

t_kable <- kable(param_df,table.attr = "style = \"color: black;\"") %>%
  kable_styling(full_width = F) %>%
  column_spec(2, background = "lightpink") %>%
  column_spec(c(1, 3:ncol(param_df)),background = 'white')

print(t)

